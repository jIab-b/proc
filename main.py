#!/usr/bin/env python3
"""
FastAPI backend server for WebGPU Minecraft Editor with fal.ai texture generation.
"""
import os
import json
from io import BytesIO
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image, ImageDraw

# Load environment variables from .env file
def load_env():
    """Load environment variables from .env file"""
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        os.environ[key] = value

load_env()

# Import fal.ai client functions
from client_example import fal_generate_edit


app = FastAPI(title="WebGPU Minecraft Editor API")

# Enable CORS for Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Content-Length",
        "X-Texture-ID",
        "X-Texture-Prompt",
        "X-Atlas-Rows",
        "X-Atlas-Cols",
        "X-Atlas-Tile",
        "X-Atlas-Sequence"
    ],
)


class TexturePrompt(BaseModel):
    prompt: str
    width: Optional[int] = 512
    height: Optional[int] = 512


TEXTURES_DIR = Path("textures")
METADATA_FILE = TEXTURES_DIR / "metadata.json"
UPLOAD_DIR = TEXTURES_DIR / ".up"
DOWNLOAD_DIR = TEXTURES_DIR / ".down"

ATLAS_ROWS = 4
ATLAS_COLS = 3
ATLAS_TILE = 64
ATLAS_WIDTH = ATLAS_COLS * ATLAS_TILE  # 192px
ATLAS_HEIGHT = ATLAS_ROWS * ATLAS_TILE  # 256px

FACE_TILE_COORDINATES = {
    "top": (1, 0),
    "bottom": (1, 3),
    "north": (1, 1),
    "south": (1, 2),
    "east": (2, 1),
    "west": (0, 1),
}
FACE_TILE_ORDER = ["top", "bottom", "north", "south", "east", "west"]

# Ensure textures directory structure exists
TEXTURES_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
DOWNLOAD_DIR.mkdir(exist_ok=True)
if not METADATA_FILE.exists():
    with open(METADATA_FILE, "w") as f:
        json.dump({"textures": [], "next_id": 1, "next_sequence": 1}, f, indent=2)

app.mount("/textures", StaticFiles(directory=str(TEXTURES_DIR)), name="texture-files")


def load_metadata():
    """Load texture metadata from JSON file"""
    with open(METADATA_FILE, "r") as f:
        return json.load(f)


def save_metadata(metadata):
    """Save texture metadata to JSON file"""
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)


def compute_uv_rect(row: int, col: int) -> dict:
    """Return normalized UV rectangle for a given cell."""
    u0 = col / ATLAS_COLS
    v0 = row / ATLAS_ROWS
    u1 = (col + 1) / ATLAS_COLS
    v1 = (row + 1) / ATLAS_ROWS
    return {
        "row": row,
        "col": col,
        "u0": u0,
        "v0": v0,
        "u1": u1,
        "v1": v1
    }


def build_default_uv_map() -> dict:
    """Generate a UV map for each column in the atlas."""
    uv_map: dict[str, dict[str, dict]] = {}
    for col in range(ATLAS_COLS):
        uv_map[f"column_{col}"] = {
            "top": compute_uv_rect(0, col),
            "side": compute_uv_rect(1, col),
            "side_secondary": compute_uv_rect(2, col),
            "bottom": compute_uv_rect(3, col)
        }
    return uv_map


def generate_checkerboard(rows: int = ATLAS_ROWS, cols: int = ATLAS_COLS, tile: int = ATLAS_TILE) -> Image.Image:
    """Create a light translucent checkerboard atlas template."""
    width = cols * tile
    height = rows * tile
    img = Image.new("RGBA", (width, height), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    colors = [
        (250, 252, 255, 48),
        (227, 233, 240, 64)
    ]
    border = (255, 255, 255, 72)
    for row in range(rows):
        for col in range(cols):
            x0 = col * tile
            y0 = row * tile
            x1 = x0 + tile
            y1 = y0 + tile
            color = colors[(row + col) % 2]
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=border)
    return img


def slice_face_tiles(atlas_image: Image.Image, sequence: int) -> dict:
    """Extract configured face tiles from the atlas into individual files."""
    sequence_dir = TEXTURES_DIR / str(sequence)
    sequence_dir.mkdir(parents=True, exist_ok=True)

    face_tiles: dict[str, dict] = {}
    for layer, face in enumerate(FACE_TILE_ORDER):
        col, row = FACE_TILE_COORDINATES[face]
        left = col * ATLAS_TILE
        upper = row * ATLAS_TILE
        right = left + ATLAS_TILE
        lower = upper + ATLAS_TILE
        tile = atlas_image.crop((left, upper, right, lower))
        filename = f"{col}_{row}.png"
        filepath = sequence_dir / filename
        tile.save(filepath)
        face_tiles[face] = {
            "col": col,
            "row": row,
            "filename": filename,
            "path": str(filepath.relative_to(TEXTURES_DIR)),
            "layer": layer
        }

    return face_tiles


def save_texture(
    metadata: dict,
    image_data: bytes,
    prompt: str,
    full_prompt: str,
    sequence: int,
    upload_filename: str,
    download_filename: str,
    atlas_info: dict
) -> dict:
    """Persist generated texture, update metadata, and return entry."""
    texture_id = metadata.get("next_id", 1)

    filename = f"texture_{texture_id}.png"
    filepath = TEXTURES_DIR / filename

    with open(filepath, "wb") as f:
        f.write(image_data)

    with Image.open(BytesIO(image_data)) as atlas_img:
        atlas_rgba = atlas_img.convert("RGBA")
    face_tiles = slice_face_tiles(atlas_rgba, sequence)
    atlas_rgba.close()

    texture_entry = {
        "id": texture_id,
        "filename": filename,
        "prompt": prompt,
        "full_prompt": full_prompt,
        "created_at": datetime.now().isoformat(),
        "size_bytes": len(image_data),
        "sequence": sequence,
        "upload_file": upload_filename,
        "download_file": download_filename,
        "atlas": {**atlas_info, "tiles": face_tiles, "directory": str(sequence)}
    }

    metadata.setdefault("textures", []).append(texture_entry)
    metadata["next_id"] = texture_id + 1
    metadata["next_sequence"] = sequence + 1

    save_metadata(metadata)

    return texture_entry


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "WebGPU Minecraft Editor API"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/api/textures")
async def list_textures():
    """List all saved textures"""
    metadata = load_metadata()
    return metadata


@app.get("/api/textures/{texture_id}")
async def get_texture(texture_id: int):
    """Get a specific texture by ID"""
    metadata = load_metadata()
    texture = next((t for t in metadata["textures"] if t["id"] == texture_id), None)

    if not texture:
        raise HTTPException(status_code=404, detail="Texture not found")

    filepath = TEXTURES_DIR / texture["filename"]
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Texture file not found")

    return FileResponse(filepath, media_type="image/png")


@app.post("/api/generate-texture")
async def generate_texture(request: TexturePrompt):
    """
    Generate a texture using fal.ai nano-banana model and save it to disk.

    Args:
        request: TexturePrompt with prompt text and optional dimensions

    Returns:
        JSON with texture metadata and image data
    """
    # Check for API key
    api_key = os.environ.get("FAL_API_KEY") or os.environ.get("FAL_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="FAL_API_KEY not configured. Set FAL_API_KEY environment variable."
        )

    # Build structured prompt for grid-based atlas generation
    full_prompt = (
        "Fill each tile of this 4-row by 3-column 64x64 checkerboard texture atlas with "
        "consistent Minecraft-style block faces. Row 0 holds top faces, rows 1-2 hold side faces, "
        "and row 3 holds bottom faces. Focus on clean pixel art details, aligned edges, and blocks "
        f"that match this description: {request.prompt}"
    )

    print(f"[API] Generating texture atlas: {full_prompt}")
    print(f"[API] Atlas dimensions: {ATLAS_WIDTH}x{ATLAS_HEIGHT}")

    try:
        metadata = load_metadata()
        sequence = metadata.get("next_sequence", 1)

        upload_filename = f"{sequence:05d}_upload.png"
        download_filename = f"{sequence:05d}_download.png"
        upload_path = UPLOAD_DIR / upload_filename
        download_path = DOWNLOAD_DIR / download_filename

        checkerboard_image = generate_checkerboard()
        with BytesIO() as buf:
            checkerboard_image.save(buf, format="PNG")
            checkerboard_bytes = buf.getvalue()

        with open(upload_path, "wb") as f:
            f.write(checkerboard_bytes)

        image_data = await fal_generate_edit(
            png_grid=checkerboard_bytes,
            prompt=full_prompt,
            width=ATLAS_WIDTH,
            height=ATLAS_HEIGHT
        )

        if not image_data:
            raise HTTPException(
                status_code=500,
                detail="Texture generation failed - no image data returned"
            )

        with open(download_path, "wb") as f:
            f.write(image_data)

        atlas_metadata = {
            "rows": ATLAS_ROWS,
            "cols": ATLAS_COLS,
            "tile_size": ATLAS_TILE,
            "width": ATLAS_WIDTH,
            "height": ATLAS_HEIGHT,
            "uv_map": build_default_uv_map(),
            "sequence": sequence
        }

        texture_entry = save_texture(
            metadata,
            image_data,
            request.prompt,
            full_prompt,
            sequence,
            upload_filename,
            download_filename,
            atlas_metadata
        )
        print(f"[API] Texture atlas saved: {texture_entry['filename']} (ID: {texture_entry['id']})")

        # Return PNG image
        return Response(
            content=image_data,
            media_type="image/png",
            headers={
                "Content-Length": str(len(image_data)),
                "Cache-Control": "no-cache",
                "X-Texture-ID": str(texture_entry["id"]),
                "X-Texture-Prompt": request.prompt,
                "X-Atlas-Rows": str(ATLAS_ROWS),
                "X-Atlas-Cols": str(ATLAS_COLS),
                "X-Atlas-Tile": str(ATLAS_TILE),
                "X-Atlas-Sequence": str(sequence)
            }
        )

    except Exception as e:
        print(f"[API] Error generating texture: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Texture generation error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    # Check for FAL_API_KEY
    if not os.environ.get("FAL_API_KEY") and not os.environ.get("FAL_KEY"):
        print("\n" + "="*60)
        print("WARNING: FAL_API_KEY not found in environment!")
        print("Texture generation will not work without it.")
        print("Set it with: export FAL_API_KEY=your_key_here")
        print("="*60 + "\n")

    print("Starting FastAPI server on http://localhost:8000")
    print("API docs available at http://localhost:8000/docs")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
