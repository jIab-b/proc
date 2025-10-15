#!/usr/bin/env python3
"""
FastAPI backend server for WebGPU Minecraft Editor with fal.ai texture generation.
"""
import base64
import os
import json
import shutil
import random
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image, ImageDraw
import httpx

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
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=[
        "Content-Length",
        "Content-Type",
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


class ImageSize(BaseModel):
    width: int
    height: int


class Intrinsics(BaseModel):
    fovYDegrees: float
    aspect: float
    near: float
    far: float


class ViewCapture(BaseModel):
    id: str
    index: int
    capturedAt: str
    position: List[float]
    forward: List[float]
    up: List[float]
    right: List[float]
    intrinsics: Intrinsics
    viewMatrix: List[float]
    projectionMatrix: List[float]
    viewProjectionMatrix: List[float]
    rgbBase64: str
    depthBase64: Optional[str] = None
    normalBase64: Optional[str] = None


class DatasetUpload(BaseModel):
    formatVersion: str
    exportedAt: str
    imageSize: ImageSize
    viewCount: int
    captureId: str
    views: List[ViewCapture]


TEXTURES_DIR = Path("textures")
METADATA_FILE = TEXTURES_DIR / "metadata.json"
UPLOAD_DIR = TEXTURES_DIR / ".up"
DOWNLOAD_DIR = TEXTURES_DIR / ".down"
DATASETS_DIR = (Path.cwd() / "datasets").resolve()
DATASET_REGISTRY_FILE = DATASETS_DIR / "registry.json"
LOGS_DIR = (Path.cwd() / "logs").resolve()

LLM_PROVIDER_DEFAULT = os.environ.get("LLM_PROVIDER", "anthropic").lower()
ANTHROPIC_MODEL_DEFAULT = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
OPENAI_MODEL_DEFAULT = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

ATLAS_ROWS = 4
ATLAS_COLS = 3
ATLAS_TILE = 64
ATLAS_WIDTH = ATLAS_COLS * ATLAS_TILE  # 192px
ATLAS_HEIGHT = ATLAS_ROWS * ATLAS_TILE  # 256px

FACE_TILE_COORDINATES = {
    "top": (1, 1),
    "bottom": (1, 3),
    "north": (1, 0),
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

DATASETS_DIR.mkdir(exist_ok=True)
if not DATASET_REGISTRY_FILE.exists():
    with open(DATASET_REGISTRY_FILE, "w") as f:
        json.dump({"next_sequence": 1, "datasets": []}, f, indent=2)

LOGS_DIR.mkdir(exist_ok=True)
LOG_SUFFIX = random.getrandbits(4)
LOG_FILE = LOGS_DIR / f"log_{LOG_SUFFIX}.jsonl"


class ReconstructionRequest(BaseModel):
    prompt: str
    datasetSequence: Optional[int] = None
    metadataFile: Optional[str] = None
    captureId: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None


class ReconstructionResponse(BaseModel):
    provider: str
    model: str
    output: str
    tokens: Optional[Dict[str, Any]] = None


class DSLLogEntry(BaseModel):
    action: str
    params: Dict[str, Any]
    result: Dict[str, Any]
    source: Optional[str] = None
    timestamp: Optional[str] = None

# Custom static file handler with CORS headers for texture tiles
@app.options("/textures/{file_path:path}")
async def options_texture_file(file_path: str):
    """Handle CORS preflight for texture files"""
    return Response(
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "86400"
        }
    )

@app.get("/textures/{file_path:path}")
async def get_texture_file(file_path: str):
    """Serve texture files with proper CORS headers"""
    file_path = file_path.lstrip('/')
    full_path = TEXTURES_DIR / file_path

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Texture file not found")

    if not full_path.is_file():
        raise HTTPException(status_code=404, detail="Not a file")

    # Check if it's a PNG file
    if full_path.suffix.lower() != '.png':
        raise HTTPException(status_code=400, detail="Only PNG files are supported")

    return FileResponse(
        full_path,
        media_type="image/png",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


def load_metadata():
    """Load texture metadata from JSON file"""
    with open(METADATA_FILE, "r") as f:
        return json.load(f)


def save_metadata(metadata):
    """Save texture metadata to JSON file"""
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)


def load_dataset_registry():
    """Load dataset registry from JSON file"""
    try:
        with open(DATASET_REGISTRY_FILE, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"next_sequence": 1, "datasets": []}
    if "next_sequence" not in data:
        data["next_sequence"] = 1
    if "datasets" not in data or not isinstance(data["datasets"], list):
        data["datasets"] = []
    return data


def save_dataset_registry(registry):
    """Persist dataset registry"""
    with open(DATASET_REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def load_captions(dataset_dir: Path) -> List[Dict[str, Any]]:
    """Load optional captions generated for a dataset."""
    captions_path = dataset_dir / "captions.json"
    if not captions_path.exists():
        return []
    try:
        with open(captions_path, "r") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data
    except Exception as exc:  # noqa: BLE001
        print(f"[LLM] Failed to load captions: {exc}")
    return []


def build_dataset_summary(metadata: Dict[str, Any], captions: List[Dict[str, Any]]) -> str:
    """Create a concise textual summary of dataset metadata and optional captions."""
    lines: List[str] = []
    capture_id = metadata.get("captureId", "unknown")
    lines.append(f"capture_id: {capture_id}")
    lines.append(f"exported_at: {metadata.get('exportedAt')}")
    image_size = metadata.get("imageSize", {})
    lines.append(
        f"image_size: {image_size.get('width')}x{image_size.get('height')}"
    )
    lines.append(f"view_count: {metadata.get('viewCount')}")

    views = metadata.get("views", [])
    for view in views[:5]:
        pos = view.get("position", [])
        forward = view.get("forward", [])
        pos_str = tuple(round(v, 3) for v in pos)
        forward_str = tuple(round(v, 3) for v in forward)
        lines.append(
            f"view #{view.get('index')} id={view.get('id')} pos={pos_str} forward={forward_str}"
        )

    if len(views) > 5:
        lines.append(f"... {len(views) - 5} additional views omitted ...")

    if captions:
        lines.append("captions:")
        for entry in captions[:5]:
            caption_text = entry.get("caption", "")
            image_name = entry.get("image", "unknown")
            lines.append(f"- {image_name}: {caption_text}")
    if len(captions) > 5:
        lines.append(f"... {len(captions) - 5} additional captions omitted ...")

    return "\n".join(lines)


async def call_llm_for_reconstruction(
    summary: str,
    user_prompt: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> ReconstructionResponse:
    """Dispatch a reconstruction request to Anthropic or OpenAI."""
    provider_choice = (provider or LLM_PROVIDER_DEFAULT).lower()
    if provider_choice not in {"anthropic", "openai"}:
        raise HTTPException(status_code=400, detail="Unsupported LLM provider")

    system_prompt = (
        "You are an expert procedural voxel world designer. "
        "Given dataset metadata from captured camera views, produce a plan to reconstruct "
        "the described area using the available block palette and DSL commands (place_block/remove_block)."
    )
    user_message = (
        "DATASET SUMMARY:\n" + summary + "\n\n" + "USER REQUEST:\n" + user_prompt.strip()
    )

    if provider_choice == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not configured")
        target_model = model or ANTHROPIC_MODEL_DEFAULT
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": target_model,
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": user_message}]}
            ],
            "temperature": 0.2,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages", json=payload, headers=headers
            )
        if response.status_code >= 400:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        data = response.json()
        content = data.get("content", [])
        text_segments = [item.get("text", "") for item in content if item.get("type") == "text"]
        output_text = "\n".join(segment.strip() for segment in text_segments if segment)
        return ReconstructionResponse(
            provider="anthropic",
            model=target_model,
            output=output_text or "",
            tokens=data.get("usage"),
        )

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured")
    target_model = model or OPENAI_MODEL_DEFAULT
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": target_model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 1024,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions", json=payload, headers=headers
        )
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    data = response.json()
    choices = data.get("choices", [])
    output_text = ""
    if choices:
        output_text = choices[0].get("message", {}).get("content", "")
    return ReconstructionResponse(
        provider="openai",
        model=target_model,
        output=output_text or "",
        tokens=data.get("usage"),
    )


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


def discover_existing_sequences(metadata: dict) -> set[int]:
    """Return all sequence numbers referenced in metadata or on disk."""
    sequences: set[int] = set()

    for texture in metadata.get("textures", []):
        seq = texture.get("sequence")
        if isinstance(seq, int) and seq > 0:
            sequences.add(seq)

    for entry in TEXTURES_DIR.iterdir():
        if entry.is_dir() and entry.name.isdigit():
            sequences.add(int(entry.name))

    return sequences


def next_available_sequence(metadata: dict) -> int:
    """Compute the next unused sequence id, considering disk state."""
    used = discover_existing_sequences(metadata)
    candidate = max(metadata.get("next_sequence", 1), 1)
    while candidate in used:
        candidate += 1
    return candidate


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
    """Extract configured face tiles from the atlas into individual files with precise 64x64 pixel boundaries."""
    sequence_dir = TEXTURES_DIR / str(sequence)
    sequence_dir.mkdir(parents=True, exist_ok=True)

    face_tiles: dict[str, dict] = {}

    # Verify atlas dimensions
    if atlas_image.width != ATLAS_WIDTH or atlas_image.height != ATLAS_HEIGHT:
        print(f"[API] Warning: Atlas dimensions {atlas_image.width}x{atlas_image.height} != expected {ATLAS_WIDTH}x{ATLAS_HEIGHT}")

    print(f"[API] Extracting tiles with precise {ATLAS_TILE}x{ATLAS_TILE} pixel boundaries")

    for layer, face in enumerate(FACE_TILE_ORDER):
        col, row = FACE_TILE_COORDINATES[face]

        # Calculate exact pixel coordinates
        left = col * ATLAS_TILE
        upper = row * ATLAS_TILE
        right = left + ATLAS_TILE
        lower = upper + ATLAS_TILE

        # Ensure coordinates are within image bounds
        if right > atlas_image.width or lower > atlas_image.height:
            print(f"[API] Warning: Tile coordinates ({left},{upper},{right},{lower}) exceed atlas bounds")
            continue

        print(f"[API] Extracting {face} tile: col={col}, row={row}, pixels=({left},{upper} to {right-1},{lower-1})")

        # Extract tile with exact 64x64 pixel boundaries
        tile = atlas_image.crop((left, upper, right, lower))

        # Verify tile dimensions
        if tile.width != ATLAS_TILE or tile.height != ATLAS_TILE:
            print(f"[API] Error: Extracted tile size {tile.width}x{tile.height} != {ATLAS_TILE}x{ATLAS_TILE}")
            continue

        filename = f"{col}_{row}.png"
        filepath = sequence_dir / filename

        # Save as PNG without compression artifacts
        tile.save(filepath, "PNG", optimize=False)

        face_tiles[face] = {
            "col": col,
            "row": row,
            "filename": filename,
            "path": str(filepath.relative_to(TEXTURES_DIR)),
            "layer": layer
        }

        print(f"[API] Saved {face} tile to {filepath}")

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

    # Only save individual face tiles, not the redundant atlas file
    with Image.open(BytesIO(image_data)) as atlas_img:
        atlas_rgba = atlas_img.convert("RGBA")
    face_tiles = slice_face_tiles(atlas_rgba, sequence)
    atlas_rgba.close()

    texture_entry = {
        "id": texture_id,
        "filename": "",  # No longer saving atlas file
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
    metadata["next_sequence"] = next_available_sequence(metadata)

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


@app.options("/api/textures/{texture_id}")
async def options_get_texture(texture_id: int):
    """Handle CORS preflight for texture retrieval"""
    return Response(
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "86400"
        }
    )

@app.get("/api/textures/{texture_id}")
async def get_texture(texture_id: int):
    """Get a specific texture by ID"""
    metadata = load_metadata()
    texture = next((t for t in metadata["textures"] if t["id"] == texture_id), None)

    if not texture:
        raise HTTPException(status_code=404, detail="Texture not found")

    # Try to serve the download file (the generated atlas)
    if "download_file" in texture and texture["download_file"]:
        filepath = DOWNLOAD_DIR / texture["download_file"]
        if filepath.exists():
            return FileResponse(
                filepath,
                media_type="image/png",
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, OPTIONS",
                    "Access-Control-Allow-Headers": "*",
                    "Cache-Control": "no-cache, no-store, must-revalidate"
                }
            )

    # Fallback: if no download file, try to reconstruct from individual tiles
    if "sequence" in texture and texture["sequence"]:
        try:
            sequence_dir = TEXTURES_DIR / str(texture["sequence"])
            if sequence_dir.exists():
                # Create a new atlas from individual tiles
                atlas_image = Image.new("RGBA", (ATLAS_WIDTH, ATLAS_HEIGHT), color=(0, 0, 0, 0))

                for face, tile_info in texture["atlas"]["tiles"].items():
                    tile_path = sequence_dir / tile_info["filename"]
                    if tile_path.exists():
                        tile_img = Image.open(tile_path)
                        col, row = tile_info["col"], tile_info["row"]
                        atlas_image.paste(tile_img, (col * ATLAS_TILE, row * ATLAS_TILE))
                        tile_img.close()

                # Save to bytes and return
                img_bytes = BytesIO()
                atlas_image.save(img_bytes, format="PNG")
                atlas_image.close()
                img_bytes.seek(0)

                return Response(
                    content=img_bytes.getvalue(),
                    media_type="image/png",
                    headers={
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Methods": "GET, OPTIONS",
                        "Access-Control-Allow-Headers": "*",
                        "Cache-Control": "no-cache, no-store, must-revalidate"
                    }
                )
        except Exception as e:
            print(f"[API] Error reconstructing atlas for texture {texture_id}: {e}")

    raise HTTPException(status_code=404, detail="Texture file not found")


@app.delete("/api/textures/{texture_id}")
async def delete_texture(texture_id: int):
    """Delete a texture by ID and remove all associated files"""
    metadata = load_metadata()
    texture = next((t for t in metadata["textures"] if t["id"] == texture_id), None)

    if not texture:
        raise HTTPException(status_code=404, detail="Texture not found")

    try:
        # Delete main texture file
        texture_file = TEXTURES_DIR / texture["filename"]
        if texture_file.exists():
            texture_file.unlink()

        # Delete upload file if it exists
        if "upload_file" in texture:
            upload_file = UPLOAD_DIR / texture["upload_file"]
            if upload_file.exists():
                upload_file.unlink()

        # Delete download file if it exists
        if "download_file" in texture:
            download_file = DOWNLOAD_DIR / texture["download_file"]
            if download_file.exists():
                download_file.unlink()

        # Delete face tiles directory if it exists
        if "sequence" in texture and texture["sequence"]:
            sequence_dir = TEXTURES_DIR / str(texture["sequence"])
            if sequence_dir.exists() and sequence_dir.is_dir():
                import shutil
                shutil.rmtree(sequence_dir)

        # Remove from metadata
        metadata["textures"] = [t for t in metadata["textures"] if t["id"] != texture_id]
        save_metadata(metadata)

        print(f"[API] Deleted texture {texture_id}: {texture.get('prompt', 'Unknown')}")
        return {"message": "Texture deleted successfully"}

    except Exception as e:
        print(f"[API] Error deleting texture {texture_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to delete texture: {str(e)}")


@app.options("/api/generate-texture")
async def options_generate_texture():
    """Handle CORS preflight for texture generation"""
    return Response(
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "86400"
        }
    )

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
        "Create a precise 4-row by 3-column Minecraft texture atlas exactly 192x256 pixels. "
        "Each tile must be exactly 64x64 pixels with pixel-perfect boundaries. "
        "Tile coordinates: (0,0) to (63,63) = top-left, (64,0) to (127,63) = top-middle, (128,0) to (191,63) = top-right, "
        "incrementing by exactly 64 pixels horizontally and vertically. "
        "Row 0 (y:0-63): top faces. Row 1 (y:64-127): side faces. Row 2 (y:128-191): secondary side faces. Row 3 (y:192-255): bottom faces. "
        "Ensure perfect 64x64 pixel alignment for each tile. Focus on clean pixel art with sharp edges. "
        f"Style: blocks that match this description: {request.prompt}"
    )

    print(f"[API] Generating texture atlas: {full_prompt}")
    print(f"[API] Atlas dimensions: {ATLAS_WIDTH}x{ATLAS_HEIGHT}")

    try:
        metadata = load_metadata()
        sequence = next_available_sequence(metadata)

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
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "*",
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


@app.post("/api/export-dataset")
async def export_dataset(payload: DatasetUpload):
    if not payload.views:
        raise HTTPException(status_code=400, detail="No views supplied")

    try:
        registry = load_dataset_registry()
        existing_entry = next((d for d in registry.get("datasets", []) if d.get("captureId") == payload.captureId), None)

        if existing_entry:
            sequence = int(existing_entry.get("sequence"))
            dataset_dir = (DATASETS_DIR / str(sequence)).resolve()
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            registry["datasets"] = [d for d in registry.get("datasets", []) if d.get("captureId") != payload.captureId]
        else:
            sequence = int(registry.get("next_sequence", 1))
            dataset_dir = (DATASETS_DIR / str(sequence)).resolve()
            while dataset_dir.exists():
                sequence += 1
                dataset_dir = (DATASETS_DIR / str(sequence)).resolve()

        dataset_dir.mkdir(parents=True, exist_ok=True)
        images_dir = dataset_dir / "images"
        images_dir.mkdir(exist_ok=True)

        metadata_views = []
        for view in payload.views:
            try:
                image_bytes = base64.b64decode(view.rgbBase64)
            except Exception as err:
                raise HTTPException(status_code=400, detail=f"Invalid base64 for view {view.id}: {err}")

            image_filename = f"{view.index:03d}_{view.id}.png"
            image_path = images_dir / image_filename
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)

            metadata_views.append({
                "id": view.id,
                "index": view.index,
                "capturedAt": view.capturedAt,
                "position": view.position,
                "forward": view.forward,
                "up": view.up,
                "right": view.right,
                "intrinsics": view.intrinsics.dict(),
                "viewMatrix": view.viewMatrix,
                "projectionMatrix": view.projectionMatrix,
                "viewProjectionMatrix": view.viewProjectionMatrix,
                "rgbPath": str(image_path.relative_to(dataset_dir)),
                "depthPath": None,
                "normalPath": None
            })

        metadata = {
            "formatVersion": payload.formatVersion,
            "exportedAt": payload.exportedAt,
            "imageSize": payload.imageSize.dict(),
            "viewCount": payload.viewCount,
            "captureId": payload.captureId,
            "views": metadata_views
        }

        metadata_path = dataset_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        registry_entry = {
            "sequence": sequence,
            "exportedAt": payload.exportedAt,
            "imageCount": len(metadata_views),
            "metadata": str(metadata_path.relative_to(DATASETS_DIR)),
            "imagesDir": str(images_dir.relative_to(DATASETS_DIR)),
            "captureId": payload.captureId
        }
        datasets_list = registry.setdefault("datasets", [])
        datasets_list.append(registry_entry)
        registry["next_sequence"] = max(sequence + 1, int(registry.get("next_sequence", sequence + 1)))
        save_dataset_registry(registry)

        dataset_dir_rel = dataset_dir.relative_to(Path.cwd())
        metadata_rel = metadata_path.relative_to(Path.cwd())

        return {
            "status": "ok",
            "datasetSequence": sequence,
            "datasetDir": str(dataset_dir_rel),
            "metadataFile": str(metadata_rel),
            "imageCount": len(metadata_views),
            "captureId": payload.captureId
        }
    except HTTPException:
        raise
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Failed to export dataset: {err}") from err


@app.post("/api/reconstruct-dataset")
async def reconstruct_dataset(request: ReconstructionRequest):
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    dataset_dir: Optional[Path] = None
    metadata_path: Optional[Path] = None

    if request.datasetSequence is not None:
        dataset_dir = (DATASETS_DIR / str(request.datasetSequence)).resolve()
        metadata_path = dataset_dir / "metadata.json"
    elif request.metadataFile:
        metadata_path = (Path.cwd() / request.metadataFile).resolve()
        dataset_dir = metadata_path.parent
    else:
        raise HTTPException(status_code=400, detail="datasetSequence or metadataFile is required")

    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset directory not found")
    if not str(dataset_dir).startswith(str(DATASETS_DIR)):
        raise HTTPException(status_code=400, detail="Dataset path is outside allowed directory")

    if not metadata_path or not metadata_path.exists():
        metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="metadata.json not found for dataset")

    try:
        with open(metadata_path, "r") as fh:
            metadata = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to load metadata: {exc}") from exc

    captions = load_captions(dataset_dir)
    summary = build_dataset_summary(metadata, captions)

    llm_response = await call_llm_for_reconstruction(
        summary=summary,
        user_prompt=prompt,
        provider=request.provider,
        model=request.model,
    )

    dataset_sequence = request.datasetSequence
    if dataset_sequence is None:
        try:
            dataset_sequence = int(dataset_dir.name)
        except ValueError:
            dataset_sequence = None

    return {
        "status": "ok",
        "datasetSequence": dataset_sequence,
        "captureId": metadata.get("captureId"),
        "provider": llm_response.provider,
        "model": llm_response.model,
        "output": llm_response.output,
        "tokens": llm_response.tokens,
        "summary": summary,
    }


@app.post("/api/log-dsl")
async def log_dsl(entry: DSLLogEntry):
    """Append DSL action logs to session file."""
    record = {
        "timestamp": entry.timestamp or datetime.utcnow().isoformat(),
        "action": entry.action,
        "params": entry.params,
        "result": entry.result,
        "source": entry.source,
    }
    try:
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"[DSL] Failed to write log entry: {exc}")
        raise HTTPException(status_code=500, detail="Failed to write DSL log") from exc
    return {"status": "ok"}


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
