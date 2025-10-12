#!/usr/bin/env python3
"""
FastAPI backend server for WebGPU Minecraft Editor with fal.ai texture generation.
"""
import os
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel

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
from client_example import fal_text_to_image


app = FastAPI(title="WebGPU Minecraft Editor API")

# Enable CORS for Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TexturePrompt(BaseModel):
    prompt: str
    width: Optional[int] = 512
    height: Optional[int] = 512


TEXTURES_DIR = Path("textures")
METADATA_FILE = TEXTURES_DIR / "metadata.json"

# Ensure textures directory exists
TEXTURES_DIR.mkdir(exist_ok=True)
if not METADATA_FILE.exists():
    with open(METADATA_FILE, "w") as f:
        json.dump({"textures": [], "next_id": 1}, f, indent=2)


def load_metadata():
    """Load texture metadata from JSON file"""
    with open(METADATA_FILE, "r") as f:
        return json.load(f)


def save_metadata(metadata):
    """Save texture metadata to JSON file"""
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)


def save_texture(image_data: bytes, prompt: str, full_prompt: str) -> dict:
    """Save texture to disk and update metadata"""
    metadata = load_metadata()
    texture_id = metadata["next_id"]

    # Save image file
    filename = f"texture_{texture_id}.png"
    filepath = TEXTURES_DIR / filename

    with open(filepath, "wb") as f:
        f.write(image_data)

    # Update metadata
    texture_entry = {
        "id": texture_id,
        "filename": filename,
        "prompt": prompt,
        "full_prompt": full_prompt,
        "created_at": datetime.now().isoformat(),
        "size_bytes": len(image_data)
    }

    metadata["textures"].append(texture_entry)
    metadata["next_id"] = texture_id + 1

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

    # Prefix prompt with minecraft style instruction
    full_prompt = f"generate a minecraft style texture {request.prompt}"

    print(f"[API] Generating texture: {full_prompt}")
    print(f"[API] Dimensions: {request.width}x{request.height}")

    try:
        # Generate texture using fal.ai
        image_data = await fal_text_to_image(
            prompt=full_prompt,
            width=request.width,
            height=request.height
        )

        if not image_data:
            raise HTTPException(
                status_code=500,
                detail="Texture generation failed - no image data returned"
            )

        print(f"[API] Texture generated successfully ({len(image_data)} bytes)")

        # Save texture to disk
        texture_entry = save_texture(image_data, request.prompt, full_prompt)
        print(f"[API] Texture saved: {texture_entry['filename']} (ID: {texture_entry['id']})")

        # Return PNG image
        return Response(
            content=image_data,
            media_type="image/png",
            headers={
                "Content-Length": str(len(image_data)),
                "Cache-Control": "no-cache",
                "X-Texture-ID": str(texture_entry["id"]),
                "X-Texture-Prompt": request.prompt
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
