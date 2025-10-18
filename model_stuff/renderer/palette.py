from __future__ import annotations
import torch


def get_face_palette(c_m: torch.Tensor) -> torch.Tensor:
    """
    Return per-material, per-face RGB colors as a tensor shaped (M, 6, 3).

    Face order matches renderer.textures FACE_ORDER and code paths:
      [top, bottom, north(-z), south(+z), east(+x), west(-x)]

    Values mirror src/chunks.ts blockPalette so the Python renderer matches
    WebGPU colors when textures are absent. For materials without a specific
    top/bottom color, side/base color is reused.
    """
    # Base colors (side) come from c_m to keep parity with current materials.
    # We override top/bottom where WebGPU palette differs.
    M = int(c_m.size(0))
    pal = torch.zeros((M, 6, 3), dtype=c_m.dtype)

    # Indices follow materials.py
    AIR, GRASS, DIRT, STONE, PLANK, SNOW, SAND, WATER = 0, 1, 2, 3, 4, 5, 6, 7

    # Helper to set faces from (top, bottom, side)
    def set_tbs(mi: int, top: list[float], bottom: list[float], side: list[float]):
        # top, bottom, north, south, east, west
        pal[mi, 0] = torch.tensor(top, dtype=c_m.dtype)
        pal[mi, 1] = torch.tensor(bottom, dtype=c_m.dtype)
        pal[mi, 2] = torch.tensor(side, dtype=c_m.dtype)
        pal[mi, 3] = torch.tensor(side, dtype=c_m.dtype)
        pal[mi, 4] = torch.tensor(side, dtype=c_m.dtype)
        pal[mi, 5] = torch.tensor(side, dtype=c_m.dtype)

    # Air
    set_tbs(AIR, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    # Grass (src/chunks.ts)
    set_tbs(GRASS, [0.34, 0.68, 0.36], [0.40, 0.30, 0.16], [0.45, 0.58, 0.30])
    # Dirt
    set_tbs(DIRT, [0.42, 0.32, 0.20], [0.38, 0.26, 0.16], [0.40, 0.30, 0.18])
    # Stone
    set_tbs(STONE, [0.58, 0.60, 0.64], [0.55, 0.57, 0.60], [0.56, 0.58, 0.62])
    # Plank
    set_tbs(PLANK, [0.78, 0.68, 0.50], [0.72, 0.60, 0.42], [0.74, 0.63, 0.45])
    # Snow
    set_tbs(SNOW, [0.92, 0.94, 0.96], [0.90, 0.92, 0.94], [0.88, 0.90, 0.93])
    # Sand
    set_tbs(SAND, [0.88, 0.82, 0.60], [0.86, 0.78, 0.56], [0.87, 0.80, 0.58])
    # Water (keep side color; top/bottom slightly varied for visual parity if needed)
    set_tbs(WATER, [0.22, 0.40, 0.66], [0.20, 0.34, 0.60], [0.20, 0.38, 0.64])

    return pal

