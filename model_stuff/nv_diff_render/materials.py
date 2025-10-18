"""
Material definitions matching WebGPU block palette exactly.

References:
- src/chunks.ts:163-200 (blockPalette)
- src/blockUtils.ts (BlockType enum)
"""

import torch
from typing import List, Tuple


# Material names matching frontend BlockType enum
MATERIALS = ["Air", "Grass", "Dirt", "Stone", "Plank", "Snow", "Sand", "Water"]


class FaceIndex:
    """Face indices matching chunks.ts:22-29"""
    PX = 0  # East  (+X)
    NX = 1  # West  (-X)
    PY = 2  # Top   (+Y)
    NY = 3  # Bottom (-Y)
    PZ = 4  # South (+Z)
    NZ = 5  # North (-Z)


# Face key strings for texture lookup (chunks.ts:38-45)
FACE_KEYS = ['east', 'west', 'top', 'bottom', 'south', 'north']


# Complete face definitions matching chunks.ts:92-159
FACE_DEFS = [
    {  # FaceIndex.PX (East, +X)
        'normal': (1, 0, 0),
        'offset': (1, 0, 0),
        'corners': [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
        'uvs': [(0, 1), (0, 0), (1, 0), (1, 1)],
        'palette_slot': 2  # side
    },
    {  # FaceIndex.NX (West, -X)
        'normal': (-1, 0, 0),
        'offset': (-1, 0, 0),
        'corners': [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
        'uvs': [(1, 1), (0, 1), (0, 0), (1, 0)],
        'palette_slot': 2  # side
    },
    {  # FaceIndex.PY (Top, +Y)
        'normal': (0, 1, 0),
        'offset': (0, 1, 0),
        'corners': [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
        'uvs': [(0, 1), (0, 0), (1, 0), (1, 1)],
        'palette_slot': 0  # top
    },
    {  # FaceIndex.NY (Bottom, -Y)
        'normal': (0, -1, 0),
        'offset': (0, -1, 0),
        'corners': [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
        'uvs': [(0, 0), (1, 0), (1, 1), (0, 1)],
        'palette_slot': 1  # bottom
    },
    {  # FaceIndex.PZ (South, +Z)
        'normal': (0, 0, 1),
        'offset': (0, 0, 1),
        'corners': [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
        'uvs': [(0, 1), (1, 1), (1, 0), (0, 0)],
        'palette_slot': 2  # side
    },
    {  # FaceIndex.NZ (North, -Z)
        'normal': (0, 0, -1),
        'offset': (0, 0, -1),
        'corners': [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
        'uvs': [(1, 1), (1, 0), (0, 0), (0, 1)],
        'palette_slot': 2  # side
    }
]


# Triangle indices for quad (chunks.ts:161)
FACE_INDICES = [0, 1, 2, 0, 2, 3]


def get_material_palette() -> torch.Tensor:
    """
    Get material color palette matching chunks.ts:163-200.

    Returns:
        Tensor of shape (M, 3, 3) where:
            - M = 8 materials
            - First dim = material index
            - Second dim = face type (0=top, 1=bottom, 2=side)
            - Third dim = RGB
    """
    # Exact values from src/chunks.ts blockPalette
    palette = [
        # Air (index 0, not rendered)
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],

        # Grass (index 1)
        [[0.34, 0.68, 0.36],  # top
         [0.40, 0.30, 0.16],  # bottom
         [0.45, 0.58, 0.30]], # side

        # Dirt (index 2)
        [[0.42, 0.32, 0.20],  # top
         [0.38, 0.26, 0.16],  # bottom
         [0.40, 0.30, 0.18]], # side

        # Stone (index 3)
        [[0.58, 0.60, 0.64],  # top
         [0.55, 0.57, 0.60],  # bottom
         [0.56, 0.58, 0.62]], # side

        # Plank (index 4)
        [[0.78, 0.68, 0.50],  # top
         [0.72, 0.60, 0.42],  # bottom
         [0.74, 0.63, 0.45]], # side

        # Snow (index 5)
        [[0.92, 0.94, 0.96],  # top
         [0.90, 0.92, 0.94],  # bottom
         [0.88, 0.90, 0.93]], # side

        # Sand (index 6)
        [[0.88, 0.82, 0.60],  # top
         [0.86, 0.78, 0.56],  # bottom
         [0.87, 0.80, 0.58]], # side

        # Water (index 7)
        [[0.22, 0.40, 0.66],  # top
         [0.20, 0.34, 0.60],  # bottom
         [0.20, 0.38, 0.64]], # side
    ]

    return torch.tensor(palette, dtype=torch.float32)


def get_material_densities() -> torch.Tensor:
    """
    Get material densities for volumetric rendering compatibility.

    Returns:
        Tensor of shape (M,) with density values
    """
    # From model_stuff/materials.py:14-15
    densities = [
        0.0,   # Air
        20.0,  # Grass
        22.0,  # Dirt
        30.0,  # Stone
        15.0,  # Plank
        10.0,  # Snow
        18.0,  # Sand
        2.0    # Water
    ]

    return torch.tensor(densities, dtype=torch.float32)


def material_name_to_index(name: str) -> int:
    """Convert material name to index."""
    try:
        return MATERIALS.index(name)
    except ValueError:
        raise ValueError(f"Unknown material: {name}. Valid: {MATERIALS}")


def material_index_to_name(index: int) -> str:
    """Convert material index to name."""
    if 0 <= index < len(MATERIALS):
        return MATERIALS[index]
    raise ValueError(f"Invalid material index: {index}. Range: [0, {len(MATERIALS)-1}]")
