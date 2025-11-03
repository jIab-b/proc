"""
Material definitions matching WebGPU block palette exactly.

References:
- src/chunks.ts:163-200 (blockPalette)
- src/blockUtils.ts (BlockType enum)
"""

import torch
from typing import List, Tuple

from ..palette import (
    BLOCK_NAMES,
    NAME_TO_BLOCK,
    get_default_densities,
    get_palette_tensor,
)


# Material names matching frontend BlockType enum
MATERIALS = [BLOCK_NAMES[i] for i in range(len(BLOCK_NAMES))]


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
    Get material color palette matching webgpu/src/core.ts blockPalette.
    Returns tensor of shape (M, 3, 3).
    """
    return get_palette_tensor()


def get_material_densities() -> torch.Tensor:
    """
    Get material densities for volumetric rendering compatibility.
    """
    return get_default_densities()


def material_name_to_index(name: str) -> int:
    """Convert material name to index."""
    if name not in NAME_TO_BLOCK:
        raise ValueError(f"Unknown material: {name}. Valid: {list(NAME_TO_BLOCK.keys())}")
    return int(NAME_TO_BLOCK[name])


def material_index_to_name(index: int) -> str:
    """Convert material index to name."""
    if index in BLOCK_NAMES:
        return BLOCK_NAMES[index]
    raise ValueError(f"Invalid material index: {index}. Range: [0, {len(BLOCK_NAMES)-1}]")
