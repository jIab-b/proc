"""
Shared palette definitions synced with the WebGPU frontend.

See webgpu/src/core.ts for the source of these values.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List

import torch


class BlockType(IntEnum):
    AIR = 0
    GRASS = 1
    DIRT = 2
    STONE = 3
    PLANK = 4
    SNOW = 5
    SAND = 6
    WATER = 7
    ALPINE_ROCK = 8
    ALPINE_GRASS = 9
    GLACIER_ICE = 10
    GRAVEL = 11


BLOCK_NAMES: Dict[int, str] = {
    BlockType.AIR: "Air",
    BlockType.GRASS: "Grass",
    BlockType.DIRT: "Dirt",
    BlockType.STONE: "Stone",
    BlockType.PLANK: "Plank",
    BlockType.SNOW: "Snow",
    BlockType.SAND: "Sand",
    BlockType.WATER: "Water",
    BlockType.ALPINE_ROCK: "AlpineRock",
    BlockType.ALPINE_GRASS: "AlpineGrass",
    BlockType.GLACIER_ICE: "GlacierIce",
    BlockType.GRAVEL: "Gravel",
}

NAME_TO_BLOCK: Dict[str, BlockType] = {v: BlockType(k) for k, v in BLOCK_NAMES.items()}


# Palette order: (top, bottom, side) per block
_PALETTE_DATA: List[List[List[float]]] = [
    # Air (unused)
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    # Grass
    [[0.34, 0.68, 0.36], [0.40, 0.30, 0.16], [0.45, 0.58, 0.30]],
    # Dirt
    [[0.42, 0.32, 0.20], [0.38, 0.26, 0.16], [0.40, 0.30, 0.18]],
    # Stone
    [[0.58, 0.60, 0.64], [0.55, 0.57, 0.60], [0.56, 0.58, 0.62]],
    # Plank
    [[0.78, 0.68, 0.50], [0.72, 0.60, 0.42], [0.74, 0.63, 0.45]],
    # Snow
    [[0.92, 0.94, 0.96], [0.90, 0.92, 0.94], [0.88, 0.90, 0.93]],
    # Sand
    [[0.88, 0.82, 0.60], [0.86, 0.78, 0.56], [0.87, 0.80, 0.58]],
    # Water
    [[0.22, 0.40, 0.66], [0.20, 0.34, 0.60], [0.20, 0.38, 0.64]],
    # Alpine Rock
    [[0.45, 0.48, 0.52], [0.43, 0.46, 0.50], [0.44, 0.47, 0.51]],
    # Alpine Grass
    [[0.26, 0.58, 0.32], [0.22, 0.44, 0.28], [0.24, 0.50, 0.30]],
    # Glacier Ice
    [[0.78, 0.88, 0.96], [0.72, 0.82, 0.90], [0.74, 0.84, 0.92]],
    # Gravel
    [[0.52, 0.52, 0.50], [0.48, 0.48, 0.46], [0.50, 0.50, 0.48]],
]


def get_palette_tensor(device: torch.device | None = None) -> torch.Tensor:
    """Return palette as (M, 3, 3) tensor on the requested device."""
    tensor = torch.tensor(_PALETTE_DATA, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def get_block_count() -> int:
    """Number of materials including air."""
    return len(_PALETTE_DATA)


def name_for_index(index: int) -> str:
    return BLOCK_NAMES.get(index, f"Block{index}")


def index_for_name(name: str) -> int:
    if name not in NAME_TO_BLOCK:
        raise KeyError(f"Unknown block name '{name}'. Valid names: {list(NAME_TO_BLOCK.keys())}")
    return int(NAME_TO_BLOCK[name])


def get_default_densities(device: torch.device | None = None) -> torch.Tensor:
    """
    Provide heuristic densities for volumetric rendering compatibility.
    These values loosely follow relative opacities of the blocks.
    """
    densities = torch.tensor(
        [
            0.0,   # Air
            20.0,  # Grass
            22.0,  # Dirt
            30.0,  # Stone
            15.0,  # Plank
            10.0,  # Snow
            18.0,  # Sand
            2.0,   # Water
            28.0,  # Alpine Rock
            18.0,  # Alpine Grass
            8.0,   # Glacier Ice
            24.0,  # Gravel
        ],
        dtype=torch.float32,
    )
    if device is not None:
        densities = densities.to(device)
    return densities


@dataclass(frozen=True)
class AdjacencyPattern:
    size: torch.Size
    kernel: torch.Tensor
    threshold: float
    weight: float
    reward: bool

