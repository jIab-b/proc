"""Thin wrapper around the differentiable renderer for convenience."""

from __future__ import annotations

from typing import Optional

import torch

from .voxel_grid import DenoisingVoxelGrid


class VoxelRenderer:
    """Helper that dispatches renders through the denoising voxel grid."""

    def __init__(self, voxel_grid: DenoisingVoxelGrid) -> None:
        self.grid = voxel_grid

    def render(
        self,
        view: torch.Tensor,
        proj: torch.Tensor,
        height: int,
        width: int,
        temperature: float = 1.0,
        occupancy_threshold: float = 0.01,
        max_blocks: Optional[int] = None,
    ) -> torch.Tensor:
        return self.grid(
            camera_view=view,
            camera_proj=proj,
            img_h=height,
            img_w=width,
            temperature=temperature,
            occupancy_threshold=occupancy_threshold,
            max_blocks=max_blocks,
        )
