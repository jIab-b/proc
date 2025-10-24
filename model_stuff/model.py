"""High-level wrapper around the differentiable voxel grid."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import torch
import torch.nn as nn

from .map_io import load_map_to_grid, save_grid_to_map
from .voxel_grid import DenoisingVoxelGrid


class VoxelScene(nn.Module):
    def __init__(
        self,
        grid_size: Tuple[int, int, int],
        world_scale: float,
        num_materials: int = 8,
        device: torch.device | str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        super().__init__()
        device = torch.device(device)
        self.grid = DenoisingVoxelGrid(
            grid_size=grid_size,
            num_materials=num_materials,
            world_scale=world_scale,
            device=device,
        )
        self.world_scale = world_scale

    # ------------------------------------------------------------------
    @classmethod
    def from_map(cls, map_path: str | Path, device: torch.device | str = "cuda") -> "VoxelScene":
        mat_logits, grid_size, world_scale = load_map_to_grid(map_path, device=device)
        scene = cls(grid_size=grid_size, world_scale=world_scale, device=device)
        scene.grid.load_state(mat_logits)
        return scene

    # ------------------------------------------------------------------
    def render(
        self,
        view: torch.Tensor,
        proj: torch.Tensor,
        height: int,
        width: int,
        temperature: float = 1.0,
        occupancy_threshold: float = 0.01,
        max_blocks: int | None = None,
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

    # ------------------------------------------------------------------
    def occupancy_probs(self) -> torch.Tensor:
        return self.grid.get_occupancy_probs()

    def material_probs(self, temperature: float = 1.0) -> torch.Tensor:
        return self.grid.get_material_probs(temperature)

    def mask_probs(self) -> torch.Tensor:
        return self.grid.get_mask_probs()

    def stats(self) -> dict:
        return self.grid.get_stats()

    def save_map(
        self,
        path: str | Path,
        threshold: float = 0.5,
        metadata: dict | None = None,
    ) -> int:
        return save_grid_to_map(
            material_logits=self.grid.final_logits().detach(),
            output_path=path,
            world_scale=self.world_scale,
            threshold=threshold,
            metadata=metadata or {},
        )

    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        return self.render(*args, **kwargs)

    def parameters(self) -> Iterable[torch.nn.Parameter]:  # type: ignore[override]
        return self.grid.parameters()
