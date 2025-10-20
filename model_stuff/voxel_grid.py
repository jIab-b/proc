"""
Differentiable voxel grid for SDS optimization.

This module provides the DifferentiableVoxelGrid class which maintains
occupancy and material logits for all voxels, enabling gradient-based
optimization via Score Distillation Sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .nv_diff_render import DifferentiableBlockRenderer
from .nv_diff_render.materials import MATERIALS


class DifferentiableVoxelGrid(nn.Module):
    """
    Dense voxel grid for SDS optimization.

    Learnable Parameters:
    - occupancy_logits: (X, Y, Z) - sigmoid → [0, 1] occupancy probabilities
    - material_logits: (X, Y, Z, M) - softmax → material probabilities

    Forward pass renders the grid from a camera view, maintaining gradients
    through both occupancy and material parameters.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int, int],
        num_materials: int = 8,
        world_scale: float = 2.0,
        device: torch.device = torch.device('cuda')
    ):
        """
        Initialize differentiable voxel grid.

        Args:
            grid_size: (X, Y, Z) dimensions
            num_materials: Number of material types (default 8)
            world_scale: World scale parameter (default 2.0)
            device: Torch device
        """
        super().__init__()

        self.grid_size = grid_size
        self.num_materials = num_materials
        self.world_scale = world_scale
        self.device = device

        X, Y, Z = grid_size

        # Learnable parameters
        # Start with low occupancy (air) and uniform materials
        self.occupancy_logits = nn.Parameter(
            torch.full((X, Y, Z), -5.0, dtype=torch.float32, device=device)
        )

        self.material_logits = nn.Parameter(
            torch.zeros((X, Y, Z, num_materials), dtype=torch.float32, device=device)
        )

        # Initialize renderer
        self.renderer = DifferentiableBlockRenderer(
            grid_size=grid_size,
            world_scale=world_scale,
            device=device
        )

    def forward(
        self,
        camera_view: torch.Tensor,
        camera_proj: torch.Tensor,
        img_h: int,
        img_w: int,
        temperature: float = 1.0,
        occupancy_threshold: float = 0.01
    ) -> torch.Tensor:
        """
        Render grid from camera view.

        Args:
            camera_view: (4, 4) view matrix
            camera_proj: (4, 4) projection matrix
            img_h: Image height
            img_w: Image width
            temperature: Softmax temperature for materials
            occupancy_threshold: Minimum occupancy to render

        Returns:
            (1, 4, H, W) RGBA tensor
        """
        # Compute occupancy probabilities
        occ_probs = torch.sigmoid(self.occupancy_logits)

        # Find active voxels
        active_mask = occ_probs > occupancy_threshold
        active_indices = torch.nonzero(active_mask, as_tuple=False)

        # Handle empty scene
        if len(active_indices) == 0:
            sky_color = self.renderer.shader.sky_color
            rgb = sky_color.view(1, 3, 1, 1).expand(1, 3, img_h, img_w)
            alpha = torch.zeros(1, 1, img_h, img_w, device=self.device)
            return torch.cat([rgb, alpha], dim=1)

        # Extract positions (convert to list of tuples for set hashing in mesh_builder)
        positions = [tuple(pos) for pos in active_indices.tolist()]

        # Extract occupancy and material logits for active voxels
        active_occ = occ_probs[active_indices[:, 0], active_indices[:, 1], active_indices[:, 2]]
        active_mat_logits = self.material_logits[active_indices[:, 0], active_indices[:, 1], active_indices[:, 2]]

        # CRITICAL: Modulate material logits by occupancy
        # This creates gradient path: pixels → materials → occupancy
        modulated_logits = active_mat_logits * active_occ.unsqueeze(-1)

        # Render
        rgba = self.renderer.render(
            positions,
            modulated_logits,
            camera_view,
            camera_proj,
            img_h,
            img_w,
            temperature=temperature,
            hard_materials=False
        )

        return rgba

    def get_occupancy_probs(self) -> torch.Tensor:
        """Get current occupancy probabilities."""
        return torch.sigmoid(self.occupancy_logits)

    def get_material_probs(self, temperature: float = 1.0) -> torch.Tensor:
        """Get current material probabilities."""
        return F.softmax(self.material_logits / temperature, dim=-1)

    def get_stats(self) -> dict:
        """Get grid statistics."""
        occ_probs = self.get_occupancy_probs()
        mat_probs = self.get_material_probs()

        active_mask = occ_probs > 0.5
        num_active = active_mask.sum().item()

        # Material distribution
        mat_dist = {}
        if active_mask.any():
            active_mats = mat_probs[active_mask].sum(dim=0)
            total = active_mats.sum().item()

            for i, mat_name in enumerate(MATERIALS):
                count = active_mats[i].item()
                mat_dist[mat_name] = float(count / total * 100) if total > 0 else 0.0

        return {
            "num_active_voxels": num_active,
            "total_voxels": self.occupancy_logits.numel(),
            "density": num_active / self.occupancy_logits.numel(),
            "occupancy_mean": float(occ_probs.mean().item()),
            "occupancy_std": float(occ_probs.std().item()),
            "material_distribution": mat_dist
        }

    def load_state(self, occupancy_logits: torch.Tensor, material_logits: torch.Tensor):
        """
        Load occupancy and material logits.

        Args:
            occupancy_logits: (X, Y, Z) tensor
            material_logits: (X, Y, Z, M) tensor
        """
        self.occupancy_logits.data.copy_(occupancy_logits)
        self.material_logits.data.copy_(material_logits)

    def to(self, device: torch.device) -> 'DifferentiableVoxelGrid':
        """Move grid to device."""
        super().to(device)
        self.device = device
        self.renderer = self.renderer.to(device)
        return self

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (f"DifferentiableVoxelGrid(grid_size={self.grid_size}, "
                f"active_voxels={stats['num_active_voxels']}, "
                f"density={stats['density']:.3f})")
