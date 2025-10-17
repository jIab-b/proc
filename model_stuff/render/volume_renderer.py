"""Differentiable volume renderer for voxel grids.

The renderer operates directly on the probabilistic voxel grid defined in
``model_stuff.dsl.probabilistic_world``.  It performs alpha compositing along
camera rays using standard NeRF-style volume rendering and is fully
autograd-compatible in PyTorch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F

from model_stuff.dsl.probabilistic_world import BlockPalette, ProbabilisticVoxelGrid


@dataclass
class CameraView:
    width: int
    height: int
    position: torch.Tensor  # (3,)
    forward: torch.Tensor  # (3,)
    up: torch.Tensor  # (3,)
    right: torch.Tensor  # (3,)
    fov_y_radians: float
    near: float
    far: float

    @classmethod
    def from_metadata(
        cls,
        view: Dict[str, object],
        image_size: Dict[str, int],
        device: torch.device,
    ) -> "CameraView":
        intrinsics = view.get("intrinsics", {})
        fov_y = float(intrinsics.get("fovYDegrees", 60.0))
        near = float(intrinsics.get("near", 0.1))
        far = float(intrinsics.get("far", 500.0))

        position = torch.tensor(view.get("position"), dtype=torch.float32, device=device)
        forward = F.normalize(torch.tensor(view.get("forward"), dtype=torch.float32, device=device), dim=0)
        up = F.normalize(torch.tensor(view.get("up"), dtype=torch.float32, device=device), dim=0)
        right = F.normalize(torch.tensor(view.get("right"), dtype=torch.float32, device=device), dim=0)

        return cls(
            width=int(image_size.get("width", 512)),
            height=int(image_size.get("height", 512)),
            position=position,
            forward=forward,
            up=up,
            right=right,
            fov_y_radians=math.radians(fov_y),
            near=near,
            far=far,
        )

    def sample_rays(
        self,
        pixel_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ray origins and directions for a batch of pixels.

        Parameters
        ----------
        pixel_indices:
            Tensor with shape ``(N, 2)`` storing ``(y, x)`` integer indices.
        """

        device = pixel_indices.device
        n = pixel_indices.shape[0]
        origins = self.position.expand(n, 3)

        ys = pixel_indices[:, 0].to(torch.float32) + 0.5
        xs = pixel_indices[:, 1].to(torch.float32) + 0.5

        # Normalised device coordinates.
        tan_half_fov = math.tan(self.fov_y_radians / 2.0)
        aspect = self.width / self.height
        x_norm = (2.0 * xs / self.width - 1.0) * aspect * tan_half_fov
        y_norm = (1.0 - 2.0 * ys / self.height) * tan_half_fov

        dirs = (
            self.forward.expand_as(origins)
            + x_norm.unsqueeze(1) * self.right.expand_as(origins)
            + y_norm.unsqueeze(1) * self.up.expand_as(origins)
        )
        dirs = F.normalize(dirs, dim=1)
        return origins, dirs


class VolumeRenderer:
    def __init__(
        self,
        grid: ProbabilisticVoxelGrid,
        device: torch.device | None = None,
        background: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        step_size: Optional[float] = None,
    ) -> None:
        self.grid = grid
        self.device = device or grid.logits.device
        self.background = torch.tensor(background, dtype=torch.float32, device=self.device)
        self.step_size = step_size or grid.metadata.world_scale

        self.refresh_volumes()

    def _prepare_volumes(self) -> Tuple[torch.Tensor, torch.Tensor]:
        densities, colours = self.grid.densities_and_colours()
        # Dimensions from ProbabilisticVoxelGrid are (X, Y, Z, 3).  PyTorch
        # grid_sample expects (N, C, D, H, W) with axes ordered as Z, Y, X.
        sigma = densities.permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
        colour = colours.permute(2, 1, 0, 3).permute(3, 0, 1, 2).unsqueeze(0)
        return sigma.contiguous().to(self.device), colour.contiguous().to(self.device)

    def refresh_volumes(self) -> None:
        self._sigma_volume, self._colour_volume = self._prepare_volumes()

    def _world_to_grid(self, points: torch.Tensor) -> torch.Tensor:
        """Convert world coordinates to grid_sample's normalised coordinates."""

        dims = self.grid.metadata.dimensions
        scale = self.grid.metadata.world_scale
        size = torch.tensor(dims, dtype=torch.float32, device=points.device) * scale

        # Points are in world units with origin at voxel (0,0,0).
        norm = points / size - 0.5
        norm = norm * 2.0
        # grid_sample expects order (z, y, x).
        return torch.stack([norm[..., 2], norm[..., 1], norm[..., 0]], dim=-1)

    def _max_distance(self) -> float:
        dims = self.grid.metadata.dimensions
        scale = self.grid.metadata.world_scale
        return float(scale * math.sqrt(sum(d * d for d in dims)))

    def render_rays(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
        num_samples: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render colours for a batch of rays.

        Returns ``(rgb, depth, weights)`` where ``rgb`` has shape ``(N, 3)``.
        """

        device = origins.device
        n_rays = origins.shape[0]

        base_lin = torch.linspace(0.0, 1.0, num_samples, device=device).view(1, num_samples)
        t_vals = near.unsqueeze(1) + (far - near).unsqueeze(1) * base_lin
        points = origins.unsqueeze(1) + directions.unsqueeze(1) * t_vals.unsqueeze(2)

        grid_coords = self._world_to_grid(points)
        grid = grid_coords.view(1, num_samples, n_rays, 1, 3)

        sigma_samples = F.grid_sample(
            self._sigma_volume,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).view(1, num_samples, n_rays)
        sigma_samples = sigma_samples.squeeze(0).transpose(0, 1)  # (n_rays, num_samples)

        colour_samples = F.grid_sample(
            self._colour_volume,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).view(3, num_samples, n_rays)
        colour_samples = colour_samples.permute(2, 1, 0)  # (n_rays, num_samples, 3)

        delta = self.step_size
        alpha = 1.0 - torch.exp(-torch.clamp(sigma_samples, min=0.0) * delta)
        transmittance = torch.cumprod(
            torch.cat([torch.ones(n_rays, 1, device=device), 1.0 - alpha + 1e-6], dim=1), dim=1
        )
        weights = alpha * transmittance[:, :-1]

        rgb = torch.sum(weights.unsqueeze(-1) * colour_samples, dim=1)
        acc_alpha = torch.sum(weights, dim=1)
        rgb = rgb + (1.0 - acc_alpha).unsqueeze(-1) * self.background

        depth = torch.sum(weights * t_vals, dim=1) / torch.clamp(acc_alpha, min=1e-6)
        return rgb, depth, weights

    def render_pixels(
        self,
        camera: CameraView,
        pixel_indices: torch.Tensor,
        num_samples: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pixel_indices = pixel_indices.to(self.device)
        origins, dirs = camera.sample_rays(pixel_indices)
        origins = origins.to(self.device)
        dirs = dirs.to(self.device)

        n = origins.shape[0]
        far_clip = min(camera.far, self._max_distance())
        near = torch.full((n,), camera.near, device=self.device)
        far = torch.full((n,), far_clip, device=self.device)

        return self.render_rays(origins, dirs, near=near, far=far, num_samples=num_samples)
