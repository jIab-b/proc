"""Camera sampling helpers used during SDS voxel training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .nv_diff_render.utils import create_look_at_matrix, create_perspective_matrix
from .dataset import MultiViewDataset, ViewRecord


@dataclass
class CameraSample:
    view: torch.Tensor
    proj: torch.Tensor
    from_dataset: bool
    index: Optional[int] = None
    rgb: Optional[torch.Tensor] = None
    image_path: Optional[str] = None


class CameraSampler:
    """Sample dataset or random orbit cameras for SDS training."""

    def __init__(
        self,
        dataset: MultiViewDataset,
        grid_size: tuple[int, int, int],
        world_scale: float,
        render_height: int,
        render_width: int,
        novel_view_prob: float = 0.2,
        device: torch.device | str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        self.dataset = dataset
        self.grid_size = grid_size
        self.world_scale = world_scale
        self.novel_view_prob = float(novel_view_prob)
        self.device = torch.device(device)

        self.render_height = int(render_height)
        self.render_width = int(render_width)

        self._height, self._width = dataset.image_size
        self._default_intrinsics = dataset.metadata["views"][0]["intrinsics"]
        self._focus_center = None
        self._focus_radius = None

    # ------------------------------------------------------------------
    def sample(self) -> CameraSample:
        """Sample either a dataset view or a random orbit view."""
        if torch.rand(1).item() < self.novel_view_prob:
            return self._sample_random_orbit()
        return self._sample_dataset()

    def _sample_dataset(self) -> CameraSample:
        rec = self.dataset.get_view(torch.randint(0, len(self.dataset), (1,)).item())
        rgb = self.dataset.load_image(rec)

        intr = rec.intrinsics or {}
        fov = math.radians(float(intr.get("fovYDegrees", 60.0)))
        near = float(intr.get("near", 0.1))
        far = float(intr.get("far", 500.0))
        aspect = float(self.render_width / max(self.render_height, 1))
        proj = create_perspective_matrix(fov, aspect, near, far).to(self.device)

        return CameraSample(
            view=rec.view_matrix,
            proj=proj,
            from_dataset=True,
            index=rec.index,
            rgb=rgb,
            image_path=str(rec.rgb_path),
        )

    def _sample_random_orbit(self) -> CameraSample:
        gx, gy, gz = self.grid_size
        if self._focus_radius is not None:
            radius = max(self._focus_radius, self.world_scale * max(gx, gz) * 0.5)
        else:
            radius = self.world_scale * max(gx, gz) * 1.4
        min_elev = math.radians(15.0)
        max_elev = math.radians(50.0)

        azimuth = torch.rand(1).item() * math.tau
        elevation = min_elev + (max_elev - min_elev) * torch.rand(1).item()

        x = radius * math.cos(elevation) * math.cos(azimuth)
        z = radius * math.cos(elevation) * math.sin(azimuth)
        y = radius * math.sin(elevation) + self.world_scale * gy * 0.25

        if self._focus_center is not None:
            target = tuple(self._focus_center.tolist())
            eye = (target[0] + x, target[1] + y, target[2] + z)
        else:
            eye = (x, y, z)
            target = (0.0, self.world_scale * gy * 0.25, 0.0)
        up = (0.0, 1.0, 0.0)

        view = create_look_at_matrix(eye, target, up).to(self.device)

        intr = self._default_intrinsics
        fov = math.radians(float(intr.get("fovYDegrees", 60.0)))
        aspect = float(self.render_width / max(self.render_height, 1))
        near = float(intr.get("near", 0.1))
        far = float(intr.get("far", 500.0))
        proj = create_perspective_matrix(fov, aspect, near, far).to(self.device)

        return CameraSample(
            view=view,
            proj=proj,
            from_dataset=False,
            index=None,
            rgb=None,
            image_path=None,
        )

    # ------------------------------------------------------------------
    def update_focus(self, mask_probs: torch.Tensor, threshold: float = 0.6) -> None:
        if mask_probs.numel() == 0:
            self._focus_center = None
            self._focus_radius = None
            return
        mask = mask_probs > threshold
        if not mask.any():
            self._focus_center = None
            self._focus_radius = None
            return
        idx = torch.nonzero(mask, as_tuple=False).float()
        center = idx.mean(dim=0)
        X, Y, Z = self.grid_size
        offsets = torch.tensor([-(X / 2.0), 0.0, -(Z / 2.0)], device=idx.device)
        world_center = (center + 0.5 + offsets) * self.world_scale
        extent = idx.max(dim=0).values - idx.min(dim=0).values + 1.0
        radius = extent.norm().item() * self.world_scale
        radius = max(radius, self.world_scale * max(self.grid_size) * 0.2)
        self._focus_center = world_center.detach().cpu()
        self._focus_radius = radius

    def sample_dataset_view(self) -> CameraSample:
        return self._sample_dataset()
