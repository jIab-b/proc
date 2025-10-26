"""Voxel grid with residual material edits and learnable palette."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nv_diff_render import DifferentiableBlockRenderer
from .nv_diff_render.materials import MATERIALS, get_material_palette
from .nv_diff_render.utils import world_to_clip


def in_frustum(
    positions: torch.Tensor,
    camera_view: torch.Tensor,
    camera_proj: torch.Tensor,
    img_w: int,
    img_h: int,
    near_clip: float = 0.1,
    far_clip: Optional[float] = None,
) -> torch.Tensor:
    device = positions.device
    camera_view = camera_view.to(device)
    camera_proj = camera_proj.to(device)

    clip = world_to_clip(positions, camera_view, camera_proj)
    w = clip[:, 3].clamp(min=1e-6)
    ndc = clip[:, :3] / w.unsqueeze(1)

    mask = (ndc[:, 0].abs() <= 1.0) & (ndc[:, 1].abs() <= 1.0) & (ndc[:, 2].abs() <= 1.0)
    if far_clip is not None:
        view_pos = positions @ camera_view[:3, :3].T + camera_view[:3, 3]
        depth_ok = (view_pos[:, 2] >= -far_clip) & (view_pos[:, 2] <= -near_clip)
        mask = mask & depth_ok
    return mask


class DenoisingVoxelGrid(nn.Module):
    """Frozen baseline map plus sparse residual edits."""

    def __init__(
        self,
        grid_size: Tuple[int, int, int],
        num_materials: int = len(MATERIALS),
        world_scale: float = 2.0,
        device: torch.device | str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        super().__init__()
        device = torch.device(device)

        self.grid_size = grid_size
        self.num_materials = num_materials
        self.world_scale = world_scale
        self.device = device
        self.sky_index = MATERIALS.index("Air")

        self.register_buffer(
            "base_logits",
            torch.zeros((*grid_size, num_materials), dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "base_reference",
            torch.zeros((*grid_size, num_materials), dtype=torch.float32, device=device),
        )
        self.edit_logits = nn.Parameter(torch.zeros_like(self.base_logits))
        self.mask_logits = nn.Parameter(torch.zeros(grid_size, dtype=torch.float32, device=device))

        palette = get_material_palette().to(torch.float32)
        self.register_buffer("palette_target", palette.clone())
        self.palette_embed = nn.Parameter(palette.clone())

        self.renderer = DifferentiableBlockRenderer(
            grid_size=grid_size,
            world_scale=world_scale,
            device=device,
        )
        self.training_occ_threshold = 0.05

    # ------------------------------------------------------------------
    def final_logits(self) -> torch.Tensor:
        mask = torch.sigmoid(self.mask_logits)[..., None]
        return self.base_logits + mask * self.edit_logits

    def palette(self) -> torch.Tensor:
        return self.palette_embed.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    def forward(
        self,
        camera_view: torch.Tensor,
        camera_proj: torch.Tensor,
        img_h: int,
        img_w: int,
        temperature: float = 1.0,
        occupancy_threshold: float = 0.01,
        max_blocks: Optional[int] = None,
    ) -> torch.Tensor:
        logits = self.final_logits()
        temp = max(float(temperature), 1e-4)
        probs = F.softmax(logits / temp, dim=-1)
        occ = 1.0 - probs[..., self.sky_index]

        train_thr = min(self.training_occ_threshold, occupancy_threshold)
        thr = train_thr if self.training else occupancy_threshold

        active_mask = occ > thr
        active_indices = torch.nonzero(active_mask, as_tuple=False)
        if active_indices.shape[0] == 0:
            sky = self.renderer.shader.sky_color.to(self.device)
            rgb = sky.view(1, 3, 1, 1).expand(1, 3, img_h, img_w)
            alpha = torch.zeros(1, 1, img_h, img_w, device=self.device)
            return torch.cat([rgb, alpha], dim=1)

        X, Y, Z = self.grid_size
        offsets = torch.tensor([-(X / 2.0), 0.0, -(Z / 2.0)], device=self.device)
        centers = active_indices.float() + 0.5
        world_centers = (centers + offsets) * self.world_scale

        visible = in_frustum(world_centers, camera_view, camera_proj, img_w, img_h)
        if visible.any():
            active_indices = active_indices[visible]
            world_centers = world_centers[visible]
        else:
            sky = self.renderer.shader.sky_color.to(self.device)
            rgb = sky.view(1, 3, 1, 1).expand(1, 3, img_h, img_w)
            alpha = torch.zeros(1, 1, img_h, img_w, device=self.device)
            return torch.cat([rgb, alpha], dim=1)

        if max_blocks is not None and active_indices.shape[0] > max_blocks:
            view_pos = world_centers @ camera_view[:3, :3].T + camera_view[:3, 3]
            depth = (-view_pos[:, 2]).clamp(min=0.0)
            keep = torch.argsort(depth, descending=False)[:max_blocks]
            active_indices = active_indices[keep]

        pruned_mask = torch.zeros_like(active_mask)
        pruned_mask[active_indices[:, 0], active_indices[:, 1], active_indices[:, 2]] = True

        pruned_weights = torch.zeros_like(occ)
        pruned_weights[pruned_mask] = occ[pruned_mask]

        return self.renderer.render_from_grid(
            pruned_mask,
            logits,
            camera_view,
            camera_proj,
            img_h,
            img_w,
            occupancy_probs=pruned_weights,
            temperature=temp,
            palette=self.palette(),
            hard_materials=False,
        )

    # ------------------------------------------------------------------
    def get_material_probs(self, temperature: float = 1.0) -> torch.Tensor:
        temp = max(float(temperature), 1e-4)
        return F.softmax(self.final_logits() / temp, dim=-1)

    def get_occupancy_probs(self) -> torch.Tensor:
        probs = self.get_material_probs()
        return 1.0 - probs[..., self.sky_index]

    def get_mask_probs(self) -> torch.Tensor:
        return torch.sigmoid(self.mask_logits)

    def get_stats(self) -> dict:
        occ = self.get_occupancy_probs()
        mask = self.get_mask_probs()
        mat = self.get_material_probs()

        active = occ > 0.5
        num_active = int(active.sum().item())
        sx, sy, sz = self.grid_size
        total_voxels = int(sx * sy * sz)

        mat_dist = {}
        if active.any():
            active_mats = mat[active].sum(dim=0)
            total = active_mats.sum().item()
            for idx, name in enumerate(MATERIALS):
                value = active_mats[idx].item()
                mat_dist[name] = float(value / total * 100.0) if total > 0 else 0.0

        return {
            "num_active_voxels": num_active,
            "total_voxels": total_voxels,
            "density": num_active / max(total_voxels, 1),
            "occupancy_mean": float(occ.mean().item()),
            "mask_mean": float(mask.mean().item()),
            "material_distribution": mat_dist,
        }

    # ------------------------------------------------------------------
    def load_state(self, base_logits: torch.Tensor) -> None:
        logits = base_logits.detach().to(self.base_logits.device)
        self.base_reference.copy_(logits)
        self.base_logits.copy_(logits)
        self.edit_logits.data.zero_()
        self.mask_logits.data.zero_()

    def to(self, device: torch.device) -> "DenoisingVoxelGrid":
        super().to(device)
        self.device = device
        self.renderer = self.renderer.to(device)
        return self

    # ------------------------------------------------------------------
    @torch.no_grad()
    def apply_noise(
        self,
        mask_std: float,
        edit_std: float,
        *,
        fraction: float = 0.02,
        bias_power: float = 1.5,
    ) -> dict:
        mask_std = float(mask_std)
        edit_std = float(edit_std)
        fraction = float(fraction)
        bias_power = float(bias_power)

        stats = {
            "mask_std": mask_std,
            "edit_std": edit_std,
            "fraction": fraction,
            "bias_power": bias_power,
            "mask_voxels": 0,
            "edit_voxels": 0,
            "edit_channels": 0,
            "mask_mean_abs_noise": 0.0,
            "edit_mean_abs_noise": 0.0,
            "selector_mean": 0.0,
        }

        if (mask_std <= 0.0 and edit_std <= 0.0) or fraction <= 0.0:
            return stats

        base_probs = F.softmax(self.base_reference, dim=-1)
        base_occ = 1.0 - base_probs[..., self.sky_index]
        current_occ = self.get_occupancy_probs()
        bias = 0.5 * (base_occ + current_occ)
        bias = bias.clamp_min(1e-4).pow(bias_power)

        mask_prob = (fraction * bias).clamp(0.0, 1.0)
        mask_selector = torch.bernoulli(mask_prob)
        mask_selector_bool = mask_selector > 0.0
        stats["mask_voxels"] = int(mask_selector_bool.sum().item())
        stats["selector_mean"] = float(mask_selector.mean().item())

        if mask_std > 0.0:
            noise = torch.randn_like(self.mask_logits) * mask_std
            self.mask_logits.add_(noise * mask_selector)
            if mask_selector_bool.any():
                mask_vals = (noise * mask_selector)[mask_selector_bool]
                stats["mask_mean_abs_noise"] = float(mask_vals.abs().mean().item())

        if edit_std > 0.0:
            noise = torch.randn_like(self.edit_logits) * edit_std
            edit_mask = mask_selector_bool[..., None].expand_as(noise)
            self.edit_logits.add_(noise * edit_mask)
            if mask_selector_bool.any():
                edit_vals = (noise * edit_mask)[edit_mask]
                stats["edit_mean_abs_noise"] = float(edit_vals.abs().mean().item())
                stats["edit_voxels"] = stats["mask_voxels"]
                stats["edit_channels"] = int(edit_mask.sum().item())

        return stats

    @torch.no_grad()
    def integrate_base_logits(
        self,
        new_logits: torch.Tensor,
        rate: float,
        bias: float = 1.0,
    ) -> dict:
        rate = float(rate)
        bias = float(bias)
        stats = {
            "rate": rate,
            "bias": bias,
            "mean_abs_update": 0.0,
            "max_abs_update": 0.0,
            "weight_mean": 0.0,
            "weight_min": 0.0,
            "weight_max": 0.0,
            "updated_voxels": 0,
        }
        if rate <= 0.0:
            return stats

        new_logits = new_logits.detach().to(self.base_reference.device)
        diff = new_logits - self.base_reference
        diff_energy = diff.pow(2).mean(dim=-1)
        if bias > 0.0:
            weights = torch.exp(-bias * diff_energy).clamp_min(0.0)
        else:
            weights = torch.ones_like(diff_energy)
        stats["weight_mean"] = float(weights.mean().item())
        stats["weight_min"] = float(weights.min().item())
        stats["weight_max"] = float(weights.max().item())

        blend = (rate * weights).clamp(0.0, 1.0)[..., None]
        update = diff * blend
        stats["mean_abs_update"] = float(update.abs().mean().item())
        stats["max_abs_update"] = float(update.abs().max().item())
        stats["updated_voxels"] = int((blend.squeeze(-1) > 0).sum().item())

        self.base_reference.add_(update)
        self.base_logits.copy_(self.base_reference)
        return stats

    @torch.no_grad()
    def harden(self, strength: float = 5.0, reset_prob: float = 0.5) -> None:
        strength = float(strength)
        reset_prob = float(reset_prob)
        reset_prob = min(max(reset_prob, 1e-4), 1 - 1e-4)
        probs = self.get_material_probs()
        hard_idx = probs.argmax(dim=-1)
        one_hot = F.one_hot(hard_idx, self.num_materials).to(self.base_logits.dtype) * strength
        self.base_logits.copy_(one_hot)
        self.base_reference.copy_(one_hot)
        self.edit_logits.zero_()
        reset_value = torch.logit(torch.tensor(reset_prob, device=self.device, dtype=self.mask_logits.dtype))
        self.mask_logits.fill_(reset_value)

    @torch.no_grad()
    def reset_palette(self) -> None:
        self.palette_embed.data.copy_(self.palette_target)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"DenoisingVoxelGrid(grid_size={self.grid_size}, "
            f"active_voxels={stats['num_active_voxels']}, density={stats['density']:.3f})"
        )
