"""Loss helpers for residual voxel optimisation."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def photometric_loss(pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred_rgb, target_rgb)


def total_variation_3d(x: torch.Tensor) -> torch.Tensor:
    dx = x[1:, :, :] - x[:-1, :, :]
    dy = x[:, 1:, :] - x[:, :-1, :]
    dz = x[:, :, 1:] - x[:, :, :-1]
    return (dx.abs().mean() + dy.abs().mean() + dz.abs().mean())


def regularisation_losses(
    occ_probs: torch.Tensor,
    mat_probs: torch.Tensor,
    occ_delta: torch.Tensor,
    mat_delta: torch.Tensor,
    palette_embed: torch.Tensor,
    palette_target: torch.Tensor,
    lambda_mask: float = 1e-3,
    lambda_entropy: float = 1e-4,
    lambda_edit_tv: float = 0.0,
    lambda_edit_l2: float = 0.0,
    lambda_palette: float = 0.0,
) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}

    if lambda_mask > 0.0:
        losses["occupancy_reg"] = occ_delta.abs().mean() * lambda_mask
    else:
        losses["occupancy_reg"] = torch.zeros((), device=occ_delta.device, dtype=occ_delta.dtype)

    entropy = -(mat_probs * torch.log(mat_probs.clamp_min(1e-8))).sum(dim=-1)
    losses["material_entropy"] = entropy.mean() * lambda_entropy

    if lambda_edit_tv > 0.0:
        tv_volume = mat_delta.norm(dim=-1)
        losses["mat_tv"] = total_variation_3d(tv_volume) * lambda_edit_tv
    else:
        losses["mat_tv"] = torch.zeros((), device=mat_delta.device, dtype=mat_delta.dtype)

    if lambda_edit_l2 > 0.0:
        losses["mat_l2"] = (mat_delta.pow(2).mean()) * lambda_edit_l2
    else:
        losses["mat_l2"] = torch.zeros((), device=mat_delta.device, dtype=mat_delta.dtype)

    if lambda_palette > 0.0:
        losses["palette_l2"] = F.mse_loss(palette_embed, palette_target) * lambda_palette
    else:
        losses["palette_l2"] = torch.zeros((), device=palette_embed.device, dtype=palette_embed.dtype)

    return losses
