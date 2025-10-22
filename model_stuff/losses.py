"""Loss helpers for voxel SDS optimisation."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def photometric_loss(pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
    """Simple L1 photometric loss between prediction and ground truth."""
    return F.l1_loss(pred_rgb, target_rgb)


def total_variation_3d(x: torch.Tensor) -> torch.Tensor:
    """3D total variation (anisotropic) for occupancy probability volume."""
    dx = x[1:, :, :] - x[:-1, :, :]
    dy = x[:, 1:, :] - x[:, :-1, :]
    dz = x[:, :, 1:] - x[:, :, :-1]
    return (dx.abs().mean() + dy.abs().mean() + dz.abs().mean())


def regularisation_losses(
    occ_probs: torch.Tensor,
    mat_probs: torch.Tensor,
    lambda_sparsity: float = 1e-3,
    lambda_entropy: float = 1e-4,
    lambda_tv: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Compute sparsity/entropy/TV losses for the voxel grid."""
    losses: Dict[str, torch.Tensor] = {}
    losses["sparsity"] = occ_probs.mean() * lambda_sparsity

    entropy = -(mat_probs * torch.log(mat_probs.clamp_min(1e-8))).sum(dim=-1)
    losses["entropy"] = entropy.mean() * lambda_entropy

    if lambda_tv > 0.0:
        losses["tv"] = total_variation_3d(occ_probs) * lambda_tv
    else:
        losses["tv"] = torch.zeros((), device=occ_probs.device, dtype=occ_probs.dtype)

    return losses
