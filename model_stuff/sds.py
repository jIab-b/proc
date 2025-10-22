"""Score Distillation Sampling utilities for SDXL-Lightning guidance."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Tuple

import torch

from .sdxl_lightning import SDXLLightning, LATENT_SCALING


PromptEmbeddings = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def render_to_latents(rgb: torch.Tensor, sdxl: SDXLLightning) -> torch.Tensor:
    """Encode an RGB tensor in `[0,1]` (B,3,H,W) to SDXL latent space."""
    rgb = rgb.clamp(0.0, 1.0)
    rgb = (rgb * 2.0 - 1.0).to(sdxl.dtype)
    device = getattr(sdxl.pipe, "device", None)
    if device is None:
        device = torch.device(sdxl.device)
    else:
        device = torch.device(device)
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=sdxl.dtype)
        if device.type == "cuda" and sdxl.dtype != torch.float32
        else nullcontext()
    )
    with amp_ctx:
        latents = sdxl.vae.encode(rgb).latent_dist.mean
    return latents * LATENT_SCALING


def score_distillation_loss(
    rgba: torch.Tensor,
    sdxl: SDXLLightning,
    embeddings: PromptEmbeddings,
    cfg_scale: float = 7.5,
) -> torch.Tensor:
    """Compute SDS loss for a rendered RGBA image."""
    pe, pe_pooled, ue, ue_pooled, add_time_ids = embeddings
    rgb = rgba[:, :3, :, :]
    latents = render_to_latents(rgb, sdxl)

    timesteps = sdxl.sample_timesteps(latents.shape[0])
    noise = torch.randn_like(latents)
    noisy = sdxl.add_noise(latents, noise, timesteps)

    device = getattr(sdxl.pipe, "device", None)
    if device is None:
        device = torch.device(sdxl.device)
    else:
        device = torch.device(device)
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=sdxl.dtype)
        if device.type == "cuda" and sdxl.dtype != torch.float32
        else nullcontext()
    )
    with amp_ctx:
        eps = sdxl.eps_pred_cfg(
            noisy,
            timesteps,
            pe,
            pe_pooled,
            ue,
            ue_pooled,
            add_time_ids,
            cfg_scale,
        )

    return (eps - noise).pow(2).mean()
