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
    rgb = (rgb * 2.0 - 1.0).float()

    vae = sdxl.vae if hasattr(sdxl, "vae") else sdxl.pipe.vae
    vae_config = getattr(vae, "config", None)
    force_upcast = bool(getattr(vae_config, "force_upcast", False))

    device_attr = getattr(sdxl.pipe, "device", None)
    device = torch.device(device_attr) if device_attr is not None else torch.device(sdxl.device)

    rgb = rgb.to(device)

    if force_upcast:
        # SDXL VAE sets force_upcast=True; mirror diffusers by casting module & input to fp32.
        vae = vae.to(dtype=torch.float32)
        encode_ctx = nullcontext()
        rgb_encode = rgb.to(dtype=torch.float32)
    else:
        encode_ctx = (
            torch.autocast(device_type=device.type, dtype=sdxl.dtype)
            if device.type == "cuda" and sdxl.dtype != torch.float32
            else nullcontext()
        )
        rgb_encode = rgb.to(dtype=sdxl.dtype if device.type == "cuda" else torch.float32)

    with encode_ctx:
        latents = vae.encode(rgb_encode).latent_dist.mean

    if force_upcast:
        latents = latents.to(dtype=torch.float32)
    else:
        latents = latents.float()

    return latents * LATENT_SCALING


def score_distillation_loss(
    rgba: torch.Tensor,
    sdxl: SDXLLightning,
    embeddings: PromptEmbeddings,
    cfg_scale: float = 7.5,
    mask_sky: bool = False,
    use_lightning_timesteps: bool = True,
    lightning_steps: int = 4,
) -> torch.Tensor:
    """Compute SDS loss for a rendered RGBA image.
    Optionally mask sky pixels via alpha and use Lightning trailing timesteps.
    """
    pe, pe_pooled, ue, ue_pooled, add_time_ids = embeddings
    rgb = rgba[:, :3, :, :]
    latents = render_to_latents(rgb, sdxl).float()

    if use_lightning_timesteps:
        timesteps = sdxl.sample_lightning_timesteps(latents.shape[0], steps=lightning_steps)
    else:
        timesteps = sdxl.sample_timesteps(latents.shape[0])
    noise = torch.randn_like(latents)
    noisy = sdxl.add_noise(latents, noise, timesteps).float()

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
            noisy.to(sdxl.dtype),
            timesteps,
            pe,
            pe_pooled,
            ue,
            ue_pooled,
            add_time_ids,
            cfg_scale,
        )

    loss = (eps.float() - noise).pow(2)

    if mask_sky and rgba.shape[1] >= 4:
        alpha = rgba[:, 3:4, :, :]
        w = (alpha > 1e-3).float()
        denom = w.mean().clamp_min(1e-6)
        return (loss * w).mean() / (denom)

    return loss.mean()
