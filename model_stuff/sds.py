"""Score Distillation Sampling utilities for SDXL-Lightning guidance."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .sdxl_lightning import SDXLLightning, LATENT_SCALING


PromptEmbeddings = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class SDSDebugArtifacts:
    """Detached tensors needed to visualise SDS guidance behaviour."""

    latents: torch.Tensor
    noise: torch.Tensor
    noisy_latents: torch.Tensor
    timesteps: torch.Tensor
    eps_pred: torch.Tensor
    x0_latents: torch.Tensor
    noisy_rgb: torch.Tensor
    x0_rgb: torch.Tensor


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


def decode_latents(latents: torch.Tensor, sdxl: SDXLLightning) -> torch.Tensor:
    """Decode SDXL latents back to RGB tensors in `[0,1]` (B,3,H,W)."""
    vae = sdxl.vae if hasattr(sdxl, "vae") else sdxl.pipe.vae
    vae_config = getattr(vae, "config", None)
    force_upcast = bool(getattr(vae_config, "force_upcast", False))

    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype

    if force_upcast:
        vae = vae.to(dtype=torch.float32)
        decode_dtype = torch.float32
        decode_ctx = nullcontext()
    else:
        decode_dtype = vae_dtype if vae_device.type == "cuda" else torch.float32
        decode_ctx = (
            torch.autocast(device_type=vae_device.type, dtype=vae_dtype)
            if vae_device.type == "cuda" and vae_dtype != torch.float32
            else nullcontext()
        )

    latents = (latents / LATENT_SCALING).to(device=vae_device, dtype=decode_dtype)

    with decode_ctx:
        decoded = vae.decode(latents).sample

    decoded = decoded.to(dtype=torch.float32)
    return (decoded * 0.5 + 0.5).clamp(0.0, 1.0)


def predict_x0_from_eps(
    noisy_latents: torch.Tensor,
    eps_pred: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler,
) -> torch.Tensor:
    """Reconstruct the predicted clean latents from epsilon predictions."""
    device = noisy_latents.device
    dtype = noisy_latents.dtype

    alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=dtype)
    t_idx = timesteps.to(device=alphas_cumprod.device, dtype=torch.long)
    a_t = alphas_cumprod[t_idx].to(device=device, dtype=dtype)

    # Broadcast coefficients over latent spatial dims
    view_shape = (-1,) + (1,) * (noisy_latents.ndim - 1)
    sqrt_a = a_t.sqrt().view(view_shape)
    sqrt_oma = (1.0 - a_t).sqrt().view(view_shape)

    x0 = (noisy_latents - sqrt_oma * eps_pred) / sqrt_a.clamp_min(1e-6)
    return x0.clamp(-3 * LATENT_SCALING, 3 * LATENT_SCALING)


def score_distillation_loss(
    rgba: torch.Tensor,
    sdxl: SDXLLightning,
    embeddings: PromptEmbeddings,
    cfg_scale: float = 7.5,
    mask_sky: bool = False,
    use_lightning_timesteps: bool = True,
    lightning_steps: int = 4,
    collect_debug: bool = False,
) -> Tuple[torch.Tensor, Optional[SDSDebugArtifacts]]:
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

    eps = eps.float()
    loss = (eps - noise).pow(2)

    if mask_sky and rgba.shape[1] >= 4:
        alpha = rgba[:, 3:4, :, :]
        w = (alpha > 1e-3).float()
        denom = w.mean().clamp_min(1e-6)
        loss_value = (loss * w).mean() / denom
    else:
        loss_value = loss.mean()

    debug: Optional[SDSDebugArtifacts] = None
    if collect_debug:
        with torch.no_grad():
            x0_pred = predict_x0_from_eps(noisy, eps, timesteps, sdxl.scheduler)
            noisy_rgb = decode_latents(noisy, sdxl)
            x0_rgb = decode_latents(x0_pred, sdxl)

        debug = SDSDebugArtifacts(
            latents=latents.detach().to(device="cpu", dtype=torch.float32),
            noise=noise.detach().to(device="cpu", dtype=torch.float32),
            noisy_latents=noisy.detach().to(device="cpu", dtype=torch.float32),
            timesteps=timesteps.detach().to(device="cpu"),
            eps_pred=eps.detach().to(device="cpu", dtype=torch.float32),
            x0_latents=x0_pred.detach().to(device="cpu", dtype=torch.float32),
            noisy_rgb=noisy_rgb.detach().to(device="cpu", dtype=torch.float32),
            x0_rgb=x0_rgb.detach().to(device="cpu", dtype=torch.float32),
        )

    return loss_value, debug
