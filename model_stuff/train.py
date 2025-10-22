"""End-to-end SDS training loop for voxel scenes."""

from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .cameras import CameraSampler
from .dataset import MultiViewDataset
from .losses import photometric_loss, regularisation_losses
from .model import VoxelScene
from .renderer_nvd import VoxelRenderer
from .sds import score_distillation_loss, render_to_latents
from .sdxl_lightning import SDXLLightning, LATENT_SCALING


# ----------------------------------------------------------------------
# Configuration container
# ----------------------------------------------------------------------


@dataclass
class TrainConfig:
    prompt: str
    dataset_id: int = 1
    map_path: Optional[str] = None
    steps: int = 100
    learning_rate: float = 0.01
    cfg_scale: float = 7.5
    temperature_start: float = 2.0
    temperature_end: float = 0.5
    sds_weight: float = 1.0
    photo_weight: float = 1.0
    lambda_sparsity: float = 1e-3
    lambda_entropy: float = 1e-4
    lambda_tv: float = 0.0
    novel_view_prob: float = 0.2
    train_height: int = 192
    train_width: int = 192
    max_blocks: Optional[int] = 50000
    # SDS controls
    sds_views_per_step: int = 1
    sds_mask_sky: bool = True
    sds_use_lightning_ts: bool = True
    sds_lightning_steps: int = 4
    # Scheduling
    occupancy_threshold_start: float = 0.0
    occupancy_threshold_end: float = 0.01
    max_blocks_start: Optional[int] = None
    max_blocks_end: Optional[int] = 50000
    output_dir: str = "out_local/voxel_sds"
    fuckdump: bool = False
    log_interval: int = 10
    image_interval: int = 50
    map_interval: int = 100
    seed: int = 42
    sdxl_dtype: str = "auto"  # auto|fp16|fp32


# ----------------------------------------------------------------------


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_image(rgb: torch.Tensor, path: Path) -> None:
    rgb = torch.nan_to_num(rgb)
    arr = rgb.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    arr = (arr * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


# ----------------------------------------------------------------------


def train(config: TrainConfig) -> Path:
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MultiViewDataset(dataset_id=config.dataset_id, device=device)
    map_path = Path(config.map_path) if config.map_path else Path(f"maps/{config.dataset_id}/map.json")
    if not map_path.exists():
        raise FileNotFoundError(f"Map not found: {map_path}")

    scene = VoxelScene.from_map(map_path, device=device)
    renderer = VoxelRenderer(scene.grid)

    sampler = CameraSampler(
        dataset=dataset,
        grid_size=scene.grid.grid_size,
        world_scale=scene.world_scale,
        novel_view_prob=config.novel_view_prob,
        device=device,
    )

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    if config.sdxl_dtype == "fp32":
        sdxl_dtype = torch.float32
    elif config.sdxl_dtype == "fp16" and device.type == "cuda":
        sdxl_dtype = torch.float16
    else:
        sdxl_dtype = torch.float16 if device.type == "cuda" else torch.float32
    sdxl = SDXLLightning(
        model_id=model_id,
        device=device,
        dtype=sdxl_dtype,
        height=config.train_height,
        width=config.train_width,
    )
    embeddings = sdxl.encode_prompt(config.prompt)

    optimizer = torch.optim.Adam(scene.parameters(), lr=config.learning_rate)

    run_dir = ensure_dir(Path(config.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S"))
    images_dir = ensure_dir(run_dir / "images")
    maps_dir = ensure_dir(run_dir / "maps")
    log_path = run_dir / "train.jsonl"

    with (run_dir / "config.json").open("w") as f:
        json.dump(asdict(config), f, indent=2)

    # Preview initial state
    jpg_view = dataset.get_view(0)
    preview_rgba = renderer.render(
        jpg_view.view_matrix,
        jpg_view.proj_matrix,
        config.train_height,
        config.train_width,
        temperature=config.temperature_start,
        max_blocks=config.max_blocks,
    )
    save_image(preview_rgba[0, :3], images_dir / "preview_init.png")

    for step in range(1, config.steps + 1):
        t = (step - 1) / max(config.steps - 1, 1)
        temperature = lerp(config.temperature_start, config.temperature_end, t)

        sample = sampler.sample()
        # Schedules
        occ_thr = lerp(config.occupancy_threshold_start, config.occupancy_threshold_end, t)
        if config.max_blocks_start is not None or config.max_blocks_end is not None:
            start = config.max_blocks_start
            end = config.max_blocks_end
            if start is None:
                max_blocks_cur = None if t < 0.5 else end
            elif end is None:
                max_blocks_cur = start if t < 0.5 else None
            else:
                max_blocks_cur = int(lerp(float(start), float(end), t))
        else:
            max_blocks_cur = config.max_blocks
        rgba = renderer.render(
            sample.view,
            sample.proj,
            config.train_height,
            config.train_width,
            temperature=temperature,
            occupancy_threshold=occ_thr,
            max_blocks=max_blocks_cur,
        )

        rgb_pred = rgba[:, :3]

        total_loss = torch.zeros((), device=device)
        losses: Dict[str, float] = {}

        # Photometric supervision (dataset cameras only)
        if sample.from_dataset and config.photo_weight > 0.0:
            gt = sample.rgb.unsqueeze(0).to(device)
            gt_resized = F.interpolate(
                gt,
                size=(config.train_height, config.train_width),
                mode="bilinear",
                align_corners=False,
            )
            l_photo = photometric_loss(rgb_pred, gt_resized) * config.photo_weight
            total_loss = total_loss + l_photo
            losses["photometric"] = float(l_photo.item())

        if config.sds_weight > 0.0:
            l_sds_total = torch.zeros((), device=device)
            num_views = max(1, int(config.sds_views_per_step))
            for i in range(num_views):
                s_i = sample if i == 0 else sampler.sample()
                rgba_i = renderer.render(
                    s_i.view,
                    s_i.proj,
                    config.train_height,
                    config.train_width,
                    temperature=temperature,
                    occupancy_threshold=occ_thr,
                    max_blocks=max_blocks_cur,
                )
                l_sds_i = score_distillation_loss(
                    rgba_i,
                    sdxl,
                    embeddings,
                    cfg_scale=config.cfg_scale,
                    mask_sky=config.sds_mask_sky,
                    use_lightning_timesteps=config.sds_use_lightning_ts,
                    lightning_steps=config.sds_lightning_steps,
                )
                if torch.isfinite(l_sds_i):
                    l_sds_total = l_sds_total + l_sds_i
            l_sds = (l_sds_total / float(num_views)) * config.sds_weight
            total_loss = total_loss + l_sds
            losses["sds"] = float(l_sds.item())

        occ = scene.occupancy_probs()
        mats = scene.material_probs(temperature=max(temperature, 1e-4))
        reg = regularisation_losses(
            occ_probs=occ,
            mat_probs=mats,
            lambda_sparsity=config.lambda_sparsity,
            lambda_entropy=config.lambda_entropy,
            lambda_tv=config.lambda_tv,
        )

        for key, value in reg.items():
            total_loss = total_loss + value
            losses[key] = float(value.item())

        optimizer.zero_grad()
        if torch.isfinite(total_loss):
            total_loss.backward()
            optimizer.step()

        # FUCKDUMP MODE: Extensive debugging output
        if config.fuckdump:
            stats = scene.stats()

            # Create fuckdump subdirectory
            fuckdump_dir = images_dir / "fuckdump"
            fuckdump_dir.mkdir(exist_ok=True)

            # Save rendered image at every step
            save_image(rgb_pred[0], fuckdump_dir / f"step_{step:04d}_render.png")

            # Save ground truth comparison if available
            if sample.from_dataset and sample.rgb is not None:
                gt_resized = F.interpolate(
                    sample.rgb.unsqueeze(0).to(device),
                    size=(config.train_height, config.train_width),
                    mode="bilinear",
                    align_corners=False,
                )
                save_image(gt_resized[0], fuckdump_dir / f"step_{step:04d}_ground_truth.png")

            # Render and save from all dataset cameras
            print(f"\nFUCKDUMP Step {step}/{config.steps} - Rendering all {len(dataset)} dataset views...")
            for cam_idx in range(len(dataset)):
                cam_view = dataset.get_view(cam_idx)
                cam_rgba = renderer.render(
                    cam_view.view_matrix,
                    cam_view.proj_matrix,
                    config.train_height,
                    config.train_width,
                    temperature=temperature,
                    occupancy_threshold=occ_thr,
                    max_blocks=max_blocks_cur,
                )
                cam_rgb = cam_rgba[:, :3]
                save_image(cam_rgb[0], fuckdump_dir / f"step_{step:04d}_cam{cam_idx:02d}.png")

            # Create noisy/x0 versions for SDS debugging
            if config.sds_weight > 0.0:
                with torch.no_grad():
                    latents = render_to_latents(rgb_pred, sdxl).float()
                    if config.sds_use_lightning_ts:
                        timestep = sdxl.sample_lightning_timesteps(1, steps=config.sds_lightning_steps)
                    else:
                        timestep = sdxl.sample_timesteps(1)
                    noise = torch.randn_like(latents)
                    noisy_latents = sdxl.add_noise(latents, noise, timestep).float()
                    noisy_latents = noisy_latents.clamp(-3 * LATENT_SCALING, 3 * LATENT_SCALING)
                    vae = sdxl.vae if hasattr(sdxl, 'vae') else sdxl.pipe.vae
                    vae_config = getattr(vae, 'config', None)
                    force_upcast = bool(getattr(vae_config, 'force_upcast', False))
                    vae_dtype = next(vae.parameters()).dtype
                    vae_device = next(vae.parameters()).device

                    def decode_latents(latents: torch.Tensor) -> torch.Tensor:
                        if force_upcast:
                            vae.to(dtype=torch.float32)
                            decode_dtype = torch.float32
                            decode_ctx = nullcontext()
                        else:
                            decode_dtype = vae_dtype if vae_device.type == 'cuda' else torch.float32
                            decode_ctx = (
                                torch.autocast(device_type=vae_device.type, dtype=vae_dtype)
                                if vae_device.type == 'cuda' and vae_dtype != torch.float32
                                else nullcontext()
                            )
                        z_local = (latents / LATENT_SCALING).to(device=vae_device, dtype=decode_dtype)
                        with decode_ctx:
                            decoded = vae.decode(z_local).sample
                        decoded = decoded.to(dtype=torch.float32)
                        return (decoded * 0.5 + 0.5).clamp(0, 1)

                    noisy_rgb = decode_latents(noisy_latents)
                    save_image(noisy_rgb[0], fuckdump_dir / f"step_{step:04d}_noisy.png")

                    # x0 prediction visualization
                    pe, pe_pooled, ue, ue_pooled, add_time_ids = embeddings
                    eps_hat = sdxl.eps_pred_cfg(
                        noisy_latents.to(sdxl.dtype),
                        timestep,
                        pe,
                        pe_pooled,
                        ue,
                        ue_pooled,
                        add_time_ids,
                        config.cfg_scale,
                    ).float()
                    # Compute x0 from scheduler alphas
                    alphas_cumprod = sdxl.scheduler.alphas_cumprod.to(noisy_latents.device, dtype=noisy_latents.dtype)
                    timestep_indices = timestep.to(device=alphas_cumprod.device, dtype=torch.long)
                    a_t = alphas_cumprod[timestep_indices]
                    sqrt_a = a_t.sqrt()
                    sqrt_oma = (1.0 - a_t).sqrt()
                    x0_pred = (noisy_latents - sqrt_oma * eps_hat) / sqrt_a.clamp_min(1e-6)
                    x0_pred = x0_pred.clamp(-3 * LATENT_SCALING, 3 * LATENT_SCALING)
                    x0_rgb = decode_latents(x0_pred)
                    save_image(x0_rgb[0], fuckdump_dir / f"step_{step:04d}_x0.png")

            # Detailed loss breakdown
            print(f"  📊 LOSSES:")
            for loss_name, loss_value in losses.items():
                print(f"    {loss_name}: {loss_value:.6f}")
            print(f"    TOTAL: {total_loss.item():.6f}")

            # Voxel statistics
            print(f"  🧊 VOXELS: active={stats['num_active_voxels']}/{stats['total_voxels']} "
                  f"({stats['density']*100:.2f}%)")
            print(f"  🌡️ TEMP: {temperature:.3f}")

            # Material distribution
            mat_dist = stats.get('material_distribution', {})
            if mat_dist:
                print(f"  🎨 MATERIALS:")
                for mat, pct in sorted(mat_dist.items(), key=lambda x: -x[1])[:5]:
                    if pct > 0.01:
                        print(f"    {mat}: {pct:.1f}%")

            # Camera info
            cam_type = "DATASET" if sample.from_dataset else "NOVEL"
            cam_info = f"idx={sample.index}" if sample.from_dataset else "random"
            print(f"  📷 CAMERA: {cam_type} {cam_info}")

        if step % config.log_interval == 0 or step == 1:
            stats = scene.stats()
            log_entry = {
                "step": step,
                "temperature": temperature,
                "loss_total": float(total_loss.item()),
                "losses": losses,
                "stats": stats,
                "sample": {
                    "from_dataset": sample.from_dataset,
                    "index": sample.index,
                },
            }
            with log_path.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")
            print(
                f"Step {step}/{config.steps} "
                f"loss={total_loss.item():.4f} "
                f"active={stats['num_active_voxels']}/{stats['total_voxels']}"
            )

        if step % config.image_interval == 0:
            save_image(rgb_pred[0], images_dir / f"step_{step:04d}.png")

        if step % config.map_interval == 0 or step == config.steps:
            scene.save_map(
                maps_dir / f"step_{step:04d}.json",
                metadata={"prompt": config.prompt, "step": step},
            )

    final_map = run_dir / "final_map.json"
    scene.save_map(final_map, metadata={"prompt": config.prompt, "steps": config.steps})
    save_image(rgb_pred[0], images_dir / "final.png")

    print(f"Training finished. Outputs saved in {run_dir}")
    return run_dir


# ----------------------------------------------------------------------


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Voxel SDS training")
    parser.add_argument("--prompt", required=True, help="Text prompt for SDS guidance")
    parser.add_argument("--dataset_id", type=int, default=1)
    parser.add_argument("--map_path", type=str, default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--temperature_start", type=float, default=2.0)
    parser.add_argument("--temperature_end", type=float, default=0.5)
    parser.add_argument("--sds_weight", type=float, default=1.0)
    parser.add_argument("--photo_weight", type=float, default=1.0)
    parser.add_argument("--lambda_sparsity", type=float, default=1e-3)
    parser.add_argument("--lambda_entropy", type=float, default=1e-4)
    parser.add_argument("--lambda_tv", type=float, default=0.0)
    parser.add_argument("--novel_view_prob", type=float, default=0.2)
    parser.add_argument("--train_height", type=int, default=192)
    parser.add_argument("--train_width", type=int, default=192)
    parser.add_argument("--max_blocks", type=int, default=50000)
    parser.add_argument("--sds_views_per_step", type=int, default=1)
    parser.add_argument("--sds_mask_sky", action="store_true")
    parser.add_argument("--sds_use_lightning_ts", action="store_true")
    parser.add_argument("--sds_lightning_steps", type=int, default=4)
    parser.add_argument("--occupancy_threshold_start", type=float, default=0.0)
    parser.add_argument("--occupancy_threshold_end", type=float, default=0.01)
    parser.add_argument("--max_blocks_start", type=int, default=None)
    parser.add_argument("--max_blocks_end", type=int, default=50000)
    parser.add_argument("--output_dir", type=str, default="out_local/voxel_sds")
    parser.add_argument("--fuckdump", action="store_true", help="Enable extensive debugging output")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--image_interval", type=int, default=50)
    parser.add_argument("--map_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sdxl_dtype", type=str, default="auto", choices=["auto", "fp16", "fp32"])
    args = parser.parse_args()

    config = TrainConfig(**vars(args))
    if args.max_blocks <= 0:
        config.max_blocks = None
    return config


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
