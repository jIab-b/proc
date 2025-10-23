"""End-to-end SDS training loop for voxel scenes."""

from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
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
from .nv_diff_render.utils import create_perspective_matrix


# ----------------------------------------------------------------------
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


def build_projection(intrinsics: Dict, height: int, width: int, device: torch.device) -> torch.Tensor:
    fov = math.radians(float(intrinsics.get("fovYDegrees", 60.0)))
    near = float(intrinsics.get("near", 0.1))
    far = float(intrinsics.get("far", 500.0))
    aspect = float(width / max(height, 1))
    proj = create_perspective_matrix(fov, aspect, near, far)
    return proj.to(device)


# ----------------------------------------------------------------------


def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MultiViewDataset(dataset_id=args.dataset_id, device=device)
    map_path = Path(args.map_path) if args.map_path else Path(f"maps/{args.dataset_id}/map.json")
    if not map_path.exists():
        raise FileNotFoundError(f"Map not found: {map_path}")

    scene = VoxelScene.from_map(map_path, device=device)
    renderer = VoxelRenderer(scene.grid)

    sampler = CameraSampler(
        dataset=dataset,
        grid_size=scene.grid.grid_size,
        world_scale=scene.world_scale,
        render_height=args.train_height,
        render_width=args.train_width,
        novel_view_prob=args.novel_view_prob,
        device=device,
    )

    debug_view_indices = list(range(min(5, len(dataset))))

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    if args.sdxl_dtype == "fp32":
        sdxl_dtype = torch.float32
    elif args.sdxl_dtype == "fp16" and device.type == "cuda":
        sdxl_dtype = torch.float16
    else:
        sdxl_dtype = torch.float16 if device.type == "cuda" else torch.float32
    sdxl = SDXLLightning(
        model_id=model_id,
        device=device,
        dtype=sdxl_dtype,
        height=args.train_height,
        width=args.train_width,
    )
    embeddings = sdxl.encode_prompt(args.prompt)

    optimizer = torch.optim.Adam(scene.parameters(), lr=args.learning_rate)

    run_dir = ensure_dir(Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S"))
    images_dir = ensure_dir(run_dir / "images")
    maps_dir = ensure_dir(run_dir / "maps")
    log_path = run_dir / "train.jsonl"

    with (run_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    fuckdump_dir = images_dir / "fuckdump"

    if args.fuckdump:
        fuckdump_dir.mkdir(exist_ok=True)
        
        print("FUCKDUMP: Saving initial primitives")
        initial_occ_tensor = scene.occupancy_probs().detach().cpu()
        initial_mat_tensor = scene.material_probs().detach().cpu()
        
        initial_dict = {
            "step": 0,
            "occupancy_probs": initial_occ_tensor.numpy().tolist(),
            "material_probs": initial_mat_tensor.numpy().tolist(),
            "grid_size": list(scene.grid.grid_size),
            "world_scale": scene.world_scale,
        }
        with (fuckdump_dir / "initial_primitives.json").open("w") as f:
            json.dump(initial_dict, f, indent=2)

        print("FUCKDUMP: Saving initial summary")
        initial_occ = scene.occupancy_probs().detach().cpu().numpy()
        initial_mat = scene.material_probs().detach().cpu().numpy()
        
        total_voxels = initial_occ.size
        init_summary = {
            "step": 0,
            "total_voxels": total_voxels,
            "num_changed_voxels": 0,
            "mean_abs_occ_diff": 0.0,
            "mean_abs_mat_diff": 0.0,
            "occ_diff_histogram": [0] * 10,
            "top_occ_changes": [],
            "slice_means_occ": [0.0] * initial_occ.shape[2],
            "mat_mean_changes": [0.0] * initial_mat.shape[-1],
            "threshold": 0.1,
        }
        summary_path = fuckdump_dir / "summaries.json"
        with summary_path.open("w") as f:
            json.dump([init_summary], f, indent=2)

    def render_debug_views(step_idx: int, temp: float, occ_threshold: float, max_blocks: Optional[int]) -> None:
        if not debug_view_indices:
            return
        fuckdump_dir.mkdir(exist_ok=True)
        for cam_idx in debug_view_indices:
            cam_view = dataset.get_view(cam_idx)
            cam_proj = build_projection(cam_view.intrinsics, args.train_height, args.train_width, device)
            cam_rgba = renderer.render(
                cam_view.view_matrix,
                cam_proj,
                args.train_height,
                args.train_width,
                temperature=temp,
                occupancy_threshold=occ_threshold,
                max_blocks=max_blocks,
            )
            cam_rgb = cam_rgba[:, :3]
            save_image(cam_rgb[0], fuckdump_dir / f"step_{step_idx:04d}_cam{cam_idx:02d}.png")

    # Preview initial state
    jpg_view = dataset.get_view(0)
    preview_proj = build_projection(jpg_view.intrinsics, args.train_height, args.train_width, device)
    preview_rgba = renderer.render(
        jpg_view.view_matrix,
        preview_proj,
        args.train_height,
        args.train_width,
        temperature=args.temperature_start,
        max_blocks=args.max_blocks,
    )
    save_image(preview_rgba[0, :3], images_dir / "preview_init.png")

    for step in range(1, args.steps + 1):
        t = (step - 1) / max(args.steps - 1, 1)
        temperature = lerp(args.temperature_start, args.temperature_end, t)

        sample = sampler.sample()
        # Schedules
        occ_thr = lerp(args.occupancy_threshold_start, args.occupancy_threshold_end, t)
        if args.max_blocks_start is not None or args.max_blocks_end is not None:
            start = args.max_blocks_start
            end = args.max_blocks_end
            if start is None:
                max_blocks_cur = None if t < 0.5 else end
            elif end is None:
                max_blocks_cur = start if t < 0.5 else None
            else:
                max_blocks_cur = int(lerp(float(start), float(end), t))
        else:
            max_blocks_cur = args.max_blocks
        rgba = renderer.render(
            sample.view,
            sample.proj,
            args.train_height,
            args.train_width,
            temperature=temperature,
            occupancy_threshold=occ_thr,
            max_blocks=max_blocks_cur,
        )

        rgb_pred = rgba[:, :3]

        total_loss = torch.zeros((), device=device)
        losses: Dict[str, float] = {}

        # Photometric supervision (dataset cameras only)
        if sample.from_dataset and args.photo_weight > 0.0:
            gt = sample.rgb.unsqueeze(0).to(device)
            gt_resized = F.interpolate(
                gt,
                size=(args.train_height, args.train_width),
                mode="bilinear",
                align_corners=False,
            )
            l_photo = photometric_loss(rgb_pred, gt_resized) * args.photo_weight
            total_loss = total_loss + l_photo
            losses["photometric"] = float(l_photo.item())

        if args.sds_weight > 0.0:
            l_sds_total = torch.zeros((), device=device)
            num_views = max(1, int(args.sds_views_per_step))
            for i in range(num_views):
                s_i = sample if i == 0 else sampler.sample()
                rgba_i = renderer.render(
                    s_i.view,
                    s_i.proj,
                    args.train_height,
                    args.train_width,
                    temperature=temperature,
                    occupancy_threshold=occ_thr,
                    max_blocks=max_blocks_cur,
                )
                l_sds_i = score_distillation_loss(
                    rgba_i,
                    sdxl,
                    embeddings,
                    cfg_scale=args.cfg_scale,
                    mask_sky=args.sds_mask_sky,
                    use_lightning_timesteps=args.sds_use_lightning_ts,
                    lightning_steps=args.sds_lightning_steps,
                )
                if torch.isfinite(l_sds_i):
                    l_sds_total = l_sds_total + l_sds_i
            l_sds = (l_sds_total / float(num_views)) * args.sds_weight
            total_loss = total_loss + l_sds
            losses["sds"] = float(l_sds.item())

        occ = scene.occupancy_probs()
        mats = scene.material_probs(temperature=max(temperature, 1e-4))
        reg = regularisation_losses(
            occ_probs=occ,
            mat_probs=mats,
            lambda_sparsity=args.lambda_sparsity,
            lambda_entropy=args.lambda_entropy,
            lambda_tv=args.lambda_tv,
        )

        for key, value in reg.items():
            total_loss = total_loss + value
            losses[key] = float(value.item())

        optimizer.zero_grad()
        if torch.isfinite(total_loss):
            total_loss.backward()
        optimizer.step()

        # FUCKDUMP MODE: Extensive debugging output
        if args.fuckdump:
            stats = scene.stats()

            # Create fuckdump subdirectory
            fuckdump_dir.mkdir(exist_ok=True)

            # Save rendered image at every step
            save_image(rgb_pred[0], fuckdump_dir / f"step_{step:04d}_render.png")

            # Save ground truth comparison if available
            if sample.from_dataset and sample.rgb is not None:
                gt_resized = F.interpolate(
                    sample.rgb.unsqueeze(0).to(device),
                    size=(args.train_height, args.train_width),
                    mode="bilinear",
                    align_corners=False,
                )
                save_image(gt_resized[0], fuckdump_dir / f"step_{step:04d}_ground_truth.png")

            # Render and save from all dataset cameras
            if debug_view_indices:
                print(
                    f"\nFUCKDUMP Step {step}/{args.steps} - Rendering {len(debug_view_indices)} dataset views..."
                )
                render_debug_views(step, temperature, occ_thr, max_blocks_cur)

            # Create noisy/x0 versions for SDS debugging
            if args.sds_weight > 0.0:
                with torch.no_grad():
                    latents = render_to_latents(rgb_pred, sdxl).float()
                    if args.sds_use_lightning_ts:
                        timestep = sdxl.sample_lightning_timesteps(1, steps=args.sds_lightning_steps)
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
                        args.cfg_scale,
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

        elif step % 500 == 0:
            render_debug_views(step, temperature, occ_thr, max_blocks_cur)

        if step % args.log_interval == 0 or step == 1:
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
                f"Step {step}/{args.steps} "
                f"loss={total_loss.item():.4f} "
                f"active={stats['num_active_voxels']}/{stats['total_voxels']}"
            )

        if step % args.image_interval == 0:
            save_image(rgb_pred[0], images_dir / f"step_{step:04d}.png")

        if step % args.map_interval == 0 or step == args.steps:
            scene.save_map(
                maps_dir / f"step_{step:04d}.json",
                metadata={"prompt": args.prompt, "step": step},
            )
            
            if args.fuckdump:
                print(f"FUCKDUMP: Computing stats for step {step}")
                curr_occ = scene.occupancy_probs().detach().cpu().numpy()
                curr_mat = scene.material_probs().detach().cpu().numpy()
                
                diff_occ = curr_occ - initial_occ
                diff_mat = curr_mat - initial_mat
                
                threshold = 0.1
                changed_mask = (np.abs(diff_occ) > threshold) | (np.abs(diff_mat).max(axis=-1) > threshold)
                num_changed = int(np.sum(changed_mask))
                
                mean_abs_occ = float(np.abs(diff_occ).mean())
                mean_abs_mat = float(np.abs(diff_mat).mean())
                
                # Histogram for occ diffs (10 bins -1 to 1)
                hist_occ, _ = np.histogram(diff_occ.flatten(), bins=10, range=(-1,1))
                hist_occ = [int(x) for x in hist_occ]
                
                # Top 10 occ changes
                flat_occ = diff_occ.flatten()
                top_indices = np.argsort(np.abs(flat_occ))[-10:][::-1]
                top_occ = [{'pos': tuple(int(x) for x in np.unravel_index(idx, diff_occ.shape)), 'diff': float(flat_occ[idx])} for idx in top_indices]
                
                # Slice means: mean |diff| per Z-slice
                slice_means_occ = []
                for z in range(diff_occ.shape[2]):
                    slice_means_occ.append(float(np.abs(diff_occ[:,:,z].mean())))
                
                # Mat top: simple mean change per material
                mat_mean_changes = []
                for m in range(diff_mat.shape[-1]):
                    mat_mean_changes.append(float(diff_mat[:,:,:,m].mean()))
                
                summary_entry = {
                    "step": int(step),
                    "num_changed_voxels": int(num_changed),
                    "mean_abs_occ_diff": mean_abs_occ,
                    "mean_abs_mat_diff": mean_abs_mat,
                    "occ_diff_histogram": hist_occ,
                    "top_occ_changes": top_occ,
                    "slice_means_occ": slice_means_occ,
                    "mat_mean_changes": mat_mean_changes,
                    "threshold": threshold,
                }
                
                with summary_path.open("r") as f:
                    summaries = json.load(f)
                summaries.append(summary_entry)
                with summary_path.open("w") as f:
                    json.dump(summaries, f, indent=2)
                
                print(f"  Stats: {num_changed} changed voxels, occ mean diff {mean_abs_occ:.4f}")

    final_map = run_dir / "final_map.json"
    scene.save_map(final_map, metadata={"prompt": args.prompt, "steps": args.steps})
    
    if args.fuckdump:
        print(f"FUCKDUMP: Computing final stats")
        final_step = int(args.steps)
        curr_occ = scene.occupancy_probs().detach().cpu().numpy()
        curr_mat = scene.material_probs().detach().cpu().numpy()
        
        diff_occ = curr_occ - initial_occ
        diff_mat = curr_mat - initial_mat
        
        threshold = 0.1
        changed_mask = (np.abs(diff_occ) > threshold) | (np.abs(diff_mat).max(axis=-1) > threshold)
        num_changed = int(np.sum(changed_mask))
        
        mean_abs_occ = float(np.abs(diff_occ).mean())
        mean_abs_mat = float(np.abs(diff_mat).mean())
        
        hist_occ, _ = np.histogram(diff_occ.flatten(), bins=10, range=(-1,1))
        hist_occ = [int(x) for x in hist_occ]
        
        flat_occ = diff_occ.flatten()
        top_indices = np.argsort(np.abs(flat_occ))[-10:][::-1]
        top_occ = [{'pos': tuple(int(x) for x in np.unravel_index(idx, diff_occ.shape)), 'diff': float(flat_occ[idx])} for idx in top_indices]
        
        slice_means_occ = []
        for z in range(diff_occ.shape[2]):
            slice_means_occ.append(float(np.abs(diff_occ[:,:,z].mean())))
        
        mat_mean_changes = []
        for m in range(diff_mat.shape[-1]):
            mat_mean_changes.append(float(diff_mat[:,:,:,m].mean()))
        
        summary_entry = {
            "step": final_step,
            "num_changed_voxels": int(num_changed),
            "mean_abs_occ_diff": mean_abs_occ,
            "mean_abs_mat_diff": mean_abs_mat,
            "occ_diff_histogram": hist_occ,
            "top_occ_changes": top_occ,
            "slice_means_occ": slice_means_occ,
            "mat_mean_changes": mat_mean_changes,
            "threshold": threshold,
        }
        
        with summary_path.open("r") as f:
            summaries = json.load(f)
        summaries.append(summary_entry)
        with summary_path.open("w") as f:
            json.dump(summaries, f, indent=2)
        
        print(f"  Final stats: {num_changed} changed voxels, occ mean diff {mean_abs_occ:.4f}")

    save_image(rgb_pred[0], images_dir / "final.png")

    print(f"Training finished. Outputs saved in {run_dir}")
    return run_dir


# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voxel SDS training")
    parser.add_argument("--prompt", required=True, help="Text prompt for SDS guidance")
    parser.add_argument("--dataset_id", type=int, default=1)
    parser.add_argument("--map_path", type=str, default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--temperature_start", type=float, default=0.1)
    parser.add_argument("--temperature_end", type=float, default=0.1)
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
    parser.add_argument("--occupancy_threshold_start", type=float, default=0.1)
    parser.add_argument("--occupancy_threshold_end", type=float, default=0.1)
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

    if args.max_blocks <= 0:
        args.max_blocks = None
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
