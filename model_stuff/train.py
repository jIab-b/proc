"""End-to-end SDS training loop for voxel scenes."""

from __future__ import annotations

import argparse
import json
import math
import random
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
from .sds import SDSDebugArtifacts, score_distillation_loss
from .sdxl_lightning import SDXLLightning
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
    scene.train()
    scene.grid.training_occ_threshold = float(args.train_occupancy_threshold)
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
        initial_mask_tensor = scene.mask_probs().detach().cpu()
        initial_edit_tensor = scene.grid.edit_logits.detach().cpu()
        initial_palette_tensor = scene.grid.palette().detach().cpu()

        initial_dict = {
            "step": 0,
            "occupancy_probs": initial_occ_tensor.numpy().tolist(),
            "material_probs": initial_mat_tensor.numpy().tolist(),
            "mask_probs": initial_mask_tensor.numpy().tolist(),
            "edit_logits": initial_edit_tensor.numpy().tolist(),
            "palette": initial_palette_tensor.numpy().tolist(),
            "grid_size": list(scene.grid.grid_size),
            "world_scale": scene.world_scale,
        }
        with (fuckdump_dir / "initial_primitives.json").open("w") as f:
            json.dump(initial_dict, f, indent=2)

        print("FUCKDUMP: Saving initial summary")
        change_threshold = max(float(args.fuckdump_change_threshold), 1e-4)

        initial_occ = scene.occupancy_probs().detach().cpu().numpy()
        initial_mat = scene.material_probs().detach().cpu().numpy()
        initial_mask = scene.mask_probs().detach().cpu().numpy()
        initial_edit = scene.grid.edit_logits.detach().cpu().numpy()
        
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
            "threshold": change_threshold,
        }
        summary_path = fuckdump_dir / "summaries.json"
        with summary_path.open("w") as f:
            json.dump([init_summary], f, indent=2)

    def render_debug_views(step_idx: int, temp: float, occ_threshold: float, max_blocks: Optional[int]) -> None:
        if not debug_view_indices:
            return
        fuckdump_dir.mkdir(exist_ok=True)
        prev_mode = scene.training
        scene.eval()
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
        if prev_mode:
            scene.train()

    # Preview initial state
    jpg_view = dataset.get_view(0)
    preview_proj = build_projection(jpg_view.intrinsics, args.train_height, args.train_width, device)
    prev_training_mode = scene.training
    scene.eval()
    preview_rgba = renderer.render(
        jpg_view.view_matrix,
        preview_proj,
        args.train_height,
        args.train_width,
        temperature=args.temperature_start,
        max_blocks=args.max_blocks,
    )
    if prev_training_mode:
        scene.train()
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
        sds_debug_artifacts: Optional[SDSDebugArtifacts] = None

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
                collect_debug = args.fuckdump and i == 0
                l_sds_i, debug_info = score_distillation_loss(
                    rgba_i,
                    sdxl,
                    embeddings,
                    cfg_scale=args.cfg_scale,
                    mask_sky=args.sds_mask_sky,
                    use_lightning_timesteps=args.sds_use_lightning_ts,
                    lightning_steps=args.sds_lightning_steps,
                    collect_debug=collect_debug,
                )
                if torch.isfinite(l_sds_i):
                    l_sds_total = l_sds_total + l_sds_i
                    if debug_info is not None:
                        sds_debug_artifacts = debug_info
            l_sds = (l_sds_total / float(num_views)) * args.sds_weight
            total_loss = total_loss + l_sds
            losses["sds"] = float(l_sds.item())

        occ = scene.occupancy_probs()
        mats = scene.material_probs(temperature=max(temperature, 1e-4))
        mask_probs = scene.mask_probs()
        reg = regularisation_losses(
            occ_probs=occ,
            mat_probs=mats,
            mask_probs=mask_probs,
            edit_logits=scene.grid.edit_logits,
            palette_embed=scene.grid.palette_embed,
            palette_target=scene.grid.palette_target,
            lambda_mask=args.lambda_sparsity,
            lambda_entropy=args.lambda_entropy,
            lambda_edit_tv=args.lambda_tv,
            lambda_edit_l2=args.lambda_edit_l2,
            lambda_palette=args.lambda_palette_l2,
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

            # Create noisy/x0 versions for SDS debugging using training artifacts
            if args.sds_weight > 0.0 and sds_debug_artifacts is not None:
                save_image(
                    sds_debug_artifacts.noisy_rgb[0],
                    fuckdump_dir / f"step_{step:04d}_noisy.png",
                )
                save_image(
                    sds_debug_artifacts.x0_rgb[0],
                    fuckdump_dir / f"step_{step:04d}_x0.png",
                )

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
                threshold=args.export_threshold,
                metadata={"prompt": args.prompt, "step": step},
            )
            
            if args.fuckdump:
                print(f"FUCKDUMP: Computing stats for step {step}")
                curr_occ = scene.occupancy_probs().detach().cpu().numpy()
                curr_mat = scene.material_probs().detach().cpu().numpy()
                curr_mask = scene.mask_probs().detach().cpu().numpy()
                curr_edit = scene.grid.edit_logits.detach().cpu().numpy()

                diff_occ = curr_occ - initial_occ
                diff_mat = curr_mat - initial_mat
                diff_mask = curr_mask - initial_mask
                diff_edit = curr_edit - initial_edit
                
                changed_mask = (
                    np.abs(diff_occ) > change_threshold
                ) | (np.abs(diff_mat).max(axis=-1) > change_threshold)
                num_changed = int(np.sum(changed_mask))
                
                mean_abs_occ = float(np.abs(diff_occ).mean())
                mean_abs_mat = float(np.abs(diff_mat).mean())
                mean_abs_mask = float(np.abs(diff_mask).mean())
                mean_abs_edit = float(np.abs(diff_edit).mean())
                
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
                    mat_mean_changes.append(float(diff_mat[:, :, :, m].mean()))

                mask_mean_change = float(diff_mask.mean())
                edit_mean_change = float(diff_edit.mean())

                summary_entry = {
                    "step": int(step),
                    "num_changed_voxels": int(num_changed),
                    "mean_abs_occ_diff": mean_abs_occ,
                    "mean_abs_mat_diff": mean_abs_mat,
                    "mean_abs_mask_diff": mean_abs_mask,
                    "mean_abs_edit_diff": mean_abs_edit,
                    "occ_diff_histogram": hist_occ,
                    "top_occ_changes": top_occ,
                    "slice_means_occ": slice_means_occ,
                    "mat_mean_changes": mat_mean_changes,
                    "mask_mean_change": mask_mean_change,
                    "edit_mean_change": edit_mean_change,
                    "threshold": change_threshold,
                }
                
                with summary_path.open("r") as f:
                    summaries = json.load(f)
                summaries.append(summary_entry)
                with summary_path.open("w") as f:
                    json.dump(summaries, f, indent=2)
                
                print(f"  Stats: {num_changed} changed voxels, occ mean diff {mean_abs_occ:.4f}")

    final_map = run_dir / "final_map.json"
    scene.save_map(
        final_map,
        threshold=args.export_threshold,
        metadata={"prompt": args.prompt, "steps": args.steps},
    )
    
    if args.fuckdump:
        print(f"FUCKDUMP: Computing final stats")
        final_step = int(args.steps)
        curr_occ = scene.occupancy_probs().detach().cpu().numpy()
        curr_mat = scene.material_probs().detach().cpu().numpy()
        curr_mask = scene.mask_probs().detach().cpu().numpy()
        curr_edit = scene.grid.edit_logits.detach().cpu().numpy()

        diff_occ = curr_occ - initial_occ
        diff_mat = curr_mat - initial_mat
        diff_mask = curr_mask - initial_mask
        diff_edit = curr_edit - initial_edit
        
        changed_mask = (
            np.abs(diff_occ) > change_threshold
        ) | (np.abs(diff_mat).max(axis=-1) > change_threshold)
        num_changed = int(np.sum(changed_mask))
        
        mean_abs_occ = float(np.abs(diff_occ).mean())
        mean_abs_mat = float(np.abs(diff_mat).mean())
        mean_abs_mask = float(np.abs(diff_mask).mean())
        mean_abs_edit = float(np.abs(diff_edit).mean())
        
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
            mat_mean_changes.append(float(diff_mat[:, :, :, m].mean()))

        mask_mean_change = float(diff_mask.mean())
        edit_mean_change = float(diff_edit.mean())

        summary_entry = {
            "step": final_step,
            "num_changed_voxels": int(num_changed),
            "mean_abs_occ_diff": mean_abs_occ,
            "mean_abs_mat_diff": mean_abs_mat,
            "mean_abs_mask_diff": mean_abs_mask,
            "mean_abs_edit_diff": mean_abs_edit,
            "occ_diff_histogram": hist_occ,
            "top_occ_changes": top_occ,
            "slice_means_occ": slice_means_occ,
            "mat_mean_changes": mat_mean_changes,
            "mask_mean_change": mask_mean_change,
            "edit_mean_change": edit_mean_change,
            "threshold": change_threshold,
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
    parser.add_argument("--prompt", default="Inside a vast, ancient fridge, discover a sprawling, ominous cityscape where Lovecraftian horrors lurk amid Gothic Eastern European spires. Shadows play on cobblestone streets, and the chilling mist of refrigerated air weaves through intricate stone arches, whispering untold eldritch secrets. Mysterious, glowing runes pulse on cold walls, illuminating the night with", help="Text prompt for SDS guidance")
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
    parser.add_argument("--lambda_edit_l2", type=float, default=1e-4)
    parser.add_argument("--lambda_palette_l2", type=float, default=1e-4)
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
    parser.add_argument("--export_threshold", type=float, default=0.5)
    parser.add_argument("--fuckdump_change_threshold", type=float, default=0.01)
    parser.add_argument("--train_occupancy_threshold", type=float, default=0.05)
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
