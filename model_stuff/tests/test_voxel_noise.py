#!/usr/bin/env python3
"""
Test script for voxel grid noise addition.

Loads a voxel grid from map, adds noise to logits, renders preview image.
No SDXL dependency - pure voxel grid testing.
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
import json
from datetime import datetime
import time
import gc

# Import voxel grid components
from ..voxel_grid import DifferentiableVoxelGrid
from ..map_io import load_map_to_grid, get_grid_stats
from ..nv_diff_render.utils import load_camera_matrices_from_metadata
from ..sdxl_lightning import SDXLLightning
from contextlib import nullcontext


def load_test_camera(dataset_path="datasets/1"):
    """Load first camera from dataset for testing."""
    metadata_path = Path(dataset_path) / "metadata.json"

    if not metadata_path.exists():
        # Fallback: create a simple camera looking at origin
        print("No dataset found, using default camera...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Simple camera matrices (looking at origin from closer position)
        # Voxel grid is scaled by world_scale=2.0, so coords are roughly [-2, 2]
        view = torch.eye(4, device=device)
        view[:3, 3] = torch.tensor([5, 5, 5], device=device)  # closer position
        view = torch.inverse(view)  # view matrix

        proj = torch.zeros(4, 4, device=device)
        proj[0, 0] = 2.0 / 1.0  # fov approximation
        proj[1, 1] = 2.0 / 1.0
        proj[2, 2] = -1.0
        proj[2, 3] = -0.1
        proj[3, 2] = -1.0

        return view, proj

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Load first camera
    view, proj = load_camera_matrices_from_metadata(metadata, 0)
    return view, proj


def add_noise_to_logits(occ_logits, mat_logits, noise_scale=0.1, temp: float = 1.0, edge_strength: float = 0.25, far_strength: float = 0.02):
    print(f"Adding noise with scale {noise_scale}")

    start_time = time.time()

    occ_orig = torch.sigmoid(occ_logits)
    solid = occ_orig > 0.5
    X, Y, Z = occ_orig.shape

    nb = torch.zeros_like(solid)
    nb[:-1, :, :] |= solid[1:, :, :]
    nb[1:, :, :] |= solid[:-1, :, :]
    nb[:, :-1, :] |= solid[:, 1:, :]
    nb[:, 1:, :] |= solid[:, :-1, :]
    nb[:, :, :-1] |= solid[:, :, 1:]
    nb[:, :, 1:] |= solid[:, :, :-1]
    near_solid = nb | solid

    occ_alpha = torch.where(near_solid, torch.full_like(occ_orig, noise_scale * edge_strength), torch.full_like(occ_orig, noise_scale * far_strength))

    occ_target = torch.rand_like(occ_orig)
    occ_mixed = occ_orig * (1.0 - occ_alpha) + occ_target * occ_alpha
    occ_mixed = occ_mixed.clamp(1e-6, 1 - 1e-6)
    noisy_occ = torch.log(occ_mixed) - torch.log(1.0 - occ_mixed)

    probs_orig = torch.softmax(mat_logits / max(temp, 1e-6), dim=-1)
    probs_rand = torch.softmax(torch.randn_like(mat_logits), dim=-1)
    probs_mixed = (1.0 - noise_scale) * probs_orig + noise_scale * probs_rand
    probs_mixed = probs_mixed / probs_mixed.sum(dim=-1, keepdim=True)
    probs_mixed = probs_mixed.clamp(min=1e-6)
    noisy_mat = torch.log(probs_mixed)

    noise_gen_time = time.time() - start_time
    print(".4f")

    return noisy_occ, noisy_mat


def render_and_save_preview(grid, camera_view, camera_proj, output_path, img_size=(192, 192)):
    """Render preview and save as PNG."""
    print(f"Rendering preview image ({img_size[0]}x{img_size[1]})...")

    device = grid.device

    # Render
    rgba = grid(camera_view, camera_proj, img_size[0], img_size[1])

    # Convert to RGB image
    rgb = rgba[0, :3, :, :].detach().cpu().permute(1, 2, 0).numpy()
    rgb = np.clip(rgb, 0, 1)
    rgb = (rgb * 255).astype(np.uint8)

    # Save image
    img = Image.fromarray(rgb)
    img.save(output_path)
    print(f"Saved preview to: {output_path}")

    return img


def main():
    parser = argparse.ArgumentParser(description="Test voxel grid noise addition")
    parser.add_argument("--map_path", type=str, default="maps/1/map.json",
                        help="Path to map.json file")
    parser.add_argument("--noise_scale", type=float, default=0.1,
                        help="Scale of noise to add to logits (single preview mode)")
    parser.add_argument("--noise_start", type=float, default=0.5, help="Start noise scale for schedule")
    parser.add_argument("--noise_end", type=float, default=0.1, help="End noise scale for schedule")
    parser.add_argument("--steps", type=int, default=0, help="Number of scheduled steps (0 = single preview)")
    parser.add_argument("--num_views", type=int, default=4, help="Number of dataset views to render per step")
    parser.add_argument("--temp", type=float, default=1.0, help="Material temperature for softmax")
    parser.add_argument("--edge_strength", type=float, default=0.25, help="Occupancy mix near solids multiplier")
    parser.add_argument("--far_strength", type=float, default=0.02, help="Occupancy mix far from solids multiplier")
    parser.add_argument("--output_dir", type=str, default="model_stuff/tests/outs",
                        help="Base output directory")
    parser.add_argument("--img_size", type=int, default=192,
                        help="Image size for rendering")
    # SDS preview-training flags
    parser.add_argument("--prompt", type=str, default="a vibrant voxel diorama", help="Text prompt for SDXL")
    parser.add_argument("--train_steps", type=int, default=0, help="Run SDS preview training for N steps (0=disabled)")
    parser.add_argument("--train_views", type=int, default=2, help="Views per step for SDS")
    parser.add_argument("--train_h", type=int, default=192, help="Train image height")
    parser.add_argument("--train_w", type=int, default=192, help="Train image width")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="CFG scale for SDS")
    parser.add_argument("--lambda_sparsity", type=float, default=1e-3, help="Sparsity weight")
    parser.add_argument("--lambda_entropy", type=float, default=1e-4, help="Entropy weight")
    parser.add_argument("--lambda_smooth", type=float, default=0.0, help="Smoothness weight on occupancy")
    parser.add_argument("--max_blocks", type=int, default=50000, help="Cap active voxels during render")

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    base_out = Path(args.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)
    run_dir = base_out / datetime.now().strftime("%Y%m%d_%H%M%S")
    (run_dir / "clean").mkdir(parents=True, exist_ok=True)
    (run_dir / "noisy").mkdir(parents=True, exist_ok=True)

    # Load map
    print(f"Loading map from: {args.map_path}")
    start_time = time.time()
    map_path = Path(args.map_path)
    if not map_path.exists():
        raise FileNotFoundError(f"Map not found: {map_path}")

    occ_logits, mat_logits, grid_size, world_scale = load_map_to_grid(map_path, device=device)
    load_time = time.time() - start_time
    print(f"Map loading took {load_time:.3f}s")
    print(f"Loaded grid: {grid_size}, scale: {world_scale}")

    # Create clean version for comparison
    print("\n--- Testing Clean Version ---")
    start_time = time.time()
    grid_clean = DifferentiableVoxelGrid(
        grid_size=grid_size,
        num_materials=8,
        world_scale=world_scale,
        device=device
    )
    grid_clean.load_state(occ_logits.clone(), mat_logits.clone())
    clean_grid_time = time.time() - start_time
    print(f"Clean grid creation took {clean_grid_time:.3f}s")

    # Load dataset metadata and select views
    dataset_path = Path(args.map_path).parent.parent / "datasets/1"
    meta_path = dataset_path / "metadata.json"
    if not meta_path.exists():
        view, proj = load_test_camera()
        views = [(view.to(device), proj.to(device))]
    else:
        with open(meta_path) as f:
            metadata = json.load(f)
        views = []
        for i, v in enumerate(metadata['views'][:max(1, max(args.num_views, args.train_views))]):
            view, proj = load_camera_matrices_from_metadata(metadata, i)
            # keep on CPU; move to device when used to reduce persistent GPU mem
            views.append((view, proj))

    print(f"Rendering {len(views)} clean views...")
    clean_start = time.time()
    for i, (cv, cp) in enumerate(views):
        out_path = run_dir / "clean" / f"clean_cam{i:02d}.png"
        with torch.no_grad():
            render_and_save_preview(grid_clean, cv.to(device), cp.to(device), out_path, (args.img_size, args.img_size))
    clean_render_time = time.time() - clean_start
    print(f"Clean views rendered in {clean_render_time:.3f}s")
    camera_time = 0.0
    # Clean up clean grid if not needed further
    del grid_clean
    torch.cuda.empty_cache()

    if args.steps <= 0:
        print(f"\n--- Testing Noisy Version (noise={args.noise_scale}) ---")
        noisy_occ, noisy_mat = add_noise_to_logits(
            occ_logits, mat_logits, args.noise_scale, temp=args.temp,
            edge_strength=args.edge_strength, far_strength=args.far_strength
        )
        start_time = time.time()
        grid_noisy = DifferentiableVoxelGrid(
            grid_size=grid_size,
            num_materials=8,
            world_scale=world_scale,
            device=device
        )
        grid_noisy.load_state(noisy_occ, noisy_mat)
        noisy_grid_time = time.time() - start_time
        print(f"Noisy grid creation took {noisy_grid_time:.3f}s")
        print("Rendering noisy views...")
        nstart = time.time()
        for i, (cv, cp) in enumerate(views):
            out_path = run_dir / "noisy" / f"noisy_{args.noise_scale}_cam{i:02d}.png"
            with torch.no_grad():
                render_and_save_preview(grid_noisy, cv.to(device), cp.to(device), out_path, (args.img_size, args.img_size))
        noisy_render_time = time.time() - nstart
        print(f"Noisy views rendered in {noisy_render_time:.3f}s")
        del grid_noisy
        torch.cuda.empty_cache()
    else:
        print(f"\n--- Scheduled Noisy Rendering: steps={args.steps}, noise {args.noise_start}->{args.noise_end} ---")
        for step in range(args.steps):
            u = step / max(args.steps - 1, 1)
            ns = args.noise_start + (args.noise_end - args.noise_start) * u
            print(f"Step {step+1}/{args.steps}: noise_scale={ns:.3f}")
            noisy_occ, noisy_mat = add_noise_to_logits(
                occ_logits, mat_logits, ns, temp=args.temp,
                edge_strength=args.edge_strength, far_strength=args.far_strength
            )
            grid_noisy = DifferentiableVoxelGrid(
                grid_size=grid_size,
                num_materials=8,
                world_scale=world_scale,
                device=device
            )
            t0 = time.time()
            grid_noisy.load_state(noisy_occ, noisy_mat)
            noisy_grid_time = time.time() - t0
            print(f"Noisy grid creation: {noisy_grid_time:.3f}s")
            for i, (cv, cp) in enumerate(views):
                out_path = run_dir / "noisy" / f"step_{step:04d}_cam{i:02d}.png"
                t_r0 = time.time()
                with torch.no_grad():
                    render_and_save_preview(grid_noisy, cv.to(device), cp.to(device), out_path, (args.img_size, args.img_size))
                noisy_render_time = time.time() - t_r0
                print(f"Noisy rendering: {noisy_render_time:.3f}s")
            if (step + 1) % max(1, args.steps // 10) == 0:
                print(f"Progress: {step+1}/{args.steps} steps complete")
            del grid_noisy
            torch.cuda.empty_cache()

    # SDS preview training loop (optional)
    if args.train_steps > 0:
        print(f"\n--- SDS Preview Training ---")
        print(f"Prompt: {args.prompt}")
        print(f"Steps: {args.train_steps}, views/step: {args.train_views}")
        # Initialize grid train copy
        grid = DifferentiableVoxelGrid(
            grid_size=grid_size,
            num_materials=8,
            world_scale=world_scale,
            device=device
        )
        grid.load_state(occ_logits.clone().requires_grad_(True), mat_logits.clone().requires_grad_(True))
        optim = torch.optim.Adam([
            {"params": [grid.occupancy_logits], "lr": args.lr},
            {"params": [grid.material_logits], "lr": args.lr}
        ])
        print(f"Params require_grad -> occ: {grid.occupancy_logits.requires_grad}, mat: {grid.material_logits.requires_grad}")
        # Init SDXL
        sdxl = SDXLLightning(
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            device=str(device),
            dtype=torch.float32,
            height=args.train_h,
            width=args.train_w
        )
        pe, pe_pooled, ue, ue_pooled, add_time_ids = sdxl.encode_prompt(args.prompt)
        use_amp = False
        amp_ctx = nullcontext()

        def compute_sds_loss_minimal(rgba: torch.Tensor) -> torch.Tensor:
            rgb = rgba[:, :3, :, :].clamp(0, 1)
            rgb = torch.nan_to_num(rgb)
            rgb_norm = (rgb * 2.0 - 1.0).to(sdxl.dtype)
            rgb_norm = torch.nan_to_num(rgb_norm)
            with amp_ctx:
                latent = sdxl.vae.encode(rgb_norm).latent_dist.mean
                z0 = latent * 0.18215
            z0 = torch.nan_to_num(z0).clamp(-10.0, 10.0)
            t = sdxl.sample_timesteps(1)
            noise = torch.randn_like(z0)
            x_t = sdxl.add_noise(z0, noise, t)
            x_t = torch.nan_to_num(x_t).clamp(-10.0, 10.0)
            with amp_ctx:
                eps = sdxl.eps_pred_cfg(x_t, t, pe, pe_pooled, ue, ue_pooled, add_time_ids, args.cfg_scale)
            eps = torch.nan_to_num(eps).clamp(-10.0, 10.0)
            return (eps - noise).float().pow(2).mean()

        def reg_losses_minimal() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            occ = torch.sigmoid(grid.occupancy_logits)
            sparsity = occ.mean() * args.lambda_sparsity
            mat = torch.softmax(grid.material_logits, dim=-1)
            entropy = (-(mat * torch.log(mat.clamp_min(1e-8))).sum(dim=-1)).mean() * args.lambda_entropy
            if args.lambda_smooth > 0:
                dx = (occ[1:, :, :] - occ[:-1, :, :]).pow(2).mean()
                dy = (occ[:, 1:, :] - occ[:, :-1, :]).pow(2).mean()
                dz = (occ[:, :, 1:] - occ[:, :, :-1]).pow(2).mean()
                smooth = (dx + dy + dz) * args.lambda_smooth
            else:
                smooth = torch.tensor(0.0, device=device)
            return sparsity, entropy, smooth

        train_dir = run_dir / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        for step in range(args.train_steps):
            u = step / max(args.train_steps - 1, 1)
            t_temp = 1.5 + (1.0 - 1.5) * u
            K = max(1, args.train_views)
            loss_total = 0.0
            optim.zero_grad(set_to_none=True)
            occ_before = grid.occupancy_logits.detach().clone()
            mat_before = grid.material_logits.detach().clone()
            for k in range(K):
                cam_idx = torch.randint(0, len(views), (1,)).item()
                cv, cp = views[cam_idx]
                rgba = grid(cv.to(device), cp.to(device), args.train_h, args.train_w, temperature=t_temp, max_blocks=args.max_blocks)
                loss_total = loss_total + compute_sds_loss_minimal(rgba)
                del rgba
            loss_total = loss_total / K
            s, e, sm = reg_losses_minimal()
            loss = loss_total + s + e + sm
            loss.backward()
            occ_grad_norm = float('nan') if grid.occupancy_logits.grad is None else grid.occupancy_logits.grad.data.norm().item()
            mat_grad_norm = float('nan') if grid.material_logits.grad is None else grid.material_logits.grad.data.norm().item()
            print(f"Grad norms -> occ: {occ_grad_norm:.4e}, mat: {mat_grad_norm:.4e}")
            optim.step()
            optim.zero_grad(set_to_none=True)
            with torch.no_grad():
                occ_delta = (grid.occupancy_logits - occ_before).abs().mean().item()
                mat_delta = (grid.material_logits - mat_before).abs().mean().item()
            stats = get_grid_stats(grid.occupancy_logits.detach(), grid.material_logits.detach())
            print(f"Param delta -> occ|mean|: {occ_delta:.4e}, mat|mean|: {mat_delta:.4e}")
            print(f"Grid stats -> active_50: {stats['active_voxels_50']}, occ_mean: {stats['occupancy_mean']:.4f}, occ_std: {stats['occupancy_std']:.4f}")
            if (step + 1) % max(1, args.train_steps // 10) == 0:
                print(f"Step {step+1}/{args.train_steps} - loss={float(loss.item()):.4f} (sds={float(loss_total.item()):.4f}, spars={float(s.item()):.4f}, ent={float(e.item()):.4f})")
            # Save preview from a random view to reflect material field updates
            prev_idx = torch.randint(0, len(views), (1,)).item()
            cvp, cpp = views[prev_idx]
            out_path = train_dir / f"step_{step:04d}_cam{prev_idx:02d}.png"
            with torch.no_grad():
                render_and_save_preview(grid, cvp.to(device), cpp.to(device), out_path, (args.img_size, args.img_size))
            print(f"Saved step preview: {out_path}")
            print(f"Step {step+1}/{args.train_steps} complete.\n")
            torch.cuda.empty_cache()

        # Training cleanup
        try:
            del optim
        except Exception:
            pass
        try:
            del sdxl.pipe
            del sdxl.unet
            del sdxl.vae
        except Exception:
            pass
        try:
            del sdxl
        except Exception:
            pass
        try:
            del grid
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

    # Print statistics
    print("\n--- Statistics ---")
    print(f"Occupancy logits - Clean: mean={occ_logits.mean().item():.3f}, std={occ_logits.std().item():.3f}")
    print(f"Occupancy logits - Noisy: mean={noisy_occ.mean().item():.3f}, std={noisy_occ.std().item():.3f}")
    print(f"Material logits - Clean: mean={mat_logits.mean().item():.3f}, std={mat_logits.std().item():.3f}")
    print(f"Material logits - Noisy: mean={noisy_mat.mean().item():.3f}, std={noisy_mat.std().item():.3f}")

    # Print timing summary
    print("\n--- Timing Summary ---")
    print(f"Map loading: {load_time:.3f}s")
    print(f"Clean grid creation: {clean_grid_time:.3f}s")
    print(f"Camera loading: {camera_time:.3f}s")
    print(f"Clean rendering: {clean_render_time:.3f}s")
    print(f"Noisy grid creation: {noisy_grid_time:.3f}s")
    print(f"Noisy rendering: {noisy_render_time:.3f}s")
    total_time = load_time + clean_grid_time + camera_time + clean_render_time + noisy_grid_time + noisy_render_time
    print(f"Total time: {total_time:.3f}s")

    print(f"\n✅ Test complete! Check {run_dir} for images.")
    # Final cleanup
    try:
        del occ_logits
        del mat_logits
        del views
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
