"""
SDS Training Pipeline - Final Implementation

Optimizes voxel grids using Score Distillation Sampling with SDXL.
Works with datasets/1 cameras and maps/1 data.

Usage:
    python -m model_stuff.train_sds_final --prompt "a medieval castle" --steps 500
"""

import torch
import torch.nn.functional as F
import json
import random
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from datetime import datetime

from .voxel_grid import DifferentiableVoxelGrid
from .map_io import load_map_to_grid, save_grid_to_map, init_grid_from_primitive, get_grid_stats
from .sdxl_lightning import SDXLLightning
from .nv_diff_render.utils import load_camera_matrices_from_metadata
from .config import get_preset


def compute_sds_loss(
    rgba: torch.Tensor,
    sdxl: SDXLLightning,
    pe: torch.Tensor,
    pe_pooled: torch.Tensor,
    ue: torch.Tensor,
    ue_pooled: torch.Tensor,
    add_time_ids: torch.Tensor,
    cfg_scale: float,
    timestep_range: tuple = (0, 1000)
) -> torch.Tensor:
    """
    Compute Score Distillation Sampling loss.

    Args:
        rgba: (1, 4, H, W) rendered image
        sdxl: SDXL model wrapper
        pe, pe_pooled: Positive prompt embeddings
        ue, ue_pooled: Negative prompt embeddings
        add_time_ids: Additional time embeddings
        cfg_scale: Classifier-free guidance scale
        timestep_range: (min, max) timestep range

    Returns:
        SDS loss scalar
    """
    rgb = rgba[:, :3, :, :].clamp(0, 1)
    print(f"    [SDS DEBUG] RGB: min={rgb.min().item():.4f}, max={rgb.max().item():.4f}, nan={torch.isnan(rgb).any().item()}")

    # SDXL VAE expects input in [-1, 1] and same dtype as VAE weights
    rgb_normalized = (rgb * 2.0 - 1.0).to(sdxl.dtype)
    print(f"    [SDS DEBUG] RGB normalized: min={rgb_normalized.min().item():.4f}, max={rgb_normalized.max().item():.4f}")

    # Encode to latent; keep VAE frozen but allow dtype-correct forward
    with torch.no_grad():
        latent_dist = sdxl.vae.encode(rgb_normalized).latent_dist
        z0 = latent_dist.mean * 0.18215  # mean for stability
    print(f"    [SDS DEBUG] Latent z0: min={z0.min().item():.4f}, max={z0.max().item():.4f}, nan={torch.isnan(z0).any().item()}")

    # Sample timestep
    t_min, t_max = timestep_range
    ts = torch.randint(t_min, t_max, (1,), device=sdxl.device, dtype=torch.long)
    print(f"    [SDS DEBUG] Timestep: {ts.item()}")

    # Add noise
    noise = torch.randn_like(z0)
    x_t = sdxl.add_noise(z0, noise, ts)
    print(f"    [SDS DEBUG] Noised latent x_t: min={x_t.min().item():.4f}, max={x_t.max().item():.4f}, nan={torch.isnan(x_t).any().item()}")

    # Predict noise with CFG
    eps_cfg = sdxl.eps_pred_cfg(
        x_t, ts, pe, pe_pooled, ue, ue_pooled, add_time_ids, cfg_scale
    )
    print(f"    [SDS DEBUG] Predicted noise eps_cfg: min={eps_cfg.min().item():.4f}, max={eps_cfg.max().item():.4f}, nan={torch.isnan(eps_cfg).any().item()}")

    # SDS loss
    # TODO: Add timestep weighting w(t) = (1 - alpha_t) / sqrt(1 - alpha_t)
    loss = (eps_cfg - noise).pow(2).mean()
    print(f"    [SDS DEBUG] Loss: {loss.item()}, nan={torch.isnan(loss).any().item()}")

    return loss


def compute_regularization_losses(
    grid: DifferentiableVoxelGrid,
    temperature: float,
    lambda_sparsity: float = 0.001,
    lambda_entropy: float = 0.0001,
    lambda_smooth: float = 0.0
) -> dict:
    """
    Compute regularization losses.

    Args:
        grid: Voxel grid
        temperature: Current temperature
        lambda_sparsity: Weight for sparsity loss
        lambda_entropy: Weight for entropy loss
        lambda_smooth: Weight for smoothness loss

    Returns:
        Dict of losses
    """
    losses = {}

    # Sparsity loss (penalize too many occupied voxels)
    occ_probs = grid.get_occupancy_probs()
    losses['sparsity'] = occ_probs.mean() * lambda_sparsity

    # Entropy loss (penalize uncertain material assignments)
    mat_probs = grid.get_material_probs(temperature)
    entropy = -(mat_probs * torch.log(mat_probs.clamp_min(1e-8))).sum(dim=-1)
    losses['entropy'] = entropy.mean() * lambda_entropy

    # Smoothness loss (penalize abrupt changes)
    if lambda_smooth > 0:
        dx = (occ_probs[1:, :, :] - occ_probs[:-1, :, :]).pow(2).mean()
        dy = (occ_probs[:, 1:, :] - occ_probs[:, :-1, :]).pow(2).mean()
        dz = (occ_probs[:, :, 1:] - occ_probs[:, :, :-1]).pow(2).mean()
        losses['smooth'] = (dx + dy + dz) * lambda_smooth
    else:
        losses['smooth'] = torch.tensor(0.0, device=grid.device)

    return losses


def train_sds(
    prompt: str,
    dataset_id: int = 1,
    steps: int = 500,
    lr: float = 0.01,
    cfg_scale: float = 7.5,
    temp_start: float = 2.0,
    temp_end: float = 0.5,
    lambda_sparsity: float = 0.001,
    lambda_entropy: float = 0.0001,
    lambda_smooth: float = 0.0,
    log_every: int = 10,
    image_every: int = 5,
    save_map_every: int = 50,
    output_dir: str = "out_local/sds_training",
    init_mode: str = "from_map",  # 'from_map', 'ground_plane', 'cube'
    seed: int = 42,
    # New: training render resolution (VRAM control)
    train_h: int = 256,
    train_w: int = 256,
    # Safety cap for active blocks (prevents OOM/segfault on small GPUs)
    max_blocks: int | None = None,
):
    """
    Train voxel grid with SDS.

    Args:
        prompt: Text prompt for SDXL
        dataset_id: Dataset ID (default 1)
        steps: Training steps
        lr: Learning rate
        cfg_scale: Classifier-free guidance scale
        temp_start: Initial temperature
        temp_end: Final temperature
        lambda_sparsity: Sparsity loss weight
        lambda_entropy: Entropy loss weight
        lambda_smooth: Smoothness loss weight
        log_every: Log frequency
        image_every: Image save frequency (default: every 5 steps)
        save_map_every: Intermediate map save frequency (default: every 50 steps)
        output_dir: Output directory
        init_mode: Initialization mode
        seed: Random seed
    """
    # Set random seed
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create timestamped run directory under output_dir
    base_dir = Path(output_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = base_dir / run_id
    output_path.mkdir(parents=True, exist_ok=True)
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)
    maps_dir = output_path / "maps"
    maps_dir.mkdir(exist_ok=True)
    log_file = output_path / "train.jsonl"
    session_file = output_path / "session.json"

    # Load dataset cameras
    dataset_path = Path(f"datasets/{dataset_id}")
    metadata_path = dataset_path / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Dataset not found: {metadata_path}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    print(f"\nLoading dataset {dataset_id}...")
    print(f"  Views: {len(metadata['views'])}")

    # Extract camera matrices
    cameras = []
    for i in range(len(metadata['views'])):
        view, proj = load_camera_matrices_from_metadata(metadata, i)
        cameras.append((view.to(device), proj.to(device)))

    img_w = metadata['imageSize']['width']
    img_h = metadata['imageSize']['height']
    print(f"  Dataset image size: {img_w}×{img_h}")

    # Training resolution (configurable)
    print(f"  Training resolution: {train_w}×{train_h}")
    print(f"  Run directory: {output_path}")

    # Initialize voxel grid
    print(f"\nInitializing voxel grid (mode={init_mode})...")

    if init_mode == "from_map":
        # Load from existing map
        map_path = Path(f"maps/{dataset_id}/map.json")
        if not map_path.exists():
            raise FileNotFoundError(f"Map not found: {map_path}")

        occ_logits, mat_logits, grid_size, world_scale = load_map_to_grid(map_path, device=device)

        grid = DifferentiableVoxelGrid(
            grid_size=grid_size,
            num_materials=8,
            world_scale=world_scale,
            device=device
        )
        grid.load_state(occ_logits, mat_logits)

    else:
        # Initialize from primitive
        # Use dataset map dimensions if available
        try:
            map_path = Path(f"maps/{dataset_id}/map.json")
            with open(map_path) as f:
                map_data = json.load(f)
            dims = map_data['worldConfig']['dimensions']
            grid_size = (dims['x'], dims['y'], dims['z'])
            world_scale = map_data['worldConfig'].get('worldScale', 2.0)
        except:
            grid_size = (32, 32, 32)
            world_scale = 2.0

        grid = DifferentiableVoxelGrid(
            grid_size=grid_size,
            num_materials=8,
            world_scale=world_scale,
            device=device
        )

        occ_logits, mat_logits = init_grid_from_primitive(grid_size, init_mode, device=device)
        grid.load_state(occ_logits, mat_logits)

    print(f"  Grid: {grid}")

    # Save session metadata early
    try:
        session = {
            "run_id": run_id,
            "prompt": prompt,
            "dataset_id": dataset_id,
            "train_h": train_h,
            "train_w": train_w,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "init_mode": init_mode,
            "output_dir": str(output_path),
            "grid_size": grid.grid_size,
            "world_scale": grid.world_scale,
        }
        with open(session_file, "w") as f:
            json.dump(session, f, indent=2)
    except Exception as e:
        print(f"⚠️  Failed to write session.json: {e}")

    # Initialize SDXL
    print(f"\nInitializing SDXL...")
    print(f"  Prompt: '{prompt}'")

    sdxl = SDXLLightning(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        device=device,
        dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        height=train_h,
        width=train_w
    )

    pe, pe_pooled, ue, ue_pooled, add_time_ids = sdxl.encode_prompt(prompt)

    # Preview render before training (view 0)
    try:
        pv_view, pv_proj = cameras[0]
        rgba_prev = grid(pv_view, pv_proj, train_h, train_w, temperature=temp_start, max_blocks=max_blocks // 4 if max_blocks else None)
        rgb_prev = rgba_prev[0, :3, :, :].detach().cpu().permute(1, 2, 0).numpy()
        rgb_prev = (np.clip(rgb_prev, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(rgb_prev).save(images_dir / "preview_init_cam0.png")
        # Save initial map snapshot
        save_grid_to_map(
            grid.occupancy_logits.data,
            grid.material_logits.data,
            maps_dir / "step_0000.json",
            world_scale=grid.world_scale,
            threshold=0.5,
            metadata={
                "prompt": prompt,
                "step": 0,
                "total_steps": steps,
                "cfg_scale": cfg_scale
            }
        )
    except Exception as e:
        print(f"⚠️  Preview render/save failed: {e}")

    # Optimizer
    optimizer = torch.optim.Adam(grid.parameters(), lr=lr)

    print(f"\nStarting training for {steps} steps...")
    print(f"  Learning rate: {lr}")
    print(f"  CFG scale: {cfg_scale}")
    print(f"  Temperature: {temp_start} → {temp_end}")

    # Training loop
    for step in range(steps):
        # Anneal temperature
        t = temp_start + (temp_end - temp_start) * (step / max(steps - 1, 1))

        # Sample random camera
        cam_idx = random.randint(0, len(cameras) - 1)
        view, proj = cameras[cam_idx]

        # Forward pass
        rgba = grid(view, proj, train_h, train_w, temperature=t, max_blocks=max_blocks)

        # DEBUG: Check render output
        print(f"  [DEBUG] RGBA shape: {rgba.shape}, min: {rgba.min().item():.4f}, max: {rgba.max().item():.4f}, has_nan: {torch.isnan(rgba).any().item()}")

        # SDS loss
        loss_sds = compute_sds_loss(
            rgba, sdxl, pe, pe_pooled, ue, ue_pooled, add_time_ids, cfg_scale
        )

        # DEBUG: Check SDS loss
        print(f"  [DEBUG] SDS loss: {loss_sds.item()}, has_nan: {torch.isnan(loss_sds).any().item()}")

        # Regularization losses
        reg_losses = compute_regularization_losses(
            grid, t, lambda_sparsity, lambda_entropy, lambda_smooth
        )

        # DEBUG: Check reg losses
        print(f"  [DEBUG] Reg losses - sparsity: {reg_losses['sparsity'].item():.6f}, entropy: {reg_losses['entropy'].item():.6f}")

        # Total loss
        loss_total = loss_sds + reg_losses['sparsity'] + reg_losses['entropy'] + reg_losses['smooth']

        # DEBUG: Check if loss is valid before backward
        if torch.isnan(loss_total).any():
            print(f"  [ERROR] NaN loss detected! Skipping backward pass.")
            continue

        # Optimize (and capture gradient diagnostics before step)
        optimizer.zero_grad()
        loss_total.backward()

        # Gradient diagnostics
        occ_grad_norm = float('nan')
        mat_grad_norm = float('nan')
        try:
            if grid.occupancy_logits.grad is not None:
                occ_grad_norm = float(grid.occupancy_logits.grad.norm().item())
            if grid.material_logits.grad is not None:
                mat_grad_norm = float(grid.material_logits.grad.norm().item())
        except Exception:
            pass

        optimizer.step()

        print(f"  [DEBUG] Backward pass completed")

        # Logging
        if (step + 1) % log_every == 0 or step == 0:
            stats = grid.get_stats()

            log_entry = {
                "step": step + 1,
                "temperature": float(t),
                "camera_index": cam_idx,
                "losses": {
                    "total": float(loss_total.item()),
                    "sds": float(loss_sds.item()),
                    "sparsity": float(reg_losses['sparsity'].item()),
                    "entropy": float(reg_losses['entropy'].item()),
                    "smooth": float(reg_losses['smooth'].item())
                },
                "grid_stats": stats
            }

            # Attach gradient norms
            log_entry["grads"] = {
                "occupancy_l2": occ_grad_norm,
                "materials_l2": mat_grad_norm,
            }

            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            print(f"Step {step+1}/{steps}: "
                  f"loss={loss_total.item():.4f} "
                  f"(sds={loss_sds.item():.4f}) "
                  f"active={stats['num_active_voxels']}/{stats['total_voxels']} "
                  f"({stats['density']*100:.1f}%)")

        # Save images (every 5 steps by default)
    if (step + 1) % image_every == 0 or step == 0 or step == steps - 1:
        rgb = rgba[0, :3, :, :].detach().cpu().permute(1, 2, 0).numpy()
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(rgb).save(images_dir / f"step_{step+1:04d}_cam{cam_idx}.png")

        # Save intermediate maps (every 50 steps by default)
        if (step + 1) % save_map_every == 0 or step == steps - 1:
            intermediate_map_path = maps_dir / f"step_{step+1:04d}.json"
            save_grid_to_map(
                grid.occupancy_logits.data,
                grid.material_logits.data,
                intermediate_map_path,
                world_scale=grid.world_scale,
                threshold=0.5,
                metadata={
                    "prompt": prompt,
                    "step": step + 1,
                    "total_steps": steps,
                    "cfg_scale": cfg_scale
                }
            )

    # Export final grid with distinct name
    print(f"\nTraining complete!")

    # Save as final_map.json (distinct name)
    final_map_path = output_path / "final_map.json"
    num_blocks = save_grid_to_map(
        grid.occupancy_logits.data,
        grid.material_logits.data,
        final_map_path,
        world_scale=grid.world_scale,
        threshold=0.5,
        metadata={"prompt": prompt, "steps": steps, "cfg_scale": cfg_scale}
    )

    # Also save as map_optimized.json (for backward compatibility)
    output_map_path = output_path / "map_optimized.json"
    save_grid_to_map(
        grid.occupancy_logits.data,
        grid.material_logits.data,
        output_map_path,
        world_scale=grid.world_scale,
        threshold=0.5,
        metadata={"prompt": prompt, "steps": steps, "cfg_scale": cfg_scale}
    )

    print(f"\nOutput:")
    print(f"  Final map: {final_map_path} ({num_blocks} blocks)")
    print(f"  Logs: {log_file}")
    print(f"  Images: {images_dir} ({len(list(images_dir.glob('*.png')))} images)")
    print(f"  Intermediate maps: {maps_dir} ({len(list(maps_dir.glob('*.json')))} checkpoints)")

    # Print final stats
    final_stats = grid.get_stats()
    print(f"\nFinal grid stats:")
    print(f"  Active voxels: {final_stats['num_active_voxels']}/{final_stats['total_voxels']}")
    print(f"  Density: {final_stats['density']*100:.2f}%")
    print(f"  Material distribution:")
    for mat, pct in sorted(final_stats['material_distribution'].items(), key=lambda x: -x[1])[:5]:
        if pct > 0.1:
            print(f"    {mat}: {pct:.1f}%")

    # Auto-export to maps/ directory
    print(f"\nExporting to maps/ directory...")
    try:
        from .export_trained_map import export_trained_map
        exported_path = export_trained_map(
            training_dir=str(output_path),
            source_map_id=dataset_id,
            maps_base_dir="maps"
        )
        print(f"✅ Exported to: {exported_path}")
    except Exception as e:
        print(f"⚠️  Auto-export failed: {e}")
        print(f"   You can manually export with:")
        print(f"   python -m model_stuff.export_trained_map --training_dir {output_path} --source_map_id {dataset_id}")

    return grid


def main():
    parser = argparse.ArgumentParser(description="SDS Training for Voxel Worlds")

    parser.add_argument("--prompt", type=str, default="a medieval stone castle",
                        help="Text prompt for SDXL")
    parser.add_argument("--dataset_id", type=int, default=1,
                        help="Dataset ID")
    parser.add_argument("--steps", type=int, default=500,
                        help="Training steps")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--cfg_scale", type=float, default=7.5,
                        help="CFG scale")
    parser.add_argument("--temp_start", type=float, default=2.0,
                        help="Initial temperature")
    parser.add_argument("--temp_end", type=float, default=0.5,
                        help="Final temperature")
    parser.add_argument("--lambda_sparsity", type=float, default=0.001,
                        help="Sparsity loss weight")
    parser.add_argument("--lambda_entropy", type=float, default=0.0001,
                        help="Entropy loss weight")
    parser.add_argument("--lambda_smooth", type=float, default=0.0,
                        help="Smoothness loss weight")
    parser.add_argument("--log_every", type=int, default=10,
                        help="Log frequency")
    parser.add_argument("--image_every", type=int, default=5,
                        help="Image save frequency (default: every 5 steps)")
    parser.add_argument("--save_map_every", type=int, default=50,
                        help="Intermediate map save frequency (default: every 50 steps)")
    parser.add_argument("--output_dir", type=str, default="out_local/sds_training",
                        help="Output directory")
    parser.add_argument("--init_mode", type=str, default="from_map",
                        choices=["from_map", "ground_plane", "cube", "empty"],
                        help="Initialization mode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--train_h", type=int, default=256,
                        help="Training image height for SDS")
    parser.add_argument("--train_w", type=int, default=256,
                        help="Training image width for SDS")
    parser.add_argument("--preset", type=str, default=None,
                        choices=["small", "medium", "large"],
                        help="Preset to override hyperparameters and training resolution")
    parser.add_argument("--max_blocks", type=int, default=None,
                        help="Cap number of active voxels rendered each step (prevents OOM)")

    args = parser.parse_args()

    # Apply preset if provided: preset fills only values the user didn't change from defaults
    if args.preset:
        preset = get_preset(args.preset, args.output_dir)
        defaults = parser.parse_args([])
        applied = {}
        for k, v in preset.items():
            # Only apply if user did not override from default
            if getattr(args, k, None) == getattr(defaults, k, None):
                setattr(args, k, v)
                applied[k] = v
        print(f"\nUsing preset: {args.preset} → {applied}")

    kwargs = vars(args).copy()
    # Remove CLI-only keys not in train_sds signature
    kwargs.pop("preset", None)
    train_sds(**kwargs)


if __name__ == "__main__":
    main()
