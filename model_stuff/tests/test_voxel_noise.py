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
import time

# Import voxel grid components
from ..voxel_grid import DifferentiableVoxelGrid
from ..map_io import load_map_to_grid
from ..nv_diff_render.utils import load_camera_matrices_from_metadata


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


def add_noise_to_logits(occ_logits, mat_logits, noise_scale=0.1, temp: float = 1.0):
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

    edge_strength = 0.25
    far_strength = 0.02
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
                        help="Scale of noise to add to logits")
    parser.add_argument("--output_dir", type=str, default="model_stuff/tests",
                        help="Output directory")
    parser.add_argument("--img_size", type=int, default=192,
                        help="Image size for rendering")

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

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

    # Load camera
    start_time = time.time()
    camera_view, camera_proj = load_test_camera()
    camera_time = time.time() - start_time
    print(f"Camera loading took {camera_time:.3f}s")

    # Render clean version
    start_time = time.time()
    clean_output = output_dir / "preview_clean.png"
    render_and_save_preview(grid_clean, camera_view, camera_proj, clean_output, (args.img_size, args.img_size))
    clean_render_time = time.time() - start_time
    print(f"Clean rendering took {clean_render_time:.3f}s")

    # Add noise
    print(f"\n--- Testing Noisy Version (noise={args.noise_scale}) ---")
    noisy_occ, noisy_mat = add_noise_to_logits(occ_logits, mat_logits, args.noise_scale)

    # Create noisy grid
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

    # Render noisy version
    start_time = time.time()
    noisy_output = output_dir / f"preview_noisy_{args.noise_scale}.png"
    render_and_save_preview(grid_noisy, camera_view, camera_proj, noisy_output, (args.img_size, args.img_size))
    noisy_render_time = time.time() - start_time
    print(f"Noisy rendering took {noisy_render_time:.3f}s")

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

    print(f"\n✅ Test complete! Check {output_dir} for images.")


if __name__ == "__main__":
    main()
