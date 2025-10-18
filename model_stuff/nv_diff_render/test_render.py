"""
Test renderer matching test_render_dataset.py interface.

Usage:
    python -m model_stuff.nv_diff_render.test_render --dataset_id 1 --view_index 0
    python -m model_stuff.nv_diff_render.test_render --dataset_id 1 --all_views
"""

import argparse
import json
import base64
from pathlib import Path
from typing import Dict, Any, Tuple
import torch
import numpy as np
from PIL import Image

from . import (
    DifferentiableBlockRenderer,
    material_name_to_index,
    load_camera_matrices_from_metadata
)


def load_dataset_metadata(dataset_id: int) -> Tuple[Dict[str, Any], Path]:
    """Load metadata.json for dataset."""
    dataset_dir = Path(f"datasets/{dataset_id}")
    metadata_file = dataset_dir / "metadata.json"

    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_file}")

    with open(metadata_file) as f:
        metadata = json.load(f)

    return metadata, dataset_dir


def load_map_data(dataset_id: int) -> Dict[str, Any]:
    """Load map.json for dataset."""
    map_file = Path(f"maps/{dataset_id}/map.json")

    if not map_file.exists():
        raise FileNotFoundError(f"Map not found: {map_file}")

    with open(map_file) as f:
        map_data = json.load(f)

    return map_data


def render_view(
    renderer: DifferentiableBlockRenderer,
    map_data: Dict[str, Any],
    metadata: Dict[str, Any],
    view_index: int,
    device: torch.device
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Render a single view using nvdiffrast.

    Args:
        renderer: DifferentiableBlockRenderer instance
        map_data: Map data dict
        metadata: Dataset metadata
        view_index: Which view to render
        device: Torch device

    Returns:
        rgb_array: (H, W, 3) uint8 array
        view_info: Dict with view metadata
    """
    view_info = metadata['views'][view_index]
    img_h, img_w = metadata['imageSize']['height'], metadata['imageSize']['width']

    # Load camera matrices
    camera_view, camera_proj = load_camera_matrices_from_metadata(metadata, view_index)
    camera_view = camera_view.to(device)
    camera_proj = camera_proj.to(device)

    # Extract block placements
    blocks = map_data.get('blocks', [])
    world_scale = map_data.get('worldScale', 2.0)

    if len(blocks) == 0:
        print(f"Warning: No blocks in map {dataset_id}")
        # Return sky-colored image
        sky_color = np.array([0.53, 0.81, 0.92])
        rgb_array = (sky_color * 255).astype(np.uint8)
        rgb_array = np.tile(rgb_array[None, None, :], (img_h, img_w, 1))
        return rgb_array, view_info

    # Parse block positions and materials
    positions = []
    material_indices = []

    for block_data in blocks:
        pos = tuple(block_data['position'])
        block_type_name = block_data['blockType']

        try:
            mat_idx = material_name_to_index(block_type_name)
        except ValueError:
            print(f"Warning: Unknown material '{block_type_name}', skipping")
            continue

        positions.append(pos)
        material_indices.append(mat_idx)

    if len(positions) == 0:
        print(f"Warning: No valid blocks to render")
        sky_color = np.array([0.53, 0.81, 0.92])
        rgb_array = (sky_color * 255).astype(np.uint8)
        rgb_array = np.tile(rgb_array[None, None, :], (img_h, img_w, 1))
        return rgb_array, view_info

    # Create material logits (hard assignment to specified materials)
    M = 8  # Number of materials
    material_logits = torch.zeros((len(positions), M), device=device)
    for i, mat_idx in enumerate(material_indices):
        material_logits[i, mat_idx] = 10.0  # Strong bias

    # Update renderer grid size and world scale
    if 'dimensions' in map_data.get('worldConfig', {}):
        dims = map_data['worldConfig']['dimensions']
        renderer.grid_size = (dims['x'], dims['y'], dims['z'])
    renderer.world_scale = world_scale

    # Render
    print(f"Rendering {len(positions)} blocks at {img_w}x{img_h}...")
    with torch.no_grad():
        rgba = renderer.render(
            positions,
            material_logits,
            camera_view,
            camera_proj,
            img_h,
            img_w,
            temperature=0.01,  # Very sharp (nearly hard assignment)
            hard_materials=True
        )

    # Convert to numpy (H, W, 3)
    rgb = rgba[0, :3, :, :].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)

    return rgb, view_info


def decode_base64_image(base64_str: str, img_h: int, img_w: int) -> np.ndarray:
    """Decode base64 image string to numpy array."""
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    img = img.convert('RGB')
    img_array = np.array(img)

    # Ensure correct size
    if img_array.shape[:2] != (img_h, img_w):
        img = img.resize((img_w, img_h), Image.BILINEAR)
        img_array = np.array(img)

    return img_array


def compute_metrics(rendered: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """
    Compute comparison metrics between rendered and ground truth.

    Args:
        rendered: (H, W, 3) uint8
        ground_truth: (H, W, 3) uint8

    Returns:
        Dict with metrics
    """
    # Convert to float for metrics
    rendered_f = rendered.astype(np.float32) / 255.0
    gt_f = ground_truth.astype(np.float32) / 255.0

    # MSE
    mse = np.mean((rendered_f - gt_f) ** 2)

    # PSNR
    if mse > 0:
        psnr = 10.0 * np.log10(1.0 / mse)
    else:
        psnr = float('inf')

    # Max error
    max_error = np.abs(rendered_f - gt_f).max()

    # Geometry match (binary mask agreement)
    rendered_mask = rendered.sum(axis=-1) > 0
    gt_mask = ground_truth.sum(axis=-1) > 0
    geometry_match = (rendered_mask == gt_mask).mean()

    return {
        'mse': float(mse),
        'psnr': float(psnr),
        'max_error': float(max_error),
        'geometry_match': float(geometry_match)
    }


def main():
    parser = argparse.ArgumentParser(description="Test nvdiffrast renderer on dataset")
    parser.add_argument('--dataset_id', type=int, required=True, help="Dataset sequence number")
    parser.add_argument('--view_index', type=int, default=0, help="View index to render")
    parser.add_argument('--all_views', action='store_true', help="Render all views")
    parser.add_argument('--device', type=str, default='cuda', help="Device (cuda or cpu)")
    parser.add_argument('--compare', action='store_true', help="Compare with ground truth")
    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"Loading dataset {args.dataset_id}...")
    metadata, dataset_dir = load_dataset_metadata(args.dataset_id)
    map_data = load_map_data(args.dataset_id)

    # Create output directory
    output_dir = Path("out_local/nvdiff_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create renderer
    print("Initializing renderer...")
    renderer = DifferentiableBlockRenderer(device=device)

    # Determine which views to render
    if args.all_views:
        view_indices = range(metadata['viewCount'])
    else:
        view_indices = [args.view_index]

    # Render views
    for view_idx in view_indices:
        print(f"\n=== Rendering view {view_idx} ===")

        # Render
        rgb, view_info = render_view(renderer, map_data, metadata, view_idx, device)

        # Save output
        output_file = output_dir / f"dataset_{args.dataset_id}_view_{view_idx}.png"
        Image.fromarray(rgb).save(output_file)
        print(f"Saved: {output_file}")

        # Compare with ground truth if requested
        if args.compare and 'rgbBase64' in view_info:
            print("Comparing with ground truth...")
            import io

            img_h, img_w = metadata['imageSize']['height'], metadata['imageSize']['width']
            gt_rgb = decode_base64_image(view_info['rgbBase64'], img_h, img_w)

            metrics = compute_metrics(rgb, gt_rgb)
            print(f"Metrics:")
            print(f"  MSE:            {metrics['mse']:.6f}")
            print(f"  PSNR:           {metrics['psnr']:.2f} dB")
            print(f"  Max Error:      {metrics['max_error']:.4f}")
            print(f"  Geometry Match: {metrics['geometry_match']:.4f}")

            # Save comparison
            comparison = np.concatenate([gt_rgb, rgb], axis=1)
            comparison_file = output_dir / f"compare_dataset_{args.dataset_id}_view_{view_idx}.png"
            Image.fromarray(comparison).save(comparison_file)
            print(f"Saved comparison: {comparison_file}")

    print(f"\n=== Done! Rendered {len(view_indices)} view(s) ===")


if __name__ == '__main__':
    main()
