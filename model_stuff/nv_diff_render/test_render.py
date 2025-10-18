"""
Test renderer matching test_render_dataset.py interface.

Usage:
    python -m model_stuff.nv_diff_render.test_render --dataset_id 1 --view_index 0
    python -m model_stuff.nv_diff_render.test_render --dataset_id 1 --all_views
"""

import argparse
import json
import base64
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List
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


class DifferentiabilityLogger:
    """Logs differentiability metrics for test renders with comprehensive gradient tracking."""

    def __init__(self, log_dir: Path, dataset_id: int):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_id = dataset_id
        self.view_logs = []

    def log_view(
        self,
        view_index: int,
        num_blocks: int,
        render_time: float,
        rgba: torch.Tensor,
        material_logits: torch.Tensor,
        gradient_test_result: Dict[str, Any]
    ):
        """Log metrics for a single view with detailed gradient information."""
        log_entry = {
            # View identification
            'view_index': view_index,
            'num_blocks': num_blocks,

            # Rendering metrics
            'render_time_ms': render_time * 1000,
            'image_shape': list(rgba.shape),

            # RGBA output statistics
            'rgba_statistics': {
                'min': float(rgba.min().item()),
                'max': float(rgba.max().item()),
                'mean': float(rgba.mean().item()),
                'std': float(rgba.std().item())
            },

            # Material logits information
            'material_logits_shape': list(material_logits.shape),
            'material_logits_statistics': {
                'min': float(material_logits.min().item()),
                'max': float(material_logits.max().item()),
                'mean': float(material_logits.mean().item()),
                'std': float(material_logits.std().item())
            },

            # Gradient test results (comprehensive)
            'gradient_test': gradient_test_result
        }
        self.view_logs.append(log_entry)

    def save(self):
        """Save all logs to JSON file with summary statistics."""
        log_file = self.log_dir / f"dataset_{self.dataset_id}_diff_log.json"

        # Compute aggregate statistics across all views
        total_render_time = sum(v['render_time_ms'] for v in self.view_logs)
        avg_render_time = total_render_time / len(self.view_logs) if self.view_logs else 0

        # Gradient statistics aggregation
        gradient_stats = {
            'views_with_gradients': 0,
            'views_differentiable': 0,
            'avg_gradient_norm_l2': 0.0,
            'avg_gradient_health_score': 0.0,
            'views_with_nan_gradients': 0,
            'views_with_inf_gradients': 0,
            'total_blocks_with_gradients': 0,
            'total_blocks': 0
        }

        for view in self.view_logs:
            if 'gradient_test' in view and view['gradient_test']:
                gt = view['gradient_test']
                if gt.get('has_gradient', False):
                    gradient_stats['views_with_gradients'] += 1
                if gt.get('differentiable', False):
                    gradient_stats['views_differentiable'] += 1
                if gt.get('has_nan_gradients', False):
                    gradient_stats['views_with_nan_gradients'] += 1
                if gt.get('has_inf_gradients', False):
                    gradient_stats['views_with_inf_gradients'] += 1

                gradient_stats['avg_gradient_norm_l2'] += gt.get('gradient_norm_l2', 0.0)
                gradient_stats['avg_gradient_health_score'] += gt.get('gradient_health_score', 0.0)
                gradient_stats['total_blocks_with_gradients'] += gt.get('blocks_with_gradients', 0)
                gradient_stats['total_blocks'] += gt.get('blocks_total', 0)

        if self.view_logs:
            gradient_stats['avg_gradient_norm_l2'] /= len(self.view_logs)
            gradient_stats['avg_gradient_health_score'] /= len(self.view_logs)

        summary = {
            'dataset_id': self.dataset_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),

            # Overall statistics
            'num_views': len(self.view_logs),
            'total_render_time_ms': total_render_time,
            'avg_render_time_ms': avg_render_time,

            # Gradient summary
            'gradient_summary': gradient_stats,

            # Detailed per-view logs
            'views': self.view_logs
        }

        with open(log_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nDifferentiability log saved: {log_file}")
        print(f"  Views with gradients: {gradient_stats['views_differentiable']}/{len(self.view_logs)}")
        print(f"  Avg gradient health: {gradient_stats['avg_gradient_health_score']:.3f}")
        print(f"  Blocks with gradients: {gradient_stats['total_blocks_with_gradients']}/{gradient_stats['total_blocks']}")

        return log_file


def test_gradient_flow(
    renderer: DifferentiableBlockRenderer,
    positions: List[Tuple[int, int, int]],
    material_logits: torch.Tensor,
    camera_view: torch.Tensor,
    camera_proj: torch.Tensor,
    img_h: int,
    img_w: int,
    view_index: int
) -> Dict[str, Any]:
    """
    Test if gradients flow through the renderer with comprehensive logging.

    Returns dict with gradient flow metrics and per-material gradient statistics.
    """
    # Clone logits to avoid affecting main render
    test_logits = material_logits.clone().detach().requires_grad_(True)

    # Render with gradients enabled
    rgba = renderer.render(
        positions,
        test_logits,
        camera_view,
        camera_proj,
        img_h,
        img_w,
        temperature=1.0,
        hard_materials=False
    )

    # Compute multiple loss components for comprehensive gradient testing
    # Loss 1: RGB sum (tests color gradients)
    loss_rgb = rgba[:, :3].sum()

    # Loss 2: Alpha-weighted RGB (tests alpha gradients)
    alpha = rgba[:, 3:4]
    loss_alpha_weighted = (rgba[:, :3] * alpha).sum()

    # Loss 3: Combined loss
    loss = loss_rgb + 0.1 * loss_alpha_weighted

    # Check if any pixels were rendered
    pixels_rendered = (alpha > 0).sum().item()
    coverage = pixels_rendered / (img_h * img_w)

    # Backprop
    loss.backward()

    # Check gradients with detailed statistics
    if test_logits.grad is not None:
        grad = test_logits.grad

        # Overall gradient stats
        grad_norm_l2 = grad.norm().item()
        grad_norm_l1 = grad.abs().sum().item()
        grad_max = grad.abs().max().item()
        grad_mean = grad.abs().mean().item()
        grad_std = grad.std().item()
        grad_min = grad.min().item()
        grad_max_signed = grad.max().item()

        # Per-material gradient statistics (across all blocks)
        num_materials = grad.shape[1]
        per_material_grads = {}
        for mat_idx in range(num_materials):
            mat_grad = grad[:, mat_idx]
            per_material_grads[f'material_{mat_idx}'] = {
                'mean': float(mat_grad.mean().item()),
                'std': float(mat_grad.std().item()),
                'max': float(mat_grad.max().item()),
                'min': float(mat_grad.min().item()),
                'norm_l2': float(mat_grad.norm().item())
            }

        # Check for gradient flow issues
        has_gradient = grad_norm_l2 > 1e-6
        has_nan = torch.isnan(grad).any().item()
        has_inf = torch.isinf(grad).any().item()

        # Count how many blocks have significant gradients
        block_grad_norms = grad.norm(dim=1)
        blocks_with_grads = (block_grad_norms > 1e-6).sum().item()

    else:
        grad_norm_l2 = 0.0
        grad_norm_l1 = 0.0
        grad_max = 0.0
        grad_mean = 0.0
        grad_std = 0.0
        grad_min = 0.0
        grad_max_signed = 0.0
        per_material_grads = {}
        has_gradient = False
        has_nan = False
        has_inf = False
        blocks_with_grads = 0

    # Compute gradient health score (0-1, higher is better)
    health_score = 0.0
    if has_gradient and not has_nan and not has_inf:
        # Penalize if very few blocks have gradients
        block_ratio = blocks_with_grads / len(positions) if len(positions) > 0 else 0
        # Penalize if gradients are too small or too large
        magnitude_score = min(1.0, grad_norm_l2 / 100.0) if grad_norm_l2 < 100 else max(0.0, 1.0 - (grad_norm_l2 - 100) / 1000)
        health_score = 0.5 * block_ratio + 0.5 * magnitude_score

    return {
        # Overall gradient metrics
        'view_index': view_index,
        'has_gradient': has_gradient,
        'gradient_norm_l2': float(grad_norm_l2),
        'gradient_norm_l1': float(grad_norm_l1),
        'gradient_max_abs': float(grad_max),
        'gradient_mean_abs': float(grad_mean),
        'gradient_std': float(grad_std),
        'gradient_min': float(grad_min),
        'gradient_max': float(grad_max_signed),

        # Gradient health indicators
        'has_nan_gradients': has_nan,
        'has_inf_gradients': has_inf,
        'blocks_with_gradients': int(blocks_with_grads),
        'blocks_total': len(positions),
        'gradient_coverage_ratio': float(blocks_with_grads / len(positions)) if len(positions) > 0 else 0.0,
        'gradient_health_score': float(health_score),

        # Loss breakdown
        'loss_total': float(loss.item()),
        'loss_rgb': float(loss_rgb.item()),
        'loss_alpha_weighted': float(loss_alpha_weighted.item()),

        # Rendering metrics
        'pixels_rendered': int(pixels_rendered),
        'pixel_coverage': float(coverage),

        # Per-material gradients
        'per_material_gradients': per_material_grads,

        # Overall differentiability flag
        'differentiable': has_gradient and pixels_rendered > 0 and not has_nan and not has_inf
    }


def render_view(
    renderer: DifferentiableBlockRenderer,
    map_data: Dict[str, Any],
    metadata: Dict[str, Any],
    view_index: int,
    device: torch.device,
    test_gradients: bool = False
) -> Tuple[np.ndarray, Dict[str, Any], float, torch.Tensor, torch.Tensor, List, Dict[str, Any]]:
    """
    Render a single view using nvdiffrast.

    Args:
        renderer: DifferentiableBlockRenderer instance
        map_data: Map data dict
        metadata: Dataset metadata
        view_index: Which view to render
        device: Torch device
        test_gradients: Whether to test gradient flow

    Returns:
        rgb_array: (H, W, 3) uint8 array
        view_info: Dict with view metadata
        render_time: Time taken to render (seconds)
        rgba: Raw RGBA tensor output
        material_logits: Material logits tensor
        positions: List of block positions
        gradient_result: Dict with gradient test results (empty if test_gradients=False)
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
    start_time = time.time()
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
    render_time = time.time() - start_time

    # Test gradients if requested
    gradient_result = {}
    if test_gradients:
        print("  Testing gradient flow...")
        gradient_result = test_gradient_flow(
            renderer,
            positions,
            material_logits,
            camera_view,
            camera_proj,
            img_h,
            img_w,
            view_index
        )
        print(f"  Gradients: {'✓ FLOW' if gradient_result['differentiable'] else '✗ BLOCKED'}")
        print(f"    Pixel coverage: {gradient_result['pixel_coverage']*100:.1f}%")
        print(f"    Gradient L2 norm: {gradient_result['gradient_norm_l2']:.6f}")
        print(f"    Gradient health score: {gradient_result['gradient_health_score']:.3f}")
        print(f"    Blocks with gradients: {gradient_result['blocks_with_gradients']}/{gradient_result['blocks_total']}")

    # Convert to numpy (H, W, 3)
    rgb = rgba[0, :3, :, :].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)

    return rgb, view_info, render_time, rgba, material_logits, positions, gradient_result


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
    parser.add_argument('--test_gradients', action='store_true', default=True, help="Test gradient flow for each view (default: True)")
    parser.add_argument('--no_test_gradients', action='store_true', help="Disable gradient testing")
    args = parser.parse_args()

    # Override test_gradients if --no_test_gradients is set
    if args.no_test_gradients:
        args.test_gradients = False

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

    # Create logger
    log_dir = Path("out_local/nv_diff_logs")
    logger = DifferentiabilityLogger(log_dir, args.dataset_id)

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
        rgb, view_info, render_time, rgba, material_logits, positions, gradient_result = render_view(
            renderer, map_data, metadata, view_idx, device, test_gradients=args.test_gradients
        )

        print(f"Render time: {render_time*1000:.1f}ms")

        # Log metrics
        logger.log_view(
            view_idx,
            len(positions),
            render_time,
            rgba,
            material_logits,
            gradient_result
        )

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

    # Save logs
    logger.save()

    # Print summary
    print(f"\n=== Done! Rendered {len(view_indices)} view(s) ===")
    if args.test_gradients:
        num_diff = sum(1 for v in logger.view_logs if v['gradient_test'].get('differentiable', False))
        print(f"Differentiable views: {num_diff}/{len(view_indices)}")


if __name__ == '__main__':
    main()
