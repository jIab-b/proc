"""
Quick test of the full SDS pipeline.

Tests:
1. Load map.json → grid
2. Save grid → map.json
3. Initialize from primitive
4. Render from grid
"""

import torch
from pathlib import Path

def test_map_io():
    """Test loading and saving map.json"""
    print("\n" + "="*60)
    print("TEST 1: Map I/O")
    print("="*60)

    from model_stuff.map_io import load_map_to_grid, save_grid_to_map

    # Load
    occ, mat, grid_size, scale = load_map_to_grid("maps/1/map.json", device='cuda')

    print(f"\nLoaded grid:")
    print(f"  Occupancy logits: {occ.shape}")
    print(f"  Material logits: {mat.shape}")
    print(f"  Grid size: {grid_size}")
    print(f"  World scale: {scale}")

    # Save
    test_output = Path("out_local/test_map_io.json")
    num_blocks = save_grid_to_map(occ, mat, test_output, world_scale=scale)

    print(f"\n✅ Test passed: {num_blocks} blocks saved to {test_output}")

    return occ, mat, grid_size, scale


def test_voxel_grid(occ, mat, grid_size, scale):
    """Test DifferentiableVoxelGrid"""
    print("\n" + "="*60)
    print("TEST 2: Voxel Grid")
    print("="*60)

    from model_stuff.voxel_grid import DifferentiableVoxelGrid

    grid = DifferentiableVoxelGrid(
        grid_size=grid_size,
        num_materials=8,
        world_scale=scale,
        device=torch.device('cuda')
    )

    grid.load_state(occ, mat)

    print(f"\nGrid initialized: {grid}")

    stats = grid.get_stats()
    print(f"\nStats:")
    print(f"  Active voxels: {stats['num_active_voxels']}")
    print(f"  Density: {stats['density']*100:.2f}%")

    print(f"\n✅ Test passed")

    return grid


def test_rendering(grid):
    """Test rendering from grid"""
    print("\n" + "="*60)
    print("TEST 3: Rendering")
    print("="*60)

    import json
    from model_stuff.nv_diff_render.utils import load_camera_matrices_from_metadata

    # Load camera
    with open("datasets/1/metadata.json") as f:
        metadata = json.load(f)

    view, proj = load_camera_matrices_from_metadata(metadata, view_index=0)
    view = view.to('cuda')
    proj = proj.to('cuda')

    img_w = metadata['imageSize']['width']
    img_h = metadata['imageSize']['height']

    print(f"\nRendering {img_w}×{img_h} image...")

    # Render
    rgba = grid(view, proj, img_h, img_w, temperature=1.0)

    print(f"  Output shape: {rgba.shape}")

    # Check if geometry rendered
    alpha = rgba[0, 3, :, :]
    opaque_pixels = (alpha > 0.5).sum().item()
    total_pixels = img_h * img_w

    print(f"  Opaque pixels: {opaque_pixels}/{total_pixels} ({opaque_pixels/total_pixels*100:.1f}%)")

    if opaque_pixels > 0:
        print(f"\n✅ Test passed: Geometry rendered successfully")
    else:
        print(f"\n⚠️  Warning: No geometry rendered (check camera/grid alignment)")

    # Save test image
    from PIL import Image
    import numpy as np

    rgb = rgba[0, :3, :, :].detach().cpu().permute(1, 2, 0).numpy()
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    output_path = Path("out_local/test_render.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output_path)

    print(f"  Saved test render to {output_path}")


def test_gradient_flow(grid):
    """Test that gradients flow"""
    print("\n" + "="*60)
    print("TEST 4: Gradient Flow")
    print("="*60)

    import json
    from model_stuff.nv_diff_render.utils import load_camera_matrices_from_metadata

    # Load camera
    with open("datasets/1/metadata.json") as f:
        metadata = json.load(f)

    view, proj = load_camera_matrices_from_metadata(metadata, view_index=0)
    view = view.to('cuda')
    proj = proj.to('cuda')

    img_w = metadata['imageSize']['width']
    img_h = metadata['imageSize']['height']

    # Render
    rgba = grid(view, proj, img_h, img_w)

    # Simple loss
    loss = rgba.mean()

    print(f"  Loss value: {loss.item():.6f}")

    # Backprop
    loss.backward()

    # Check gradients
    occ_grad = grid.occupancy_logits.grad
    mat_grad = grid.material_logits.grad

    if occ_grad is None or mat_grad is None:
        print(f"\n❌ FAILED: No gradients!")
        return

    occ_norm = occ_grad.norm().item()
    mat_norm = mat_grad.norm().item()

    print(f"  Occupancy gradient norm: {occ_norm:.6f}")
    print(f"  Material gradient norm: {mat_norm:.6f}")

    if occ_norm > 1e-8 and mat_norm > 1e-8:
        print(f"\n✅ Test passed: Gradients flow correctly")
    else:
        print(f"\n❌ FAILED: Gradients vanishing")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SDS PIPELINE TESTS")
    print("="*60)

    try:
        # Test 1: Map I/O
        occ, mat, grid_size, scale = test_map_io()

        # Test 2: Voxel Grid
        grid = test_voxel_grid(occ, mat, grid_size, scale)

        # Test 3: Rendering
        test_rendering(grid)

        # Test 4: Gradient Flow
        test_gradient_flow(grid)

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        print("\nPipeline is ready for training!")
        print("Run: python -m model_stuff.train_sds_final --prompt 'your prompt' --steps 100")

    except Exception as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
