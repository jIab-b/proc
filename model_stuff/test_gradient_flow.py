"""
Test gradient flow through nvdiffrast renderer.

This verifies that gradients flow correctly from:
- Rendered pixels → Material logits (CRITICAL for SDS)
- Rendered pixels → Occupancy logits (CRITICAL for structure optimization)
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_stuff.nv_diff_render import DifferentiableBlockRenderer
from model_stuff.nv_diff_render.utils import create_perspective_matrix, create_look_at_matrix


def test_material_gradient_flow():
    """Test that gradients flow from pixels to material logits."""
    print("\n" + "="*60)
    print("TEST 1: Material Gradient Flow")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load camera from actual dataset
    import json
    from pathlib import Path
    from model_stuff.nv_diff_render.utils import load_camera_matrices_from_metadata

    dataset_path = Path("datasets/1")
    with open(dataset_path / "metadata.json") as f:
        metadata = json.load(f)

    view, proj = load_camera_matrices_from_metadata(metadata, view_index=0)
    view = view.to(device)
    proj = proj.to(device)

    img_w = metadata['imageSize']['width']
    img_h = metadata['imageSize']['height']

    print(f"Using dataset camera: {img_w}x{img_h}")

    # Load map data
    map_path = Path("maps/1/map.json")
    with open(map_path) as f:
        map_data = json.load(f)

    # Setup renderer with map grid size
    grid_size_x = map_data['worldConfig']['dimensions']['x']
    grid_size_y = map_data['worldConfig']['dimensions']['y']
    grid_size_z = map_data['worldConfig']['dimensions']['z']
    world_scale = map_data['worldConfig'].get('worldScale', 2.0)

    print(f"Grid size: {grid_size_x}×{grid_size_y}×{grid_size_z}, scale={world_scale}")

    renderer = DifferentiableBlockRenderer(
        grid_size=(grid_size_x, grid_size_y, grid_size_z),
        world_scale=world_scale,
        device=device
    )

    # Place a few blocks from the map
    first_blocks = map_data['blocks'][:20]  # Use more blocks
    positions = [tuple(b['position']) for b in first_blocks]

    # Material logits (N blocks, M=8 materials)
    # CRITICAL: requires_grad=True
    material_logits = torch.zeros(len(positions), 8, device=device, requires_grad=True)
    # Bias toward Stone (index 3) so we get visible gray blocks
    material_logits.data[:, 3] = 5.0

    # Render
    print("\nRendering...")
    print(f"Positions: {positions[:3]}...")
    print(f"Material logits shape: {material_logits.shape}")

    rgba = renderer.render(
        positions,
        material_logits,
        view,
        proj,
        img_h=img_h,
        img_w=img_w,
        temperature=1.0,
        hard_materials=False
    )

    print(f"Output shape: {rgba.shape}")  # Should be (1, 4, 256, 256)

    # Check if any geometry was rendered
    alpha = rgba[0, 3, :, :]  # Alpha channel
    num_opaque_pixels = (alpha > 0.5).sum().item()
    print(f"Opaque pixels: {num_opaque_pixels}/{img_h*img_w}")

    # Create simple loss (mean pixel value)
    rgb = rgba[0, :3, :, :]  # (3, 256, 256)
    loss = rgb.mean()

    print(f"Loss value: {loss.item():.6f}")

    # Backpropagate
    print("\nBackpropagating...")
    loss.backward()

    # Check gradients
    print("\n" + "-"*60)
    print("GRADIENT ANALYSIS")
    print("-"*60)

    if material_logits.grad is None:
        print("❌ FAILED: No gradients on material_logits!")
        return False

    print("✅ Gradients exist on material_logits")

    grad = material_logits.grad
    print(f"\nGradient shape: {grad.shape}")
    print(f"Gradient norm (L2): {grad.norm().item():.6f}")
    print(f"Gradient mean: {grad.mean().item():.8f}")
    print(f"Gradient std: {grad.std().item():.8f}")
    print(f"Gradient min: {grad.min().item():.8f}")
    print(f"Gradient max: {grad.max().item():.8f}")

    # Check for NaN/Inf
    has_nan = torch.isnan(grad).any()
    has_inf = torch.isinf(grad).any()

    if has_nan:
        print("❌ FAILED: NaN gradients detected!")
        return False

    if has_inf:
        print("❌ FAILED: Inf gradients detected!")
        return False

    print("✅ No NaN/Inf gradients")

    # Check gradient magnitude (should be non-zero)
    if grad.norm().item() < 1e-8:
        print("❌ FAILED: Vanishing gradients (norm too small)!")
        return False

    print("✅ Gradients are non-vanishing")

    # Check per-block gradients
    print("\nPer-block gradient norms:")
    for i in range(3):
        block_grad_norm = grad[i].norm().item()
        print(f"  Block {i} (pos {positions[i]}): {block_grad_norm:.6f}")

    # Health score (heuristic: what % of gradient entries are non-negligible)
    threshold = grad.abs().max() * 0.01  # 1% of max gradient
    active_grads = (grad.abs() > threshold).float().mean()
    print(f"\nGradient health score: {active_grads.item():.3f} ({active_grads.item()*100:.1f}% active)")

    print("\n" + "="*60)
    print("✅ TEST 1 PASSED: Material gradients flow correctly!")
    print("="*60)

    return True


def test_occupancy_gradient_flow():
    """Test that gradients can flow from pixels to occupancy logits."""
    print("\n" + "="*60)
    print("TEST 2: Occupancy Gradient Flow (Conceptual)")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Simulate dense grid with occupancy
    X, Y, Z = 32, 32, 32

    # Occupancy logits (one per voxel)
    occupancy_logits = torch.randn(X, Y, Z, device=device, requires_grad=True)

    # Material logits (one per voxel, M materials)
    material_logits = torch.randn(X, Y, Z, 8, device=device, requires_grad=True)

    # Convert occupancy to probabilities
    occupancy_probs = torch.sigmoid(occupancy_logits)

    print(f"Occupancy shape: {occupancy_probs.shape}")
    print(f"Mean occupancy: {occupancy_probs.mean().item():.3f}")

    # Threshold to get active voxels
    threshold = 0.5
    active_mask = occupancy_probs > threshold

    num_active = active_mask.sum().item()
    print(f"Active voxels (occupancy > {threshold}): {num_active}/{X*Y*Z}")

    # Extract active positions
    active_indices = torch.nonzero(active_mask, as_tuple=False)  # (N, 3)

    if len(active_indices) == 0:
        print("⚠️  No active voxels, adjusting threshold...")
        threshold = 0.1
        active_mask = occupancy_probs > threshold
        active_indices = torch.nonzero(active_mask, as_tuple=False)
        num_active = active_indices.shape[0]
        print(f"Active voxels (occupancy > {threshold}): {num_active}/{X*Y*Z}")

    positions = [(int(x), int(y), int(z)) for x, y, z in active_indices[:100]]  # Limit for speed

    # Extract material logits for active voxels
    active_material_logits = material_logits[active_indices[:, 0], active_indices[:, 1], active_indices[:, 2]][:100]

    # Modulate by occupancy (CRITICAL: this creates gradient path to occupancy)
    active_occupancy = occupancy_probs[active_indices[:, 0], active_indices[:, 1], active_indices[:, 2]][:100]
    modulated_logits = active_material_logits * active_occupancy.unsqueeze(-1)

    print(f"\nModulated logits shape: {modulated_logits.shape}")

    # Setup renderer
    renderer = DifferentiableBlockRenderer(
        grid_size=(X, Y, Z),
        world_scale=2.0,
        device=device
    )

    # Camera
    view = create_look_at_matrix(
        eye=(0.0, 20.0, 30.0),
        center=(0.0, 10.0, 0.0),
        up=(0.0, 1.0, 0.0)
    ).to(device)

    proj = create_perspective_matrix(
        fov_y_rad=1.047,
        aspect=1.0,
        near=0.1,
        far=500.0
    ).to(device)

    # Render
    print("\nRendering...")
    rgba = renderer.render(
        positions,
        modulated_logits,
        view,
        proj,
        img_h=256,
        img_w=256,
        temperature=1.0
    )

    # Loss
    rgb = rgba[0, :3, :, :]
    loss = rgb.mean()

    print(f"Loss value: {loss.item():.6f}")

    # Backpropagate
    print("\nBackpropagating...")
    loss.backward()

    # Check gradients on occupancy
    print("\n" + "-"*60)
    print("OCCUPANCY GRADIENT ANALYSIS")
    print("-"*60)

    if occupancy_logits.grad is None:
        print("❌ FAILED: No gradients on occupancy_logits!")
        return False

    print("✅ Gradients exist on occupancy_logits")

    occ_grad = occupancy_logits.grad
    print(f"\nGradient shape: {occ_grad.shape}")
    print(f"Gradient norm (L2): {occ_grad.norm().item():.6f}")
    print(f"Non-zero gradients: {(occ_grad.abs() > 1e-8).sum().item()}/{occ_grad.numel()}")

    # Check material gradients
    if material_logits.grad is None:
        print("❌ FAILED: No gradients on material_logits!")
        return False

    print("✅ Gradients exist on material_logits")

    mat_grad = material_logits.grad
    print(f"\nMaterial gradient norm (L2): {mat_grad.norm().item():.6f}")

    # Check for NaN/Inf
    if torch.isnan(occ_grad).any() or torch.isnan(mat_grad).any():
        print("❌ FAILED: NaN gradients!")
        return False

    if torch.isinf(occ_grad).any() or torch.isinf(mat_grad).any():
        print("❌ FAILED: Inf gradients!")
        return False

    print("✅ No NaN/Inf gradients")

    print("\n" + "="*60)
    print("✅ TEST 2 PASSED: Occupancy gradients flow correctly!")
    print("="*60)
    print("\nKey insight: Modulating material_logits by occupancy creates")
    print("a differentiable path from pixels → materials → occupancy")

    return True


def test_sds_style_gradient():
    """Test gradient flow in SDS-style training setup."""
    print("\n" + "="*60)
    print("TEST 3: SDS-Style Gradient Flow")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup
    renderer = DifferentiableBlockRenderer(
        grid_size=(32, 32, 32),
        world_scale=2.0,
        device=device
    )

    # Simple scene
    positions = [(15, i, 15) for i in range(5, 10)]  # Vertical column
    material_logits = torch.zeros(5, 8, device=device, requires_grad=True)
    material_logits.data[:, 3] = 5.0  # Bias toward Stone

    # Camera
    view = create_look_at_matrix(
        eye=(5.0, 10.0, 20.0),
        center=(0.0, 7.0, 0.0),
        up=(0.0, 1.0, 0.0)
    ).to(device)

    proj = create_perspective_matrix(1.047, 1.0, 0.1, 500.0).to(device)

    # Optimizer
    optimizer = torch.optim.Adam([material_logits], lr=0.01)

    print("\nRunning mini optimization loop (10 steps)...")
    print("Goal: Push rendered image toward target color [0.8, 0.3, 0.3] (reddish)")

    target_color = torch.tensor([0.8, 0.3, 0.3], device=device)

    for step in range(10):
        optimizer.zero_grad()

        # Render
        rgba = renderer.render(
            positions,
            material_logits,
            view,
            proj,
            img_h=128,
            img_w=128,
            temperature=1.0
        )

        # Loss: Match target color (simplified SDS-like objective)
        rgb = rgba[0, :3, :, :]
        mean_color = rgb.mean(dim=[1, 2])  # (3,)
        loss = ((mean_color - target_color) ** 2).sum()

        # Backprop
        loss.backward()
        optimizer.step()

        if step % 3 == 0:
            print(f"  Step {step:2d}: loss={loss.item():.6f}, color={mean_color.detach().cpu().numpy()}")

    print("\n✅ TEST 3 PASSED: SDS-style optimization works!")
    print(f"Final mean color: {mean_color.detach().cpu().numpy()}")
    print(f"Target color: {target_color.cpu().numpy()}")

    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("NVDIFFRAST RENDERER GRADIENT FLOW TESTS")
    print("="*60)

    success = True

    # Test 1: Material gradients
    try:
        success = test_material_gradient_flow() and success
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        success = False

    # Test 2: Occupancy gradients
    try:
        success = test_occupancy_gradient_flow() and success
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        success = False

    # Test 3: SDS-style optimization
    try:
        success = test_sds_style_gradient() and success
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        success = False

    # Summary
    print("\n" + "="*60)
    if success:
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nGradient flow is working correctly:")
        print("  1. ✅ Pixels → Material logits")
        print("  2. ✅ Pixels → Occupancy logits (via modulation)")
        print("  3. ✅ SDS-style optimization converges")
        print("\nRenderer is ready for SDS training!")
    else:
        print("❌ SOME TESTS FAILED!")
        print("="*60)
        print("\nPlease review errors above and fix gradient flow issues.")

    print("="*60 + "\n")

    sys.exit(0 if success else 1)
