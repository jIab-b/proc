"""
Test gradient flow through nvdiffrast renderer.
"""

import torch
from model_stuff.nv_diff_render import DifferentiableBlockWorld
from model_stuff.nv_diff_render.utils import create_look_at_matrix, create_perspective_matrix

def test_gradient_flow():
    """Test that gradients flow through the renderer to material parameters."""
    print("Testing gradient flow through nvdiffrast renderer...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create world
    world = DifferentiableBlockWorld(
        grid_size=(64, 48, 64),
        world_scale=2.0,
        device=device
    )

    # Place 3 blocks with WEAK bias to enable gradients
    print("\nPlacing blocks with weak material bias...")
    world.place_block((32, 5, 32), material='Stone', bias_strength=0.1)
    world.place_block((33, 5, 32), material='Grass', bias_strength=0.1)
    world.place_block((32, 6, 32), material='Dirt', bias_strength=0.1)

    print("\nActual logit values:")
    for i, (pos, logits) in enumerate(world.blocks):
        print(f"  Block {i}: {logits.data}")

    # Create camera - looking directly at the blocks
    # Blocks are at (32, 5, 32), (33, 5, 32), (32, 6, 32)
    # In world coords that's roughly (0, 10, 0) after centering
    view = create_look_at_matrix(
        eye=(10.0, 15.0, 10.0),  # Closer and from the side
        center=(0.0, 10.0, 0.0),  # Looking at blocks
        up=(0.0, 1.0, 0.0)
    ).to(device)

    proj = create_perspective_matrix(
        fov_y_rad=1.047,
        aspect=1.0,
        near=0.1,
        far=500.0
    ).to(device)

    # Check parameters before rendering
    print("\nChecking parameters before rendering:")
    for i, (pos, logits) in enumerate(world.blocks):
        print(f"  Block {i}: is_leaf={logits.is_leaf}, requires_grad={logits.requires_grad}")

    # Render with soft materials for gradient flow
    print("\nRendering with temperature=1.0 (soft materials)...")
    rgba = world.render(
        view, proj,
        img_h=128, img_w=128,
        temperature=1.0,
        hard_materials=False
    )

    # Retain gradient on rgba to debug
    rgba.retain_grad()

    # Compute simple loss (sum of RGB values - more sensitive)
    print("\nComputing loss...")
    print(f"rgba.requires_grad: {rgba.requires_grad}")
    print(f"rgba min/max: {rgba[0, :3].min().item():.4f} / {rgba[0, :3].max().item():.4f}")
    loss = rgba[0, :3].sum()
    print(f"Loss value: {loss.item():.6f}")
    print(f"loss.requires_grad: {loss.requires_grad}")

    # Backpropagate
    print("\nBackpropagating...")
    loss.backward()

    # Check if gradient reached rgba
    print(f"\nrgba.grad (should exist): {rgba.grad is not None}")
    if rgba.grad is not None:
        print(f"rgba.grad norm: {rgba.grad.norm().item():.6f}")

    # Check gradients
    print("\nChecking parameter gradients:")
    has_gradients = False
    for i, (pos, logits) in enumerate(world.blocks):
        if logits.grad is not None:
            grad_norm = logits.grad.norm().item()
            print(f"  Block {i} at {pos}: grad_norm = {grad_norm:.6f}")
            if grad_norm > 0:
                has_gradients = True
        else:
            print(f"  Block {i} at {pos}: NO GRADIENT")

    if has_gradients:
        print("\n✓ SUCCESS: Gradients are flowing!")
        return True
    else:
        print("\n✗ FAILURE: No gradients detected")
        return False

if __name__ == '__main__':
    success = test_gradient_flow()
    exit(0 if success else 1)
