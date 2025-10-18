"""Debug nvdiffrast interpolate gradient flow."""
import torch
import torch.nn as nn
import nvdiffrast.torch as dr

device = torch.device('cuda')

# Create simple mesh: single triangle
vertices = torch.tensor([
    [-1.0, -1.0, 0.5, 1.0],
    [ 1.0, -1.0, 0.5, 1.0],
    [ 0.0,  1.0, 0.5, 1.0]
], dtype=torch.float32, device=device).unsqueeze(0)  # (1, 3, 4)

faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)

# Create LEAF tensor (nn.Parameter) - THIS IS THE KEY!
color_params = nn.Parameter(torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
], dtype=torch.float32, device=device))  # (3, 3)

print(f"color_params.is_leaf: {color_params.is_leaf}")
print(f"color_params.requires_grad: {color_params.requires_grad}")

# Add batch dimension
colors = color_params.unsqueeze(0)  # (1, 3, 3)

# Create context and rasterize
glctx = dr.RasterizeCudaContext()
rast, rast_db = dr.rasterize(glctx, vertices, faces, resolution=[64, 64])

print(f"rast.shape: {rast.shape}")

# Interpolate colors
interp_colors, _ = dr.interpolate(colors, rast, faces)
print(f"interp_colors.shape: {interp_colors.shape}")
print(f"interp_colors.requires_grad: {interp_colors.requires_grad}")

# Compute loss
loss = interp_colors.mean()
print(f"loss: {loss.item()}")

# Backprop
loss.backward()

print(f"\ncolor_params.grad (LEAF tensor):")
print(color_params.grad)
if color_params.grad is not None:
    print(f"grad norm: {color_params.grad.norm().item()}")
else:
    print("ERROR: No gradient!")
