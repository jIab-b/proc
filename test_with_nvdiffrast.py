"""Test gradient flow through mesh builder + nvdiffrast."""
import torch
import torch.nn as nn
from model_stuff.nv_diff_render.mesh_builder import build_block_mesh
from model_stuff.nv_diff_render.utils import create_look_at_matrix, create_perspective_matrix, world_to_clip
import nvdiffrast.torch as dr

device = torch.device('cuda')

# Create material logits as parameters
logits = nn.Parameter(torch.zeros(1, 8, device=device))
logits.data[0, 3] = 0.1  # Stone

print(f"logits.is_leaf: {logits.is_leaf}")

# Build mesh
positions = [(32, 5, 32)]
vertices, faces, attributes = build_block_mesh(
    positions,
    logits,
    grid_size=(64, 48, 64),
    world_scale=2.0,
    neighbor_check=None,
    temperature=1.0,
    hard_assignment=False
)

print(f"colors.requires_grad: {attributes['colors'].requires_grad}")

# Camera
view = create_look_at_matrix(
    eye=(0.0, 20.0, 30.0),
    center=(0.0, 5.0, 0.0),
    up=(0.0, 1.0, 0.0)
).to(device)

proj = create_perspective_matrix(
    fov_y_rad=1.047,
    aspect=1.0,
    near=0.1,
    far=500.0
).to(device)

# Transform to clip space
clip_pos = world_to_clip(vertices, view, proj)
clip_pos_batch = clip_pos.unsqueeze(0)
faces_int32 = faces.int()

# Rasterize
glctx = dr.RasterizeCudaContext()
rast, rast_db = dr.rasterize(glctx, clip_pos_batch, faces_int32, resolution=[128, 128])

# Interpolate colors
interp_colors, _ = dr.interpolate(
    attributes['colors'].unsqueeze(0),
    rast,
    faces_int32
)

print(f"interp_colors.requires_grad: {interp_colors.requires_grad}")

# Loss
loss = interp_colors.sum()
print(f"\nloss: {loss.item():.6f}")

# Backprop
loss.backward()

print(f"\nlogits.grad: {logits.grad}")
if logits.grad is not None:
    print(f"grad_norm: {logits.grad.norm().item():.6f}")
else:
    print("NO GRADIENT!")
