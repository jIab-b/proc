"""Simplest possible end-to-end gradient test."""
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
print(f"logits.requires_grad: {logits.requires_grad}")

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

print(f"\ncolors.shape: {attributes['colors'].shape}")
print(f"colors.requires_grad: {attributes['colors'].requires_grad}")

# Simple loss: sum of colors
loss = attributes['colors'].sum()
print(f"\nloss: {loss.item():.6f}")

# Backprop
loss.backward()

print(f"\nlogits.grad: {logits.grad}")
if logits.grad is not None:
    print(f"grad_norm: {logits.grad.norm().item():.6f}")
