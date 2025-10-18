"""Debug mesh builder gradient flow."""
import torch
import torch.nn.functional as F

# Simulate the mesh builder logic
device = torch.device('cuda')

# Create material logits with gradients
material_logits = torch.randn(3, 8, device=device, requires_grad=True)
print(f"material_logits.requires_grad: {material_logits.requires_grad}")

# Softmax
material_probs = F.softmax(material_logits / 1.0, dim=-1)
print(f"material_probs.requires_grad: {material_probs.requires_grad}")

# Palette (constant)
palette = torch.randn(8, 3, 3, device=device)
print(f"palette.requires_grad: {palette.requires_grad}")

# Simulate weighted color computation
all_weighted_colors = []
for block_idx in range(3):
    block_mat_probs = material_probs[block_idx]  # (M,)
    print(f"\nBlock {block_idx}: block_mat_probs.requires_grad = {block_mat_probs.requires_grad}")

    # Face loop
    for face_idx in range(2):  # Simplified: just 2 faces
        palette_slot = 2  # side
        face_colors = palette[:, palette_slot, :]  # (M, 3)
        weighted_color = (block_mat_probs.unsqueeze(-1) * face_colors).sum(dim=0)  # (3,)
        print(f"  Face {face_idx}: weighted_color.requires_grad = {weighted_color.requires_grad}")

        # Append to list
        all_weighted_colors.append(weighted_color)

# Stack
print(f"\nBefore stack: len(all_weighted_colors) = {len(all_weighted_colors)}")
colors = torch.stack(all_weighted_colors, dim=0)
print(f"After stack: colors.shape = {colors.shape}")
print(f"colors.requires_grad: {colors.requires_grad}")

# Compute loss and backprop
loss = colors.mean()
print(f"\nloss: {loss.item()}")
loss.backward()

print(f"\nmaterial_logits.grad:")
print(material_logits.grad)
print(f"grad norm: {material_logits.grad.norm().item()}")
