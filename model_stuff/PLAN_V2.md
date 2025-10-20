# SDS Training Pipeline Plan V2
**Updated:** 2025-01-20
**Goal:** Use Score Distillation Sampling (SDS) to optimize voxel worlds for structure/geometry and basic color consistency (not photorealism)

---

## Executive Summary

### What Works (Verified)
✅ **nvdiffrast renderer** - Pixel-perfect WebGPU match, gradients flow correctly
✅ **Material optimization** - Gradients from pixels → material logits (tested: L2 norm ~3.0)
✅ **SDS loss** - SDXL integration ready via `sdxl_lightning.py`
✅ **8-material palette** - Sufficient for basic color semantics

### Architecture Decision
**Two-parameter dense voxel grid:**
- `occupancy_logits`: (X, Y, Z) → Controls which voxels exist
- `material_logits`: (X, Y, Z, M) → Controls what material each voxel is

### Key Insight
For **geometry + basic colors** (not photorealism), this approach is **well-suited**:
- 32×32×32 voxels ≈ Architectural scale (blocks are ~2m)
- SDS excels at discovering coarse 3D structure
- 8 colors sufficient for material semantics (stone=gray, grass=green, etc.)

---

## Phase 0: Gradient Flow Verification ✅ COMPLETE

### Status
**Renderer gradients verified working:**
```bash
python -m model_stuff.nv_diff_render.test_render --dataset_id 1 --view_index 0
# Output: Gradient L2 norm: 3.082408
#         Blocks with gradients: 3493/88403
#         ✓ GRADIENTS FLOW
```

### Critical Files
- `nv_diff_render/renderer.py` - DifferentiableBlockRenderer ✅
- `nv_diff_render/mesh_builder.py` - Differentiable mesh generation ✅
- `nv_diff_render/shading.py` - TerrainShader (lighting) ✅

### Gradient Flow Path (VERIFIED)
```
SDS Loss → ∂loss/∂pixels
  ↓
∂pixels/∂lit_colors (shader)
  ↓
∂lit_colors/∂interpolated_colors (nvdiffrast.interpolate)
  ↓
∂interpolated_colors/∂vertex_colors (mesh)
  ↓
∂vertex_colors/∂material_probs (softmax weighting)
  ↓
∂material_probs/∂material_logits ✓ LEARNED
```

**Key implementation detail (mesh_builder.py:95):**
```python
# Weighted color computation (CRITICAL for gradients)
face_colors = palette[:, palette_slot, :]  # (M, 3)
weighted_color = (block_mat_probs.unsqueeze(-1) * face_colors).sum(dim=0)
# This maintains gradient flow from colors → material_probs
```

---

## Phase 1: Implement Differentiable Voxel Grid

### Architecture

```python
class DifferentiableVoxelGrid(nn.Module):
    """
    Dense voxel grid for SDS optimization.

    Learnable Parameters:
    - occupancy_logits: nn.Parameter((X, Y, Z))
    - material_logits: nn.Parameter((X, Y, Z, M))

    Forward pass:
    1. occupancy_probs = sigmoid(occupancy_logits)
    2. active_mask = occupancy_probs > threshold
    3. Extract active positions
    4. Modulate: effective_logits = material_logits * occupancy.unsqueeze(-1)
    5. Render via DifferentiableBlockRenderer
    """

    def __init__(self, grid_size, num_materials=8, device='cuda'):
        super().__init__()
        X, Y, Z = grid_size

        # Learnable parameters
        self.occupancy_logits = nn.Parameter(
            torch.zeros(X, Y, Z, device=device)
        )
        self.material_logits = nn.Parameter(
            torch.zeros(X, Y, Z, num_materials, device=device)
        )

        # Renderer
        self.renderer = DifferentiableBlockRenderer(
            grid_size=grid_size,
            world_scale=2.0,
            device=device
        )

    def forward(self, camera_view, camera_proj, img_h, img_w, temperature=1.0):
        # Occupancy probabilities
        occ_probs = torch.sigmoid(self.occupancy_logits)

        # Threshold for active voxels
        active_mask = occ_probs > 0.01
        active_indices = torch.nonzero(active_mask, as_tuple=False)

        if len(active_indices) == 0:
            # Return sky
            return self.renderer.shader.sky_color.view(1, 3, 1, 1).expand(1, 3, img_h, img_w)

        # Extract positions
        positions = [(int(x), int(y), int(z)) for x, y, z in active_indices]

        # Extract and modulate material logits
        active_occ = occ_probs[active_indices[:, 0], active_indices[:, 1], active_indices[:, 2]]
        active_mat_logits = self.material_logits[active_indices[:, 0], active_indices[:, 1], active_indices[:, 2]]

        # CRITICAL: Modulate by occupancy to create gradient path
        modulated_logits = active_mat_logits * active_occ.unsqueeze(-1)

        # Render
        rgba = self.renderer.render(
            positions,
            modulated_logits,
            camera_view,
            camera_proj,
            img_h,
            img_w,
            temperature=temperature
        )

        return rgba
```

### Initialization Strategies

#### From Existing Map
```python
def init_from_map(grid, map_data):
    """Initialize from map.json with strong priors."""
    for block in map_data['blocks']:
        x, y, z = block['position']
        mat_idx = MATERIALS.index(block['blockType'])

        # High occupancy
        grid.occupancy_logits.data[x, y, z] = 5.0  # sigmoid(5) ≈ 0.993

        # Strong material preference
        grid.material_logits.data[x, y, z, :] = 0.0
        grid.material_logits.data[x, y, z, mat_idx] = 10.0
```

#### From Scratch (Primitive + Noise)
```python
def init_from_primitive(grid, primitive_type='ground_plane'):
    """Initialize with architectural primitive."""
    X, Y, Z = grid.occupancy_logits.shape

    if primitive_type == 'ground_plane':
        # Solid ground layer
        grid.occupancy_logits.data[:, 0, :] = 3.0  # sigmoid(3) ≈ 0.95
        grid.material_logits.data[:, 0, :, GRASS_IDX] = 5.0

        # Sparse noise above
        grid.occupancy_logits.data[:, 1:, :] = -3.0 + torch.randn(X, Y-1, Z) * 0.5

    elif primitive_type == 'cube':
        # Central cube structure
        cx, cy, cz = X//2, 2, Z//2
        size = 4
        grid.occupancy_logits.data[cx-size:cx+size, cy:cy+size*2, cz-size:cz+size] = 3.0
        grid.material_logits.data[cx-size:cx+size, cy:cy+size*2, cz-size:cz+size, STONE_IDX] = 5.0
```

---

## Phase 2: SDS Training Loop

### Core Training Script

```python
# model_stuff/train_sds_v2.py

import torch
import torch.nn as nn
from pathlib import Path
import json
import random

from model_stuff.nv_diff_render import DifferentiableBlockRenderer
from model_stuff.sdxl_lightning import SDXLLightning
from model_stuff.materials import MATERIALS

class DifferentiableVoxelGrid(nn.Module):
    # ... (implementation from Phase 1)
    pass

def train_sds(
    prompt: str,
    grid_size: tuple = (32, 32, 32),
    steps: int = 500,
    lr: float = 0.01,
    output_path: str = "out_local/sds_output",
    dataset_id: int = 1,
    init_mode: str = 'ground_plane',  # or 'from_map', 'cube'
    cfg_scale: float = 7.5,
    temp_start: float = 2.0,
    temp_end: float = 0.5,
    lambda_sparsity: float = 0.001,
    lambda_entropy: float = 0.0001,
    lambda_smooth: float = 0.0,
    log_every: int = 10,
    image_every: int = 50,
):
    device = torch.device('cuda')

    # 1. Load dataset cameras
    dataset_path = Path(f"datasets/{dataset_id}")
    with open(dataset_path / "metadata.json") as f:
        metadata = json.load(f)

    # Extract camera matrices
    from model_stuff.nv_diff_render.utils import load_camera_matrices_from_metadata
    cameras = []
    for i in range(len(metadata['views'])):
        view, proj = load_camera_matrices_from_metadata(metadata, i)
        cameras.append((view.to(device), proj.to(device)))

    img_h = metadata['imageSize']['height']
    img_w = metadata['imageSize']['width']

    print(f"Loaded {len(cameras)} camera views ({img_w}×{img_h})")

    # 2. Initialize voxel grid
    grid = DifferentiableVoxelGrid(grid_size, num_materials=8, device=device)

    if init_mode == 'from_map':
        map_path = Path(f"maps/{dataset_id}/map.json")
        with open(map_path) as f:
            map_data = json.load(f)
        init_from_map(grid, map_data)
    else:
        init_from_primitive(grid, init_mode)

    print(f"Initialized grid: {grid_size}, mode={init_mode}")

    # 3. Setup SDXL
    sdxl = SDXLLightning(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        device=device,
        dtype=torch.float16,
        height=img_h,
        width=img_w
    )

    pe, pe_pooled, ue, ue_pooled, add_time_ids = sdxl.encode_prompt(prompt)
    print(f"SDXL ready, prompt: '{prompt}'")

    # 4. Optimizer
    optimizer = torch.optim.Adam(grid.parameters(), lr=lr)

    # 5. Logging setup
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "train.jsonl"
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # 6. Training loop
    for step in range(steps):
        # Anneal temperature
        t = temp_start + (temp_end - temp_start) * (step / max(steps - 1, 1))

        # Sample random camera
        cam_idx = random.randint(0, len(cameras) - 1)
        view, proj = cameras[cam_idx]

        # Forward pass
        rgba = grid(view, proj, img_h, img_w, temperature=t)
        rgb = rgba[:, :3, :, :].clamp(0, 1)

        # === SDS LOSS ===
        # Encode to latent
        z0 = sdxl.vae_encode(rgb.to(sdxl.dtype))

        # Sample timestep
        ts = sdxl.sample_timesteps(batch_size=1)

        # Add noise
        noise = torch.randn_like(z0)
        x_t = sdxl.add_noise(z0, noise, ts)

        # Predict noise with CFG
        eps_cfg = sdxl.eps_pred_cfg(
            x_t, ts, pe, pe_pooled, ue, ue_pooled, add_time_ids, cfg_scale
        )

        # SDS loss (should add timestep weighting)
        loss_sds = (eps_cfg - noise).pow(2).mean()

        # === REGULARIZATION ===
        # Sparsity loss
        occ_probs = torch.sigmoid(grid.occupancy_logits)
        loss_sparse = occ_probs.mean()

        # Entropy loss
        mat_probs = torch.softmax(grid.material_logits / t, dim=-1)
        entropy = -(mat_probs * torch.log(mat_probs.clamp_min(1e-8))).sum(dim=-1)
        loss_entropy = entropy.mean()

        # Smoothness loss (optional)
        if lambda_smooth > 0:
            dx = (occ_probs[1:, :, :] - occ_probs[:-1, :, :]).pow(2).mean()
            dy = (occ_probs[:, 1:, :] - occ_probs[:, :-1, :]).pow(2).mean()
            dz = (occ_probs[:, :, 1:] - occ_probs[:, :, :-1]).pow(2).mean()
            loss_smooth = dx + dy + dz
        else:
            loss_smooth = torch.tensor(0.0)

        # Total loss
        loss = loss_sds + lambda_sparsity * loss_sparse + lambda_entropy * loss_entropy + lambda_smooth * loss_smooth

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if (step + 1) % log_every == 0 or step == 0:
            log_entry = {
                "step": step + 1,
                "temperature": float(t),
                "camera_index": cam_idx,
                "losses": {
                    "total": float(loss.item()),
                    "sds": float(loss_sds.item()),
                    "sparsity": float(loss_sparse.item()),
                    "entropy": float(loss_entropy.item()),
                    "smooth": float(loss_smooth.item()) if lambda_smooth > 0 else 0.0
                },
                "grid_stats": {
                    "num_active": int((occ_probs > 0.5).sum().item()),
                    "occupancy_mean": float(occ_probs.mean().item()),
                    "occupancy_std": float(occ_probs.std().item())
                }
            }

            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            print(f"Step {step+1}/{steps}: loss={loss.item():.4f} "
                  f"(sds={loss_sds.item():.4f}, active={log_entry['grid_stats']['num_active']})")

        # Save images
        if (step + 1) % image_every == 0 or step == 0 or step == steps - 1:
            from PIL import Image
            import numpy as np

            img = rgb[0].detach().cpu().permute(1, 2, 0).numpy()
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(img).save(images_dir / f"step_{step+1:04d}_view{cam_idx}.png")

    # 7. Export final grid
    # TODO: Implement grid_to_map_json
    print(f"\nTraining complete!")
    print(f"Logs: {log_file}")
    print(f"Images: {images_dir}")

    return grid

if __name__ == "__main__":
    train_sds(
        prompt="a medieval stone castle",
        steps=500,
        grid_size=(32, 32, 32),
        init_mode='ground_plane'
    )
```

---

## Phase 3: Improvements & Tuning

### 1. Timestep Weighting (IMPORTANT)

Current SDS loss is missing importance weighting:

```python
# BEFORE (current):
loss_sds = (eps_cfg - noise).pow(2).mean()

# AFTER (proper SDS):
# Get timestep-dependent weight
alpha_t = sdxl.scheduler.alphas_cumprod[ts]
w_t = (1 - alpha_t) / torch.sqrt(1 - alpha_t)

loss_sds = (w_t * (eps_cfg - noise).pow(2)).mean()
```

**Why:** High-noise timesteps learn global structure, low-noise refines details.

### 2. Coarse-to-Fine Training

```python
def train_coarse_to_fine(prompt, max_steps=1000):
    # Phase 1: 16×16×16 grid (steps 0-300)
    grid_coarse = train_sds(prompt, grid_size=(16, 16, 16), steps=300)

    # Phase 2: Upsample to 32×32×32 (steps 300-1000)
    grid_fine = DifferentiableVoxelGrid((32, 32, 32))
    upsample_grid(grid_coarse, grid_fine)  # Trilinear interpolation

    train_sds(prompt, grid=grid_fine, steps=700, init_mode='from_grid')
```

**Why:** Avoids local minima, builds structure hierarchically.

### 3. Adaptive Timestep Scheduling

```python
def schedule_timesteps(step, total_steps):
    """Start with high noise (structure), end with low noise (details)."""
    progress = step / total_steps

    if progress < 0.5:
        # Coarse phase: High noise (t ∈ [500, 900])
        t_min, t_max = 500, 900
    else:
        # Fine phase: Low noise (t ∈ [0, 500])
        t_min, t_max = 0, 500

    return torch.randint(t_min, t_max, (1,), device='cuda')
```

### 4. Connectivity Regularization

```python
def compute_connectivity_loss(occupancy_probs):
    """Penalize isolated voxels (encourage connected structures)."""
    occ = occupancy_probs

    # Count neighbors (6-connected)
    neighbors = (
        F.pad(occ[:-1, :, :], (0,0, 0,0, 1,0)) +  # -X
        F.pad(occ[1:, :, :],  (0,0, 0,0, 0,1)) +  # +X
        F.pad(occ[:, :-1, :], (0,0, 1,0, 0,0)) +  # -Y
        F.pad(occ[:, 1:, :],  (0,0, 0,1, 0,0)) +  # +Y
        F.pad(occ[:, :, :-1], (1,0, 0,0, 0,0)) +  # -Z
        F.pad(occ[:, :, 1:],  (0,1, 0,0, 0,0))    # +Z
    )

    # Loss is high if occupied voxel has few neighbors
    isolated_penalty = occ * torch.exp(-neighbors / 3)
    return isolated_penalty.mean()
```

---

## Expected Results

### Input
```
Prompt: "a medieval stone castle"
Grid: 32×32×32 voxels
Init: Ground plane (y=0 filled with Grass)
Steps: 500
```

### After Training
```
Occupancy:
├─ Ground layer: ~100% filled (y=0)
├─ Towers: 4 corners, height ~10-15 voxels
├─ Walls: Connecting towers, height ~5-8 voxels
├─ Central keep: Taller structure, height ~12-18 voxels
└─ Total active voxels: ~800-1500 / 32768

Materials:
├─ Stone: ~70% (walls, towers, keep)
├─ Grass: ~20% (ground)
├─ Dirt: ~5% (ground under structures)
└─ Other: ~5% (noise/variation)

Quality:
✅ Recognizable castle structure
✅ Correct proportions (tall towers, lower walls)
✅ Spatially coherent (connected geometry)
✅ Color consistency (stone = gray)
❌ Blocky appearance (expected for voxels)
❌ No fine details (windows, doors)
```

### Comparison to State-of-Art

| Method | Resolution | Quality | Speed | Suitable for Architecture |
|--------|-----------|---------|-------|---------------------------|
| **This (Voxel SDS)** | 32³ | Blocky | ~10 min | ✅ Perfect fit |
| DreamFusion | 128³ NeRF | Smooth | ~2 hours | ⚠️  Overkill |
| Magic3D | Mesh | High | ~30 min | ⚠️  Too detailed |
| Point-E | Point cloud | Medium | ~15 sec | ❌ No solid structures |

**Verdict:** For **geometry + basic colors** at **architectural scale**, voxel SDS is well-matched.

---

## Implementation Checklist

### Phase 1: Voxel Grid
- [ ] Implement `DifferentiableVoxelGrid` class
- [ ] Test occupancy gradient flow (verify modulation technique)
- [ ] Implement `init_from_map()` and `init_from_primitive()`
- [ ] Test forward pass (grid → render)

### Phase 2: Training Loop
- [ ] Implement `train_sds_v2.py` with multi-view sampling
- [ ] Add timestep weighting to SDS loss
- [ ] Add all regularization losses
- [ ] Test on simple prompt ("a cube")

### Phase 3: Validation
- [ ] Run on "a castle" prompt (500 steps)
- [ ] Verify structure emerges (towers, walls)
- [ ] Check material distribution matches semantics
- [ ] Export to map.json and load in WebGPU editor

### Phase 4: Improvements
- [ ] Implement coarse-to-fine training
- [ ] Add adaptive timestep scheduling
- [ ] Add connectivity regularization
- [ ] Tune hyperparameters

---

## Success Criteria

### Minimum Viable (MVP)
✅ SDS loss decreases over training
✅ Recognizable 3D structure emerges matching prompt
✅ Material colors match prompt semantics (castle → stone/gray)
✅ No NaN/Inf gradients
✅ Output is valid map.json

### Stretch Goals
✅ Coarse-to-fine training accelerates convergence
✅ Works from random init (not just primitive)
✅ Multi-prompt support ("castle at sunset")
✅ Export quality suitable for game level design

---

## Files to Create/Modify

### New Files
```
model_stuff/
├── train_sds_v2.py              # Main training script (this plan)
├── voxel_grid.py                # DifferentiableVoxelGrid class
├── sds_losses.py                # SDS + regularization losses
└── test_gradient_flow_v2.py     # Simpler gradient test
```

### Modified Files
```
model_stuff/
├── train_sds.py                 # Legacy (keep for reference)
└── sdxl_lightning.py           # Add timestep weighting method
```

---

## Risk Assessment

### High Risk ❌
- **Random init convergence**: Likely fails from pure noise
  - **Mitigation**: Always use primitive init (ground plane, cube, etc.)

### Medium Risk ⚠️
- **Hyperparameter sensitivity**: May need significant tuning
  - **Mitigation**: Start with proven values from DreamFusion

### Low Risk ✅
- **Gradient flow**: Already verified working
- **Material optimization**: Known to work from test_render
- **SDXL integration**: Already implemented in train_sds.py

---

## Next Steps (Immediate)

1. **Create `voxel_grid.py`** with `DifferentiableVoxelGrid` class
2. **Test occupancy gradients** with simple modulation example
3. **Implement `train_sds_v2.py`** minimal version (100 lines)
4. **Run test**: "a stone cube" (should converge quickly)
5. **If test passes**: Run "a castle" (500 steps)

**Estimated time to MVP**: 1-2 days of focused implementation

---

## References

- **DreamFusion**: "DreamFusion: Text-to-3D using 2D Diffusion" (2022)
- **Magic3D**: "Magic3D: High-Resolution Text-to-3D Content Creation" (2023)
- **SDS Loss**: Poole et al., Score Distillation Sampling
- **nvdiffrast**: NVIDIA Differentiable Rasterizer
- **SDXL**: Stable Diffusion XL (Stability AI)

---

**End of Plan V2**
