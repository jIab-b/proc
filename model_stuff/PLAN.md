# SDS Training Pipeline Plan

## Project Overview

**Goal:** Use Score Distillation Sampling (SDS) with SDXL Lightning to optimize voxel worlds toward text prompts, maintaining differentiability through both block positions (via occupancy) and materials.

**Input:** `map.json` + dataset with posed cameras
**Output:** Optimized `map.json` + training logs + test images

---

## Phase 1: Get Differentiable Rendering Right ⭐ CURRENT FOCUS

### Objective
Verify that the nvdiffrast renderer has proper gradient flow from:
- Final RGB pixels → material logits (already implemented)
- **TODO:** Final RGB pixels → occupancy logits (for DifferentiableVoxelGrid)

### Current Status

#### ✅ What Works:
- `DifferentiableBlockRenderer` - renders blocks with nvdiffrast
- `mesh_builder.py` - differentiable material → color conversion
- `TerrainShader` - differentiable lighting
- `test_render.py` - can render test images from datasets

#### ❌ What Needs Testing:
- **Gradient flow verification**: Need to run `test_render.py --test_gradients` and confirm:
  - Gradients reach material_logits
  - Gradient magnitudes are reasonable
  - No NaN/Inf issues
  - Per-material gradients are meaningful

#### 🔨 What Needs Building:
- `DifferentiableVoxelGrid` - dense grid with occupancy + materials
- Gradient flow from pixels → occupancy logits
- Map I/O for initializing grids from map.json

### Test Plan for Differentiability

#### Test 1: Material Gradient Flow (Current Architecture)

**Setup:**
```bash
python -m model_stuff.nv_diff_render.test_render \
  --dataset_id 1 \
  --view_index 0 \
  --test_gradients
```

**Expected Results:**
```json
{
  "gradient_test": {
    "has_gradient": true,
    "differentiable": true,
    "gradient_norm_l2": 100-1000,  // Should be non-zero
    "gradient_health_score": 0.5-1.0,  // Should be healthy
    "blocks_with_gradients": "most blocks",
    "per_material_gradients": {
      "material_0": {"norm_l2": ...},
      "material_1": {"norm_l2": ...},
      // etc.
    }
  }
}
```

**Success Criteria:**
- [ ] `differentiable: true`
- [ ] `gradient_norm_l2 > 1.0` (not vanishing)
- [ ] `has_nan_gradients: false`
- [ ] `has_inf_gradients: false`
- [ ] `gradient_health_score > 0.5`
- [ ] At least 50% of blocks have gradients

**If Test Fails:**
- Check if `dr.interpolate()` is breaking gradient flow
- Verify `torch.no_grad()` is not wrapping the test render
- Check material_logits `requires_grad=True`

#### Test 2: End-to-End SDS-Style Gradient Flow

**Purpose:** Simulate actual SDS training loop

**Test Code:**
```python
# model_stuff/nv_diff_render/test_gradient_e2e.py
def test_sds_style_gradient():
    """
    Test gradient flow matching SDS training:
    pixels → loss → backprop → material_logits
    """
    # Load map
    map_data = load_map_data(1)

    # Setup renderer
    renderer = DifferentiableBlockRenderer()

    # Extract blocks with requires_grad
    positions = [tuple(b['position']) for b in map_data['blocks']]
    material_logits = torch.randn(len(positions), 8, requires_grad=True)

    # Render
    rgba = renderer.render(positions, material_logits, camera_view, camera_proj, H, W)

    # Simulate SDS loss (simple MSE to target color)
    target = torch.ones_like(rgba) * 0.5
    loss = (rgba - target).pow(2).mean()

    # Backprop
    loss.backward()

    # Check gradients
    assert material_logits.grad is not None
    assert not torch.isnan(material_logits.grad).any()
    print(f"Gradient norm: {material_logits.grad.norm().item()}")
```

**Run:**
```bash
python -m model_stuff.nv_diff_render.test_gradient_e2e
```

**Success Criteria:**
- [ ] Gradients flow to material_logits
- [ ] Loss decreases over multiple steps
- [ ] Rendered colors move toward target

---

## Phase 2: Implement DifferentiableVoxelGrid

### Architecture

```python
class DifferentiableVoxelGrid(nn.Module):
    """
    Dense voxel grid for SDS optimization.

    Differentiable Parameters:
    - occupancy_logits: (X, Y, Z) → sigmoid → [0, 1]
    - material_logits: (X, Y, Z, M) → softmax → material probs

    Forward Pass:
    1. occupancy_probs = sigmoid(occupancy_logits)
    2. active_mask = occupancy_probs > threshold
    3. Extract positions where active
    4. Get material_logits for active positions
    5. Render via DifferentiableBlockRenderer
    """
```

### Key Design Decisions

#### Occupancy Representation
- **Logits** (not probabilities) for better optimization
- **Sigmoid** activation for smooth [0, 1] range
- **Threshold** (default 0.01) for rendering efficiency

#### Material Representation
- **Logits** per voxel: `(X, Y, Z, M)` where M=8
- **Modulated by occupancy**: `material_logits * occupancy.unsqueeze(-1)`
  - Low occupancy → weak material preferences
  - High occupancy → strong material preferences

#### Initialization from map.json
```python
def initialize_from_map(grid, map_data):
    for block in map_data['blocks']:
        x, y, z = block['position']
        material_idx = material_name_to_index(block['blockType'])

        # Set high occupancy
        grid.occupancy_logits[x, y, z] = 5.0  # sigmoid(5) ≈ 0.993

        # Set strong material preference
        grid.material_logits[x, y, z, :] = 0.0
        grid.material_logits[x, y, z, material_idx] = 10.0
```

### Gradient Flow Path

```
SDS Loss (from SDXL noise prediction)
  ↓
∂loss/∂pixels (RGB image)
  ↓
∂pixels/∂lit_colors (shader output)
  ↓
∂lit_colors/∂interpolated_colors (nvdiffrast)
  ↓
∂interpolated_colors/∂vertex_colors (mesh)
  ↓ (BRANCH 1: Material gradients)
∂vertex_colors/∂material_probs
  ↓
∂material_probs/∂material_logits ✓ LEARNED
  ↓ (BRANCH 2: Occupancy gradients - NEW)
∂vertex_colors/∂(occupancy * material_logits)
  ↓
∂/∂occupancy_logits ✓ LEARNED
```

### Files to Create

```
model_stuff/nv_diff_render/
├── voxel_grid.py              # DifferentiableVoxelGrid class
├── map_io.py                  # load_map_json, initialize_from_map, grid_to_map_json
└── test_gradient_e2e.py       # End-to-end gradient tests
```

---

## Phase 3: SDS Training Pipeline

### Training Loop Architecture

```python
def train_sds(
    input_map_path: str,
    dataset_dir: str,
    prompt: str,
    steps: int,
    lr: float
):
    # 1. Load dataset metadata (camera poses)
    metadata = load_metadata(dataset_dir / "metadata.json")
    cameras = extract_camera_matrices(metadata)

    # 2. Load map.json
    map_data = load_map_json(input_map_path)

    # 3. Determine grid size from map bounds
    grid_size = compute_grid_size_from_map(map_data)

    # 4. Initialize voxel grid
    grid = DifferentiableVoxelGrid(grid_size)
    initialize_from_map(grid, map_data)

    # 5. Setup SDXL
    sdxl = SDXLLightning(model_id, device='cuda')
    pe, pe_pooled, ue, ue_pooled, time_ids = sdxl.encode_prompt(prompt)

    # 6. Setup optimizer
    optimizer = Adam(grid.parameters(), lr=lr)

    # 7. Training loop
    for step in range(steps):
        # Sample random camera from dataset poses
        cam_idx = random.randint(0, len(cameras) - 1)
        camera_view, camera_proj = cameras[cam_idx]

        # Render
        temperature = schedule_temperature(step, steps)
        rgba = grid(camera_view, camera_proj, H, W, temperature)

        # SDS loss
        loss_sds = compute_sds_loss(rgba, sdxl, pe, pe_pooled, ...)

        # Regularization
        loss_sparsity = compute_sparsity_loss(grid)
        loss_entropy = compute_entropy_loss(grid)

        # Total loss
        loss = loss_sds + λ_sparse * loss_sparsity + λ_entropy * loss_entropy

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if step % LOG_EVERY == 0:
            log_metrics(step, loss, grid.get_stats())

        # Test images
        if step % IMAGE_EVERY == 0:
            render_test_views(grid, step, cameras[0:3])  # Fixed views

    # 8. Export final map
    save_map_json(output_map_path, grid, metadata={"prompt": prompt})
```

### Camera Handling

**From Dataset:**
- Load `datasets/1/metadata.json`
- Extract all camera matrices: `views[i].viewMatrix`, `views[i].projectionMatrix`
- During training: randomly sample from these camera poses
- For test images: use first 3 views (or specific indices)

**Grid Size from Map:**
```python
def compute_grid_size_from_map(map_data):
    """
    Compute minimal bounding grid from block positions.
    Add padding if needed.
    """
    blocks = map_data['blocks']
    if not blocks:
        return (32, 24, 32)  # Default

    positions = [b['position'] for b in blocks]
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]

    max_x = max(xs) + 1
    max_y = max(ys) + 1
    max_z = max(zs) + 1

    # Add 10% padding
    return (
        int(max_x * 1.1),
        int(max_y * 1.1),
        int(max_z * 1.1)
    )
```

### Logging Strategy

**Metrics to Track:**

```json
{
  "step": 100,
  "timestamp": "2025-10-19T10:30:00",
  "temperature": 1.2,
  "camera_index": 7,
  "losses": {
    "total": 0.345,
    "sds": 0.320,
    "sparsity": 0.015,
    "entropy": 0.010
  },
  "grid_stats": {
    "num_active_voxels": 1234,
    "occupancy_mean": 0.25,
    "occupancy_std": 0.18,
    "density": 0.012,
    "material_distribution": {
      "Air": 0.05,
      "Grass": 0.30,
      "Dirt": 0.15,
      "Stone": 0.25,
      "Plank": 0.10,
      "Snow": 0.05,
      "Sand": 0.05,
      "Water": 0.05
    }
  },
  "gradient_stats": {
    "occupancy_grad_norm": 2.45,
    "material_grad_norm": 1.83,
    "grad_max": 5.2,
    "grad_health_score": 0.87
  }
}
```

**File:** `out_local/logs/train.jsonl` (one JSON object per line)

### Test Image Rendering

**Fixed Views:**
- Use first 3 camera poses from dataset
- Or define canonical views: front, top, isometric

**Frequency:**
- Step 0 (initial state)
- Every IMAGE_EVERY steps (e.g., 50)
- Final step

**Naming:**
```
out_local/test_imgs/
├── step_0000_view_0.png
├── step_0000_view_1.png
├── step_0000_view_2.png
├── step_0050_view_0.png
├── ...
└── step_0500_view_2.png
```

---

## Phase 4: Regularization & Optimization

### Loss Components

#### 1. SDS Loss (Primary)
```python
def compute_sds_loss(rgba, sdxl, pe, pe_pooled, ue, ue_pooled, time_ids, cfg_scale):
    """
    Score Distillation Sampling loss.
    Guides rendered image toward text prompt.
    """
    rgb = rgba[:, :3, :, :]
    z0 = sdxl.vae_encode(rgb)
    ts = sdxl.sample_timesteps(batch_size=1)
    noise = torch.randn_like(z0)
    x_t = sdxl.add_noise(z0, noise, ts)
    eps_cfg = sdxl.eps_pred_cfg(x_t, ts, pe, pe_pooled, ue, ue_pooled, time_ids, cfg_scale)
    loss = (eps_cfg - noise).pow(2).mean()
    return loss
```

#### 2. Sparsity Loss
```python
def compute_sparsity_loss(grid):
    """
    Penalize too many active voxels.
    Encourages sparse, efficient structures.
    """
    occupancy_probs = torch.sigmoid(grid.occupancy_logits)
    return occupancy_probs.mean()
```

#### 3. Entropy Loss
```python
def compute_entropy_loss(grid):
    """
    Penalize uncertain material assignments.
    Encourages decisive material choices.
    """
    material_probs = F.softmax(grid.material_logits, dim=-1)
    entropy = -(material_probs * torch.log(material_probs + 1e-8)).sum(dim=-1)
    return entropy.mean()
```

#### 4. Spatial Smoothness (Optional)
```python
def compute_smoothness_loss(grid):
    """
    Penalize rapid changes in occupancy/materials.
    Encourages connected structures.
    """
    # Occupancy smoothness (L2 of gradients)
    occ = torch.sigmoid(grid.occupancy_logits)
    dx = (occ[1:, :, :] - occ[:-1, :, :]).pow(2).mean()
    dy = (occ[:, 1:, :] - occ[:, :-1, :]).pow(2).mean()
    dz = (occ[:, :, 1:] - occ[:, :, :-1]).pow(2).mean()
    return dx + dy + dz
```

### Hyperparameters

```python
# SDS
CFG_SCALE = 7.5
TIMESTEP_RANGE = (0.02, 0.98)  # Avoid extreme noise levels

# Regularization weights
LAMBDA_SPARSITY = 0.001
LAMBDA_ENTROPY = 0.0001
LAMBDA_SMOOTHNESS = 0.0  # Start with 0, add if needed

# Optimization
LR = 0.01
TEMPERATURE_START = 2.0
TEMPERATURE_END = 0.5
```

---

## Phase 5: Export & Validation

### Export to map.json

```python
def grid_to_map_json(grid: DifferentiableVoxelGrid, threshold=0.5):
    """
    Convert optimized voxel grid to map.json format.

    Steps:
    1. Threshold occupancy > 0.5 → discrete blocks
    2. Argmax materials → single material per block
    3. Format as map.json actions
    """
    occupancy_probs = torch.sigmoid(grid.occupancy_logits)
    material_probs = F.softmax(grid.material_logits, dim=-1)

    blocks = []
    X, Y, Z = grid.occupancy_logits.shape

    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                occ = occupancy_probs[x, y, z].item()

                if occ > threshold:
                    mat_idx = material_probs[x, y, z].argmax().item()
                    mat_name = MATERIALS[mat_idx]

                    if mat_name != "Air":
                        blocks.append({
                            "position": [x, y, z],
                            "blockType": mat_name
                        })

    return {
        "sequence": 1,
        "blocks": blocks,
        "worldConfig": {
            "dimensions": {"x": X, "y": Y, "z": Z},
            "worldScale": grid.world_scale
        }
    }
```

### Validation

**Compare with initial map:**
- Number of blocks (should be similar if sparsity is balanced)
- Material distribution changes
- Visual comparison of test renders

**Gradient test on final grid:**
- Run `test_gradient_flow()` on final state
- Verify gradients still healthy

---

## Directory Structure

```
datasets/1/
├── metadata.json          # Camera poses + image size
└── images/                # (Not used for training, only for eval)

maps/1/
├── map.json              # INPUT: Initial voxel world
└── map_optimized.json    # OUTPUT: SDS-optimized world

out_local/
├── logs/
│   ├── train.jsonl       # Training metrics
│   └── gradient_stats.jsonl
├── test_imgs/
│   ├── step_0000_view_0.png
│   ├── step_0000_view_1.png
│   └── ...
└── nv_diff_logs/
    └── final_gradient_test.json

model_stuff/
└── nv_diff_render/
    ├── voxel_grid.py     # NEW
    ├── map_io.py         # NEW
    ├── test_gradient_e2e.py  # NEW
    ├── renderer.py       # EXISTS
    ├── mesh_builder.py   # EXISTS
    └── ...
```

---

## Implementation Checklist

### Phase 1: Differentiability ✓ CURRENT
- [x] Update test_render.py to enable --test_gradients by default
- [ ] Run gradient test on dataset 1
- [ ] Verify gradient flow is healthy
- [ ] Document expected gradient magnitudes
- [ ] Create test_gradient_e2e.py for SDS-style testing

### Phase 2: Voxel Grid
- [ ] Implement DifferentiableVoxelGrid
- [ ] Implement map_io.py (load/save/initialize)
- [ ] Test initialization from map.json
- [ ] Test occupancy gradient flow
- [ ] Test forward pass (render from grid)

### Phase 3: Training Pipeline
- [ ] Create train_sds_nvdiff.py
- [ ] Implement camera loading from metadata.json
- [ ] Implement grid size computation from map
- [ ] Integrate SDXL Lightning
- [ ] Add SDS loss computation
- [ ] Add regularization losses

### Phase 4: Logging & Visualization
- [ ] Implement training metrics logging
- [ ] Implement gradient stats tracking
- [ ] Implement test image rendering
- [ ] Create visualization script for logs

### Phase 5: Testing & Validation
- [ ] End-to-end test on dataset 1
- [ ] Verify output map.json is valid
- [ ] Compare initial vs optimized
- [ ] Performance profiling

---

## Success Criteria

### For Phase 1 (Differentiability):
✅ Gradients flow to material_logits
✅ Gradient health score > 0.5
✅ No NaN/Inf gradients
✅ Reasonable gradient magnitudes (1-1000 range)

### For Complete Pipeline:
✅ SDS loss decreases over training
✅ Rendered images improve toward prompt
✅ Output map.json is valid and renderable
✅ Training completes without errors
✅ Test images show clear progression

---

## Next Steps

1. **Run gradient test:**
   ```bash
   python -m model_stuff.nv_diff_render.test_render \
     --dataset_id 1 \
     --view_index 0 \
     --test_gradients
   ```

2. **Analyze results:**
   - Check `out_local/nv_diff_logs/dataset_1_diff_log.json`
   - Verify `gradient_test` is populated
   - Confirm differentiability

3. **If gradients are healthy:**
   - Proceed to implement DifferentiableVoxelGrid
   - Add occupancy gradient testing

4. **If gradients are broken:**
   - Debug nvdiffrast interpolation
   - Check material weighting computation
   - Verify no detach() calls blocking flow
