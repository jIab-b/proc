# SDS Training for Voxel Worlds

Score Distillation Sampling (SDS) optimization for voxel-based 3D worlds using SDXL diffusion models.

## Quick Start

### 1. Train from Existing Map

```bash
# Train on maps/1/map.json with datasets/1/ cameras
source venv/bin/activate
python -m model_stuff.train_sds_final \
  --prompt "a medieval stone castle" \
  --steps 500 \
  --dataset_id 1 \
  --init_mode from_map
```

**Output:** Automatically exported to `maps/trained_map_1_1/map.json`

### 2. Train from Primitive

```bash
# Start with ground plane + noise
python -m model_stuff.train_sds_final \
  --prompt "a stone castle" \
  --steps 500 \
  --init_mode ground_plane
```

### 3. Export Trained Map

If auto-export fails, manually export:

```bash
python -m model_stuff.export_trained_map \
  --training_dir out_local/sds_training \
  --source_map_id 1
```

## File Structure

### Core Files
```
model_stuff/
├── train_sds_final.py       # Main training script
├── voxel_grid.py            # DifferentiableVoxelGrid class
├── map_io.py                # map.json ↔ grid conversion
├── export_trained_map.py    # Export to maps/ directory
├── sdxl_lightning.py        # SDXL wrapper
└── nv_diff_render/          # Differentiable renderer (5 files)
    ├── renderer.py
    ├── mesh_builder.py
    ├── materials.py
    ├── shading.py
    └── utils.py
```

### Test Files
```
model_stuff/
├── test_pipeline.py         # Integration tests
└── test_gradient_flow.py    # Gradient verification
```

## Workflow

### Data Flow
```
maps/1/map.json                    # Source map (88k blocks)
    ↓ load_map_to_grid()
occupancy_logits (64×48×64)        # Learnable parameters
material_logits (64×48×64×8)
    ↓ DifferentiableVoxelGrid.forward()
RGBA image (1094×881)              # Rendered view
    ↓ compute_sds_loss()
SDS loss + regularization
    ↓ loss.backward()
Gradients → occupancy/materials    # Update parameters
    ↓ (repeat 500 steps)
optimized grid
    ↓ save_grid_to_map()
maps/trained_map_1_1/map.json     # Exported result
```

## Training Parameters

### Basic
```bash
--prompt "text description"        # SDXL text prompt
--steps 500                        # Training iterations
--dataset_id 1                     # Which dataset/map to use
```

### Initialization
```bash
--init_mode from_map               # Load from maps/{id}/map.json
--init_mode ground_plane           # Ground layer + noise
--init_mode cube                   # Central cube structure
--init_mode empty                  # Empty grid
```

### Optimization
```bash
--lr 0.01                          # Learning rate
--cfg_scale 7.5                    # CFG scale (7.5 = strong guidance)
--temp_start 2.0                   # Initial temperature (soft materials)
--temp_end 0.5                     # Final temperature (sharp materials)
```

### Regularization
```bash
--lambda_sparsity 0.001            # Penalize too many blocks
--lambda_entropy 0.0001            # Penalize uncertain materials
--lambda_smooth 0.0                # Penalize disconnected structures
```

### Logging
```bash
--log_every 10                     # Log frequency
--image_every 50                   # Image save frequency
--output_dir out_local/my_training # Output directory
```

## Output Structure

After training completes:

```
out_local/sds_training/
├── map_optimized.json             # Optimized voxel grid
├── train.jsonl                    # Training metrics
└── images/                        # Progress images
    ├── step_0001_cam3.png
    ├── step_0050_cam7.png
    └── step_0500_cam2.png

maps/trained_map_1_1/              # Auto-exported
├── map.json                       # Ready for WebGPU editor
├── training_metadata.json         # Training info
├── train.jsonl                    # Training logs (copy)
└── images/                        # Sample images
    ├── step_0001_cam3.png         # First
    ├── step_0250_cam1.png         # Middle
    └── step_0500_cam2.png         # Last
```

## Expected Results

### For Geometry/Structure Optimization

**Goal:** Use SDS to discover 3D structure matching text prompts

**Strengths:**
- ✅ Coarse geometry discovery (towers, walls, volumes)
- ✅ Material semantic consistency (stone=gray, grass=green)
- ✅ Multi-view 3D consistency
- ✅ Blocky aesthetic acceptable

**Limitations:**
- ❌ No fine details (windows, doors, textures)
- ❌ 32³ resolution = architectural scale (~2m blocks)
- ❌ 8 materials = limited color palette

### Example: "a medieval stone castle"

**Input:** maps/1/map.json (88k blocks, terrain)

**After 500 steps:**
- Recognizable castle structure (4 corner towers, connecting walls)
- Stone material for structures (~70%)
- Grass for ground (~20%)
- Spatially coherent (connected geometry)

## Testing

### Pipeline Tests
```bash
# Test map I/O, rendering, gradients
python -m model_stuff.test_pipeline
```

**Expected output:**
```
✅ Map I/O: 88403 blocks loaded/saved
✅ Voxel Grid: 64×48×64 initialized
✅ Rendering: 65% geometry coverage
✅ Gradients: Flow correctly
```

### Gradient Verification
```bash
# Verify gradient flow through renderer
python -m model_stuff.nv_diff_render.test_render \
  --dataset_id 1 \
  --view_index 0
```

## Troubleshooting

### No gradients flowing
- Check that `requires_grad=True` on parameters
- Verify occupancy modulation: `material_logits * occupancy.unsqueeze(-1)`
- Test with `test_pipeline.py`

### Training diverges
- Reduce learning rate: `--lr 0.001`
- Increase sparsity penalty: `--lambda_sparsity 0.01`
- Use lower CFG scale: `--cfg_scale 5.0`

### Out of memory
- Reduce image size (edit dataset metadata)
- Use fewer training steps
- Reduce grid size (requires recreating map)

### Map export fails
- Manually export: `python -m model_stuff.export_trained_map --training_dir ... --source_map_id 1`
- Check `out_local/sds_training/map_optimized.json` exists

## Implementation Notes

### Gradient Flow Path
```
SDS Loss
  ↓ ∂loss/∂pixels
Rendered RGB
  ↓ ∂pixels/∂lit_colors
Shader output
  ↓ ∂lit_colors/∂vertex_colors
Mesh colors
  ↓ ∂vertex_colors/∂material_probs
Material probabilities
  ↓ ∂material_probs/∂(material_logits * occupancy)
Material logits ← LEARNED
Occupancy logits ← LEARNED
```

### Key Design Decisions

**Two-parameter system:**
- `occupancy_logits`: Controls which voxels exist
- `material_logits`: Controls what material they are
- Modulation: `effective_logits = material_logits * occupancy`

**Why logits, not probabilities:**
- Unbounded optimization space
- Better gradient flow
- Standard practice in differentiable rendering

**Temperature annealing:**
- Start high (2.0): Soft, exploratory
- End low (0.5): Sharp, decisive
- Like simulated annealing for discrete optimization

## References

- **DreamFusion**: "Text-to-3D using 2D Diffusion" (2022)
- **SDS Loss**: Score Distillation Sampling (Poole et al.)
- **nvdiffrast**: NVIDIA Differentiable Rasterizer
- **SDXL**: Stable Diffusion XL (Stability AI)

## License

Same as parent project.
