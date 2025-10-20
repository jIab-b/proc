# Training Output Format

## Directory Structure

After running training, the output directory will have this structure:

```
out_local/sds_training/
├── final_map.json              # ✨ FINAL trained map (distinct name)
├── map_optimized.json          # Copy of final map (backward compatibility)
├── train.jsonl                 # Training metrics log
├── images/                     # Rendered views (saved every 5 steps)
│   ├── step_0001_cam3.png
│   ├── step_0005_cam7.png
│   ├── step_0010_cam2.png
│   ├── step_0015_cam5.png
│   ├── ...
│   └── step_0500_cam1.png      # Last step
└── maps/                       # Intermediate checkpoints (every 50 steps)
    ├── step_0050.json
    ├── step_0100.json
    ├── step_0150.json
    ├── ...
    └── step_0500.json          # Final checkpoint (same as final_map.json)
```

---

## File Details

### 1. Images (`images/`)

**Frequency:** Every 5 steps (configurable with `--image_every 5`)

**Naming:** `step_{STEP:04d}_cam{CAM_IDX}.png`
- `STEP`: Training step (padded to 4 digits)
- `CAM_IDX`: Random camera index used for this render

**Content:** RGB image rendered from a random camera view in datasets/

**Example:**
```
step_0001_cam3.png  → Step 1, camera 3
step_0005_cam7.png  → Step 5, camera 7
step_0500_cam2.png  → Step 500, camera 2
```

**For 500 steps with default settings:**
- Total images: ~100 files (500 / 5 = 100)
- File size: ~50-500KB each (depending on resolution)

---

### 2. Intermediate Maps (`maps/`)

**Frequency:** Every 50 steps (configurable with `--save_map_every 50`)

**Naming:** `step_{STEP:04d}.json`

**Content:** Full map.json snapshot at that training step

**Metadata includes:**
```json
{
  "sequence": 1,
  "blocks": [...],
  "worldConfig": {...},
  "metadata": {
    "prompt": "a medieval stone castle",
    "step": 50,
    "total_steps": 500,
    "cfg_scale": 7.5
  }
}
```

**For 500 steps with default settings:**
- Total checkpoints: 10 files (500 / 50 = 10)
- File size: ~500KB - 5MB each (depending on block count)

**Use cases:**
- Resume training from checkpoint
- Compare intermediate results
- Debug training progression
- Recover from failed training

---

### 3. Final Map (`final_map.json`)

**When saved:** After training completes

**Naming:** Always `final_map.json` (distinct, easy to find)

**Content:** Final optimized voxel grid

**Same as:** `maps/step_0500.json` but with distinct name

**This is the map that gets auto-exported to `maps/trained_map_1_X/`**

---

### 4. Training Log (`train.jsonl`)

**Format:** JSON Lines (one JSON object per line)

**Frequency:** Every 10 steps (configurable with `--log_every 10`)

**Content:**
```json
{
  "step": 100,
  "temperature": 1.5,
  "camera_index": 7,
  "losses": {
    "total": 0.345,
    "sds": 0.320,
    "sparsity": 0.015,
    "entropy": 0.010,
    "smooth": 0.0
  },
  "grid_stats": {
    "num_active_voxels": 45623,
    "total_voxels": 196608,
    "density": 0.232,
    "occupancy_mean": 0.15,
    "occupancy_std": 0.28,
    "material_distribution": {
      "Stone": 65.2,
      "Grass": 20.1,
      "Dirt": 8.3,
      "Air": 6.4
    }
  }
}
```

**Use cases:**
- Plot loss curves
- Track material distribution over time
- Monitor grid density changes
- Debug training dynamics

---

## Auto-Export to maps/

After training completes, `final_map.json` is automatically exported to:

```
maps/trained_map_{SOURCE_ID}_{X}/
├── map.json                    # Copy of final_map.json
├── training_metadata.json      # Training info
├── train.jsonl                 # Training log (copy)
└── images/                     # Sample images (first, middle, last)
    ├── step_0001_cam3.png
    ├── step_0250_cam1.png
    └── step_0500_cam2.png
```

**Numbering:**
- `SOURCE_ID`: Original map ID (e.g., 1 from maps/1/map.json)
- `X`: Auto-incremented (1, 2, 3, ...) based on existing trained maps

**Examples:**
- First training from maps/1/ → `maps/trained_map_1_1/`
- Second training from maps/1/ → `maps/trained_map_1_2/`
- First training from maps/2/ → `maps/trained_map_2_1/`

---

## Customizing Output Frequency

### Save images more frequently (every step):
```bash
python -m model_stuff.train_sds_final \
  --prompt "a castle" \
  --steps 100 \
  --image_every 1
```

**Result:** 100 images (one per step)

### Save images less frequently (every 20 steps):
```bash
python -m model_stuff.train_sds_final \
  --prompt "a castle" \
  --steps 500 \
  --image_every 20
```

**Result:** 25 images (500 / 20)

### Save more checkpoints (every 10 steps):
```bash
python -m model_stuff.train_sds_final \
  --prompt "a castle" \
  --steps 500 \
  --save_map_every 10
```

**Result:** 50 checkpoint maps (500 / 10)

### Save fewer checkpoints (every 100 steps):
```bash
python -m model_stuff.train_sds_final \
  --prompt "a castle" \
  --steps 500 \
  --save_map_every 100
```

**Result:** 5 checkpoint maps (500 / 100)

---

## Disk Space Estimates

For a typical 500-step training run:

| Item | Count | Size Each | Total |
|------|-------|-----------|-------|
| Images (every 5 steps) | 100 | ~200KB | ~20MB |
| Checkpoints (every 50 steps) | 10 | ~2MB | ~20MB |
| Final maps | 2 | ~2MB | ~4MB |
| Training log | 1 | ~100KB | ~100KB |
| **TOTAL** | - | - | **~45MB** |

For aggressive settings (images every step, maps every 10 steps):

| Item | Count | Size Each | Total |
|------|-------|-----------|-------|
| Images (every step) | 500 | ~200KB | ~100MB |
| Checkpoints (every 10 steps) | 50 | ~2MB | ~100MB |
| Final maps | 2 | ~2MB | ~4MB |
| Training log | 1 | ~200KB | ~200KB |
| **TOTAL** | - | - | **~205MB** |

---

## Example Training Session

### Command:
```bash
python -m model_stuff.train_sds_final \
  --prompt "a medieval stone castle" \
  --steps 100 \
  --dataset_id 1 \
  --image_every 5 \
  --save_map_every 25
```

### Expected Output:
```
out_local/sds_training/
├── final_map.json
├── map_optimized.json
├── train.jsonl
├── images/
│   ├── step_0001_cam2.png
│   ├── step_0005_cam5.png
│   ├── step_0010_cam1.png
│   ├── step_0015_cam8.png
│   ├── step_0020_cam3.png
│   ├── ... (15 more images)
│   └── step_0100_cam6.png
└── maps/
    ├── step_0025.json
    ├── step_0050.json
    ├── step_0075.json
    └── step_0100.json
```

**Counts:**
- Images: 20 files (100 / 5)
- Checkpoints: 4 files (100 / 25)

### Console Output:
```
Training complete!

Output:
  Final map: out_local/sds_training/final_map.json (85234 blocks)
  Logs: out_local/sds_training/train.jsonl
  Images: out_local/sds_training/images (20 images)
  Intermediate maps: out_local/sds_training/maps (4 checkpoints)

Final grid stats:
  Active voxels: 85234/196608
  Density: 43.35%
  Material distribution:
    Stone: 68.5%
    Grass: 18.2%
    Dirt: 9.1%
    Water: 2.3%
    Plank: 1.9%

Exporting to maps/ directory...
✅ Exported to: maps/trained_map_1_1
```

---

## Key Takeaways

✅ **Images saved every 5 steps** (not 50) - default changed
✅ **Intermediate maps in `sds_training/maps/`** - saved every 50 steps
✅ **Final map has distinct name: `final_map.json`** - easy to identify
✅ **All outputs organized in subdirectories** - clean structure
✅ **Auto-export to `maps/`** - ready for WebGPU editor

**The output format is now exactly as requested!**
