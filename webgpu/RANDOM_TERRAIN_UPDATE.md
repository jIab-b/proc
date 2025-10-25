# Random Terrain Generation - UI Update

## Summary

Updated the terrain generation UI to make clicking the "Generate Terrain" button create random terrain with randomized parameters.

## Changes Made

### File: `src/lib/ProceduralTerrainPanel.svelte`

1. **Added `randomizeAll()` function**
   - Randomizes terrain profile (rolling_hills, mountain, or hybrid)
   - Randomizes seed (0 to 1,000,000)
   - Randomizes amplitude (8-20, favoring middle range)
   - Randomizes roughness (2.0-2.8, favoring middle range)
   - Randomizes elevation (0.3-0.5, favoring middle range)
   - Logs the randomized parameters to console

2. **Added "🎲 Generate Random Terrain" button**
   - Primary button (highlighted in blue)
   - Calls `randomizeAll()` then `runGeneration('generate')`
   - Completely randomizes the terrain with each click

3. **Increased generation region size**
   - Changed from ±16 blocks to ±24 blocks
   - Now generates a 48×48×32 block region (instead of 32×32×32)
   - More impressive terrain generation

## How to Use

### Start the Development Server

```bash
cd /home/beed/splats/proc/webgpu
npm run dev
```

### Generate Random Terrain

1. Open the app in your browser
2. Look for the "Procedural Terrain" panel in the UI
3. Click the **"🎲 Generate Random Terrain"** button (blue button at top)
4. Watch as completely random terrain is generated around the camera!
5. Click it again to get different random terrain

### Manual Control (Optional)

You can still manually control terrain parameters:
- **Profile dropdown**: Choose rolling_hills, mountain, or hybrid
- **Seed**: Click 🎲 to randomize just the seed
- **Amplitude slider**: Control height variation
- **Roughness slider**: Control terrain detail/noise
- **Elevation slider**: Control base height
- **Generate Around Player**: Generate with current manual parameters

## What Happens When You Click

1. **Randomization**: All terrain parameters are randomized
   - Profile: Randomly picks one of 3 terrain types
   - Seed: Random number for unique terrain
   - Amplitude: 8-20 (height range)
   - Roughness: 2.0-2.8 (detail level)
   - Elevation: 0.3-0.5 (base height)

2. **Generation**: Creates terrain in a 48×48 block region around camera
   - Uses procedural noise (Fractional Brownian Motion)
   - Creates natural-looking hills, valleys, or mountains
   - Seamless terrain with proper block types

3. **Rendering**: The terrain mesh is rebuilt and displayed

## Parameter Ranges

| Parameter | Range | Randomized Range | Effect |
|-----------|-------|------------------|--------|
| Profile | 3 types | All 3 equally likely | rolling_hills = gentle, mountain = steep, hybrid = mix |
| Seed | 0 - 1M | 0 - 1M | Determines unique terrain shape |
| Amplitude | 4 - 32 | 8 - 20 | Height variation (higher = taller features) |
| Roughness | 1.2 - 3.4 | 2.0 - 2.8 | Noise octaves (higher = more detail) |
| Elevation | 0.2 - 0.7 | 0.3 - 0.5 | Base height offset |

## Examples

### Example 1: Rolling Hills
```
Profile: rolling_hills
Seed: 234567
Amplitude: 12.3
Roughness: 2.24
Elevation: 0.38
Result: Gentle, grassy hills with valleys
```

### Example 2: Mountains
```
Profile: mountain
Seed: 789012
Amplitude: 18.7
Roughness: 2.65
Elevation: 0.45
Result: Steep mountain peaks with alpine blocks
```

### Example 3: Hybrid
```
Profile: hybrid
Seed: 456789
Amplitude: 15.2
Roughness: 2.41
Elevation: 0.42
Result: Mix of gentle slopes and steep peaks
```

## Console Output

When you click "🎲 Generate Random Terrain", check the browser console for:

```javascript
Randomized terrain params: {
  profile: 'mountain',
  seed: 734821,
  amplitude: '15.4',
  roughness: '2.37',
  elevation: '0.43'
}
```

## Block Types Generated

Depending on the terrain profile and height:

**Rolling Hills:**
- Grass (top surface, height < 18)
- Dirt (subsurface, 3 blocks deep)
- Stone (deep underground)

**Mountain:**
- Alpine Rock (peaks > 28)
- Glacier Ice (very high peaks > 34)
- Gravel (subsurface)
- Stone (deep)

**Hybrid:**
- Alpine Grass (high areas > 30)
- Grass (lower areas)
- Dirt (subsurface)
- Stone (deep)

## Technical Details

### Randomization Algorithm

```typescript
// Profile: Equal probability for each type
const profiles = ['rolling_hills', 'mountain', 'hybrid']
profile = profiles[random(0, 2)]

// Seed: Uniform distribution
seed = random(0, 1_000_000)

// Amplitude: Normal distribution favoring middle
amplitude = 8 + random(0, 12)

// Roughness: Narrow range for quality
roughness = 2.0 + random(0, 0.8)

// Elevation: Narrow range for playability
elevation = 0.3 + random(0, 0.2)
```

### Region Calculation

```typescript
// Camera position (world coordinates)
const cameraPos = getCameraPosition() // e.g., [0, 40, 0]

// Define region (±24 blocks = 48×48 area)
const regionSize = 24
const region = {
  min: cameraPos.map(pos => Math.floor(pos - regionSize)),
  max: cameraPos.map(pos => Math.floor(pos + regionSize))
}

// Example:
// If camera at [0, 40, 0]:
//   min = [-24, 16, -24]
//   max = [24, 64, 24]
//   Size = 48×48×48 blocks
```

## Troubleshooting

### Button Not Working

1. Check browser console for errors
2. Ensure camera position is available (not before renderer initializes)
3. Check that backend API is running (localhost:8000)

### No Terrain Generated

1. Check console for "Terrain generation complete" message
2. Verify chunk bounds (may be clamped if outside 32×32×32 chunk)
3. Try moving camera to center (0, 40, 0)

### Terrain Looks Wrong

1. Check randomized parameters in console
2. Try clicking button multiple times for different results
3. Manually adjust sliders to fine-tune

## Integration with Grid System

**Note:** This update still uses the single-chunk system (32×32×32 blocks). For truly massive terrain (thousands of blocks), integrate the grid system:

1. Replace `ChunkManager` with `ChunkGrid` in `Canvas.svelte`
2. Update `generateTerrain` hook to use `generateGridRegion`
3. Use `buildGridMesh` for rendering

See `TERRAIN_GRID.md` for full grid system documentation.

## Future Enhancements

Potential improvements:
- [ ] Add "region size" slider (let user choose generation area)
- [ ] Add "Randomize" buttons for individual parameters
- [ ] Save/load favorite parameter presets
- [ ] Animate parameter changes
- [ ] Show preview thumbnail before generating
- [ ] Add more terrain profiles (desert, tundra, volcanic, etc.)
- [ ] Integrate with grid system for infinite terrain
