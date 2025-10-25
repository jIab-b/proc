# Quick Start - Grid-Based Terrain Generation

## Installation

```bash
cd /home/beed/splats/proc/webgpu
npm install tsx @types/node  # Install required dependencies
```

## Basic Usage

### 1. CLI Commands (LLM-Accessible)

```bash
# Generate a small region (10×10 chunks = 320×320 blocks)
npm run terrain generate 0 0 9 9 rolling_hills

# Generate a large region (50×50 chunks = 1600×1600 blocks)
npm run terrain generate 0 0 49 49 mountain 7331 18 2.8 0.5

# Generate with custom parameters
npm run terrain generate -10 -10 10 10 hybrid 4242 14 2.4 0.42

# List loaded chunks
npm run terrain list

# Show statistics
npm run terrain stats

# Clear all chunks
npm run terrain clear

# Show help
npm run terrain help
```

### 2. Programmatic Usage

```typescript
import { ChunkGrid } from './src/chunkGrid'
import { generateGridRegion } from './src/procedural/gridTerrainGenerator'
import { buildGridMesh } from './src/gridMeshBuilder'

// Create grid
const grid = new ChunkGrid({
  chunkSize: { x: 32, y: 32, z: 32 },
  maxLoadedChunks: 256,
  viewDistance: 16
})

// Generate terrain (10×10 chunks = 320×320 blocks)
generateGridRegion(grid, {
  minChunk: { cx: 0, cz: 0 },
  maxChunk: { cx: 9, cz: 9 }
}, {
  profile: 'rolling_hills',
  seed: 1337,
  amplitude: 10,
  roughness: 2.4,
  elevation: 0.35
})

// Update view center
grid.setCenterFromWorld(playerX, playerZ)

// Build mesh for rendering
const mesh = buildGridMesh(grid, worldScale)

// Upload to GPU
device.queue.writeBuffer(vertexBuffer, 0, mesh.vertexData)
// Draw with vertexCount = mesh.vertexCount
```

### 3. Run Example

```bash
npm run example:grid
```

## Configuration

Copy `.env.example` to `.env` and customize:

```env
MAX_LOADED_CHUNKS=256  # Maximum chunks in memory
VIEW_DISTANCE=16       # View distance in chunks
```

## LLM Automation Example

```bash
#!/bin/bash
# Generate multiple terrain regions

# Small rolling hills
npm run terrain generate 0 0 9 9 rolling_hills 1337 | jq '.success'

# Medium mountains
npm run terrain generate 10 0 29 19 mountain 7331 | jq '.success'

# Large hybrid region
npm run terrain generate 0 20 49 69 hybrid 4242 | jq '.success'

# Get final statistics
npm run terrain stats | jq '.data'
```

## Scale Examples

- **Small (10×10 chunks)**: 320×320 blocks = 102K blocks
- **Medium (50×50 chunks)**: 1,600×1,600 blocks = 2.5M blocks
- **Large (100×100 chunks)**: 3,200×3,200 blocks = 10M blocks
- **Huge (500×500 chunks)**: 16,000×16,000 blocks = 256M blocks

## Documentation

- `TERRAIN_GRID.md` - Full documentation with API reference
- `GRID_IMPLEMENTATION_SUMMARY.md` - Implementation details
- `examples/grid-terrain-example.ts` - Working code examples

## Files Created

- `src/chunkGrid.ts` - ChunkGrid manager
- `src/procedural/gridTerrainGenerator.ts` - Grid terrain generator
- `src/gridMeshBuilder.ts` - Multi-chunk mesh builder
- `terrain-dsl.ts` - LLM-accessible CLI
- `.env.example` - Configuration template
- `examples/grid-terrain-example.ts` - Example code

## Next Steps

1. Install dependencies: `npm install tsx @types/node`
2. Run example: `npm run example:grid`
3. Test CLI: `npm run terrain generate 0 0 9 9 rolling_hills`
4. Read full docs: `TERRAIN_GRID.md`
5. Integrate with your renderer using `buildGridMesh()`
