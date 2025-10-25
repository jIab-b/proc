# Large-Scale Grid-Based Terrain Generation

This document describes the new grid-based terrain generation system that supports creating terrain regions thousands of blocks wide.

## Overview

The system is composed of several key components:

1. **ChunkGrid** - Multi-chunk spatial grid manager
2. **Grid Terrain Generator** - Continuous terrain generation across chunks
3. **Grid Mesh Builder** - Efficient rendering of multiple chunks
4. **Terrain DSL** - LLM-accessible CLI for automated terrain generation

## Architecture

### ChunkGrid (`src/chunkGrid.ts`)

The ChunkGrid manages a large spatial grid of 32×32×32 chunks.

**Key Features:**
- Dynamic chunk loading/unloading based on view distance
- Memory-efficient (only loads chunks near the camera)
- Seamless coordinate conversion between world and chunk space
- Support for thousands of blocks wide terrain

**Example Usage:**

```typescript
import { ChunkGrid } from './src/chunkGrid'
import { BlockType } from './src/core'

// Create grid
const grid = new ChunkGrid({
  chunkSize: { x: 32, y: 32, z: 32 },
  maxLoadedChunks: 256,
  viewDistance: 16
})

// Set/get blocks at world coordinates
grid.setBlock(100, 10, 200, BlockType.Stone)
const block = grid.getBlock(100, 10, 200)

// Update view center for automatic chunk loading/unloading
grid.setCenterFromWorld(playerX, playerZ)

// Get statistics
console.log(grid.getStats())
```

### Grid Terrain Generator (`src/procedural/gridTerrainGenerator.ts`)

Generates continuous terrain across multiple chunks using the same noise functions as the single-chunk system.

**Example Usage:**

```typescript
import { ChunkGrid } from './src/chunkGrid'
import { generateGridRegion } from './src/procedural/gridTerrainGenerator'

const grid = new ChunkGrid()

// Generate a 10×10 chunk region (320×320 blocks)
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

// Generate a massive 100×100 chunk region (3200×3200 blocks)
generateGridRegion(grid, {
  minChunk: { cx: 0, cz: 0 },
  maxChunk: { cx: 99, cz: 99 }
}, {
  profile: 'mountain',
  seed: 7331
})
```

### Grid Mesh Builder (`src/gridMeshBuilder.ts`)

Builds combined meshes for efficient rendering of multiple chunks.

**Example Usage:**

```typescript
import { buildGridMesh } from './src/gridMeshBuilder'

// Build mesh for all visible chunks
const mesh = buildGridMesh(grid, worldScale)
console.log(`Rendering ${mesh.chunkCount} chunks with ${mesh.vertexCount} vertices`)

// Upload to GPU
device.queue.writeBuffer(vertexBuffer, 0, mesh.vertexData)
```

## Terrain DSL - LLM-Accessible CLI

The Terrain DSL provides a simple command-line interface designed for LLM automation.

### Installation

```bash
cd webgpu
npm install
```

### Configuration

Copy the example configuration:

```bash
cp .env.example .env
```

Edit `.env` to customize:

```env
MAX_LOADED_CHUNKS=256
VIEW_DISTANCE=16
```

### Commands

#### Generate Terrain

Generate terrain in a grid region from chunk (cx1, cz1) to (cx2, cz2):

```bash
npm run terrain generate <cx1> <cz1> <cx2> <cz2> <profile> [seed] [amplitude] [roughness] [elevation]
```

**Arguments:**
- `cx1, cz1` - Starting chunk coordinates
- `cx2, cz2` - Ending chunk coordinates (inclusive)
- `profile` - Terrain profile: `rolling_hills`, `mountain`, or `hybrid`
- `seed` - Optional: Random seed (integer)
- `amplitude` - Optional: Height amplitude (float, 8-18)
- `roughness` - Optional: Terrain roughness (float, 2.2-2.8)
- `elevation` - Optional: Base elevation (float, 0.35-0.5)

**Examples:**

```bash
# Small region (5×5 chunks = 160×160 blocks)
npm run terrain generate 0 0 4 4 rolling_hills

# Medium region (20×20 chunks = 640×640 blocks)
npm run terrain generate 0 0 19 19 mountain 7331 18 2.8 0.5

# Large region (50×50 chunks = 1600×1600 blocks)
npm run terrain generate 0 0 49 49 hybrid 4242

# Huge region (100×100 chunks = 3200×3200 blocks)
npm run terrain generate 0 0 99 99 mountain 1337 15 2.6 0.45

# Custom seed
npm run terrain generate -10 -10 10 10 rolling_hills 42

# Negative coordinates work too
npm run terrain generate -25 -25 25 25 hybrid 9999
```

#### List Chunks

List all currently loaded chunks:

```bash
npm run terrain list
```

#### Show Statistics

Show grid statistics (loaded chunks, view distance, etc.):

```bash
npm run terrain stats
```

#### Clear Chunks

Clear all loaded chunks from memory:

```bash
npm run terrain clear
```

#### Help

Show help message:

```bash
npm run terrain help
```

### Output Format

All commands output JSON to stdout for easy parsing by LLMs:

```json
{
  "success": true,
  "command": "generate",
  "data": {
    "region": {
      "chunks": {
        "min": { "cx": 0, "cz": 0 },
        "max": { "cx": 9, "cz": 9 }
      },
      "dimensions": {
        "chunksX": 10,
        "chunksZ": 10,
        "blocksX": 320,
        "blocksZ": 320,
        "blocksY": 32
      }
    },
    "params": {
      "profile": "rolling_hills",
      "seed": 1337,
      "amplitude": 10,
      "roughness": 2.4,
      "elevation": 0.35
    },
    "elapsed_ms": 1234
  },
  "stats": {
    "loadedChunks": 100,
    "centerChunk": { "cx": 0, "cz": 0 },
    "viewDistance": 16,
    "maxLoadedChunks": 256,
    "dirtyChunks": 100
  }
}
```

## LLM Automation Examples

### Example 1: Generate Multiple Regions

An LLM can generate terrain in multiple regions by calling the CLI multiple times:

```bash
# Generate a patchwork of different terrain types
npm run terrain generate 0 0 9 9 rolling_hills 1337
npm run terrain generate 10 0 19 9 mountain 7331
npm run terrain generate 0 10 9 19 hybrid 4242
```

### Example 2: Progressive Generation

Generate terrain progressively outward from a center point:

```bash
# Center region
npm run terrain generate -5 -5 5 5 rolling_hills 1337

# Expand outward
npm run terrain generate -10 -10 10 10 rolling_hills 1337

# Expand further
npm run terrain generate -20 -20 20 20 rolling_hills 1337
```

### Example 3: Scripted Generation

LLMs can read the .env file for configuration and generate terrain accordingly:

```bash
# Read configuration from .env
export $(cat .env | xargs)

# Generate terrain using env variables
npm run terrain generate 0 0 99 99 mountain $DEFAULT_SEED $DEFAULT_AMPLITUDE $DEFAULT_ROUGHNESS $DEFAULT_ELEVATION
```

## Performance Considerations

### Memory Usage

Each chunk stores 32×32×32 blocks = 32,768 bytes (~32 KB).

**Examples:**
- 100 chunks = ~3.2 MB
- 1,000 chunks = ~32 MB
- 10,000 chunks = ~320 MB

Configure `MAX_LOADED_CHUNKS` based on available memory.

### Generation Performance

Terrain generation is CPU-bound:

**Approximate times (on modern CPU):**
- 10×10 chunks (100 chunks) = ~0.5 seconds
- 50×50 chunks (2,500 chunks) = ~10 seconds
- 100×100 chunks (10,000 chunks) = ~40 seconds

For very large regions (>100×100), consider:
1. Generating in batches
2. Running generation in parallel
3. Using worker threads

### Rendering Performance

Rendering is GPU-bound:

**Recommendations:**
- Use `VIEW_DISTANCE` to limit rendered chunks
- Typical value: 8-16 chunks
- GPU can easily handle 256+ chunks at 60 FPS

## Scale Examples

### Small Scale
- **Region:** 10×10 chunks
- **Blocks:** 320×320 (102,400 blocks)
- **Use case:** Testing, small maps

### Medium Scale
- **Region:** 50×50 chunks
- **Blocks:** 1,600×1,600 (2.56M blocks)
- **Use case:** Small game levels

### Large Scale
- **Region:** 100×100 chunks
- **Blocks:** 3,200×3,200 (10.24M blocks)
- **Use case:** Large open worlds

### Massive Scale
- **Region:** 500×500 chunks
- **Blocks:** 16,000×16,000 (256M blocks)
- **Use case:** Procedural infinite worlds

## Integration with Existing Code

To integrate with the existing renderer:

```typescript
import { ChunkGrid } from './src/chunkGrid'
import { generateGridRegion } from './src/procedural/gridTerrainGenerator'
import { buildGridMesh } from './src/gridMeshBuilder'

// Replace ChunkManager with ChunkGrid
const grid = new ChunkGrid()

// Generate terrain
generateGridRegion(grid, {
  minChunk: { cx: -10, cz: -10 },
  maxChunk: { cx: 10, cz: 10 }
}, {
  profile: 'rolling_hills',
  seed: 1337
})

// Update view center from player position
grid.setCenterFromWorld(playerX, playerZ)

// Build mesh for visible chunks
const mesh = buildGridMesh(grid, worldScale)

// Render (same as before)
device.queue.writeBuffer(vertexBuffer, 0, mesh.vertexData)
// ... render with vertexCount = mesh.vertexCount
```

## Future Enhancements

Potential improvements:

1. **Chunk Persistence** - Save/load chunks to disk
2. **Async Generation** - Generate chunks in worker threads
3. **Level of Detail (LOD)** - Lower detail for distant chunks
4. **Infinite Terrain** - Generate chunks on-demand as player explores
5. **Biome System** - Different terrain types based on world position
6. **Structure Generation** - Trees, buildings, caves, etc.

## Troubleshooting

### Out of Memory

If you run out of memory:
1. Reduce `MAX_LOADED_CHUNKS` in `.env`
2. Reduce `VIEW_DISTANCE` in `.env`
3. Generate smaller regions

### Slow Generation

If generation is too slow:
1. Generate smaller regions at a time
2. Use lower `roughness` values (fewer octaves)
3. Consider running in worker threads

### Visual Seams Between Chunks

The grid terrain generator uses continuous noise functions, so there should be no visual seams. If you see seams:
1. Check that you're using `generateGridRegion` (not the old single-chunk generator)
2. Verify that chunks are being meshed with correct offsets
3. Check for floating-point precision issues at very large coordinates

## API Reference

See individual source files for detailed API documentation:

- `src/chunkGrid.ts` - ChunkGrid class
- `src/procedural/gridTerrainGenerator.ts` - Grid terrain generation
- `src/gridMeshBuilder.ts` - Multi-chunk mesh building
- `terrain-dsl.ts` - CLI commands
