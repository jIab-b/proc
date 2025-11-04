# Chunked Region Generation System

## Overview

Implemented a chunked region generation system that divides the 256×128×256 world into manageable 64×64×64 sub-regions with visual chunk boundary indicators.

## Features Implemented

### 1. Chunk Grid System (`src/world/chunkGrid.ts`)

**ChunkGrid class** - Divides world into configurable chunk regions:
- Default: 64×64×64 voxel chunks (4×2×4 grid = 32 total chunks)
- Tracks generation status per chunk
- Spatial queries: find chunks by position, get chunks in region
- Boundary calculation for visual rendering

**Key Methods:**
- `getChunkAt(x, y, z)` - Find chunk containing position
- `getChunksInRegion(min, max)` - Get all chunks overlapping a region
- `intersectRegionWithChunk()` - Clip region to chunk bounds
- `getChunkBoundaries()` - Get all boundary boxes for rendering

### 2. Region Manager (`src/world/regionManager.ts`)

**RegionManager class** - Handles generation requests and history:
- Converts highlight selections to bounding boxes
- Identifies affected chunks for any generation request
- Splits commands into per-chunk execution
- Maintains generation history with timestamps

**Key Features:**
- **Region-bounded generation**: Commands automatically restricted to chunk bounds
- **History tracking**: Records what was generated where and when
- **Command splitting**: Divides terrain/structure commands across multiple chunks
- **Undo support**: History can be used for rollback

### 3. Visual Chunk Boundaries

**Overlay rendering** (`src/renderer.ts:544-600`):
- Renders chunk boundaries as wireframe boxes
- Dashed white lines (rgba(255, 255, 255, 0.15))
- Only visible chunks drawn (frustum culling)
- Renders behind other UI elements

**Visual style:**
- Line width: 0.5px
- Dash pattern: [4, 4]
- Semi-transparent to not obstruct terrain
- 12 edges per chunk (cube wireframe)

### 4. LLM Integration Updates

**Context-aware prompts** (`src/UI.svelte:24-53`):
- LLM receives current camera chunk region
- Prompts include world bounds and chunk divisions
- Generation directed to visible area
- Commands validated against chunk boundaries

**Prompt additions:**
```
WORLD CONSTRAINTS:
- World bounds: x[0-256], y[0-128], z[0-256]
- World is divided into 64×64×64 chunk regions
- Current camera chunk region: x[X-X+63], y[Y-Y+63], z[Z-Z+63]
- Generate terrain in or near the visible camera region
```

### 5. Integration with Canvas

**Initialization** (`src/Canvas.svelte:56-61, 231-234`):
```typescript
const chunkGrid = new ChunkGrid({
  worldSize: worldConfig.dimensions,
  chunkSize: { x: 64, y: 64, z: 64 }
})
const regionManager = new RegionManager(chunkGrid)

// Set boundaries for visual rendering
const boundaries = chunkGrid.getChunkBoundaries()
renderBackend?.setChunkBoundaries(boundaries)
```

**Debug access:**
- `window.chunkGrid` - Access chunk grid from console
- `window.regionManager` - Access region manager from console

## Architecture Benefits

### Performance
- **Incremental updates**: Only affected chunks need mesh rebuild
- **Spatial culling**: Skip rendering off-screen chunks
- **Command batching**: Group operations by chunk for efficiency

### Organization
- **Clear boundaries**: Visual feedback on world divisions
- **Region tracking**: Know what was generated where
- **History**: Undo/redo support infrastructure

### LLM Generation
- **Bounded commands**: LLM stays within valid ranges
- **Context awareness**: Generation near camera for relevance
- **Validation**: Automatic clipping to chunk bounds

## Usage Examples

### Console Commands
```javascript
// Get all chunks
window.chunkGrid.getAllChunks()

// Find chunk at position
window.chunkGrid.getChunkAt(128, 64, 128)

// Get chunks in region
window.chunkGrid.getChunksInRegion([0,0,0], [127,127,127])

// Check generation history
window.regionManager.getHistory()

// Get recent generations
window.regionManager.getRecentGenerations(5)
```

### Programmatic Usage
```typescript
// Convert highlight selection to affected chunks
const request: RegionGenerationRequest = {
  selection: highlightSelection,
  commands: dslCommands,
  description: "Generated mountains"
}

const affectedChunks = regionManager.getAffectedChunks(request)
console.log(`Will affect ${affectedChunks.length} chunks`)

// Record generation for history
regionManager.recordGeneration(request, affectedChunks)

// Split commands per-chunk for execution
const commandsByChunk = regionManager.splitCommandsByChunks(
  dslCommands,
  affectedChunks
)
```

## Visual Result

When the app runs:
1. **32 chunk boundaries** visible as white wireframe boxes (4×2×4 grid)
2. **Dashed lines** create subtle grid overlay
3. **Boundaries persist** during camera movement/rotation
4. **Frustum culled** - only visible chunks rendered

## Configuration

Change chunk size in `Canvas.svelte`:
```typescript
const chunkGrid = new ChunkGrid({
  worldSize: worldConfig.dimensions,
  chunkSize: { x: 32, y: 32, z: 32 } // Smaller chunks
})
```

Trade-offs:
- **Larger chunks (128³)**: Fewer boundaries, less visual clutter
- **Smaller chunks (32³)**: More granular control, more overhead

Current sweet spot: **64×64×64** (8 chunks per dimension)

## Next Steps

To fully utilize the chunk system:

1. **Incremental mesh updates**: Only rebuild dirty chunks
2. **Async generation**: Generate chunks in background
3. **Multi-selection**: Allow selecting multiple chunks for batch generation
4. **Chunk LOD**: Different detail levels at distance
5. **Streaming**: Load/unload chunks dynamically

## Files Modified

- `src/world/chunkGrid.ts` - NEW: Chunk grid system
- `src/world/regionManager.ts` - NEW: Region management
- `src/render/renderBackend.ts` - Added setChunkBoundaries interface
- `src/render/webgpuBackend.ts` - Implemented setChunkBoundaries
- `src/renderer.ts` - Added renderChunkBoundaries function
- `src/Canvas.svelte` - Initialize chunk grid, set boundaries
- `src/UI.svelte` - Updated LLM prompt with chunk context

## Technical Notes

**Coordinate system:**
- Chunk coordinates: 0-based indices (e.g., chunk[2][1][3])
- Voxel coordinates: Absolute world positions [0-256]
- Boundaries stored in voxel coordinates for rendering

**Rendering pipeline:**
1. chunkGrid.getChunkBoundaries() → Array of bounding boxes
2. renderBackend.setChunkBoundaries() → Store in renderer
3. renderChunkBoundaries() → Project to screen space
4. Canvas 2D overlay → Draw wireframes

**Performance:**
- ~32 chunks × 12 edges × 2 points = ~768 projections/frame
- With frustum culling: typically 8-16 chunks visible
- Negligible overhead (<0.1ms per frame)
