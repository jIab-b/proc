/**
 * Example: Large-scale grid-based terrain generation
 *
 * This example demonstrates how to use the ChunkGrid and grid terrain generator
 * to create massive terrain regions (thousands of blocks wide).
 *
 * Run with: npm run example:grid
 */

import { ChunkGrid } from '../src/chunkGrid'
import {
  generateGridRegion,
  getRegionDimensions,
  type GridRegion
} from '../src/procedural/gridTerrainGenerator'
import { buildGridMesh } from '../src/gridMeshBuilder'
import { BlockType } from '../src/core'

console.log('=== Large-Scale Grid Terrain Generation Example ===\n')

// Create a chunk grid
const grid = new ChunkGrid({
  chunkSize: { x: 32, y: 32, z: 32 },
  maxLoadedChunks: 256,
  viewDistance: 16
})

console.log('1. Creating ChunkGrid...')
console.log(`   Max loaded chunks: 256`)
console.log(`   View distance: 16 chunks\n`)

// Example 1: Generate a small region (10x10 chunks = 320x320 blocks)
console.log('2. Generating small region (10x10 chunks)...')
const smallRegion: GridRegion = {
  minChunk: { cx: 0, cz: 0 },
  maxChunk: { cx: 9, cz: 9 }
}

const smallDims = getRegionDimensions(smallRegion)
console.log(`   Region size: ${smallDims.blocksX}×${smallDims.blocksZ} blocks`)
console.log(`   Chunks: ${smallDims.chunksX}×${smallDims.chunksZ}`)

const startTime1 = Date.now()
generateGridRegion(grid, smallRegion, {
  profile: 'rolling_hills',
  seed: 1337,
  amplitude: 10,
  roughness: 2.4,
  elevation: 0.35
})
const elapsed1 = Date.now() - startTime1
console.log(`   Generated in ${elapsed1}ms\n`)

// Check some blocks
console.log('3. Checking generated blocks...')
const block1 = grid.getBlock(100, 10, 100)
const block2 = grid.getBlock(200, 15, 200)
console.log(`   Block at (100, 10, 100): ${BlockType[block1]}`)
console.log(`   Block at (200, 15, 200): ${BlockType[block2]}\n`)

// Get statistics
console.log('4. Grid statistics:')
const stats1 = grid.getStats()
console.log(`   Loaded chunks: ${stats1.loadedChunks}`)
console.log(`   Dirty chunks: ${stats1.dirtyChunks}`)
console.log(`   Center chunk: (${stats1.centerChunk.cx}, ${stats1.centerChunk.cz})\n`)

// Example 2: Expand to a larger region
console.log('5. Expanding to larger region (30x30 chunks)...')
const largeRegion: GridRegion = {
  minChunk: { cx: 0, cz: 0 },
  maxChunk: { cx: 29, cz: 29 }
}

const largeDims = getRegionDimensions(largeRegion)
console.log(`   Region size: ${largeDims.blocksX}×${largeDims.blocksZ} blocks`)
console.log(`   Chunks: ${largeDims.chunksX}×${largeDims.chunksZ}`)

const startTime2 = Date.now()
generateGridRegion(grid, largeRegion, {
  profile: 'mountain',
  seed: 7331,
  amplitude: 18,
  roughness: 2.8,
  elevation: 0.5
})
const elapsed2 = Date.now() - startTime2
console.log(`   Generated in ${elapsed2}ms\n`)

// Update center and build mesh
console.log('6. Building mesh for visible chunks...')
grid.setCenterFromWorld(500, 500) // Set view center at world position (500, 500)

const mesh = buildGridMesh(grid, 2.0) // worldScale = 2.0
console.log(`   Mesh vertices: ${mesh.vertexCount}`)
console.log(`   Chunks in mesh: ${mesh.chunkCount}`)
console.log(`   Vertex buffer size: ${(mesh.vertexData.byteLength / 1024).toFixed(2)} KB\n`)

// Final statistics
console.log('7. Final statistics:')
const stats2 = grid.getStats()
console.log(`   Total loaded chunks: ${stats2.loadedChunks}`)
console.log(`   View distance: ${stats2.viewDistance}`)
console.log(`   Max loaded chunks: ${stats2.maxLoadedChunks}`)
console.log(`   Center chunk: (${stats2.centerChunk.cx}, ${stats2.centerChunk.cz})\n`)

// Example 3: Generate with negative coordinates
console.log('8. Generating region with negative coordinates...')
const negativeRegion: GridRegion = {
  minChunk: { cx: -10, cz: -10 },
  maxChunk: { cx: 10, cz: 10 }
}

const negativeDims = getRegionDimensions(negativeRegion)
console.log(`   Region size: ${negativeDims.blocksX}×${negativeDims.blocksZ} blocks`)
console.log(`   Chunks: ${negativeDims.chunksX}×${negativeDims.chunksZ}`)

const startTime3 = Date.now()
generateGridRegion(grid, negativeRegion, {
  profile: 'hybrid',
  seed: 4242
})
const elapsed3 = Date.now() - startTime3
console.log(`   Generated in ${elapsed3}ms\n`)

// Check blocks in negative region
console.log('9. Checking blocks in negative region...')
const block3 = grid.getBlock(-100, 10, -100)
const block4 = grid.getBlock(-50, 15, 50)
console.log(`   Block at (-100, 10, -100): ${BlockType[block3]}`)
console.log(`   Block at (-50, 15, 50): ${BlockType[block4]}\n`)

console.log('=== Example Complete ===')
console.log('\nNext steps:')
console.log('1. Try the CLI: npm run terrain generate 0 0 9 9 rolling_hills')
console.log('2. Read TERRAIN_GRID.md for full documentation')
console.log('3. Integrate with your renderer using buildGridMesh()')
