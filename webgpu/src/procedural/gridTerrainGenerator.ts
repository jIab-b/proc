// Grid-aware terrain generator for large-scale continuous terrain
// Generates terrain across multiple chunks with seamless boundaries

import { BlockType } from '../core'
import { ChunkGrid } from '../chunkGrid'
import {
  sampleHeight,
  type TerrainProfile,
  type TerrainParams,
  createTerrainGeneratorState
} from './terrainGenerator'

export interface GridRegion {
  minChunk: { cx: number; cz: number }
  maxChunk: { cx: number; cz: number }
}

export interface GridTerrainParams extends TerrainParams {
  profile: TerrainProfile
}

/**
 * Generate terrain across a grid region (multiple chunks)
 * @param grid - The chunk grid to populate
 * @param region - The chunk region to generate (e.g., chunks [0,0] to [10,10])
 * @param params - Terrain generation parameters
 */
export function generateGridRegion(
  grid: ChunkGrid,
  region: GridRegion,
  params: GridTerrainParams
) {
  const state = createTerrainGeneratorState(params.profile, params)
  const terrainSeed = params.seed ?? state.params.seed

  // Normalize region bounds
  const minCx = Math.min(region.minChunk.cx, region.maxChunk.cx)
  const maxCx = Math.max(region.minChunk.cx, region.maxChunk.cx)
  const minCz = Math.min(region.minChunk.cz, region.maxChunk.cz)
  const maxCz = Math.max(region.minChunk.cz, region.maxChunk.cz)

  console.log(
    `[GridTerrain] Generating region: chunks [${minCx},${minCz}] to [${maxCx},${maxCz}]`
  )
  console.log(
    `[GridTerrain] Region size: ${maxCx - minCx + 1}x${maxCz - minCz + 1} chunks`
  )

  // Generate terrain for each chunk in the region
  for (let cx = minCx; cx <= maxCx; cx++) {
    for (let cz = minCz; cz <= maxCz; cz++) {
      generateChunk(grid, cx, cz, params.profile, state.params, terrainSeed)
    }
  }

  const totalChunks = (maxCx - minCx + 1) * (maxCz - minCz + 1)
  console.log(`[GridTerrain] Generated ${totalChunks} chunks`)
}

/**
 * Generate terrain for a single chunk
 * Uses continuous noise functions so terrain is seamless across chunk boundaries
 */
function generateChunk(
  grid: ChunkGrid,
  cx: number,
  cz: number,
  profile: TerrainProfile,
  params: Required<TerrainParams>,
  seed: number
) {
  const chunk = grid.getChunk(cx, cz)
  const { x: sx, y: sy, z: sz } = chunk.size

  // World coordinates of chunk origin
  const worldOriginX = cx * sx
  const worldOriginZ = cz * sz

  // Generate height map for this chunk
  const heights: number[][] = []
  for (let lx = 0; lx < sx; lx++) {
    const row: number[] = []
    for (let lz = 0; lz < sz; lz++) {
      // Use world coordinates for continuous noise
      const worldX = worldOriginX + lx
      const worldZ = worldOriginZ + lz
      const height = sampleHeight(worldX, worldZ, profile, params, seed)
      row.push(height)
    }
    heights.push(row)
  }

  // Fill chunk with blocks based on height map
  for (let lx = 0; lx < sx; lx++) {
    for (let lz = 0; lz < sz; lz++) {
      const columnHeight = clamp(Math.floor(heights[lx]![lz]!), 0, sy - 1)

      for (let ly = 0; ly < sy; ly++) {
        if (ly > columnHeight) {
          chunk.setBlock(lx, ly, lz, BlockType.Air)
          continue
        }

        const block = selectBlock(ly, columnHeight, profile)
        chunk.setBlock(lx, ly, lz, block)
      }
    }
  }

  grid.markDirty(cx, cz)
}

/**
 * Select block type based on height and profile
 */
function selectBlock(
  y: number,
  columnHeight: number,
  profile: TerrainProfile
): BlockType {
  const surfaceThreshold = columnHeight

  // Surface block
  if (y === surfaceThreshold) {
    if (profile === 'mountain' && columnHeight > 28) return BlockType.AlpineRock
    if (profile === 'mountain' && columnHeight > 34) return BlockType.GlacierIce
    if (profile === 'hybrid' && columnHeight > 30) return BlockType.AlpineGrass
    if (profile === 'rolling_hills' && columnHeight > 18) return BlockType.Grass
    return BlockType.AlpineGrass
  }

  // Subsurface layers
  if (y >= surfaceThreshold - 2) {
    if (profile === 'rolling_hills') return BlockType.Dirt
    if (profile === 'mountain') return BlockType.Gravel
    return BlockType.Dirt
  }

  // Deep layers
  if (profile === 'mountain') return BlockType.Stone
  return BlockType.Stone
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

/**
 * Calculate total block count for a grid region
 */
export function getRegionBlockCount(region: GridRegion, chunkSize = 32): number {
  const chunksX = Math.abs(region.maxChunk.cx - region.minChunk.cx) + 1
  const chunksZ = Math.abs(region.maxChunk.cz - region.minChunk.cz) + 1
  const blocksX = chunksX * chunkSize
  const blocksZ = chunksZ * chunkSize
  return blocksX * blocksZ * chunkSize
}

/**
 * Get region dimensions in blocks
 */
export function getRegionDimensions(region: GridRegion, chunkSize = 32) {
  const chunksX = Math.abs(region.maxChunk.cx - region.minChunk.cx) + 1
  const chunksZ = Math.abs(region.maxChunk.cz - region.minChunk.cz) + 1
  return {
    chunksX,
    chunksZ,
    blocksX: chunksX * chunkSize,
    blocksZ: chunksZ * chunkSize,
    blocksY: chunkSize
  }
}
