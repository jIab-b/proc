// ChunkGrid - Multi-chunk terrain management for large-scale worlds
// Supports thousands of blocks wide with dynamic chunk loading/unloading

import { ChunkManager, type ChunkDimensions, BlockType } from './core'

export type ChunkCoord = { cx: number; cz: number }
export type BlockCoord = { x: number; y: number; z: number }

export interface ChunkData {
  coord: ChunkCoord
  chunk: ChunkManager
  dirty: boolean
  lastAccessed: number
}

export interface GridConfig {
  chunkSize: ChunkDimensions
  maxLoadedChunks: number
  viewDistance: number // in chunks
}

const DEFAULT_GRID_CONFIG: GridConfig = {
  chunkSize: { x: 32, y: 32, z: 32 },
  maxLoadedChunks: 64, // Maximum chunks to keep in memory
  viewDistance: 8 // Load chunks within 8 chunk radius
}

/**
 * ChunkGrid manages a large grid of chunks for massive terrain generation.
 * - Supports thousands of blocks wide (e.g., 100 chunks = 3200 blocks)
 * - Dynamic chunk loading/unloading based on view distance
 * - Continuous terrain generation across chunk boundaries
 */
export class ChunkGrid {
  private chunks: Map<string, ChunkData> = new Map()
  private config: GridConfig
  private centerChunk: ChunkCoord = { cx: 0, cz: 0 }

  constructor(config: Partial<GridConfig> = {}) {
    this.config = { ...DEFAULT_GRID_CONFIG, ...config }
  }

  /**
   * Get or create a chunk at the given chunk coordinates
   */
  getChunk(cx: number, cz: number): ChunkManager {
    const key = this.chunkKey(cx, cz)
    let chunkData = this.chunks.get(key)

    if (!chunkData) {
      const chunk = new ChunkManager(this.config.chunkSize)
      chunkData = {
        coord: { cx, cz },
        chunk,
        dirty: false,
        lastAccessed: Date.now()
      }
      this.chunks.set(key, chunkData)
      this.unloadDistantChunks()
    }

    chunkData.lastAccessed = Date.now()
    return chunkData.chunk
  }

  /**
   * Get block at world coordinates
   */
  getBlock(x: number, y: number, z: number): BlockType {
    const { cx, cz, lx, ly, lz } = this.worldToChunk(x, y, z)
    const chunk = this.chunks.get(this.chunkKey(cx, cz))
    if (!chunk) return BlockType.Air
    return chunk.chunk.getBlock(lx, ly, lz)
  }

  /**
   * Set block at world coordinates
   */
  setBlock(x: number, y: number, z: number, type: BlockType) {
    const { cx, cz, lx, ly, lz } = this.worldToChunk(x, y, z)
    const chunk = this.getChunk(cx, cz)
    chunk.setBlock(lx, ly, lz, type)
    this.markDirty(cx, cz)
  }

  /**
   * Mark chunk as needing mesh rebuild
   */
  markDirty(cx: number, cz: number) {
    const key = this.chunkKey(cx, cz)
    const chunkData = this.chunks.get(key)
    if (chunkData) {
      chunkData.dirty = true
    }
  }

  /**
   * Convert world coordinates to chunk coordinates and local coordinates
   */
  worldToChunk(x: number, y: number, z: number) {
    const { x: sx, y: sy, z: sz } = this.config.chunkSize

    // Floor division for chunk coordinates
    const cx = Math.floor(x / sx)
    const cz = Math.floor(z / sz)

    // Modulo for local coordinates (handle negatives correctly)
    const lx = ((x % sx) + sx) % sx
    const ly = Math.max(0, Math.min(y, sy - 1))
    const lz = ((z % sz) + sz) % sz

    return { cx, cz, lx, ly, lz }
  }

  /**
   * Convert chunk coordinates and local coordinates to world coordinates
   */
  chunkToWorld(cx: number, cz: number, lx: number, ly: number, lz: number) {
    const { x: sx, z: sz } = this.config.chunkSize
    return {
      x: cx * sx + lx,
      y: ly,
      z: cz * sz + lz
    }
  }

  /**
   * Update center chunk for view distance culling
   */
  setCenterChunk(cx: number, cz: number) {
    this.centerChunk = { cx, cz }
    this.unloadDistantChunks()
  }

  /**
   * Update center based on world position
   */
  setCenterFromWorld(x: number, z: number) {
    const { cx, cz } = this.worldToChunk(x, 0, z)
    this.setCenterChunk(cx, cz)
  }

  /**
   * Get all loaded chunks
   */
  getLoadedChunks(): ChunkData[] {
    return Array.from(this.chunks.values())
  }

  /**
   * Get all chunks within view distance
   */
  getVisibleChunks(): ChunkData[] {
    const visible: ChunkData[] = []
    const vd = this.config.viewDistance

    for (let dcx = -vd; dcx <= vd; dcx++) {
      for (let dcz = -vd; dcz <= vd; dcz++) {
        const cx = this.centerChunk.cx + dcx
        const cz = this.centerChunk.cz + dcz
        const key = this.chunkKey(cx, cz)
        const chunkData = this.chunks.get(key)
        if (chunkData) {
          visible.push(chunkData)
        }
      }
    }

    return visible
  }

  /**
   * Clear all chunks
   */
  clear() {
    this.chunks.clear()
  }

  /**
   * Unload chunks outside view distance
   */
  private unloadDistantChunks() {
    if (this.chunks.size <= this.config.maxLoadedChunks) return

    const vd = this.config.viewDistance
    const chunksToRemove: string[] = []

    for (const [key, chunkData] of this.chunks) {
      const dx = Math.abs(chunkData.coord.cx - this.centerChunk.cx)
      const dz = Math.abs(chunkData.coord.cz - this.centerChunk.cz)

      if (dx > vd || dz > vd) {
        chunksToRemove.push(key)
      }
    }

    // Sort by last accessed time, remove oldest first
    chunksToRemove.sort((a, b) => {
      const ta = this.chunks.get(a)?.lastAccessed ?? 0
      const tb = this.chunks.get(b)?.lastAccessed ?? 0
      return ta - tb
    })

    // Remove chunks until we're under the limit
    const toRemove = Math.min(
      chunksToRemove.length,
      this.chunks.size - this.config.maxLoadedChunks
    )

    for (let i = 0; i < toRemove; i++) {
      this.chunks.delete(chunksToRemove[i]!)
    }
  }

  /**
   * Generate chunk key from coordinates
   */
  private chunkKey(cx: number, cz: number): string {
    return `${cx}_${cz}`
  }

  /**
   * Get grid statistics
   */
  getStats() {
    return {
      loadedChunks: this.chunks.size,
      centerChunk: this.centerChunk,
      viewDistance: this.config.viewDistance,
      maxLoadedChunks: this.config.maxLoadedChunks,
      dirtyChunks: Array.from(this.chunks.values()).filter(c => c.dirty).length
    }
  }
}
