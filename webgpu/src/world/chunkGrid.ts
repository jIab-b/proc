// Chunk Grid System - Divides world into manageable regions for generation
import type { Vec3 } from '../core'

export interface ChunkGridConfig {
  worldSize: { x: number; y: number; z: number }
  chunkSize: { x: number; y: number; z: number }
}

export interface ChunkRegion {
  id: string
  index: { x: number; y: number; z: number }
  bounds: {
    min: Vec3
    max: Vec3
  }
  center: Vec3
  generated: boolean
  timestamp?: number
}

export class ChunkGrid {
  readonly config: ChunkGridConfig
  readonly chunks: Map<string, ChunkRegion>
  readonly gridDimensions: { x: number; y: number; z: number }

  constructor(config: ChunkGridConfig) {
    this.config = config
    this.chunks = new Map()

    // Calculate how many chunks fit in each dimension
    this.gridDimensions = {
      x: Math.ceil(config.worldSize.x / config.chunkSize.x),
      y: Math.ceil(config.worldSize.y / config.chunkSize.y),
      z: Math.ceil(config.worldSize.z / config.chunkSize.z)
    }

    this.initializeGrid()
  }

  private initializeGrid() {
    for (let y = 0; y < this.gridDimensions.y; y++) {
      for (let z = 0; z < this.gridDimensions.z; z++) {
        for (let x = 0; x < this.gridDimensions.x; x++) {
          const chunk = this.createChunkRegion(x, y, z)
          this.chunks.set(chunk.id, chunk)
        }
      }
    }
  }

  private createChunkRegion(x: number, y: number, z: number): ChunkRegion {
    const minX = x * this.config.chunkSize.x
    const minY = y * this.config.chunkSize.y
    const minZ = z * this.config.chunkSize.z

    const maxX = Math.min(minX + this.config.chunkSize.x - 1, this.config.worldSize.x - 1)
    const maxY = Math.min(minY + this.config.chunkSize.y - 1, this.config.worldSize.y - 1)
    const maxZ = Math.min(minZ + this.config.chunkSize.z - 1, this.config.worldSize.z - 1)

    const centerX = (minX + maxX) / 2
    const centerY = (minY + maxY) / 2
    const centerZ = (minZ + maxZ) / 2

    return {
      id: `chunk_${x}_${y}_${z}`,
      index: { x, y, z },
      bounds: {
        min: [minX, minY, minZ],
        max: [maxX, maxY, maxZ]
      },
      center: [centerX, centerY, centerZ],
      generated: false
    }
  }

  getChunkAt(x: number, y: number, z: number): ChunkRegion | null {
    const chunkX = Math.floor(x / this.config.chunkSize.x)
    const chunkY = Math.floor(y / this.config.chunkSize.y)
    const chunkZ = Math.floor(z / this.config.chunkSize.z)

    const id = `chunk_${chunkX}_${chunkY}_${chunkZ}`
    return this.chunks.get(id) || null
  }

  getChunkById(id: string): ChunkRegion | null {
    return this.chunks.get(id) || null
  }

  getAllChunks(): ChunkRegion[] {
    return Array.from(this.chunks.values())
  }

  getChunkBoundaries(): Array<{ min: Vec3; max: Vec3 }> {
    return Array.from(this.chunks.values()).map(chunk => chunk.bounds)
  }

  markChunkGenerated(id: string) {
    const chunk = this.chunks.get(id)
    if (chunk) {
      chunk.generated = true
      chunk.timestamp = Date.now()
    }
  }

  getChunksInRegion(min: Vec3, max: Vec3): ChunkRegion[] {
    const chunks: ChunkRegion[] = []

    for (const chunk of this.chunks.values()) {
      // Check if chunk bounds overlap with the given region
      const overlaps = this.boundsOverlap(chunk.bounds, { min, max })
      if (overlaps) {
        chunks.push(chunk)
      }
    }

    return chunks
  }

  boundsOverlap(
    a: { min: Vec3; max: Vec3 },
    b: { min: Vec3; max: Vec3 }
  ): boolean {
    return (
      a.min[0] <= b.max[0] && a.max[0] >= b.min[0] &&
      a.min[1] <= b.max[1] && a.max[1] >= b.min[1] &&
      a.min[2] <= b.max[2] && a.max[2] >= b.min[2]
    )
  }

  intersectRegionWithChunk(region: { min: Vec3; max: Vec3 }, chunk: ChunkRegion): { min: Vec3; max: Vec3 } {
    return {
      min: [
        Math.max(region.min[0], chunk.bounds.min[0]),
        Math.max(region.min[1], chunk.bounds.min[1]),
        Math.max(region.min[2], chunk.bounds.min[2])
      ],
      max: [
        Math.min(region.max[0], chunk.bounds.max[0]),
        Math.min(region.max[1], chunk.bounds.max[1]),
        Math.min(region.max[2], chunk.bounds.max[2])
      ]
    }
  }
}
