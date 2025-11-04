// Region Manager - Handles chunked generation and region history
import type { Vec3, HighlightSelection } from '../core'
import type { DSLCommand } from '../dsl/commands'
import { ChunkGrid, type ChunkRegion } from './chunkGrid'

export interface RegionHistory {
  id: string
  selection: HighlightSelection | null
  chunkIds: string[]
  commands: DSLCommand[]
  timestamp: number
  description?: string
}

export interface RegionGenerationRequest {
  selection?: HighlightSelection
  region?: { min: Vec3; max: Vec3 }
  commands: DSLCommand[]
  description?: string
}

export class RegionManager {
  private chunkGrid: ChunkGrid
  private history: RegionHistory[] = []
  private maxHistorySize = 50

  constructor(chunkGrid: ChunkGrid) {
    this.chunkGrid = chunkGrid
  }

  getChunkGrid(): ChunkGrid {
    return this.chunkGrid
  }

  selectionToBounds(selection: HighlightSelection): { min: Vec3; max: Vec3 } {
    const { center, shape } = selection

    switch (shape) {
      case 'cube':
      case 'sphere': {
        const r = selection.radius
        return {
          min: [center[0] - r, center[1] - r, center[2] - r],
          max: [center[0] + r, center[1] + r, center[2] + r]
        }
      }

      case 'ellipsoid': {
        const rx = selection.radiusX ?? selection.radius
        const ry = selection.radiusY ?? selection.radius
        const rz = selection.radiusZ ?? selection.radius
        return {
          min: [center[0] - rx, center[1] - ry, center[2] - rz],
          max: [center[0] + rx, center[1] + ry, center[2] + rz]
        }
      }

      case 'plane': {
        const sx = selection.planeSizeX ?? 8
        const sz = selection.planeSizeZ ?? 8
        const h = selection.planeHeight ?? 64
        return {
          min: [center[0] - sx / 2, center[1], center[2] - sz / 2],
          max: [center[0] + sx / 2, center[1] + h, center[2] + sz / 2]
        }
      }

      default:
        return { min: center, max: center }
    }
  }

  getAffectedChunks(request: RegionGenerationRequest): ChunkRegion[] {
    let bounds: { min: Vec3; max: Vec3 }

    if (request.selection) {
      bounds = this.selectionToBounds(request.selection)
    } else if (request.region) {
      bounds = request.region
    } else {
      // Default to entire world
      const worldSize = this.chunkGrid.config.worldSize
      bounds = {
        min: [0, 0, 0],
        max: [worldSize.x - 1, worldSize.y - 1, worldSize.z - 1]
      }
    }

    return this.chunkGrid.getChunksInRegion(bounds.min, bounds.max)
  }

  recordGeneration(request: RegionGenerationRequest, affectedChunks: ChunkRegion[]) {
    const entry: RegionHistory = {
      id: this.generateId(),
      selection: request.selection ?? null,
      chunkIds: affectedChunks.map(c => c.id),
      commands: request.commands,
      timestamp: Date.now(),
      description: request.description
    }

    this.history.push(entry)

    // Mark chunks as generated
    affectedChunks.forEach(chunk => {
      this.chunkGrid.markChunkGenerated(chunk.id)
    })

    // Trim history if too large
    if (this.history.length > this.maxHistorySize) {
      this.history.shift()
    }
  }

  getHistory(): RegionHistory[] {
    return [...this.history]
  }

  getRecentGenerations(count: number = 10): RegionHistory[] {
    return this.history.slice(-count)
  }

  clearHistory() {
    this.history.length = 0
  }

  private generateId(): string {
    return `region_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`
  }

  // Split commands into per-chunk execution
  splitCommandsByChunks(
    commands: DSLCommand[],
    chunks: ChunkRegion[]
  ): Map<string, DSLCommand[]> {
    const commandsByChunk = new Map<string, DSLCommand[]>()

    for (const chunk of chunks) {
      const chunkCommands: DSLCommand[] = []

      for (const cmd of commands) {
        // Check if command affects this chunk
        if (this.commandAffectsChunk(cmd, chunk)) {
          // Clone command and restrict to chunk bounds
          const restrictedCmd = this.restrictCommandToChunk(cmd, chunk)
          if (restrictedCmd) {
            chunkCommands.push(restrictedCmd)
          }
        }
      }

      if (chunkCommands.length > 0) {
        commandsByChunk.set(chunk.id, chunkCommands)
      }
    }

    return commandsByChunk
  }

  private commandAffectsChunk(cmd: DSLCommand, chunk: ChunkRegion): boolean {
    switch (cmd.type) {
      case 'terrain_region':
        return this.chunkGrid.boundsOverlap(
          chunk.bounds,
          { min: cmd.params.region.min, max: cmd.params.region.max }
        )

      case 'generate_structure':
        return this.chunkGrid.boundsOverlap(
          chunk.bounds,
          { min: cmd.generator.region.min, max: cmd.generator.region.max }
        )

      case 'set_block':
        return this.pointInChunk(cmd.edit.position, chunk)

      case 'set_blocks':
        return cmd.edits.some(edit => this.pointInChunk(edit.position, chunk))

      case 'set_material':
      case 'set_lighting':
      case 'add_point_light':
      case 'remove_point_light':
      case 'clear_all':
        // These affect the entire world/rendering
        return true

      default:
        return false
    }
  }

  private pointInChunk(point: Vec3, chunk: ChunkRegion): boolean {
    return (
      point[0] >= chunk.bounds.min[0] && point[0] <= chunk.bounds.max[0] &&
      point[1] >= chunk.bounds.min[1] && point[1] <= chunk.bounds.max[1] &&
      point[2] >= chunk.bounds.min[2] && point[2] <= chunk.bounds.max[2]
    )
  }

  private restrictCommandToChunk(cmd: DSLCommand, chunk: ChunkRegion): DSLCommand | null {
    switch (cmd.type) {
      case 'terrain_region': {
        const intersection = this.chunkGrid.intersectRegionWithChunk(
          { min: cmd.params.region.min, max: cmd.params.region.max },
          chunk
        )
        return {
          ...cmd,
          params: {
            ...cmd.params,
            region: {
              min: intersection.min,
              max: intersection.max
            }
          }
        }
      }

      case 'generate_structure': {
        const intersection = this.chunkGrid.intersectRegionWithChunk(
          { min: cmd.generator.region.min, max: cmd.generator.region.max },
          chunk
        )
        return {
          ...cmd,
          generator: {
            ...cmd.generator,
            region: {
              min: intersection.min,
              max: intersection.max
            }
          }
        }
      }

      case 'set_blocks': {
        const filteredEdits = cmd.edits.filter(edit => this.pointInChunk(edit.position, chunk))
        if (filteredEdits.length === 0) return null
        return {
          ...cmd,
          edits: filteredEdits
        }
      }

      default:
        // Commands that don't need restriction
        return cmd
    }
  }
}
