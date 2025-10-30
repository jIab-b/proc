import { BlockType, ChunkManager, type Vec3, type TerrainGenerateParams } from '../core'
import {
  createTerrainGeneratorState,
  generateRegion,
  type TerrainRegion,
  type TerrainGeneratorState
} from '../procedural/terrainGenerator'
import { type WorldCommand, type VoxelEdit } from './commands'

export type WorldChange =
  | { kind: 'set'; edits?: VoxelEdit[]; count: number; source?: string }
  | { kind: 'clear_all'; source?: string }
  | { kind: 'terrain'; region: TerrainRegion; action: TerrainGenerateParams['action']; source?: string }
  | { kind: 'snapshot_loaded'; count: number; source?: string }

export type WorldResult = { success: boolean; message?: string; change?: WorldChange }

type Listener = (change: WorldChange) => void

export interface WorldConfig {
  worldScale: number
  chunkOriginOffset: Vec3
}

export class WorldState {
  private listeners = new Set<Listener>()
  private dirty = false

  constructor(private chunk: ChunkManager, private config: WorldConfig) {}

  getChunk() {
    return this.chunk
  }

  getWorldScale() {
    return this.config.worldScale
  }

  setWorldScale(scale: number) {
    this.config.worldScale = scale
  }

  getChunkOriginOffset(): Vec3 {
    return [...this.config.chunkOriginOffset] as Vec3
  }

  setChunkOriginOffset(offset: Vec3) {
    this.config.chunkOriginOffset = [...offset] as Vec3
  }

  onChange(listener: Listener) {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  isDirty() {
    return this.dirty
  }

  consumeDirty() {
    const wasDirty = this.dirty
    this.dirty = false
    return wasDirty
  }

  apply(command: WorldCommand): WorldResult {
    switch (command.type) {
      case 'set_block':
        return this.applySetBlocks([command.edit], command.source)
      case 'set_blocks':
        return this.applySetBlocks(command.edits, command.source)
      case 'clear_all':
        return this.applyClearAll(command.source)
      case 'load_snapshot':
        return this.applyLoadSnapshot(command.blocks, command.clear, command.worldScale, command.source)
      case 'terrain_region':
        return this.applyTerrainRegion(command.params, command.source)
      default:
        return { success: false, message: `Unhandled command ${(command as any).type}` }
    }
  }

  private applySetBlocks(edits: VoxelEdit[], source?: string): WorldResult {
    if (!edits.length) {
      return { success: true, change: { kind: 'set', count: 0, source } }
    }

    const bounded: VoxelEdit[] = []
    for (const edit of edits) {
      const [x, y, z] = edit.position
      if (!this.inBounds(x, y, z)) continue
      this.chunk.setBlock(x, y, z, edit.blockType as BlockType)
      bounded.push({ position: [...edit.position] as Vec3, blockType: edit.blockType })
    }

    const change: WorldChange = {
      kind: 'set',
      edits: bounded.length <= 256 ? bounded : undefined,
      count: bounded.length,
      source
    }
    this.notify(change)
    return { success: true, change }
  }

  private applyClearAll(source?: string): WorldResult {
    const { x: sx, y: sy, z: sz } = this.chunk.size
    for (let y = 0; y < sy; y++)
      for (let z = 0; z < sz; z++)
        for (let x = 0; x < sx; x++)
          this.chunk.setBlock(x, y, z, BlockType.Air)

    const change: WorldChange = { kind: 'clear_all', source }
    this.notify(change)
    return { success: true, change }
  }

  private applyLoadSnapshot(blocks: VoxelEdit[], clear?: boolean, worldScale?: number, source?: string): WorldResult {
    if (clear) this.applyClearAll(source)
    if (typeof worldScale === 'number') this.setWorldScale(worldScale)
    const result = this.applySetBlocks(blocks, source)
    const change: WorldChange = { kind: 'snapshot_loaded', count: blocks.length, source }
    this.notify(change)
    return { success: true, change, message: result.message }
  }

  private applyTerrainRegion(params: TerrainGenerateParams, source?: string): WorldResult {
    const chunkRegion = this.computeChunkRegion(params.region.min, params.region.max)
    if (!chunkRegion) {
      return { success: false, message: 'Region outside chunk bounds' }
    }

    const terrainState = createTerrainGeneratorState(params.profile, params.params)

    if (params.action === 'clear') {
      this.clearRegion(chunkRegion, params, source)
    } else {
      this.generateTerrain(chunkRegion, terrainState, params)
    }

    const change: WorldChange = {
      kind: 'terrain',
      region: chunkRegion,
      action: params.action,
      source
    }
    this.notify(change)
    return { success: true, change }
  }

  private clearRegion(region: TerrainRegion, params: TerrainGenerateParams, source?: string) {
    const { min, max } = region
    for (let x = min[0]; x <= max[0]; x++) {
      for (let y = min[1]; y <= max[1]; y++) {
        for (let z = min[2]; z <= max[2]; z++) {
          if (!this.inBounds(x, y, z)) continue
          if (this.isInsideMask(x, y, z, params)) {
            this.chunk.setBlock(x, y, z, BlockType.Air)
          }
        }
      }
    }
  }

  private generateTerrain(region: TerrainRegion, state: TerrainGeneratorState, params: TerrainGenerateParams) {
    const floorHeightFn =
      params.ellipsoidMask
        ? (x: number, z: number) => this.ellipsoidFloorHeight(x, z, params)
        : params.selectionType === 'plane'
        ? () => this.planeFloorHeight(params.region.min[1])
        : undefined

    generateRegion(this.chunk, region, state, floorHeightFn)
  }

  private ellipsoidFloorHeight(worldX: number, worldZ: number, params: TerrainGenerateParams) {
    const mask = params.ellipsoidMask!
    const [offsetX, offsetY, offsetZ] = this.config.chunkOriginOffset
    const scale = this.config.worldScale

    const posX = worldX * scale + offsetX
    const posZ = worldZ * scale + offsetZ

    const dx = (posX - mask.center[0]) / mask.radiusX
    const dz = (posZ - mask.center[2]) / mask.radiusZ
    const horizontalDistSq = dx * dx + dz * dz
    if (horizontalDistSq >= 1) return (mask.center[1] - offsetY) / scale
    const dy = mask.radiusY * Math.sqrt(1 - horizontalDistSq)
    const worldY = mask.center[1] + dy
    return (worldY - offsetY) / scale
  }

  private planeFloorHeight(worldY: number) {
    const [, offsetY] = this.config.chunkOriginOffset
    const scale = this.config.worldScale || 1
    return (worldY - offsetY) / scale
  }

  private isInsideMask(chunkX: number, chunkY: number, chunkZ: number, params: TerrainGenerateParams) {
    const mask = params.ellipsoidMask
    if (!mask) return true

    const [offsetX, offsetY, offsetZ] = this.config.chunkOriginOffset
    const scale = this.config.worldScale

    const worldX = chunkX * scale + offsetX
    const worldY = chunkY * scale + offsetY
    const worldZ = chunkZ * scale + offsetZ

    const dx = (worldX - mask.center[0]) / mask.radiusX
    const dy = (worldY - mask.center[1]) / mask.radiusY
    const dz = (worldZ - mask.center[2]) / mask.radiusZ

    return dx * dx + dy * dy + dz * dz <= 1
  }

  private computeChunkRegion(minWorld: Vec3, maxWorld: Vec3): TerrainRegion | null {
    const toChunk = (worldCoord: Vec3): Vec3 => {
      const [ox, oy, oz] = this.config.chunkOriginOffset
      const scale = this.config.worldScale || 1
      return [
        (worldCoord[0] - ox) / scale,
        (worldCoord[1] - oy) / scale,
        (worldCoord[2] - oz) / scale
      ]
    }

    const minChunk = toChunk(minWorld)
    const maxChunk = toChunk(maxWorld)
    const region: TerrainRegion = {
      min: [
        Math.max(0, Math.floor(Math.min(minChunk[0], maxChunk[0]))),
        Math.max(0, Math.floor(Math.min(minChunk[1], maxChunk[1]))),
        Math.max(0, Math.floor(Math.min(minChunk[2], maxChunk[2])))
      ],
      max: [
        Math.min(this.chunk.size.x - 1, Math.floor(Math.max(minChunk[0], maxChunk[0]))),
        Math.min(this.chunk.size.y - 1, Math.floor(Math.max(minChunk[1], maxChunk[1]))),
        Math.min(this.chunk.size.z - 1, Math.floor(Math.max(minChunk[2], maxChunk[2])))
      ]
    }

    if (region.min[0] > region.max[0] || region.min[1] > region.max[1] || region.min[2] > region.max[2]) {
      return null
    }
    return region
  }

  private inBounds(x: number, y: number, z: number) {
    return (
      x >= 0 &&
      y >= 0 &&
      z >= 0 &&
      x < this.chunk.size.x &&
      y < this.chunk.size.y &&
      z < this.chunk.size.z
    )
  }

  private notify(change: WorldChange) {
    this.dirty = true
    for (const listener of this.listeners) {
      listener(change)
    }
  }
}
