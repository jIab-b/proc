// Consolidated WebGPU Engine Module
// Combines: terrain generation, map management, camera capture, DSL engine

import type { ChunkManager, BlockType } from './core'
import { API_BASE_URL } from './core'
import { createTerrainGeneratorState, generateRegion, type TerrainProfile, type TerrainParams } from './procedural/terrainGenerator'
import { createTerrainGeneratorState, generateRegion, type TerrainProfile, type TerrainParams } from './procedural/terrainGenerator'

// ============================================================================
// TERRAIN GENERATION
// ============================================================================

export type WorldConfig = {
  seed: number
  dimensions: { x: number; y: number; z: number }
}

export function createWorldConfig(seed: number = Date.now()): WorldConfig {
  return { seed, dimensions: { x: 64, y: 48, z: 64 } }
}

export function generateTerrain(chunk: ChunkManager, config: WorldConfig, BlockType: any) {
  const { x: sx, y: sy, z: sz } = chunk.size
  const baseFreq = 1 / 32
  const minHeight = Math.floor(sy * 0.2)
  const maxHeight = Math.floor(sy * 0.6)
  const heightRange = maxHeight - minHeight

  for (let x = 0; x < sx; x++) {
    for (let z = 0; z < sz; z++) {
      const elevation = fbm(x * baseFreq, z * baseFreq, config.seed, 4, 2.0, 0.5)
      const height = Math.floor(minHeight + ((elevation + 1) / 2) * heightRange)
      for (let y = 0; y < sy; y++) {
        if (y > height) chunk.setBlock(x, y, z, BlockType.Air)
        else if (y === height) chunk.setBlock(x, y, z, BlockType.Grass)
        else if (y >= height - 3) chunk.setBlock(x, y, z, BlockType.Dirt)
        else chunk.setBlock(x, y, z, BlockType.Stone)
      }
    }
  }
}

function fbm(x: number, z: number, seed: number, octaves: number, lacunarity: number, gain: number) {
  let freq = 1, amp = 1, sum = 0, max = 0
  for (let i = 0; i < octaves; i++) {
    sum += noise2D(x * freq, z * freq, seed + i * 131) * amp
    max += amp
    freq *= lacunarity
    amp *= gain
  }
  return max > 0 ? sum / max : 0
}

function noise2D(x: number, z: number, seed: number) {
  const xi = Math.floor(x), zi = Math.floor(z)
  const xf = x - xi, zf = z - zi
  const u = xf * xf * (3 - 2 * xf), v = zf * zf * (3 - 2 * zf)
  const h00 = hash(xi, zi, seed), h10 = hash(xi + 1, zi, seed)
  const h01 = hash(xi, zi + 1, seed), h11 = hash(xi + 1, zi + 1, seed)
  return (h00 * (1 - u) + h10 * u) * (1 - v) + (h01 * (1 - u) + h11 * u) * v
}

function hash(x: number, z: number, seed: number) {
  let h = seed >>> 0
  h ^= Math.imul(0x27d4eb2d, x)
  h = (h ^ (h >>> 15)) >>> 0
  h ^= Math.imul(0x165667b1, z)
  h = (h ^ (h >>> 13)) >>> 0
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296
}

// ============================================================================
// MAP MANAGEMENT
// ============================================================================

export type BlockPosition = [number, number, number]

export class MapManager {
  private chunk: ChunkManager
  private worldScale: number
  private captureSessionId: string
  activeSequence: number | null = null
  isDirty = false

  constructor(chunk: ChunkManager, worldScale: number) {
    this.chunk = chunk
    this.worldScale = worldScale
    this.captureSessionId = this.generateId()
  }

  private generateId() {
    return crypto?.randomUUID?.() || `map_${Math.random().toString(36).slice(2)}_${Date.now()}`
  }

  markDirty() {
    this.isDirty = true
  }

  async save(worldConfig: WorldConfig, customBlocks: any[], BlockType: any) {
    const placements = this.serializeBlocks(BlockType)
    const payload = {
      sequence: this.activeSequence,
      captureId: this.captureSessionId,
      worldScale: this.worldScale,
      worldConfig,
      blocks: placements,
      customBlocks: this.serializeCustomBlocks(customBlocks)
    }

    const res = await fetch(`${API_BASE_URL}/api/save-map`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    if (!res.ok) throw new Error(`Save failed: ${res.status}`)
    const data = await res.json()
    if (data.sequence) this.activeSequence = data.sequence
    this.isDirty = false
  }

  async load(sequence: number, BlockType: any) {
    const res = await fetch(`${API_BASE_URL}/api/maps/${sequence}`)
    if (!res.ok) throw new Error(`Load failed: ${res.status}`)
    const data = await res.json()

    if (data.worldScale) this.worldScale = data.worldScale
    this.clearChunk(BlockType)

    const blocks = data.blocks || []
    for (const b of blocks) {
      const [x, y, z] = b.position
      const type = BlockType[b.blockType as keyof typeof BlockType]
      if (type !== undefined && this.inBounds([x, y, z])) {
        this.chunk.setBlock(x, y, z, type)
      }
    }

    this.activeSequence = sequence
    return { blocks, worldScale: this.worldScale }
  }

  async loadFromFile(jsonContent: string, BlockType: any) {
    const data = JSON.parse(jsonContent)
    if (data.worldScale) this.worldScale = data.worldScale
    this.clearChunk(BlockType)

    const blocks = data.blocks || []
    for (const b of blocks) {
      const [x, y, z] = b.position
      const type = BlockType[b.blockType as keyof typeof BlockType]
      if (type !== undefined && this.inBounds([x, y, z])) {
        this.chunk.setBlock(x, y, z, type)
      }
    }

    this.activeSequence = null
    return { blocks, worldScale: this.worldScale }
  }

  async createNew(copyFromSequence?: number, BlockType?: any) {
    if (copyFromSequence !== undefined && BlockType) {
      const { blocks } = await this.load(copyFromSequence, BlockType)
      this.activeSequence = null
      return blocks
    } else if (BlockType) {
      this.clearChunk(BlockType)
      this.activeSequence = null
      return []
    }
    return []
  }

  async loadFirstAvailable(BlockType: any, worldConfig: WorldConfig) {
    try {
      const res = await fetch(`${API_BASE_URL}/api/maps`)
      if (!res.ok) {
        generateTerrain(this.chunk, worldConfig, BlockType)
        return null
      }
      const data = await res.json()
      const maps = data.maps || []
      if (maps.length > 0) {
        return await this.load(maps[0].sequence, BlockType)
      }
      generateTerrain(this.chunk, worldConfig, BlockType)
      return null
    } catch {
      generateTerrain(this.chunk, worldConfig, BlockType)
      return null
    }
  }

  private clearChunk(BlockType: any) {
    const { x: sx, y: sy, z: sz } = this.chunk.size
    for (let y = 0; y < sy; y++)
      for (let z = 0; z < sz; z++)
        for (let x = 0; x < sx; x++)
          this.chunk.setBlock(x, y, z, BlockType.Air)
  }

  private inBounds([x, y, z]: BlockPosition) {
    const { x: sx, y: sy, z: sz } = this.chunk.size
    return x >= 0 && y >= 0 && z >= 0 && x < sx && y < sy && z < sz
  }

  private serializeBlocks(BlockType: any) {
    const placements: Array<{ position: BlockPosition; blockType: string }> = []
    const { x: sx, y: sy, z: sz } = this.chunk.size
    for (let y = 0; y < sy; y++)
      for (let z = 0; z < sz; z++)
        for (let x = 0; x < sx; x++) {
          const block = this.chunk.getBlock(x, y, z)
          if (block === BlockType.Air) continue
          placements.push({ position: [x, y, z], blockType: BlockType[block] ?? String(block) })
        }
    return placements
  }

  private serializeCustomBlocks(blocks: any[]) {
    return blocks.map(b => ({
      id: b.id,
      name: b.name,
      textureLayer: b.textureLayer ?? null,
      colors: b.colors,
      faceTiles: Object.fromEntries(
        Object.entries(b.faceTiles ?? {}).map(([face, info]: [string, any]) => [
          face,
          { path: info?.path ?? null, url: info?.url ?? null, prompt: info?.prompt ?? null, sequence: info?.sequence ?? null }
        ])
      )
    }))
  }
}

// ============================================================================
// CAMERA CAPTURE SYSTEM
// ============================================================================

type Vec3 = [number, number, number]
type Mat4 = Float32Array

export type CameraSnapshot = {
  position: Vec3
  forward: Vec3
  up: Vec3
  right: Vec3
  viewMatrix: Mat4
  projectionMatrix: Mat4
  viewProjectionMatrix: Mat4
  fovYRadians: number
  aspect: number
  near: number
  far: number
}

type CapturedView = {
  id: string
  createdAt: string
  snapshot: CameraSnapshot
}

export class CaptureSystem {
  private views: CapturedView[] = []
  private sessionId: string
  private exporting = false

  constructor() {
    this.sessionId = this.generateId()
  }

  private generateId() {
    return crypto?.randomUUID?.() || `capture_${Math.random().toString(36).slice(2)}_${Date.now()}`
  }

  capture(camera: CameraSnapshot) {
    const id = `view_${String(this.views.length + 1).padStart(3, '0')}`
    this.views.push({
      id,
      createdAt: new Date().toISOString(),
      snapshot: this.cloneSnapshot(camera)
    })
    console.log(`[capture] Stored view ${id}`)
  }

  clear() {
    this.views.length = 0
    this.sessionId = this.generateId()
    console.log('[capture] Cleared all views')
  }

  getViews() {
    return this.views
  }

  getSessionId() {
    return this.sessionId
  }

  isExporting() {
    return this.exporting
  }

  async exportDataset(device: GPUDevice, renderFn: (snapshot: CameraSnapshot) => Promise<string>, width: number, height: number) {
    if (this.exporting || this.views.length === 0) return null

    this.exporting = true
    try {
      await (device.queue as any).onSubmittedWorkDone()

      const payload = {
        formatVersion: '1.0',
        exportedAt: new Date().toISOString(),
        imageSize: { width, height },
        viewCount: this.views.length,
        captureId: this.sessionId,
        views: [] as any[]
      }

      for (let i = 0; i < this.views.length; i++) {
        const view = this.views[i]!
        const rgbBase64 = await renderFn(view.snapshot)
        payload.views.push({
          id: view.id,
          index: i,
          capturedAt: view.createdAt,
          position: view.snapshot.position,
          forward: view.snapshot.forward,
          up: view.snapshot.up,
          right: view.snapshot.right,
          intrinsics: {
            fovYDegrees: (view.snapshot.fovYRadians * 180) / Math.PI,
            aspect: view.snapshot.aspect,
            near: view.snapshot.near,
            far: view.snapshot.far
          },
          viewMatrix: Array.from(view.snapshot.viewMatrix),
          projectionMatrix: Array.from(view.snapshot.projectionMatrix),
          viewProjectionMatrix: Array.from(view.snapshot.viewProjectionMatrix),
          rgbBase64,
          depthBase64: null,
          normalBase64: null
        })
      }

      const res = await fetch(`${API_BASE_URL}/api/export-dataset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })

      if (!res.ok) throw new Error(`Export failed: ${res.status}`)
      const result = await res.json()
      this.clear()
      return result
    } finally {
      this.exporting = false
    }
  }

  private cloneSnapshot(s: CameraSnapshot): CameraSnapshot {
    return {
      position: [...s.position] as Vec3,
      forward: [...s.forward] as Vec3,
      up: [...s.up] as Vec3,
      right: [...s.right] as Vec3,
      viewMatrix: new Float32Array(s.viewMatrix),
      projectionMatrix: new Float32Array(s.projectionMatrix),
      viewProjectionMatrix: new Float32Array(s.viewProjectionMatrix),
      fovYRadians: s.fovYRadians,
      aspect: s.aspect,
      near: s.near,
      far: s.far
    }
  }
}

// ============================================================================
// DSL ENGINE
// ============================================================================

type DSLAction =
  | { type: 'place_block'; params: { position: BlockPosition; blockType: string; customBlockId?: number } }
  | { type: 'remove_block'; params: { position: BlockPosition } }

export class DSLEngine {
  async parse(text: string): Promise<DSLAction[]> {
    const res = await fetch(`${API_BASE_URL}/api/parse-dsl`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    })
    if (!res.ok) return []
    const data = await res.json()
    return data.actions || []
  }

  async logAction(action: string, params: any, result: any) {
    const payload = {
      action,
      params: this.prepareParams(params),
      result: this.prepareResult(result),
      source: params.source ?? 'unknown',
      timestamp: new Date().toISOString()
    }
    fetch(`${API_BASE_URL}/api/log-dsl`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }).catch(() => {})
  }

  private prepareParams(params: any) {
    const base = { ...params }
    if (Array.isArray(params.position)) base.position = params.position
    if (typeof params.blockType === 'number') base.blockTypeName = params.blockType
    return base
  }

  private prepareResult(result: any) {
    return { ...result }
  }
}
