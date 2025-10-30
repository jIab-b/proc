// Consolidated WebGPU Engine Module
// Combines: map management, camera capture, DSL engine

import type { BlockType } from './core'
import { API_BASE_URL } from './core'
import type { TerrainGenerateParams, Vec3 } from './core'
import { WorldState } from './world'

export type WorldConfig = {
  seed: number
  dimensions: { x: number; y: number; z: number }
}

export function createWorldConfig(seed: number = Date.now()): WorldConfig {
  return { seed, dimensions: { x: 256, y: 128, z: 256 } }
}

// ============================================================================
// MAP MANAGEMENT
// ============================================================================

export type BlockPosition = [number, number, number]

export class MapManager {
  private chunk: ReturnType<WorldState['getChunk']>
  private worldScale: number
  private captureSessionId: string
  activeSequence: number | null = null
  isDirty = false

  constructor(private world: WorldState) {
    this.chunk = this.world.getChunk()
    this.worldScale = this.world.getWorldScale()
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
      worldScale: this.world.getWorldScale(),
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

    if (data.worldScale) {
      this.worldScale = data.worldScale
      this.world.setWorldScale(data.worldScale)
    }
    this.world.apply({ type: 'clear_all', source: 'map.load' })

    const blocks = (data.blocks || []).flatMap((b: any) => {
      const type = BlockType[b.blockType as keyof typeof BlockType]
      const position = b.position as Vec3
      if (!this.inBounds(position)) return []
      return [{
        position: [...position] as Vec3,
        blockType: type ?? BlockType.Air
      }]
    })
    if (blocks.length) {
      this.world.apply({ type: 'set_blocks', edits: blocks, source: 'map.load' })
    }

    this.activeSequence = sequence
    return { blocks, worldScale: this.worldScale }
  }

  async loadFromFile(jsonContent: string, BlockType: any) {
    const data = JSON.parse(jsonContent)
    if (data.worldScale) {
      this.worldScale = data.worldScale
      this.world.setWorldScale(data.worldScale)
    }
    this.world.apply({ type: 'clear_all', source: 'map.loadFromFile' })

    const blocks = (data.blocks || []).flatMap((b: any) => {
      const type = BlockType[b.blockType as keyof typeof BlockType]
      const position = b.position as Vec3
      if (!this.inBounds(position)) return []
      return [{
        position: [...position] as Vec3,
        blockType: type ?? BlockType.Air
      }]
    })
    if (blocks.length) {
      this.world.apply({ type: 'set_blocks', edits: blocks, source: 'map.loadFromFile' })
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
      this.world.apply({ type: 'clear_all', source: 'map.createNew' })
      this.activeSequence = null
      return []
    }
    return []
  }

  async loadFirstAvailable(BlockType: any, worldConfig: WorldConfig) {
    try {
      const res = await fetch(`${API_BASE_URL}/api/maps`)
      if (!res.ok) {
        this.generateDefaultTerrain(worldConfig)
        return null
      }
      const data = await res.json()
      const maps = data.maps || []
      if (maps.length > 0) {
        return await this.load(maps[0].sequence, BlockType)
      }
      this.generateDefaultTerrain(worldConfig)
      return null
    } catch {
      this.generateDefaultTerrain(worldConfig)
      return null
    }
  }

  private generateDefaultTerrain(worldConfig: WorldConfig) {
    const origin = this.world.getChunkOriginOffset()
    const scale = this.world.getWorldScale()
    const { x: sx, y: sy, z: sz } = this.chunk.size
    const max: Vec3 = [
      origin[0] + (sx - 1) * scale,
      origin[1] + (sy - 1) * scale,
      origin[2] + (sz - 1) * scale
    ]
    const params: TerrainGenerateParams = {
      action: 'generate',
      region: { min: origin, max },
      profile: 'rolling_hills',
      selectionType: 'default',
      params: {
        seed: worldConfig.seed,
        amplitude: 10,
        roughness: 2.4,
        elevation: 0.35
      }
    }
    this.world.apply({ type: 'terrain_region', params, source: 'map.defaultTerrain' })
  }

  private inBounds([x, y, z]: Vec3) {
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
