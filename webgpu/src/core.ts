// Core types, blocks, chunks, and utilities - consolidated module

import type { Writable } from 'svelte/store'
import { writable } from 'svelte/store'

// ============================================================================
// API CONFIGURATION
// ============================================================================

export const API_BASE_URL = 'http://localhost:8000'
export const TEXTURES_ENDPOINT = `${API_BASE_URL}/api/textures`
export const GENERATE_TILE_ENDPOINT = `${API_BASE_URL}/api/generate-tile`
export const BLOCKS_ENDPOINT = `${API_BASE_URL}/api/blocks`
export const TILE_BASE_URL = `${API_BASE_URL}/textures`

export const backendConfig: Writable<{
  mode: 'dev' | 'prod'
  requiresApiKey: boolean
}> = writable({
  mode: 'prod',
  requiresApiKey: true
})

export const openaiApiKey: Writable<string | null> = writable(null)

// ============================================================================
// BLOCK TYPES & PALETTE
// ============================================================================

export enum BlockType {
  Air = 0, Grass = 1, Dirt = 2, Stone = 3, Plank = 4, Snow = 5, Sand = 6, Water = 7,
  AlpineRock = 8, AlpineGrass = 9, GlacierIce = 10, Gravel = 11
}

export type BlockFaceKey = 'top' | 'bottom' | 'north' | 'south' | 'east' | 'west'
export const blockFaceOrder: BlockFaceKey[] = ['top', 'bottom', 'north', 'south', 'east', 'west']

type RGB = [number, number, number]
type Palette = { top: RGB; bottom: RGB; side: RGB }

export const blockPalette: Record<BlockType, Palette | undefined> = {
  [BlockType.Air]: undefined,
  [BlockType.Grass]: { top: [0.34, 0.68, 0.36], bottom: [0.40, 0.30, 0.16], side: [0.45, 0.58, 0.30] },
  [BlockType.Dirt]: { top: [0.42, 0.32, 0.20], bottom: [0.38, 0.26, 0.16], side: [0.40, 0.30, 0.18] },
  [BlockType.Stone]: { top: [0.58, 0.60, 0.64], bottom: [0.55, 0.57, 0.60], side: [0.56, 0.58, 0.62] },
  [BlockType.Plank]: { top: [0.78, 0.68, 0.50], bottom: [0.72, 0.60, 0.42], side: [0.74, 0.63, 0.45] },
  [BlockType.Snow]: { top: [0.92, 0.94, 0.96], bottom: [0.90, 0.92, 0.94], side: [0.88, 0.90, 0.93] },
  [BlockType.Sand]: { top: [0.88, 0.82, 0.60], bottom: [0.86, 0.78, 0.56], side: [0.87, 0.80, 0.58] },
  [BlockType.Water]: { top: [0.22, 0.40, 0.66], bottom: [0.20, 0.34, 0.60], side: [0.20, 0.38, 0.64] },
  [BlockType.AlpineRock]: { top: [0.45, 0.48, 0.52], bottom: [0.43, 0.46, 0.5], side: [0.44, 0.47, 0.51] },
  [BlockType.AlpineGrass]: { top: [0.26, 0.58, 0.32], bottom: [0.22, 0.44, 0.28], side: [0.24, 0.50, 0.30] },
  [BlockType.GlacierIce]: { top: [0.78, 0.88, 0.96], bottom: [0.72, 0.82, 0.90], side: [0.74, 0.84, 0.92] },
  [BlockType.Gravel]: { top: [0.52, 0.52, 0.5], bottom: [0.48, 0.48, 0.46], side: [0.5, 0.5, 0.48] }
}

export const availableBlocks = [
  { type: BlockType.Grass, name: 'Grass' },
  { type: BlockType.Dirt, name: 'Dirt' },
  { type: BlockType.Stone, name: 'Stone' },
  { type: BlockType.Plank, name: 'Plank' },
  { type: BlockType.Snow, name: 'Snow' },
  { type: BlockType.Sand, name: 'Sand' },
  { type: BlockType.Water, name: 'Water' },
  { type: BlockType.AlpineRock, name: 'Alpine Rock' },
  { type: BlockType.AlpineGrass, name: 'Alpine Grass' },
  { type: BlockType.Gravel, name: 'Gravel' },
  { type: BlockType.GlacierIce, name: 'Glacier Ice' }
]

// ============================================================================
// CUSTOM BLOCKS & STORES
// ============================================================================

export interface FaceTileInfo {
  sequence: number
  path: string
  url: string
  prompt: string
}

export interface CustomBlock {
  id: number
  name: string
  colors: { top: RGB; bottom: RGB; side: RGB }
  faceBitmaps?: Partial<Record<BlockFaceKey, ImageBitmap>>
  faceTiles?: Partial<Record<BlockFaceKey, FaceTileInfo>>
  remoteId?: number
  textureLayer?: number
}

export interface TerrainGenerateParams {
  action: 'generate' | 'preview' | 'clear'
  region: { min: [number, number, number]; max: [number, number, number] }
  selectionType: 'ellipsoid' | 'plane' | 'default'
  params: {
    amplitude: number
    roughness: number
    elevation: number
    seed?: number
  }
  ellipsoidMask?: {
    center: [number, number, number]
    radiusX: number
    radiusY: number
    radiusZ: number
  }
}

export interface GPUHooks {
  requestFaceBitmaps: ((tiles: Partial<Record<BlockFaceKey, FaceTileInfo>>) => Promise<Record<BlockFaceKey, ImageBitmap>>) | null
  uploadFaceBitmapsToGPU: ((bitmaps: Record<BlockFaceKey, ImageBitmap>, customBlock: CustomBlock) => void) | null
  getCameraPosition: (() => [number, number, number] | null) | null
  getWorldScale: (() => number) | null
  chunkToWorld: ((chunkCoord: Vec3) => Vec3) | null
  worldToChunk: ((worldCoord: Vec3) => Vec3) | null
  generateTerrain: ((params: TerrainGenerateParams) => void) | null
}

export type InteractionMode = 'block' | 'highlight'
export type HighlightShape = 'cube' | 'sphere' | 'ellipsoid' | 'plane'

export interface HighlightSelection {
  center: [number, number, number]
  radius: number
  radiusX?: number  // For ellipsoid
  radiusY?: number  // For ellipsoid
  radiusZ?: number  // For ellipsoid
  planeSizeX?: number  // For plane - size in X direction
  planeSizeZ?: number  // For plane - size in Z direction
  planeHeight?: number  // For plane - vertical extent (default 64)
  shape: HighlightShape
}

// Stores
export const selectedBlockType: Writable<BlockType> = writable(BlockType.Plank)
export const selectedCustomBlock: Writable<CustomBlock | null> = writable(null)
export const customBlocks: Writable<CustomBlock[]> = writable([])
export const selectedFace: Writable<BlockFaceKey | null> = writable(null)
export const texturePrompt: Writable<string> = writable('')
export const terrainProfile: Writable<'rolling_hills' | 'mountain' | 'hybrid'> = writable('rolling_hills')
export const terrainSeed: Writable<number> = writable(1337)
export const terrainAmplitude: Writable<number> = writable(10)
export const terrainRoughness: Writable<number> = writable(2.4)
export const terrainElevation: Writable<number> = writable(0.35)
export const gpuHooks: Writable<GPUHooks> = writable({
  requestFaceBitmaps: null,
  uploadFaceBitmapsToGPU: null,
  getCameraPosition: null,
  getWorldScale: null,
  chunkToWorld: null,
  worldToChunk: null,
  generateTerrain: null
})
export const interactionMode: Writable<InteractionMode> = writable('highlight')
export const highlightShape: Writable<HighlightShape> = writable('plane')
export const highlightRadius: Writable<number> = writable(2)
export const ellipsoidRadiusX: Writable<number> = writable(4)
export const ellipsoidRadiusY: Writable<number> = writable(2)
export const ellipsoidRadiusZ: Writable<number> = writable(4)
export const ellipsoidEditAxis: Writable<'x' | 'y' | 'z' | null> = writable(null)
export const planeSizeX: Writable<number> = writable(8)  // Size of plane in X direction
export const planeSizeZ: Writable<number> = writable(8)  // Size of plane in Z direction
export const planeHeight: Writable<number> = writable(64)  // Height of plane (vertical extent)
export type EllipsoidNode = '+x' | '-x' | '+y' | '-y' | '+z' | '-z' | 'center' | null
export const ellipsoidSelectedNode: Writable<EllipsoidNode> = writable(null)
export const highlightSelection: Writable<HighlightSelection | null> = writable(null)
export const cameraMode: Writable<'player' | 'overview'> = writable('overview')

// === V2 STORES ===
import type { MaterialParams, LightingParams, PointLight } from './dsl/commands'

// Material registry (blockType -> MaterialParams)
export const blockMaterials: Writable<Map<number, MaterialParams>> = writable(new Map())

// Scene lighting
function normalize(v: Vec3): Vec3 {
  const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
  return len > 0 ? [v[0] / len, v[1] / len, v[2] / len] : [0, 1, 0]
}

export const sceneLighting: Writable<LightingParams> = writable({
  sun: {
    direction: normalize([-0.4, -0.85, -0.5]),
    color: [1.0, 0.97, 0.9],
    intensity: 5.0  // Well-balanced brightness
  },
  sky: {
    zenithColor: [0.53, 0.81, 0.92],
    horizonColor: [0.8, 0.85, 0.9],
    groundColor: [0.18, 0.16, 0.12],
    intensity: 1.3  // Moderate sky contribution
  },
  ambient: {
    color: [1, 1, 1],
    intensity: 1.3  // Good ambient light
  }
})

// Point lights registry (id -> PointLight)
export const pointLights: Writable<Map<string, PointLight>> = writable(new Map())

// ============================================================================
// CHUNK MANAGER
// ============================================================================

export type ChunkDimensions = { x: number; y: number; z: number }
export type BlockTextureFaceIndices = Record<BlockFaceKey, number>

export class ChunkManager {
  readonly size: ChunkDimensions
  private blocks: Uint8Array

  constructor(size: ChunkDimensions = { x: 32, y: 32, z: 32 }) {
    this.size = size
    this.blocks = new Uint8Array(size.x * size.y * size.z)
  }

  getBlock(x: number, y: number, z: number): BlockType {
    if (!this.inBounds(x, y, z)) return BlockType.Air
    return this.blocks[this.index(x, y, z)] as BlockType
  }

  setBlock(x: number, y: number, z: number, type: BlockType) {
    if (this.inBounds(x, y, z)) this.blocks[this.index(x, y, z)] = type
  }

  private inBounds(x: number, y: number, z: number) {
    return x >= 0 && y >= 0 && z >= 0 && x < this.size.x && y < this.size.y && z < this.size.z
  }

  private index(x: number, y: number, z: number) {
    return x + this.size.x * (z + this.size.z * y)
  }
}

// ============================================================================
// MESH BUILDER - Compact face definitions
// ============================================================================

type Face = { n: [number, number, number]; o: [number, number, number]; c: [[number, number, number], [number, number, number], [number, number, number], [number, number, number]]; s: 'top' | 'bottom' | 'side' }

// Compact face definitions: [normal, offset, corners, colorSlot]
const faces: [Face, BlockFaceKey, [number, number][]][] = [
  [{ n: [1, 0, 0], o: [1, 0, 0], c: [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]], s: 'side' }, 'east', [[0, 1], [0, 0], [1, 0], [1, 1]]],
  [{ n: [-1, 0, 0], o: [-1, 0, 0], c: [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], s: 'side' }, 'west', [[1, 1], [0, 1], [0, 0], [1, 0]]],
  [{ n: [0, 1, 0], o: [0, 1, 0], c: [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]], s: 'top' }, 'top', [[0, 1], [0, 0], [1, 0], [1, 1]]],
  [{ n: [0, -1, 0], o: [0, -1, 0], c: [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], s: 'bottom' }, 'bottom', [[0, 0], [1, 0], [1, 1], [0, 1]]],
  [{ n: [0, 0, 1], o: [0, 0, 1], c: [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], s: 'side' }, 'south', [[0, 1], [1, 1], [1, 0], [0, 0]]],
  [{ n: [0, 0, -1], o: [0, 0, -1], c: [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], s: 'side' }, 'north', [[1, 1], [1, 0], [0, 0], [0, 1]]]
]

const indices = [0, 1, 2, 0, 2, 3]
const blockTextureLayers: Partial<Record<BlockType, BlockTextureFaceIndices>> = {}

export function setBlockTextureIndices(block: BlockType, config: BlockTextureFaceIndices | null) {
  if (!config) delete blockTextureLayers[block]
  else blockTextureLayers[block] = { ...config }
}

export function buildChunkMesh(chunk: ChunkManager, worldScale = 1) {
  const vertices: number[] = []
  const { x: sx, y: sy, z: sz } = chunk.size
  const ox = -sx / 2, oz = -sz / 2

  for (let y = 0; y < sy; y++) {
    for (let z = 0; z < sz; z++) {
      for (let x = 0; x < sx; x++) {
        const block = chunk.getBlock(x, y, z)
        if (block === BlockType.Air) continue
        const palette = blockPalette[block]!
        const texConfig = blockTextureLayers[block]

        for (const [face, faceKey, uvs] of faces) {
          const [nx, ny, nz] = [x + face.o[0], y + face.o[1], z + face.o[2]]
          if (chunk.getBlock(nx, ny, nz) !== BlockType.Air) continue

          const color = palette[face.s]
          const texLayer = texConfig?.[faceKey] ?? -1
          const bx = x + ox, by = y, bz = z + oz

          for (const i of indices) {
            const [cx, cy, cz] = face.c[i]!
            const [u, v] = uvs[i]!
            vertices.push(
              (bx + cx) * worldScale, (by + cy) * worldScale, (bz + cz) * worldScale,
              face.n[0], face.n[1], face.n[2],
              color[0], color[1], color[2],
              u, v, texLayer, block
            )
          }
        }
      }
    }
  }

  return { vertexData: new Float32Array(vertices), vertexCount: vertices.length / 13 }
}

// ============================================================================
// CAMERA UTILITIES
// ============================================================================

export type Mat4 = Float32Array
export type Vec3 = [number, number, number]

export function createPerspective(fovYRad: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovYRad / 2), nf = 1 / (near - far), out = new Float32Array(16)
  out[0] = f / aspect; out[5] = f; out[10] = (far + near) * nf; out[11] = -1; out[14] = 2 * far * near * nf
  return out
}

export function lookAt(eye: Vec3, center: Vec3, up: Vec3): Mat4 {
  const [ex, ey, ez] = eye, [cx, cy, cz] = center
  let zx = ex - cx, zy = ey - cy, zz = ez - cz
  let len = Math.hypot(zx, zy, zz)
  zx /= len; zy /= len; zz /= len
  let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx
  len = Math.hypot(xx, xy, xz)
  xx /= len; xy /= len; xz /= len
  const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx
  const out = new Float32Array(16)
  out[0] = xx; out[1] = yx; out[2] = zx; out[4] = xy; out[5] = yy; out[6] = zy
  out[8] = xz; out[9] = yz; out[10] = zz
  out[12] = -(xx * ex + xy * ey + xz * ez)
  out[13] = -(yx * ex + yy * ey + yz * ez)
  out[14] = -(zx * ex + zy * ey + zz * ez)
  out[15] = 1
  return out
}

export function multiplyMat4(a: Mat4, b: Mat4): Mat4 {
  const out = new Float32Array(16)
  for (let i = 0; i < 4; i++) {
    const ai0 = a[i]!, ai1 = a[i + 4]!, ai2 = a[i + 8]!, ai3 = a[i + 12]!
    out[i] = ai0 * b[0]! + ai1 * b[1]! + ai2 * b[2]! + ai3 * b[3]!
    out[i + 4] = ai0 * b[4]! + ai1 * b[5]! + ai2 * b[6]! + ai3 * b[7]!
    out[i + 8] = ai0 * b[8]! + ai1 * b[9]! + ai2 * b[10]! + ai3 * b[11]!
    out[i + 12] = ai0 * b[12]! + ai1 * b[13]! + ai2 * b[14]! + ai3 * b[15]!
  }
  return out
}

// ============================================================================
// UI UTILITIES - Isometric block drawing
// ============================================================================

export function drawIsometricBlock(canvas: HTMLCanvasElement, colors: { top: RGB; side: RGB; bottom: RGB }, bitmap?: Partial<Record<BlockFaceKey, ImageBitmap>>) {
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const w = canvas.width, h = canvas.height
  ctx.clearRect(0, 0, w, h)

  const size = Math.min(w, h) * 0.6, cx = w / 2, cy = h / 2 + 4
  const rgb = (c: RGB) => `rgb(${Math.floor(c[0] * 255)}, ${Math.floor(c[1] * 255)}, ${Math.floor(c[2] * 255)})`

  // Top face
  ctx.save()
  ctx.beginPath()
  ctx.moveTo(cx, cy - size / 2)
  ctx.lineTo(cx + size / 2, cy - size / 4)
  ctx.lineTo(cx, cy)
  ctx.lineTo(cx - size / 2, cy - size / 4)
  ctx.closePath()
  ctx.clip()
  if (bitmap?.top) ctx.drawImage(bitmap.top, 0, 0, bitmap.top.width, bitmap.top.height, cx - size / 2, cy - size / 2, size, size / 2)
  else { ctx.fillStyle = rgb(colors.top); ctx.fill() }
  ctx.restore()
  ctx.stroke()

  // Left face
  ctx.save()
  ctx.globalAlpha = 0.7
  ctx.beginPath()
  ctx.moveTo(cx - size / 2, cy - size / 4)
  ctx.lineTo(cx, cy)
  ctx.lineTo(cx, cy + size / 2)
  ctx.lineTo(cx - size / 2, cy + size / 4)
  ctx.closePath()
  ctx.clip()
  const bmpLeft = bitmap?.west ?? bitmap?.east ?? bitmap?.north ?? bitmap?.south
  if (bmpLeft) ctx.drawImage(bmpLeft, 0, 0, bmpLeft.width, bmpLeft.height, cx - size / 2, cy - size / 4, size / 2, size)
  else { ctx.fillStyle = rgb(colors.side.map(c => c * 0.7) as RGB); ctx.fill() }
  ctx.restore()

  // Right face
  ctx.save()
  ctx.globalAlpha = 0.85
  ctx.beginPath()
  ctx.moveTo(cx + size / 2, cy - size / 4)
  ctx.lineTo(cx, cy)
  ctx.lineTo(cx, cy + size / 2)
  ctx.lineTo(cx + size / 2, cy + size / 4)
  ctx.closePath()
  ctx.clip()
  const bmpRight = bitmap?.south ?? bitmap?.north ?? bitmap?.east ?? bitmap?.west
  if (bmpRight) ctx.drawImage(bmpRight, 0, 0, bmpRight.width, bmpRight.height, cx, cy - size / 4, size / 2, size)
  else { ctx.fillStyle = rgb(colors.side.map(c => c * 0.85) as RGB); ctx.fill() }
  ctx.restore()
}

export async function fetchTileBitmap(url: string): Promise<ImageBitmap> {
  const res = await fetch(url, { mode: 'cors', credentials: 'omit' })
  if (!res.ok) throw new Error(`Fetch failed: ${res.status}`)
  const blob = await res.blob()
  if (!blob.type.startsWith('image/')) throw new Error(`Invalid type: ${blob.type}`)
  return await createImageBitmap(blob, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' })
}
