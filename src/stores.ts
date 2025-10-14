import { writable } from 'svelte/store'
import type { BlockFaceKey } from './chunks'
import { BlockType } from './chunks'

export interface TextureAtlasInfo {
  rows: number
  cols: number
  tileSize: number
  column: number
  sequence?: number
}

export interface FaceTileInfo {
  sequence: number
  path: string
  url: string
  prompt: string
}

export interface CustomBlock {
  id: number
  name: string
  colors: { top: number[], bottom: number[], side: number[] }
  faceBitmaps?: Partial<Record<BlockFaceKey, ImageBitmap>>
  faceTiles?: Partial<Record<BlockFaceKey, FaceTileInfo>>
  remoteId?: number
  textureLayer?: number
}

// Block selection state
export const selectedBlockType = writable<BlockType>(BlockType.Plank)
export const selectedCustomBlock = writable<CustomBlock | null>(null)
export const customBlocks = writable<CustomBlock[]>([])

// Face selection state
export const selectedFace = writable<BlockFaceKey | null>(null)

// Texture generation state
export const texturePrompt = writable<string>('')

// GPU hooks (set by WebGPU engine)
export interface GPUHooks {
  requestFaceBitmaps: ((tiles: Partial<Record<BlockFaceKey, FaceTileInfo>>) => Promise<Record<BlockFaceKey, ImageBitmap>>) | null
  uploadFaceBitmapsToGPU: ((bitmaps: Record<BlockFaceKey, ImageBitmap>, customBlock: CustomBlock) => void) | null
}

export const gpuHooks = writable<GPUHooks>({
  requestFaceBitmaps: null,
  uploadFaceBitmapsToGPU: null
})

// ID counters
export const nextCustomBlockId = writable<number>(1000)
export const nextTextureLayer = writable<number>(0)

// Custom texture layers map
export const customTextureLayers = writable<Map<number, Record<BlockFaceKey, ImageBitmap>>>(new Map())
