import { writable } from 'svelte/store'
import type { BlockFaceKey } from './chunks'
import { BlockType } from './chunks'

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
export const terrainProfile = writable<'rolling_hills' | 'mountain' | 'hybrid'>('rolling_hills')
export const terrainSeed = writable<number>(1337)
export const terrainAmplitude = writable<number>(10)
export const terrainRoughness = writable<number>(2.4)
export const terrainElevation = writable<number>(0.35)

// GPU hooks (set by WebGPU engine)
export interface GPUHooks {
  requestFaceBitmaps: ((tiles: Partial<Record<BlockFaceKey, FaceTileInfo>>) => Promise<Record<BlockFaceKey, ImageBitmap>>) | null
  uploadFaceBitmapsToGPU: ((bitmaps: Record<BlockFaceKey, ImageBitmap>, customBlock: CustomBlock) => void) | null
}

export const gpuHooks = writable<GPUHooks>({
  requestFaceBitmaps: null,
  uploadFaceBitmapsToGPU: null
})

export type InteractionMode = 'block' | 'highlight'

export const interactionMode = writable<InteractionMode>('block')

export type HighlightShape = 'cube' | 'sphere'

export interface HighlightSelection {
  center: [number, number, number]
  radius: number
  shape: HighlightShape
}

export const highlightShape = writable<HighlightShape>('cube')
export const highlightRadius = writable<number>(2)
export const highlightSelection = writable<HighlightSelection | null>(null)
