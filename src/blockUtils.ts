import type { BlockFaceKey } from './chunks'
import { BlockType } from './chunks'
import type { CustomBlock, FaceTileInfo } from './stores'

// API endpoints
export const API_BASE_URL = 'http://localhost:8000'
export const TEXTURES_ENDPOINT = `${API_BASE_URL}/api/textures`
export const GENERATE_TILE_ENDPOINT = `${API_BASE_URL}/api/generate-tile`
export const BLOCKS_ENDPOINT = `${API_BASE_URL}/api/blocks`
export const TILE_BASE_URL = `${API_BASE_URL}/textures`

// Block face configuration
export const blockFaceOrder: BlockFaceKey[] = ['top', 'bottom', 'north', 'south', 'east', 'west']

export const faceLayerIndex: Record<BlockFaceKey, number> = blockFaceOrder.reduce((acc, face, index) => {
  acc[face] = index
  return acc
}, {} as Record<BlockFaceKey, number>)

// Block palette for default blocks
export const blockPalette: Record<BlockType, { top: number[], bottom: number[], side: number[] } | undefined> = {
  [BlockType.Air]: undefined,
  [BlockType.Grass]: { top: [0.34, 0.68, 0.36], bottom: [0.40, 0.30, 0.16], side: [0.45, 0.58, 0.30] },
  [BlockType.Dirt]: { top: [0.42, 0.32, 0.20], bottom: [0.38, 0.26, 0.16], side: [0.40, 0.30, 0.18] },
  [BlockType.Stone]: { top: [0.58, 0.60, 0.64], bottom: [0.55, 0.57, 0.60], side: [0.56, 0.58, 0.62] },
  [BlockType.Plank]: { top: [0.78, 0.68, 0.50], bottom: [0.72, 0.60, 0.42], side: [0.74, 0.63, 0.45] },
  [BlockType.Snow]: { top: [0.92, 0.94, 0.96], bottom: [0.90, 0.92, 0.94], side: [0.88, 0.90, 0.93] },
  [BlockType.Sand]: { top: [0.88, 0.82, 0.60], bottom: [0.86, 0.78, 0.56], side: [0.87, 0.80, 0.58] },
  [BlockType.Water]: { top: [0.22, 0.40, 0.66], bottom: [0.20, 0.34, 0.60], side: [0.20, 0.38, 0.64] }
}

// Available block types (excluding Air)
export const availableBlocks = [
  { type: BlockType.Grass, name: 'Grass' },
  { type: BlockType.Dirt, name: 'Dirt' },
  { type: BlockType.Stone, name: 'Stone' },
  { type: BlockType.Plank, name: 'Plank' },
  { type: BlockType.Snow, name: 'Snow' },
  { type: BlockType.Sand, name: 'Sand' },
  { type: BlockType.Water, name: 'Water' }
]

// Utility functions for drawing isometric blocks
export function drawIsometricBlock(
  canvas: HTMLCanvasElement,
  colors: { top: number[], side: number[], bottom: number[] }
) {
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const w = canvas.width
  const h = canvas.height
  ctx.clearRect(0, 0, w, h)

  const blockSize = Math.min(w, h) * 0.6
  const cx = w / 2
  const cy = h / 2 + 4

  const toRGB = (c: number[]) => `rgb(${Math.floor((c[0] ?? 0) * 255)}, ${Math.floor((c[1] ?? 0) * 255)}, ${Math.floor((c[2] ?? 0) * 255)})`

  // Draw top face
  ctx.beginPath()
  ctx.moveTo(cx, cy - blockSize / 2)
  ctx.lineTo(cx + blockSize / 2, cy - blockSize / 4)
  ctx.lineTo(cx, cy)
  ctx.lineTo(cx - blockSize / 2, cy - blockSize / 4)
  ctx.closePath()
  ctx.fillStyle = toRGB(colors.top)
  ctx.fill()
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)'
  ctx.lineWidth = 1
  ctx.stroke()

  // Draw left face
  ctx.beginPath()
  ctx.moveTo(cx - blockSize / 2, cy - blockSize / 4)
  ctx.lineTo(cx, cy)
  ctx.lineTo(cx, cy + blockSize / 2)
  ctx.lineTo(cx - blockSize / 2, cy + blockSize / 4)
  ctx.closePath()
  const leftColor = colors.side.map(c => c * 0.7)
  ctx.fillStyle = toRGB(leftColor)
  ctx.fill()
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)'
  ctx.stroke()

  // Draw right face
  ctx.beginPath()
  ctx.moveTo(cx + blockSize / 2, cy - blockSize / 4)
  ctx.lineTo(cx, cy)
  ctx.lineTo(cx, cy + blockSize / 2)
  ctx.lineTo(cx + blockSize / 2, cy + blockSize / 4)
  ctx.closePath()
  const rightColor = colors.side.map(c => c * 0.85)
  ctx.fillStyle = toRGB(rightColor)
  ctx.fill()
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)'
  ctx.stroke()
}

export function drawIsometricBlockWithTexture(
  canvas: HTMLCanvasElement,
  customBlock: CustomBlock
) {
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const w = canvas.width
  const h = canvas.height
  ctx.clearRect(0, 0, w, h)

  const blockSize = Math.min(w, h) * 0.6
  const cx = w / 2
  const cy = h / 2 + 4

  const bitmapTop = customBlock.faceBitmaps?.top
  const bitmapWest = customBlock.faceBitmaps?.west ?? customBlock.faceBitmaps?.east ?? customBlock.faceBitmaps?.north ?? customBlock.faceBitmaps?.south
  const bitmapSouth = customBlock.faceBitmaps?.south ?? customBlock.faceBitmaps?.north ?? customBlock.faceBitmaps?.east ?? customBlock.faceBitmaps?.west

  const fillWithColor = (color: number[]) => {
    const rgb = `rgb(${Math.floor((color[0] ?? 0) * 255)}, ${Math.floor((color[1] ?? 0) * 255)}, ${Math.floor((color[2] ?? 0) * 255)})`
    ctx.fillStyle = rgb
    ctx.fill()
  }

  // Draw top face
  ctx.save()
  ctx.beginPath()
  ctx.moveTo(cx, cy - blockSize / 2)
  ctx.lineTo(cx + blockSize / 2, cy - blockSize / 4)
  ctx.lineTo(cx, cy)
  ctx.lineTo(cx - blockSize / 2, cy - blockSize / 4)
  ctx.closePath()
  ctx.clip()
  if (bitmapTop) {
    ctx.drawImage(bitmapTop, 0, 0, bitmapTop.width, bitmapTop.height, cx - blockSize / 2, cy - blockSize / 2, blockSize, blockSize / 2)
  } else {
    ctx.beginPath()
    ctx.moveTo(cx, cy - blockSize / 2)
    ctx.lineTo(cx + blockSize / 2, cy - blockSize / 4)
    ctx.lineTo(cx, cy)
    ctx.lineTo(cx - blockSize / 2, cy - blockSize / 4)
    ctx.closePath()
    fillWithColor(customBlock.colors.top)
  }
  ctx.restore()
  ctx.stroke()

  // Draw left face (darkened)
  ctx.save()
  ctx.globalAlpha = 0.7
  ctx.beginPath()
  ctx.moveTo(cx - blockSize / 2, cy - blockSize / 4)
  ctx.lineTo(cx, cy)
  ctx.lineTo(cx, cy + blockSize / 2)
  ctx.lineTo(cx - blockSize / 2, cy + blockSize / 4)
  ctx.closePath()
  ctx.clip()
  if (bitmapWest) {
    ctx.drawImage(bitmapWest, 0, 0, bitmapWest.width, bitmapWest.height, cx - blockSize / 2, cy - blockSize / 4, blockSize / 2, blockSize)
  } else {
    ctx.beginPath()
    ctx.moveTo(cx - blockSize / 2, cy - blockSize / 4)
    ctx.lineTo(cx, cy)
    ctx.lineTo(cx, cy + blockSize / 2)
    ctx.lineTo(cx - blockSize / 2, cy + blockSize / 4)
    ctx.closePath()
    fillWithColor(customBlock.colors.side.map(c => c * 0.7))
  }
  ctx.restore()

  // Draw right face (slightly darkened)
  ctx.save()
  ctx.globalAlpha = 0.85
  ctx.beginPath()
  ctx.moveTo(cx + blockSize / 2, cy - blockSize / 4)
  ctx.lineTo(cx, cy)
  ctx.lineTo(cx, cy + blockSize / 2)
  ctx.lineTo(cx + blockSize / 2, cy + blockSize / 4)
  ctx.closePath()
  ctx.clip()
  if (bitmapSouth) {
    ctx.drawImage(bitmapSouth, 0, 0, bitmapSouth.width, bitmapSouth.height, cx, cy - blockSize / 4, blockSize / 2, blockSize)
  } else {
    ctx.beginPath()
    ctx.moveTo(cx + blockSize / 2, cy - blockSize / 4)
    ctx.lineTo(cx, cy)
    ctx.lineTo(cx, cy + blockSize / 2)
    ctx.lineTo(cx + blockSize / 2, cy + blockSize / 4)
    ctx.closePath()
    fillWithColor(customBlock.colors.side.map(c => c * 0.85))
  }
  ctx.restore()
}

export async function fetchTileBitmap(url: string): Promise<ImageBitmap> {
  const response = await fetch(url, { mode: 'cors', credentials: 'omit' })
  if (!response.ok) {
    throw new Error(`Failed to fetch tile (${response.status}) from ${url}`)
  }
  const blob = await response.blob()
  if (!blob.type.startsWith('image/')) {
    throw new Error(`Unexpected content type for tile: ${blob.type}`)
  }
  return await createImageBitmap(blob, {
    premultiplyAlpha: 'none',
    colorSpaceConversion: 'none'
  })
}
