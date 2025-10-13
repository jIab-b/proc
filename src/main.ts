import terrainWGSL from './pipelines/render/terrain.wgsl?raw'
import { createPerspective, lookAt, multiplyMat4 } from './camera'
import { ChunkManager, BlockType, BlockFaceKey, buildChunkMesh, setBlockTextureIndices } from './chunks'
import { generateRollingHills, createSimpleWorldConfig } from './worldgen'

// Create sidebar
const sidebar = document.createElement('div')
sidebar.className = 'sidebar'

// Block selection area
const blockSelection = document.createElement('div')
blockSelection.className = 'block-selection'
const blockTitle = document.createElement('h3')
blockTitle.textContent = 'Block Selection'
blockSelection.appendChild(blockTitle)

// Custom block type interface
interface TextureAtlasInfo {
  rows: number
  cols: number
  tileSize: number
  column: number
  sequence?: number
}

interface CustomBlock {
  id: number
  name: string
  colors: { top: number[], bottom: number[], side: number[] }
  texture?: HTMLImageElement
  atlas?: TextureAtlasInfo
  faceBitmaps?: Partial<Record<BlockFaceKey, ImageBitmap>>
  textureId?: number // Store original server texture ID for deletion
  textureLayer?: number // GPU texture layer index for this custom block
}

type FaceBitmapMap = Partial<Record<BlockFaceKey, ImageBitmap>>

const blockFaceOrder: BlockFaceKey[] = ['top', 'bottom', 'north', 'south', 'east', 'west']
const faceLayerIndex: Record<BlockFaceKey, number> = blockFaceOrder.reduce((acc, face, index) => {
  acc[face] = index
  return acc
}, {} as Record<BlockFaceKey, number>)
const faceTileCoordinates: Record<BlockFaceKey, [number, number]> = {
  top: [1, 1],
  bottom: [1, 3],
  north: [1, 0],
  south: [1, 2],
  east: [2, 1],
  west: [0, 1]
}

let requestFaceBitmaps: ((sequence: number) => Promise<Record<BlockFaceKey, ImageBitmap>>) | null = null
let uploadFaceBitmapsToGPU: ((bitmaps: Record<BlockFaceKey, ImageBitmap>, customBlock: CustomBlock) => void) | null = null

// Custom texture management
const customTextureLayers = new Map<number, Record<BlockFaceKey, ImageBitmap>>() // blockId -> face bitmaps
let nextTextureLayer = 0 // Next available GPU texture layer for custom blocks

// Available block types (excluding Air)
const availableBlocks = [
  { type: BlockType.Grass, name: 'Grass' },
  { type: BlockType.Dirt, name: 'Dirt' },
  { type: BlockType.Stone, name: 'Stone' },
  { type: BlockType.Plank, name: 'Plank' },
  { type: BlockType.Snow, name: 'Snow' },
  { type: BlockType.Sand, name: 'Sand' },
  { type: BlockType.Water, name: 'Water' }
]

// Custom blocks storage
const customBlocks: CustomBlock[] = []
let nextCustomBlockId = 1000 // Start custom IDs from 1000

let selectedBlockType = BlockType.Plank
let selectedCustomBlock: CustomBlock | null = null
const blockItems: HTMLDivElement[] = []

// Create block grid container
const blockGrid = document.createElement('div')
blockGrid.className = 'block-grid'

// Function to draw isometric block preview
function drawIsometricBlock(canvas: HTMLCanvasElement, colors: { top: number[], side: number[], bottom: number[] }) {
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const w = canvas.width
  const h = canvas.height
  ctx.clearRect(0, 0, w, h)

  // Isometric block dimensions
  const blockSize = Math.min(w, h) * 0.6
  const cx = w / 2
  const cy = h / 2 + 4

  // Helper to convert RGB array to CSS color
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
  const leftColor = colors.side.map(c => c * 0.7) // Darken left side
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
  const rightColor = colors.side.map(c => c * 0.85) // Slightly darken right side
  ctx.fillStyle = toRGB(rightColor)
  ctx.fill()
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)'
  ctx.stroke()
}

// Block palette for default blocks
const blockPalette: Record<BlockType, { top: number[], bottom: number[], side: number[] } | undefined> = {
  [BlockType.Air]: undefined,
  [BlockType.Grass]: { top: [0.34, 0.68, 0.36], bottom: [0.40, 0.30, 0.16], side: [0.45, 0.58, 0.30] },
  [BlockType.Dirt]: { top: [0.42, 0.32, 0.20], bottom: [0.38, 0.26, 0.16], side: [0.40, 0.30, 0.18] },
  [BlockType.Stone]: { top: [0.58, 0.60, 0.64], bottom: [0.55, 0.57, 0.60], side: [0.56, 0.58, 0.62] },
  [BlockType.Plank]: { top: [0.78, 0.68, 0.50], bottom: [0.72, 0.60, 0.42], side: [0.74, 0.63, 0.45] },
  [BlockType.Snow]: { top: [0.92, 0.94, 0.96], bottom: [0.90, 0.92, 0.94], side: [0.88, 0.90, 0.93] },
  [BlockType.Sand]: { top: [0.88, 0.82, 0.60], bottom: [0.86, 0.78, 0.56], side: [0.87, 0.80, 0.58] },
  [BlockType.Water]: { top: [0.22, 0.40, 0.66], bottom: [0.20, 0.34, 0.60], side: [0.20, 0.38, 0.64] }
}

function renderBlockGrid() {
  blockGrid.innerHTML = ''
  blockItems.length = 0

  // Render default blocks
  availableBlocks.forEach(({ type, name }) => {
    const blockItem = document.createElement('div')
    blockItem.className = 'block-item'
    if (type === selectedBlockType && !selectedCustomBlock) blockItem.classList.add('selected')

    const previewCanvas = document.createElement('canvas')
    previewCanvas.className = 'block-preview-3d'
    previewCanvas.width = 48
    previewCanvas.height = 48

    const palette = blockPalette[type]
    if (palette) {
      drawIsometricBlock(previewCanvas, palette)
    }

    const blockName = document.createElement('span')
    blockName.textContent = name

    blockItem.appendChild(previewCanvas)
    blockItem.appendChild(blockName)

    blockItem.onclick = (e) => {
      e.stopPropagation()
      blockItems.forEach(item => item.classList.remove('selected'))
      blockItem.classList.add('selected')
      selectedBlockType = type
      selectedCustomBlock = null
      const paletteForPreview = blockPalette[type]
      if (paletteForPreview) updateFacePreviews(paletteForPreview)
    }
    blockItem.addEventListener('mousedown', (e) => e.stopPropagation())

    blockItems.push(blockItem)
    blockGrid.appendChild(blockItem)
  })

  // Render custom blocks
  customBlocks.forEach((customBlock) => {
    const blockItem = document.createElement('div')
    blockItem.className = 'block-item'
    if (selectedCustomBlock?.id === customBlock.id) blockItem.classList.add('selected')

    const previewCanvas = document.createElement('canvas')
    previewCanvas.className = 'block-preview-3d'
    previewCanvas.width = 48
    previewCanvas.height = 48

    if (customBlock.texture) {
      drawIsometricBlockWithTexture(previewCanvas, customBlock.texture, customBlock.atlas, customBlock.faceBitmaps)
    } else {
      drawIsometricBlock(previewCanvas, customBlock.colors)
    }

    const blockName = document.createElement('span')
    blockName.textContent = customBlock.name

    blockItem.appendChild(previewCanvas)
    blockItem.appendChild(blockName)

    blockItem.onclick = (e) => {
      e.stopPropagation()

      // Check if Shift key is held for delete
      if (e.shiftKey) {
        const confirmDelete = confirm(`Delete "${customBlock.name}"? This will also remove the texture files.`)
        if (confirmDelete) {
          deleteCustomBlock(customBlock)
        }
        return
      }

      blockItems.forEach(item => item.classList.remove('selected'))
      blockItem.classList.add('selected')
      selectedCustomBlock = customBlock
      selectedBlockType = BlockType.Plank // Use any default as placeholder
      updateFacePreviewsWithTexture(customBlock)
    }
    blockItem.addEventListener('mousedown', (e) => e.stopPropagation())

    blockItems.push(blockItem)
    blockGrid.appendChild(blockItem)
  })

  // Add "Add New Block" button
  const addBlockBtn = document.createElement('div')
  addBlockBtn.className = 'block-item add-block-btn'
  const icon = document.createElement('div')
  icon.className = 'icon'
  icon.textContent = '+'
  const label = document.createElement('span')
  label.textContent = 'Add Block'
  addBlockBtn.appendChild(icon)
  addBlockBtn.appendChild(label)

  addBlockBtn.onclick = (e) => {
    e.stopPropagation()
    const blockName = prompt('Enter block name:')
    if (blockName && blockName.trim()) {
      const newBlock: CustomBlock = {
        id: nextCustomBlockId++,
        name: blockName.trim(),
        colors: { top: [0.5, 0.5, 0.5], bottom: [0.4, 0.4, 0.4], side: [0.45, 0.45, 0.45] }
      }
      customBlocks.push(newBlock)
      renderBlockGrid()
      console.log('Created custom block:', blockName)
    }
  }
  addBlockBtn.addEventListener('mousedown', (e) => e.stopPropagation())

  blockGrid.appendChild(addBlockBtn)
}

blockSelection.appendChild(blockGrid)

// Add helper text
const helperText = document.createElement('div')
helperText.className = 'helper-text'
helperText.textContent = 'Shift+Click on custom tiles to delete them'
blockSelection.appendChild(helperText)

renderBlockGrid()

// Function to delete a custom block
async function deleteCustomBlock(customBlock: CustomBlock) {
  try {
    // If it has a textureId, try to delete from server
    if (customBlock.textureId) {
      const response = await fetch(`http://localhost:8000/api/textures/${customBlock.textureId}`, {
        method: 'DELETE'
      })

      if (!response.ok) {
        console.warn(`Failed to delete texture from server: ${response.status}`)
        // Continue with local deletion even if server deletion fails
      } else {
        console.log(`Deleted texture ${customBlock.textureId} from server`)
      }
    }

    // Clean up face bitmaps
    if (customBlock.faceBitmaps) {
      blockFaceOrder.forEach(face => {
        const bitmap = customBlock.faceBitmaps![face]
        if (bitmap && bitmap.close) {
          bitmap.close()
        }
      })
    }

    // Remove from custom blocks array
    const index = customBlocks.findIndex(block => block.id === customBlock.id)
    if (index !== -1) {
      customBlocks.splice(index, 1)
    }

    // Clear selection if this block was selected
    if (selectedCustomBlock?.id === customBlock.id) {
      selectedCustomBlock = null
      selectedBlockType = BlockType.Plank
      const initialPalette = blockPalette[selectedBlockType]
      if (initialPalette) updateFacePreviews(initialPalette)
    }

    // Re-render block grid
    renderBlockGrid()

    console.log(`Deleted custom block: ${customBlock.name}`)
  } catch (err) {
    console.error('Error deleting custom block:', err)
    alert('Failed to delete texture. Please try again.')
  }
}

sidebar.appendChild(blockSelection)

// Face viewer area
const faceViewer = document.createElement('div')
faceViewer.className = 'face-viewer'
const faceTitle = document.createElement('h3')
faceTitle.textContent = 'Block Faces'
faceViewer.appendChild(faceTitle)

const faceGrid = document.createElement('div')
faceGrid.className = 'face-grid'
const faceLabels = ['Top (+Y)', 'Bottom (-Y)', 'Front (+Z)', 'Back (-Z)', 'Right (+X)', 'Left (-X)']
const facePreviews: HTMLCanvasElement[] = []
faceLabels.forEach(label => {
  const faceBox = document.createElement('div')
  faceBox.className = 'face-box'
  const faceLabel = document.createElement('label')
  faceLabel.textContent = label
  const facePreview = document.createElement('canvas')
  facePreview.className = 'face-preview'
  facePreview.width = 64
  facePreview.height = 64
  faceBox.appendChild(faceLabel)
  faceBox.appendChild(facePreview)
  faceGrid.appendChild(faceBox)
  facePreviews.push(facePreview)
})
faceViewer.appendChild(faceGrid)

// Function to draw isometric block with texture
function drawIsometricBlockWithTexture(
  canvas: HTMLCanvasElement,
  texture: HTMLImageElement,
  atlas?: TextureAtlasInfo,
  faceBitmaps?: FaceBitmapMap
) {
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const w = canvas.width
  const h = canvas.height
  ctx.clearRect(0, 0, w, h)

  const blockSize = Math.min(w, h) * 0.6
  const cx = w / 2
  const cy = h / 2 + 4

  const bitmapTop = faceBitmaps?.top
  const bitmapWest = faceBitmaps?.west
  const bitmapSouth = faceBitmaps?.south
  const useBitmaps = Boolean(bitmapTop && bitmapWest && bitmapSouth)

  const cols = Math.max(1, atlas ? atlas.cols : 1)
  const rows = Math.max(1, atlas ? atlas.rows : 1)
  const column = atlas ? atlas.column : 0
  const cellWidth = texture.width / cols
  const cellHeight = texture.height / rows

  const getSource = (row: number) => {
    const clampedRow = Math.min(rows - 1, Math.max(0, row))
    const clampedCol = Math.min(cols - 1, Math.max(0, column))
    return {
      sx: clampedCol * cellWidth,
      sy: clampedRow * cellHeight,
      sw: cellWidth,
      sh: cellHeight
    }
  }

  // Draw top face with texture
  ctx.save()
  ctx.beginPath()
  ctx.moveTo(cx, cy - blockSize / 2)
  ctx.lineTo(cx + blockSize / 2, cy - blockSize / 4)
  ctx.lineTo(cx, cy)
  ctx.lineTo(cx - blockSize / 2, cy - blockSize / 4)
  ctx.closePath()
  ctx.clip()
  if (useBitmaps && bitmapTop) {
    ctx.drawImage(
      bitmapTop,
      0,
      0,
      bitmapTop.width,
      bitmapTop.height,
      cx - blockSize / 2,
      cy - blockSize / 2,
      blockSize,
      blockSize / 2
    )
  } else {
    const topSrc = getSource(0)
    ctx.drawImage(
      texture,
      topSrc.sx,
      topSrc.sy,
      topSrc.sw,
      topSrc.sh,
      cx - blockSize / 2,
      cy - blockSize / 2,
      blockSize,
      blockSize / 2
    )
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
  if (useBitmaps && bitmapWest) {
    ctx.drawImage(
      bitmapWest,
      0,
      0,
      bitmapWest.width,
      bitmapWest.height,
      cx - blockSize / 2,
      cy - blockSize / 4,
      blockSize / 2,
      blockSize
    )
  } else {
    const sideSrc = getSource(Math.min(rows - 1, 1))
    ctx.drawImage(
      texture,
      sideSrc.sx,
      sideSrc.sy,
      sideSrc.sw,
      sideSrc.sh,
      cx - blockSize / 2,
      cy - blockSize / 4,
      blockSize / 2,
      blockSize
    )
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
  if (useBitmaps && bitmapSouth) {
    ctx.drawImage(
      bitmapSouth,
      0,
      0,
      bitmapSouth.width,
      bitmapSouth.height,
      cx,
      cy - blockSize / 4,
      blockSize / 2,
      blockSize
    )
  } else {
    const sideAltSrc = getSource(rows > 2 ? Math.min(rows - 1, 2) : Math.min(rows - 1, 1))
    ctx.drawImage(
      texture,
      sideAltSrc.sx,
      sideAltSrc.sy,
      sideAltSrc.sw,
      sideAltSrc.sh,
      cx,
      cy - blockSize / 4,
      blockSize / 2,
      blockSize
    )
  }
  ctx.restore()
}

// Function to update face previews based on block palette
function updateFacePreviews(palette: { top: number[], bottom: number[], side: number[] }) {
  // Top, Bottom, Front (side), Back (side), Right (side), Left (side)
  const faceColors = [palette.top, palette.bottom, palette.side, palette.side, palette.side, palette.side]

  facePreviews.forEach((canvas, index) => {
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const color = faceColors[index]!
    const rgb = `rgb(${Math.floor((color[0] ?? 0) * 255)}, ${Math.floor((color[1] ?? 0) * 255)}, ${Math.floor((color[2] ?? 0) * 255)})`
    ctx.fillStyle = rgb
    ctx.fillRect(0, 0, canvas.width, canvas.height)
  })
}

// Function to update face previews with texture
function updateFacePreviewsWithTexture(customBlock: CustomBlock) {
  const previewFaces: BlockFaceKey[] = ['top', 'bottom', 'north', 'south', 'east', 'west']
  const faceBitmaps = customBlock.faceBitmaps
  if (faceBitmaps && previewFaces.every(face => Boolean(faceBitmaps[face]))) {
    facePreviews.forEach((canvas, index) => {
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      const face = previewFaces[index]
      if (!face) return
      const bitmap = faceBitmaps[face]
      if (!bitmap) return
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height)
    })
    return
  }

  const texture = customBlock.texture
  if (!texture) {
    updateFacePreviews(customBlock.colors)
    return
  }

  const atlas = customBlock.atlas
  const cols = Math.max(1, atlas ? atlas.cols : 1)
  const rows = Math.max(1, atlas ? atlas.rows : 1)
  const column = atlas ? atlas.column : 0
  const cellWidth = texture.width / cols
  const cellHeight = texture.height / rows

  const resolveRow = (faceIndex: number) => {
    if (faceIndex === 0) return 0 // Top
    if (faceIndex === 1) return rows - 1 // Bottom
    if (faceIndex === 2 || faceIndex === 3) return Math.min(rows - 1, 1) // Front/Back
    return Math.min(rows - 1, rows > 2 ? 2 : 1) // Right/Left
  }

  facePreviews.forEach((canvas, index) => {
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const row = Math.min(rows - 1, Math.max(0, resolveRow(index)))
    const sx = Math.min(cols - 1, Math.max(0, column)) * cellWidth
    const sy = row * cellHeight
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(texture, sx, sy, cellWidth, cellHeight, 0, 0, canvas.width, canvas.height)
  })
}

// Initialize face previews with default block
const initialPalette = blockPalette[selectedBlockType]
if (initialPalette) updateFacePreviews(initialPalette)

// Texture generation area
const textureGen = document.createElement('div')
textureGen.className = 'texture-gen'
const promptInput = document.createElement('input')
promptInput.type = 'text'
promptInput.placeholder = 'Enter texture prompt...'
// Prevent canvas pointer lock when clicking input
promptInput.addEventListener('mousedown', (e) => e.stopPropagation())
promptInput.addEventListener('click', (e) => {
  e.stopPropagation()
  promptInput.focus()
})
const genButton = document.createElement('button')
genButton.textContent = 'Generate Texture'
genButton.disabled = true
// Prevent canvas pointer lock when clicking button
genButton.addEventListener('mousedown', (e) => e.stopPropagation())
genButton.addEventListener('click', (e) => e.stopPropagation())
genButton.onclick = async () => {
  const prompt = promptInput.value.trim()
  if (!prompt) return
  genButton.disabled = true
  genButton.textContent = 'Generating...'
  try {
    console.log('Generating texture with prompt:', prompt)

    // Call FastAPI backend for texture generation
    const response = await fetch('http://localhost:8000/api/generate-texture', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, width: 192, height: 256 })
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(error.detail || `HTTP ${response.status}`)
    }

    // Get texture metadata from response headers
    const textureId = response.headers.get('X-Texture-ID')
    const texturePrompt = response.headers.get('X-Texture-Prompt')
    const atlasRows = Number.parseInt(response.headers.get('X-Atlas-Rows') ?? '4', 10)
    const atlasCols = Number.parseInt(response.headers.get('X-Atlas-Cols') ?? '3', 10)
    const atlasTile = Number.parseInt(response.headers.get('X-Atlas-Tile') ?? '64', 10)
    const atlasSequence = Number.parseInt(response.headers.get('X-Atlas-Sequence') ?? '-1', 10)

    const atlasInfo: TextureAtlasInfo = {
      rows: Number.isFinite(atlasRows) && atlasRows > 0 ? atlasRows : 4,
      cols: Number.isFinite(atlasCols) && atlasCols > 0 ? atlasCols : 3,
      tileSize: Number.isFinite(atlasTile) && atlasTile > 0 ? atlasTile : 64,
      column: 0,
      sequence: Number.isFinite(atlasSequence) && atlasSequence > 0 ? atlasSequence : undefined
    }

    console.log(`Texture saved with ID: ${textureId}, prompt: "${texturePrompt}"`)

    const blob = await response.blob()
    const imageUrl = URL.createObjectURL(blob)

    // Create image object for the texture
    const textureImg = new Image()
    textureImg.onload = async () => {
      const atlasConfig: TextureAtlasInfo = { ...atlasInfo }
      let targetBlock: CustomBlock

      if (selectedCustomBlock) {
        selectedCustomBlock.texture = textureImg
        selectedCustomBlock.atlas = atlasConfig
        targetBlock = selectedCustomBlock
        console.log(`Applied texture to custom block: ${selectedCustomBlock.name}`)
      } else {
        const blockName = (prompt.split(',')[0] ?? 'Custom').trim() || 'Custom Block'
        const newBlock: CustomBlock = {
          id: nextCustomBlockId++,
          name: blockName,
          colors: { top: [0.5, 0.5, 0.5], bottom: [0.4, 0.4, 0.4], side: [0.45, 0.45, 0.45] },
          texture: textureImg,
          atlas: atlasConfig
        }
        customBlocks.push(newBlock)
        selectedCustomBlock = newBlock
        selectedBlockType = BlockType.Plank
        targetBlock = newBlock
        console.log(`Created new custom block: ${blockName}`)
      }

      const sequence = atlasConfig.sequence
      if (typeof sequence === 'number' && Number.isFinite(sequence) && sequence > 0) {
        try {
          if (!requestFaceBitmaps) {
            console.warn('GPU texture loader not ready yet')
          } else {
            const bitmaps = await requestFaceBitmaps(sequence)
            const previous = targetBlock.faceBitmaps
            if (previous) {
              blockFaceOrder.forEach(face => previous[face]?.close?.())
            }
            targetBlock.faceBitmaps = bitmaps
            if (uploadFaceBitmapsToGPU) {
              uploadFaceBitmapsToGPU(bitmaps, targetBlock)
              console.log(`Loaded face tile set for sequence ${sequence}`)
            } else {
              console.warn('GPU upload hook missing; custom block will use palette colors')
            }
          }
        } catch (bitmapErr) {
          console.error('Failed to load per-face tiles:', bitmapErr)
        }
      }

      updateFacePreviewsWithTexture(targetBlock)
      renderBlockGrid()
      URL.revokeObjectURL(imageUrl)
    }
    textureImg.onerror = () => {
      console.error('Failed to load generated texture')
      URL.revokeObjectURL(imageUrl)
    }
    textureImg.src = imageUrl

    genButton.textContent = 'Generate Texture'
    console.log('Texture generation complete')
    if (atlasInfo.sequence) {
      console.log(`Texture tiles saved to textures/${atlasInfo.sequence}`)
    }
  } catch (err) {
    console.error('Texture generation failed:', err)
    genButton.textContent = 'Failed - Retry'
    alert(`Texture generation failed: ${err instanceof Error ? err.message : 'Unknown error'}\n\nMake sure the backend server is running:\npython3 aio.py`)
  } finally {
    genButton.disabled = false
  }
}
promptInput.addEventListener('input', () => {
  genButton.disabled = !promptInput.value.trim()
})
textureGen.appendChild(promptInput)
textureGen.appendChild(genButton)
faceViewer.appendChild(textureGen)

sidebar.appendChild(faceViewer)
document.body.insertBefore(sidebar, document.body.firstChild)

// Load existing textures from server on startup
async function loadExistingTextures() {
  try {
    const response = await fetch('http://localhost:8000/api/textures')
    if (!response.ok) {
      console.warn('Failed to load existing textures from server')
      return
    }

    const data = await response.json()
    const textures = data.textures || []

    console.log(`Loading ${textures.length} existing textures from server`)

    for (const texture of textures) {
      // Skip textures without sequence info (older format)
      if (!texture.sequence || !texture.atlas) {
        console.log(`Skipping texture ${texture.id} (no sequence info)`)
        continue
      }

      // Create custom block from existing texture
      const textureImg = new Image()
      textureImg.onload = async () => {
        const blockName = texture.prompt.split(',')[0]?.trim() || `Texture ${texture.id}`

        const customBlock: CustomBlock = {
          id: nextCustomBlockId++,
          name: blockName,
          colors: { top: [0.5, 0.5, 0.5], bottom: [0.4, 0.4, 0.4], side: [0.45, 0.45, 0.45] },
          texture: textureImg,
          atlas: {
            rows: texture.atlas.rows || 4,
            cols: texture.atlas.cols || 3,
            tileSize: texture.atlas.tile_size || 64,
            column: 0,
            sequence: texture.sequence
          },
          textureId: texture.id // Store original texture ID for deletion
        }

        // Load face bitmaps if available (with limit to prevent memory issues)
        const MAX_CUSTOM_BLOCKS = 8
        if (texture.sequence && requestFaceBitmaps && nextTextureLayer < MAX_CUSTOM_BLOCKS) {
          try {
            const bitmaps = await requestFaceBitmaps(texture.sequence)
            customBlock.faceBitmaps = bitmaps
            // Assign texture layer for existing texture
            customBlock.textureLayer = nextTextureLayer++
            customTextureLayers.set(customBlock.id, bitmaps)
            if (uploadFaceBitmapsToGPU) {
              uploadFaceBitmapsToGPU(bitmaps, customBlock)
              console.log(`Loaded face tiles for existing texture: ${blockName}`)
            }
          } catch (err) {
            console.warn(`Failed to load face tiles for ${blockName}:`, err)
          }
        } else if (nextTextureLayer >= MAX_CUSTOM_BLOCKS) {
          console.warn(`Maximum custom blocks (${MAX_CUSTOM_BLOCKS}) reached. Skipping texture: ${blockName}`)
        }

        customBlocks.push(customBlock)
        renderBlockGrid()
        console.log(`Added existing texture as block: ${blockName}`)
      }

      textureImg.onerror = () => {
        console.error(`Failed to load existing texture ${texture.id}`)
      }

      // Load the texture image with proper CORS handling
      textureImg.crossOrigin = "anonymous"
      textureImg.src = `http://localhost:8000/api/textures/${texture.id}`
    }

  } catch (err) {
    console.error('Error loading existing textures:', err)
  }
}

// Load existing textures after a short delay to ensure server is ready
setTimeout(() => {
  loadExistingTextures().catch(err => {
    console.error('Failed to load existing textures:', err)
    // Retry after 2 seconds
    setTimeout(loadExistingTextures, 2000)
  })
}, 1000)

// Create main app container
const root = document.getElementById('app') as HTMLDivElement
const canvasShell = document.createElement('div')
canvasShell.className = 'canvas-shell'
const canvas = document.createElement('canvas')
canvas.style.width = '100%'
canvas.style.height = '100%'
canvasShell.appendChild(canvas)
root.appendChild(canvasShell)

const logBuffer: string[] = []
const origConsole = { log: console.log, warn: console.warn, error: console.error }
function push(msg: string) { logBuffer.push(`[${new Date().toISOString()}] ${msg}`) }
console.log = (...args: any[]) => { origConsole.log.apply(console, args); push(args.map(String).join(' ')) }
console.warn = (...args: any[]) => { origConsole.warn.apply(console, args); push('WARN ' + args.map(String).join(' ')) }
console.error = (...args: any[]) => { origConsole.error.apply(console, args); push('ERROR ' + args.map(String).join(' ')) }

async function saveLog() {
  const text = logBuffer.join('\n')
  if ('showSaveFilePicker' in window) {
    try {
      // @ts-ignore
      const handle = await (window as any).showSaveFilePicker({ suggestedName: 'webgpu-log.txt', types: [{ accept: { 'text/plain': ['.txt'] } }] })
      const writable = await handle.createWritable()
      await writable.write(text)
      await writable.close()
      return
    } catch {}
  }
  const blob = new Blob([text], { type: 'text/plain' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'webgpu-log.txt'
  a.click()
  URL.revokeObjectURL(url)
}

function installLogUI() {
  const wrap = document.createElement('div')
  wrap.className = 'log-ui'
  const btnSave = document.createElement('button')
  btnSave.textContent = 'Download Log'
  btnSave.onclick = () => {
    push('Saving log via download')
    saveLog()
  }
  wrap.appendChild(btnSave)
  document.body.appendChild(wrap)
}

function log(msg: string) { console.error(msg) }

type Vec3 = [number, number, number]

function normalize(v: Vec3): Vec3 {
  const len = Math.hypot(v[0], v[1], v[2])
  if (len < 1e-5) return [0, 0, 0]
  return [v[0] / len, v[1] / len, v[2] / len]
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
  ]
}

function addScaled(target: Vec3, dir: Vec3, amt: number) {
  target[0] += dir[0] * amt
  target[1] += dir[1] * amt
  target[2] += dir[2] * amt
}

function alignTo(n: number, alignment: number) {
  return Math.ceil(n / alignment) * alignment
}

async function init() {
  installLogUI()
  ;(window as any).onerror = (msg: any, src: any, line: any, col: any, err: any) => { push(`window.onerror ${msg} @${src}:${line}:${col} ${err?.stack || ''}`) }
  window.addEventListener('unhandledrejection', (ev) => { push(`unhandledrejection ${String((ev as any).reason)}`) })

  if (!('gpu' in navigator)) throw new Error('WebGPU not supported')
  const gpu = (navigator as any).gpu as any
  const adapter = await gpu.requestAdapter()
  if (!adapter) throw new Error('No adapter')
  const device = await adapter.requestDevice()
  try {
    ;(device as any).addEventListener && (device as any).addEventListener('uncapturederror', (ev: any) => { push(`device uncapturederror: ${ev?.error?.message || ev}`) })
  } catch {}

  const context = canvas.getContext('webgpu') as unknown as GPUCanvasContext
  const format = gpu.getPreferredCanvasFormat()
  context.configure({ device, format, alphaMode: 'opaque' })

  let depthTexture = device.createTexture({
    size: { width: canvas.width || 1, height: canvas.height || 1, depthOrArrayLayers: 1 },
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT
  })

  function resize() {
    const rect = canvasShell.getBoundingClientRect()
    const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1))
    const width = Math.max(1, Math.floor(rect.width * dpr))
    const height = Math.max(1, Math.floor(rect.height * dpr))
    if (width === canvas.width && height === canvas.height) return
    canvas.width = width
    canvas.height = height
    context.configure({ device, format, alphaMode: 'opaque' })
    depthTexture.destroy()
    depthTexture = device.createTexture({ size: { width, height, depthOrArrayLayers: 1 }, format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT })
  }

  window.addEventListener('resize', resize)
  resize()

  const cameraBuffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  const tileLayerCount = blockFaceOrder.length
  const tileSampler = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' })
  let tileTextureSize = 1
  let tileArrayTexture = device.createTexture({
    size: { width: tileTextureSize, height: tileTextureSize, depthOrArrayLayers: Math.max(1, tileLayerCount) },
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
  })
  let tileArrayView = tileArrayTexture.createView({ dimension: '2d-array', baseArrayLayer: 0, arrayLayerCount: Math.max(1, tileLayerCount) })

  const renderBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } }
    ]
  })

  let renderBindGroup: GPUBindGroup
  const refreshRenderBindGroup = () => {
    renderBindGroup = device.createBindGroup({
      layout: renderBGL,
      entries: [
        { binding: 0, resource: { buffer: cameraBuffer } },
        { binding: 1, resource: tileSampler },
        { binding: 2, resource: tileArrayView }
      ]
    })
  }
  refreshRenderBindGroup()

  device.pushErrorScope('validation')
  const shaderModule = device.createShaderModule({ code: terrainWGSL })
  const shaderInfo = await shaderModule.getCompilationInfo()
  if (shaderInfo.messages?.length) {
    for (const m of shaderInfo.messages) log(`terrain.wgsl: ${m.lineNum}:${m.linePos} ${m.message}`)
  }

  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [renderBGL] }),
    vertex: {
      module: shaderModule,
      entryPoint: 'vs_main',
      buffers: [
        {
          arrayStride: 12 * 4,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32x3' },
            { shaderLocation: 2, offset: 24, format: 'float32x3' },
            { shaderLocation: 3, offset: 36, format: 'float32x2' },
            { shaderLocation: 4, offset: 44, format: 'float32' }
          ]
        }
      ]
    },
    fragment: { module: shaderModule, entryPoint: 'fs_main', targets: [{ format }] },
    primitive: { topology: 'triangle-list', cullMode: 'back' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' }
  })
  await device.popErrorScope().catch((e: unknown) => log(`Render pipeline error: ${String(e)}`))

  const worldConfig = createSimpleWorldConfig(Math.floor(Math.random() * 1000000))
  const chunk = new ChunkManager(worldConfig.dimensions)
  const worldScale = 2  // Units per block for rendering
  setBlockTextureIndices(BlockType.Plank, null)

  console.log('Generating rolling hills terrain...')
  generateRollingHills(chunk, worldConfig)
  console.log('Terrain generation complete')
  const chunkOriginOffset: Vec3 = [-chunk.size.x * worldScale / 2, 0, -chunk.size.z * worldScale / 2]
  let meshDirty = true
  let vertexBuffer: GPUBuffer | null = null
  let vertexBufferSize = 0
  let vertexCount = 0

  function rebuildMesh() {
    const mesh = buildChunkMesh(chunk, worldScale)
    vertexCount = mesh.vertexCount
    console.log('Rebuild mesh vertexCount', vertexCount)
    if (vertexCount === 0) return
    const byteLength = alignTo(mesh.vertexData.byteLength, 4)
    if (!vertexBuffer || byteLength > vertexBufferSize) {
      ;(vertexBuffer as any)?.destroy?.()
      vertexBuffer = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST })
      vertexBufferSize = byteLength
    }
    device.queue.writeBuffer(vertexBuffer, 0, mesh.vertexData.buffer, mesh.vertexData.byteOffset, mesh.vertexData.byteLength)
  }

  const tileBaseUrl = 'http://localhost:8000/textures'

  async function fetchFaceBitmaps(sequence: number) {
    const entries = await Promise.all(blockFaceOrder.map(async (face) => {
      const [col, row] = faceTileCoordinates[face]
      const url = `${tileBaseUrl}/${sequence}/${col}_${row}.png`
      try {
        const response = await fetch(url, {
          mode: 'cors',
          credentials: 'omit'
        })
        if (!response.ok) {
          const errorText = await response.text()
          console.error(`HTTP ${response.status} for ${url}: ${errorText}`)
          throw new Error(`Failed to load face tile ${face} (${col}_${row}) - HTTP ${response.status}`)
        }
        const blob = await response.blob()
        if (!blob.type.startsWith('image/')) {
          throw new Error(`Invalid response type for ${face}: ${blob.type}`)
        }
        const bitmap = await createImageBitmap(blob, {
          premultiplyAlpha: 'none',
          colorSpaceConversion: 'none'
        })
        return [face, bitmap] as const
      } catch (err) {
        console.error(`Failed to fetch ${url}:`, err)
        throw err
      }
    }))
    const result: Record<BlockFaceKey, ImageBitmap> = {} as Record<BlockFaceKey, ImageBitmap>
    for (const [face, bitmap] of entries) {
      result[face] = bitmap
    }
    return result
  }

  function applyFaceBitmapsToGPU(bitmaps: Record<BlockFaceKey, ImageBitmap>, customBlock: CustomBlock) {
    const sampleFace = blockFaceOrder.find(face => Boolean(bitmaps[face]))
    if (!sampleFace) return
    const sampleBitmap = bitmaps[sampleFace]!
    const size = sampleBitmap.width

    // Limit custom blocks to prevent memory issues
    const MAX_CUSTOM_BLOCKS = 8
    const currentCustomLayers = Math.min(customBlocks.length, MAX_CUSTOM_BLOCKS) * blockFaceOrder.length
    const totalLayers = tileLayerCount + currentCustomLayers

    if (!tileArrayTexture || tileTextureSize !== size || tileArrayTexture.depthOrArrayLayers !== totalLayers) {
      ;(tileArrayTexture as any)?.destroy?.()
      tileArrayTexture = device.createTexture({
        size: { width: size, height: size, depthOrArrayLayers: totalLayers },
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
      })
      tileArrayView = tileArrayTexture.createView({ dimension: '2d-array', baseArrayLayer: 0, arrayLayerCount: totalLayers })
      tileTextureSize = size
      refreshRenderBindGroup()
    }

    // Assign texture layer for this custom block (with limit)
    if (customBlock.textureLayer === undefined) {
      if (nextTextureLayer >= MAX_CUSTOM_BLOCKS) {
        console.warn(`Maximum custom blocks (${MAX_CUSTOM_BLOCKS}) reached. Cannot add more custom textures.`)
        return
      }
      customBlock.textureLayer = nextTextureLayer++
    }

    const baseLayer = tileLayerCount + (customBlock.textureLayer * blockFaceOrder.length)

    blockFaceOrder.forEach((face, index) => {
      const bitmap = bitmaps[face]
      if (!bitmap) return
      device.queue.copyExternalImageToTexture(
        { source: bitmap },
        { texture: tileArrayTexture, origin: { x: 0, y: 0, z: baseLayer + index } },
        { width: bitmap.width, height: bitmap.height, depthOrArrayLayers: 1 }
      )
    })

    // Store bitmaps for this custom block
    customTextureLayers.set(customBlock.id, bitmaps)

    // Update texture indices for all custom blocks
    updateAllCustomTextureIndices()
    meshDirty = true
  }

  function updateAllCustomTextureIndices() {
    customBlocks.forEach(block => {
      if (block.textureLayer !== undefined) {
        const baseLayer = tileLayerCount + (block.textureLayer * blockFaceOrder.length)
        const textureIndices: Record<BlockFaceKey, number> = {}
        blockFaceOrder.forEach((face, index) => {
          textureIndices[face] = baseLayer + index
        })
        // For now, we'll use BlockType.Plank as placeholder but with correct texture layers
        setBlockTextureIndices(BlockType.Plank, textureIndices)
      }
    })
  }

  requestFaceBitmaps = fetchFaceBitmaps
  uploadFaceBitmapsToGPU = applyFaceBitmapsToGPU

  const pressedKeys = new Set<string>()
  window.addEventListener('keydown', (ev) => { if (!ev.repeat) pressedKeys.add(ev.code) })
  window.addEventListener('keyup', (ev) => { pressedKeys.delete(ev.code) })
  window.addEventListener('blur', () => pressedKeys.clear())

  const cameraPos: Vec3 = [0, chunk.size.y * worldScale * 0.55, chunk.size.z * worldScale * 0.45]
  const lookTarget: Vec3 = [0, chunk.size.y * worldScale * 0.35, 0]
  const initialDir = normalize([
    lookTarget[0] - cameraPos[0],
    lookTarget[1] - cameraPos[1],
    lookTarget[2] - cameraPos[2]
  ])
  let yaw = Math.atan2(initialDir[0], initialDir[2])
  let pitch = Math.asin(initialDir[1])
  let pointerActive = false
  canvas.addEventListener('click', () => canvas.requestPointerLock())
  canvas.addEventListener('contextmenu', (ev) => ev.preventDefault())
  document.addEventListener('pointerlockchange', () => { pointerActive = document.pointerLockElement === canvas })
  window.addEventListener('mousemove', (ev) => {
    if (!pointerActive) return
    const sensitivity = 0.0025
    yaw -= ev.movementX * sensitivity
    pitch -= ev.movementY * sensitivity
    const limit = Math.PI / 2 - 0.05
    pitch = Math.max(-limit, Math.min(limit, pitch))
  })

  const worldUp: Vec3 = [0, 1, 0]
  type BlockPos = [number, number, number]
  type RaycastHit = { block: BlockPos; previous: BlockPos; normal: Vec3 }
  const maxRayDistance = Math.sqrt(
    (chunk.size.x * worldScale) ** 2 +
    (chunk.size.y * worldScale) ** 2 +
    (chunk.size.z * worldScale) ** 2
  )

  function getForwardVector(): Vec3 {
    return normalize([
      Math.cos(pitch) * Math.sin(yaw),
      Math.sin(pitch),
      Math.cos(pitch) * Math.cos(yaw)
    ])
  }

  function worldToChunk(pos: Vec3): Vec3 {
    return [
      (pos[0] - chunkOriginOffset[0]) / worldScale,
      pos[1] / worldScale,
      (pos[2] - chunkOriginOffset[2]) / worldScale
    ]
  }

  function isInsideChunk([x, y, z]: BlockPos) {
    return x >= 0 && y >= 0 && z >= 0 && x < chunk.size.x && y < chunk.size.y && z < chunk.size.z
  }

  function raycast(origin: Vec3, direction: Vec3, maxDistance = maxRayDistance): RaycastHit | null {
    const dir = normalize(direction)
    if (Math.abs(dir[0]) < 1e-5 && Math.abs(dir[1]) < 1e-5 && Math.abs(dir[2]) < 1e-5) return null

    const pos = worldToChunk(origin)
    let voxel: BlockPos = [Math.floor(pos[0]), Math.floor(pos[1]), Math.floor(pos[2])]
    let prevVoxel: BlockPos = [...voxel]
    const step: BlockPos = [
      dir[0] > 0 ? 1 : dir[0] < 0 ? -1 : 0,
      dir[1] > 0 ? 1 : dir[1] < 0 ? -1 : 0,
      dir[2] > 0 ? 1 : dir[2] < 0 ? -1 : 0
    ]

    const chunkDir: Vec3 = [
      dir[0] / worldScale,
      dir[1] / worldScale,
      dir[2] / worldScale
    ]

    const tDelta: Vec3 = [
      step[0] !== 0 ? Math.abs(1 / chunkDir[0]) : Number.POSITIVE_INFINITY,
      step[1] !== 0 ? Math.abs(1 / chunkDir[1]) : Number.POSITIVE_INFINITY,
      step[2] !== 0 ? Math.abs(1 / chunkDir[2]) : Number.POSITIVE_INFINITY
    ]

    let tMaxX = step[0] > 0 ? (voxel[0] + 1 - pos[0]) * tDelta[0] : step[0] < 0 ? (pos[0] - voxel[0]) * tDelta[0] : Number.POSITIVE_INFINITY
    let tMaxY = step[1] > 0 ? (voxel[1] + 1 - pos[1]) * tDelta[1] : step[1] < 0 ? (pos[1] - voxel[1]) * tDelta[1] : Number.POSITIVE_INFINITY
    let tMaxZ = step[2] > 0 ? (voxel[2] + 1 - pos[2]) * tDelta[2] : step[2] < 0 ? (pos[2] - voxel[2]) * tDelta[2] : Number.POSITIVE_INFINITY
    if (!Number.isFinite(tMaxX)) tMaxX = Number.POSITIVE_INFINITY
    if (!Number.isFinite(tMaxY)) tMaxY = Number.POSITIVE_INFINITY
    if (!Number.isFinite(tMaxZ)) tMaxZ = Number.POSITIVE_INFINITY

    let normal: Vec3 = [0, 0, 0]
    let distance = 0

    for (let i = 0; i < 256; i++) {
      if (distance > maxDistance) break
      if (isInsideChunk(voxel)) {
        const block = chunk.getBlock(voxel[0], voxel[1], voxel[2])
        if (block !== BlockType.Air) {
          const hitNormal = (Math.abs(normal[0]) + Math.abs(normal[1]) + Math.abs(normal[2]) > 0) ? normal : ([-Math.sign(dir[0]), -Math.sign(dir[1]), -Math.sign(dir[2])] as Vec3)
          return { block: [...voxel], previous: [...prevVoxel], normal: hitNormal }
        }
      }

      let axis: 0 | 1 | 2
      if (tMaxX <= tMaxY && tMaxX <= tMaxZ) {
        axis = 0
      } else if (tMaxY <= tMaxZ) {
        axis = 1
      } else {
        axis = 2
      }

      if (axis === 0 && step[0] === 0) return null
      if (axis === 1 && step[1] === 0) return null
      if (axis === 2 && step[2] === 0) return null

      prevVoxel = [...voxel]
      if (axis === 0) {
        voxel = [voxel[0] + step[0], voxel[1], voxel[2]]
        distance = tMaxX
        tMaxX += tDelta[0]
        normal = [-step[0], 0, 0]
      } else if (axis === 1) {
        voxel = [voxel[0], voxel[1] + step[1], voxel[2]]
        distance = tMaxY
        tMaxY += tDelta[1]
        normal = [0, -step[1], 0]
      } else {
        voxel = [voxel[0], voxel[1], voxel[2] + step[2]]
        distance = tMaxZ
        tMaxZ += tDelta[2]
        normal = [0, 0, -step[2]]
      }
    }
    return null
  }

  function handleMouseDown(ev: MouseEvent) {
    if (ev.button !== 0 && ev.button !== 2) return
    ev.preventDefault()
    const hit = raycast(cameraPos, getForwardVector())
    if (!hit) return
    if (ev.button === 0) {
      const current = chunk.getBlock(hit.block[0], hit.block[1], hit.block[2])
      if (current !== BlockType.Air) {
        chunk.setBlock(hit.block[0], hit.block[1], hit.block[2], BlockType.Air)
        meshDirty = true
      }
    } else if (ev.button === 2) {
      const target = hit.previous
      if (isInsideChunk(target) && chunk.getBlock(target[0], target[1], target[2]) === BlockType.Air) {
        // Place either default block or custom block
        if (selectedCustomBlock && selectedCustomBlock.textureLayer !== undefined) {
          // Place custom block - use Plank as placeholder but set custom texture indices
          chunk.setBlock(target[0], target[1], target[2], BlockType.Plank)

          // Set custom texture indices for this specific block
          const baseLayer = tileLayerCount + (selectedCustomBlock.textureLayer * blockFaceOrder.length)
          const textureIndices: Record<BlockFaceKey, number> = {}
          blockFaceOrder.forEach((face, index) => {
            textureIndices[face] = baseLayer + index
          })
          setBlockTextureIndices(BlockType.Plank, textureIndices)

          console.log(`Placed custom block: ${selectedCustomBlock.name} with texture layer ${selectedCustomBlock.textureLayer}`)
        } else {
          chunk.setBlock(target[0], target[1], target[2], selectedBlockType)
          // Reset to default texture indices for non-custom blocks
          setBlockTextureIndices(selectedBlockType, null)
        }
        meshDirty = true
      }
    }
  }

  window.addEventListener('mousedown', handleMouseDown)

  function updateCamera(dt: number) {
    const aspect = canvas.width / Math.max(1, canvas.height)
    const proj = createPerspective((60 * Math.PI) / 180, aspect, 0.1, 500.0)

    const forward = getForwardVector()

    let right = cross(forward, worldUp)
    if (Math.hypot(right[0], right[1], right[2]) < 1e-4) {
      right = [1, 0, 0]
    }
    right = normalize(right)
    const upVec = normalize(cross(right, forward))

    const speedBase = (pressedKeys.has('ShiftLeft') || pressedKeys.has('ShiftRight')) ? 20 : 10
    const speed = speedBase * worldScale
    const move = speed * dt
    if (pressedKeys.has('KeyW')) addScaled(cameraPos, forward, move)
    if (pressedKeys.has('KeyS')) addScaled(cameraPos, forward, -move)
    if (pressedKeys.has('KeyA')) addScaled(cameraPos, right, -move)
    if (pressedKeys.has('KeyD')) addScaled(cameraPos, right, move)
    if (pressedKeys.has('KeyE') || pressedKeys.has('Space')) addScaled(cameraPos, upVec, move)
    if (pressedKeys.has('KeyQ') || pressedKeys.has('ControlLeft')) addScaled(cameraPos, upVec, -move)

    const target: Vec3 = [
      cameraPos[0] + forward[0],
      cameraPos[1] + forward[1],
      cameraPos[2] + forward[2]
    ]
    const view = lookAt(cameraPos, target, upVec)
    const viewProj = multiplyMat4(proj, view)
    device.queue.writeBuffer(cameraBuffer, 0, viewProj.buffer, viewProj.byteOffset, viewProj.byteLength)
  }

  let lastFrameTime = performance.now()

  function frame() {
    const now = performance.now()
    const dt = Math.min(0.1, (now - lastFrameTime) / 1000)
    lastFrameTime = now

    updateCamera(dt)
    if (meshDirty) {
      rebuildMesh()
      meshDirty = false
    }

    const encoder = device.createCommandEncoder()
    const colorView = context.getCurrentTexture().createView()
    const depthView = depthTexture.createView()
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: colorView, clearValue: { r: 0.53, g: 0.81, b: 0.92, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
      depthStencilAttachment: { view: depthView, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' }
    })
    if (vertexBuffer && vertexCount > 0) {
      if (!renderBindGroup) {
        console.warn('renderBindGroup missing before draw')
      }
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, renderBindGroup)
      pass.setVertexBuffer(0, vertexBuffer)
      pass.draw(vertexCount, 1, 0, 0)
    }
    pass.end()
    device.queue.submit([encoder.finish()])
    requestAnimationFrame(frame)
  }

  requestAnimationFrame(frame)
}

init().catch((err) => log(`Startup error: ${String(err)}`))
