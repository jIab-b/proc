// Streamlined WebGPU Renderer
// Core rendering engine without map/capture/dsl concerns

import { get } from 'svelte/store'
import terrainWGSL from './pipelines/render/terrain.wgsl?raw'
import {
  createPerspective,
  lookAt,
  multiplyMat4,
  ChunkManager,
  BlockType,
  buildChunkMesh,
  setBlockTextureIndices,
  blockFaceOrder,
  interactionMode,
  highlightShape,
  highlightRadius,
  ellipsoidRadiusX,
  ellipsoidRadiusY,
  ellipsoidRadiusZ,
  planeSize,
  ellipsoidEditAxis,
  ellipsoidSelectedNode,
  highlightSelection as highlightSelectionStore,
  cameraMode as cameraModeStore,
  type CustomBlock,
  type Vec3,
  type Mat4,
  type BlockFaceKey,
  type HighlightSelection,
  type EllipsoidNode
} from './core'
import type { CameraSnapshot } from './engine'

export interface RendererOptions {
  canvas: HTMLCanvasElement
  overlayCanvas?: HTMLCanvasElement | null
  getSelectedBlock: () => { type: BlockType; custom: CustomBlock | null }
}

const MAX_CUSTOM_BLOCKS = 8

function normalize(v: Vec3): Vec3 {
  const len = Math.hypot(v[0], v[1], v[2])
  return len < 1e-5 ? [0, 0, 0] : [v[0] / len, v[1] / len, v[2] / len]
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

function addScaled(target: Vec3, dir: Vec3, amt: number) {
  target[0] += dir[0] * amt
  target[1] += dir[1] * amt
  target[2] += dir[2] * amt
}

function alignTo(n: number, alignment: number) {
  return Math.ceil(n / alignment) * alignment
}

function projectToScreen(point: Vec3, viewProj: Mat4, width: number, height: number): [number, number] | null {
  const [x, y, z] = point
  const clipX = viewProj[0]! * x + viewProj[4]! * y + viewProj[8]! * z + viewProj[12]!
  const clipY = viewProj[1]! * x + viewProj[5]! * y + viewProj[9]! * z + viewProj[13]!
  const clipZ = viewProj[2]! * x + viewProj[6]! * y + viewProj[10]! * z + viewProj[14]!
  const clipW = viewProj[3]! * x + viewProj[7]! * y + viewProj[11]! * z + viewProj[15]!
  if (clipW <= 0) return null
  const ndcX = clipX / clipW, ndcY = clipY / clipW, ndcZ = clipZ / clipW
  if (ndcX < -1 || ndcX > 1 || ndcY < -1 || ndcY > 1 || ndcZ < -1 || ndcZ > 1) return null
  return [(ndcX * 0.5 + 0.5) * width, (-ndcY * 0.5 + 0.5) * height]
}

export async function createRenderer(opts: RendererOptions, chunk: ChunkManager, worldScale: number, chunkOriginOffset: Vec3) {
  const { canvas, getSelectedBlock } = opts
  const overlayCanvas = opts.overlayCanvas ?? null
  const overlayCtx = overlayCanvas?.getContext('2d') || null

  if (!('gpu' in navigator)) throw new Error('WebGPU not supported')
  const gpu = (navigator as any).gpu
  const adapter = await gpu.requestAdapter()
  if (!adapter) throw new Error('No adapter')
  const device = await adapter.requestDevice()

  try {
    (device as any).addEventListener?.('uncapturederror', (ev: any) => {
      console.error(`device uncapturederror: ${ev?.error?.message || ev}`)
    })
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
    const rect = canvas.getBoundingClientRect()
    const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1))
    const width = Math.max(1, Math.floor(rect.width * dpr))
    const height = Math.max(1, Math.floor(rect.height * dpr))
    if (width === canvas.width && height === canvas.height) return
    canvas.width = width
    canvas.height = height
    context.configure({ device, format, alphaMode: 'opaque' })
    depthTexture.destroy()
    depthTexture = device.createTexture({ size: { width, height, depthOrArrayLayers: 1 }, format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT })
    if (overlayCanvas) {
      overlayCanvas.width = width
      overlayCanvas.height = height
    }
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

  const shaderModule = device.createShaderModule({ code: terrainWGSL })
  const shaderInfo = await shaderModule.getCompilationInfo()
  if (shaderInfo.messages?.length) {
    for (const m of shaderInfo.messages) console.error(`terrain.wgsl: ${m.lineNum}:${m.linePos} ${m.message}`)
  }

  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [renderBGL] }),
    vertex: {
      module: shaderModule,
      entryPoint: 'vs_main',
      buffers: [{
        arrayStride: 12 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x3' },
          { shaderLocation: 1, offset: 12, format: 'float32x3' },
          { shaderLocation: 2, offset: 24, format: 'float32x3' },
          { shaderLocation: 3, offset: 36, format: 'float32x2' },
          { shaderLocation: 4, offset: 44, format: 'float32' }
        ]
      }]
    },
    fragment: { module: shaderModule, entryPoint: 'fs_main', targets: [{ format }] },
    primitive: { topology: 'triangle-list', cullMode: 'back' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' }
  })

  setBlockTextureIndices(BlockType.Plank, null)
  let meshDirty = true
  let vertexBuffer: GPUBuffer | null = null
  let vertexBufferSize = 0
  let vertexCount = 0
  let latestCamera: CameraSnapshot | null = null
  let highlightSelection: HighlightSelection | null = null
  let overlayViews: Array<{ position: Vec3; id: string }> = []

  function chunkToWorld(pos: Vec3): Vec3 {
    return [
      chunkOriginOffset[0] + pos[0] * worldScale,
      pos[1] * worldScale,
      chunkOriginOffset[2] + pos[2] * worldScale
    ]
  }

  function rebuildMesh() {
    const mesh = buildChunkMesh(chunk, worldScale)
    vertexCount = mesh.vertexCount
    if (vertexCount === 0) return
    const byteLength = alignTo(mesh.vertexData.byteLength, 4)
    if (!vertexBuffer || byteLength > vertexBufferSize) {
      if (vertexBuffer) (vertexBuffer as any).destroy?.()
      vertexBuffer = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST })
      vertexBufferSize = byteLength
    }
    device.queue.writeBuffer(vertexBuffer, 0, mesh.vertexData.buffer, mesh.vertexData.byteOffset, mesh.vertexData.byteLength)
  }

  function updateCustomTextures(customBlocks: CustomBlock[]) {
    const usedLayers = new Set<number>()
    customBlocks.forEach(b => {
      if (typeof b.textureLayer === 'number' && b.textureLayer >= 0 && b.textureLayer < MAX_CUSTOM_BLOCKS) {
        usedLayers.add(b.textureLayer)
      }
    })
    const totalLayers = tileLayerCount + usedLayers.size * blockFaceOrder.length

    customBlocks.forEach(block => {
      if (block.textureLayer !== undefined) {
        const baseLayer = tileLayerCount + (block.textureLayer * blockFaceOrder.length)
        const indices: Record<BlockFaceKey, number> = {} as any
        blockFaceOrder.forEach((face, i) => { indices[face] = baseLayer + i })
        setBlockTextureIndices(BlockType.Plank, indices)
      }
    })
  }

  function applyCustomBlockTextures(bitmaps: Record<BlockFaceKey, ImageBitmap>, customBlock: CustomBlock, customBlocks: CustomBlock[]) {
    const sampleFace = blockFaceOrder.find(f => Boolean(bitmaps[f]))
    if (!sampleFace) return
    const size = bitmaps[sampleFace]!.width

    if (customBlock.textureLayer === undefined) {
      const usedLayers = new Set<number>()
      customBlocks.forEach(b => {
        if (typeof b.textureLayer === 'number') usedLayers.add(b.textureLayer)
      })
      let nextLayer = null
      for (let i = 0; i < MAX_CUSTOM_BLOCKS; i++) {
        if (!usedLayers.has(i)) {
          nextLayer = i
          break
        }
      }
      if (nextLayer === null) {
        console.warn(`Max ${MAX_CUSTOM_BLOCKS} custom blocks reached`)
        return
      }
      customBlock.textureLayer = nextLayer
    }

    const usedLayers = new Set<number>()
    customBlocks.forEach(b => {
      if (typeof b.textureLayer === 'number') usedLayers.add(b.textureLayer)
    })
    const totalLayers = tileLayerCount + usedLayers.size * blockFaceOrder.length

    if (!tileArrayTexture || tileTextureSize !== size || tileArrayTexture.depthOrArrayLayers !== totalLayers) {
      tileArrayTexture?.destroy?.()
      tileArrayTexture = device.createTexture({
        size: { width: size, height: size, depthOrArrayLayers: totalLayers },
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
      })
      tileArrayView = tileArrayTexture.createView({ dimension: '2d-array', baseArrayLayer: 0, arrayLayerCount: totalLayers })
      tileTextureSize = size
      refreshRenderBindGroup()
    }

    const baseLayer = tileLayerCount + (customBlock.textureLayer * blockFaceOrder.length)
    blockFaceOrder.forEach((face, i) => {
      const bitmap = bitmaps[face]
      if (bitmap) {
        device.queue.copyExternalImageToTexture(
          { source: bitmap },
          { texture: tileArrayTexture, origin: { x: 0, y: 0, z: baseLayer + i } },
          { width: bitmap.width, height: bitmap.height, depthOrArrayLayers: 1 }
        )
      }
    })

    updateCustomTextures(customBlocks)
    meshDirty = true
  }

  function renderOverlay() {
    if (!overlayCtx || !overlayCanvas || !latestCamera) return
    const width = overlayCanvas.width || canvas.width
    const height = overlayCanvas.height || canvas.height
    overlayCtx.clearRect(0, 0, width, height)

    overlayCtx.save()
    overlayViews.forEach((view, idx) => {
      const screen = projectToScreen(view.position, latestCamera!.viewProjectionMatrix, width, height)
      if (!screen) return
      overlayCtx.fillStyle = 'rgba(255, 64, 64, 0.9)'
      overlayCtx.strokeStyle = 'rgba(0, 0, 0, 0.6)'
      overlayCtx.beginPath()
      overlayCtx.arc(screen[0], screen[1], 4, 0, Math.PI * 2)
      overlayCtx.fill()
      overlayCtx.stroke()
      overlayCtx.fillStyle = 'rgba(255, 255, 255, 0.9)'
      overlayCtx.fillText(String(idx + 1), screen[0] + 6, screen[1])
    })
    overlayCtx.restore()

    if (highlightSelection && latestCamera) {
      const center = highlightSelection.center
      const halfStep = 0.5
      const baseCenter: Vec3 = [center[0] + halfStep, center[1] + halfStep, center[2] + halfStep]
      const projectChunk = (p: Vec3) => projectToScreen(chunkToWorld(p), latestCamera!.viewProjectionMatrix, width, height)

      overlayCtx.save()
      overlayCtx.lineWidth = 1.5
      overlayCtx.strokeStyle = 'rgba(120, 200, 255, 0.9)'
      overlayCtx.fillStyle = 'rgba(120, 200, 255, 0.12)'

      if (highlightSelection.shape === 'sphere') {
        const centerScreen = projectChunk(baseCenter)
        if (!centerScreen) { overlayCtx.restore(); return }
        const radius = highlightSelection.radius + halfStep
        const edgeScreen = projectChunk([baseCenter[0] + radius, baseCenter[1], baseCenter[2]])
        if (!edgeScreen) { overlayCtx.restore(); return }
        const radiusPx = Math.max(4, Math.hypot(edgeScreen[0] - centerScreen[0], edgeScreen[1] - centerScreen[1]))
        overlayCtx.beginPath()
        overlayCtx.arc(centerScreen[0], centerScreen[1], radiusPx, 0, Math.PI * 2)
        overlayCtx.stroke()
        overlayCtx.fill()
      } else if (highlightSelection.shape === 'ellipsoid') {
        // Draw ellipsoid wireframe with 3 adjustable radii
        const rx = (highlightSelection.radiusX ?? highlightSelection.radius) + halfStep
        const ry = (highlightSelection.radiusY ?? highlightSelection.radius) + halfStep
        const rz = (highlightSelection.radiusZ ?? highlightSelection.radius) + halfStep

        // Draw 3 elliptical cross-sections (OPTIMIZED)
        const segments = 16  // Reduced from 64 to 16 for 4x performance boost
        const activeAxis = get(ellipsoidEditAxis)

        // PERFORMANCE: Removed expensive gradient fills - just use simple strokes
        // This eliminates 3 gradient creations and 100+ Math.hypot calls per frame

        // XY plane (around Z axis) - controls Z radius
        overlayCtx.lineWidth = activeAxis === 'z' ? 3 : 2
        overlayCtx.strokeStyle = activeAxis === 'z' ? 'rgba(255, 140, 100, 0.95)' : 'rgba(255, 100, 80, 0.7)'
        overlayCtx.beginPath()
        let firstPoint = true
        for (let i = 0; i <= segments; i++) {
          const angle = (i / segments) * Math.PI * 2
          const x = baseCenter[0] + rx * Math.cos(angle)
          const y = baseCenter[1] + ry * Math.sin(angle)
          const z = baseCenter[2]
          const screen = projectChunk([x, y, z])
          if (screen) {
            if (firstPoint) {
              overlayCtx.moveTo(screen[0], screen[1])
              firstPoint = false
            } else {
              overlayCtx.lineTo(screen[0], screen[1])
            }
          }
        }
        overlayCtx.stroke()

        // XZ plane (around Y axis) - controls Y radius
        overlayCtx.lineWidth = activeAxis === 'y' ? 3 : 2
        overlayCtx.strokeStyle = activeAxis === 'y' ? 'rgba(255, 140, 100, 0.95)' : 'rgba(255, 100, 80, 0.7)'
        overlayCtx.beginPath()
        firstPoint = true
        for (let i = 0; i <= segments; i++) {
          const angle = (i / segments) * Math.PI * 2
          const x = baseCenter[0] + rx * Math.cos(angle)
          const y = baseCenter[1]
          const z = baseCenter[2] + rz * Math.sin(angle)
          const screen = projectChunk([x, y, z])
          if (screen) {
            if (firstPoint) {
              overlayCtx.moveTo(screen[0], screen[1])
              firstPoint = false
            } else {
              overlayCtx.lineTo(screen[0], screen[1])
            }
          }
        }
        overlayCtx.stroke()

        // YZ plane (around X axis) - controls X radius
        overlayCtx.lineWidth = activeAxis === 'x' ? 3 : 2
        overlayCtx.strokeStyle = activeAxis === 'x' ? 'rgba(255, 140, 100, 0.95)' : 'rgba(255, 100, 80, 0.7)'
        overlayCtx.beginPath()
        firstPoint = true
        for (let i = 0; i <= segments; i++) {
          const angle = (i / segments) * Math.PI * 2
          const x = baseCenter[0]
          const y = baseCenter[1] + ry * Math.cos(angle)
          const z = baseCenter[2] + rz * Math.sin(angle)
          const screen = projectChunk([x, y, z])
          if (screen) {
            if (firstPoint) {
              overlayCtx.moveTo(screen[0], screen[1])
              firstPoint = false
            } else {
              overlayCtx.lineTo(screen[0], screen[1])
            }
          }
        }
        overlayCtx.stroke()

        // Draw axis endpoint nodes (6 nodes total: +x, -x, +y, -y, +z, -z)
        const nodeRadius = 6
        const selectedNode = get(ellipsoidSelectedNode)
        const nodes: Array<{ pos: Vec3; id: EllipsoidNode }> = [
          { pos: [baseCenter[0] + rx, baseCenter[1], baseCenter[2]], id: '+x' },
          { pos: [baseCenter[0] - rx, baseCenter[1], baseCenter[2]], id: '-x' },
          { pos: [baseCenter[0], baseCenter[1] + ry, baseCenter[2]], id: '+y' },
          { pos: [baseCenter[0], baseCenter[1] - ry, baseCenter[2]], id: '-y' },
          { pos: [baseCenter[0], baseCenter[1], baseCenter[2] + rz], id: '+z' },
          { pos: [baseCenter[0], baseCenter[1], baseCenter[2] - rz], id: '-z' },
          { pos: [baseCenter[0], baseCenter[1], baseCenter[2]], id: 'center' }
        ]

        nodes.forEach(node => {
          const screen = projectChunk(node.pos)
          if (!screen) return

          const isCenter = node.id === 'center'
          const isActivelyAdjusting = ellipsoidNodeAdjustActive && selectedNode === node.id
          const wasActivated = activatedNodes.has(node.id)
          const isSelected = selectedNode === node.id || (isCenter && centerNodeSelected)

          // Color logic:
          // - Center node: cyan/blue (unchanged)
          // - Active node (being adjusted): blue
          // - Previously activated node: red
          // - Default node: orange
          let nodeFillColor: string
          let nodeStrokeColor: string

          if (isCenter) {
            nodeFillColor = isSelected ? 'rgba(120, 220, 255, 0.95)' : 'rgba(120, 200, 255, 0.85)'
            nodeStrokeColor = isSelected ? 'rgba(255, 255, 255, 0.95)' : 'rgba(80, 180, 255, 0.9)'
          } else if (isActivelyAdjusting) {
            // Active node: bright blue
            nodeFillColor = 'rgba(80, 140, 255, 0.95)'
            nodeStrokeColor = 'rgba(255, 255, 255, 0.95)'
          } else if (wasActivated) {
            // Previously activated node: red
            nodeFillColor = 'rgba(255, 60, 60, 0.95)'
            nodeStrokeColor = 'rgba(200, 30, 30, 0.95)'
          } else {
            // Default node: orange
            nodeFillColor = 'rgba(255, 120, 100, 0.85)'
            nodeStrokeColor = 'rgba(255, 60, 40, 0.95)'
          }

          const nodeSize = (isActivelyAdjusting || isSelected) ? nodeRadius + 2 : (wasActivated ? nodeRadius + 1 : nodeRadius)

          // Draw node
          overlayCtx.fillStyle = nodeFillColor
          overlayCtx.strokeStyle = nodeStrokeColor
          overlayCtx.lineWidth = (isActivelyAdjusting || isSelected) ? 2.5 : 2
          overlayCtx.beginPath()
          overlayCtx.arc(screen[0], screen[1], nodeSize, 0, Math.PI * 2)
          overlayCtx.fill()
          overlayCtx.stroke()
        })

        // Reset style
        overlayCtx.lineWidth = 1.5
        overlayCtx.strokeStyle = 'rgba(255, 100, 80, 0.85)'
      } else if (highlightSelection.shape === 'plane') {
        // Draw horizontal plane as a grid (RED)
        const size = highlightSelection.planeSize ?? 8
        const y = baseCenter[1]  // Fixed Y coordinate (the base height)

        // Draw grid lines in red
        overlayCtx.strokeStyle = 'rgba(255, 100, 80, 0.7)'
        overlayCtx.lineWidth = 1.5

        // Draw X lines (parallel to X axis)
        for (let z = -size; z <= size; z += 2) {
          const p1: Vec3 = [baseCenter[0] - size, y, baseCenter[2] + z]
          const p2: Vec3 = [baseCenter[0] + size, y, baseCenter[2] + z]
          const s1 = projectChunk(p1)
          const s2 = projectChunk(p2)
          if (s1 && s2) {
            overlayCtx.beginPath()
            overlayCtx.moveTo(s1[0], s1[1])
            overlayCtx.lineTo(s2[0], s2[1])
            overlayCtx.stroke()
          }
        }

        // Draw Z lines (parallel to Z axis)
        for (let x = -size; x <= size; x += 2) {
          const p1: Vec3 = [baseCenter[0] + x, y, baseCenter[2] - size]
          const p2: Vec3 = [baseCenter[0] + x, y, baseCenter[2] + size]
          const s1 = projectChunk(p1)
          const s2 = projectChunk(p2)
          if (s1 && s2) {
            overlayCtx.beginPath()
            overlayCtx.moveTo(s1[0], s1[1])
            overlayCtx.lineTo(s2[0], s2[1])
            overlayCtx.stroke()
          }
        }

        // Draw corner nodes and center node
        const nodeRadius = 6
        const selectedNode = get(ellipsoidSelectedNode)
        const nodes: Array<{ pos: Vec3; id: EllipsoidNode }> = [
          { pos: [baseCenter[0] + size, y, baseCenter[2] + size], id: '+x' },  // +X+Z corner
          { pos: [baseCenter[0] - size, y, baseCenter[2] + size], id: '-x' },  // -X+Z corner
          { pos: [baseCenter[0] + size, y, baseCenter[2] - size], id: '+z' },  // +X-Z corner
          { pos: [baseCenter[0] - size, y, baseCenter[2] - size], id: '-z' },  // -X-Z corner
          { pos: [baseCenter[0], y, baseCenter[2]], id: 'center' }
        ]

        nodes.forEach(node => {
          const screen = projectChunk(node.pos)
          if (!screen) return

          const isCenter = node.id === 'center'
          const isActivelyAdjusting = ellipsoidNodeAdjustActive && selectedNode === node.id
          const wasActivated = activatedNodes.has(node.id)
          const isSelected = selectedNode === node.id || (isCenter && centerNodeSelected)

          // Red color scheme for plane
          let nodeFillColor: string
          let nodeStrokeColor: string

          if (isCenter) {
            nodeFillColor = isSelected ? 'rgba(255, 140, 120, 0.95)' : 'rgba(255, 120, 100, 0.85)'
            nodeStrokeColor = isSelected ? 'rgba(255, 255, 255, 0.95)' : 'rgba(255, 80, 60, 0.9)'
          } else if (isActivelyAdjusting) {
            nodeFillColor = 'rgba(255, 100, 80, 0.95)'
            nodeStrokeColor = 'rgba(255, 255, 255, 0.95)'
          } else if (wasActivated) {
            nodeFillColor = 'rgba(255, 60, 60, 0.95)'
            nodeStrokeColor = 'rgba(200, 30, 30, 0.95)'
          } else {
            nodeFillColor = 'rgba(255, 120, 100, 0.85)'
            nodeStrokeColor = 'rgba(255, 60, 40, 0.95)'
          }

          const nodeSize = (isActivelyAdjusting || isSelected) ? nodeRadius + 2 : (wasActivated ? nodeRadius + 1 : nodeRadius)

          // Draw node
          overlayCtx.fillStyle = nodeFillColor
          overlayCtx.strokeStyle = nodeStrokeColor
          overlayCtx.lineWidth = (isActivelyAdjusting || isSelected) ? 2.5 : 2
          overlayCtx.beginPath()
          overlayCtx.arc(screen[0], screen[1], nodeSize, 0, Math.PI * 2)
          overlayCtx.fill()
          overlayCtx.stroke()
        })
      } else {
        const r = highlightSelection.radius + halfStep
        const corners: Vec3[] = [
          [baseCenter[0] - r, baseCenter[1] - r, baseCenter[2] - r],
          [baseCenter[0] + r, baseCenter[1] - r, baseCenter[2] - r],
          [baseCenter[0] + r, baseCenter[1] - r, baseCenter[2] + r],
          [baseCenter[0] - r, baseCenter[1] - r, baseCenter[2] + r],
          [baseCenter[0] - r, baseCenter[1] + r, baseCenter[2] - r],
          [baseCenter[0] + r, baseCenter[1] + r, baseCenter[2] - r],
          [baseCenter[0] + r, baseCenter[1] + r, baseCenter[2] + r],
          [baseCenter[0] - r, baseCenter[1] + r, baseCenter[2] + r]
        ]
        const projected = corners.map(c => projectChunk(c))
        if (projected.some(p => !p)) { overlayCtx.restore(); return }

        const edges: [number, number][] = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        for (const [a, b] of edges) {
          const pA = projected[a], pB = projected[b]
          if (!pA || !pB) continue
          overlayCtx.beginPath()
          overlayCtx.moveTo(pA[0], pA[1])
          overlayCtx.lineTo(pB[0], pB[1])
          overlayCtx.stroke()
        }
      }
      overlayCtx.restore()
    }
  }

  // Camera modes: 'player' (FPS) or 'overview' (orbital)
  let cameraMode: 'player' | 'overview' = get(cameraModeStore)

  // Subscribe to camera mode changes from the store
  cameraModeStore.subscribe(mode => {
    if (mode === cameraMode) return
    cameraMode = mode

    if (cameraMode === 'overview') {
      if (pointerActive) document.exitPointerLock()
      paused = false
      orbitTarget = [...cameraPos] as Vec3
      orbitDistance = 50
      orbitYaw = yaw
      orbitPitch = pitch
    } else {
      paused = true
    }
  })

  // Player camera (FPS)
  const cameraPos: Vec3 = [0, chunk.size.y * worldScale * 0.55, chunk.size.z * worldScale * 0.45]
  const worldUp: Vec3 = [0, 1, 0]
  let yaw = 0, pitch = 0

  // Overview camera (orbital)
  let orbitTarget: Vec3 = [0, chunk.size.y * worldScale * 0.3, 0]
  let orbitDistance = 50
  let orbitYaw = 0
  let orbitPitch = -0.5

  const pressedKeys = new Set<string>()
  let pointerActive = false
  let paused = true
  let rafHandle: number | null = null
  let lastFrameTime = performance.now()

  // Mouse state for overview mode
  let isDragging = false
  let dragButton = -1
  let lastMouseX = 0
  let lastMouseY = 0

  // Mouse state for ellipsoid node dragging
  let nodeEditStartX = 0
  let nodeEditStartY = 0
  let nodeEditStartRadius = 0

  canvas.addEventListener('click', () => {
    if (cameraMode === 'player' && !pointerActive) {
      canvas.requestPointerLock()
    }
  })

  document.addEventListener('pointerlockchange', () => {
    pointerActive = document.pointerLockElement === canvas
    if (cameraMode === 'player') {
      paused = !pointerActive
    }
    pressedKeys.clear()
    lastFrameTime = performance.now()
  })

  // Overview mode: mouse drag controls and block placement
  canvas.addEventListener('mousedown', (ev) => {
    if (cameraMode === 'overview' && !pointerActive) {
      const mode = get(interactionMode)
      const shape = get(highlightShape)
      const overviewPos = getCameraWorldPosition()
      const overviewForward: Vec3 = normalize([
        orbitTarget[0] - overviewPos[0],
        orbitTarget[1] - overviewPos[1],
        orbitTarget[2] - overviewPos[2]
      ])

      console.log('=== OVERVIEW CLICK ===')
      console.log('Button:', ev.button === 0 ? 'LEFT' : ev.button === 1 ? 'MIDDLE' : 'RIGHT')
      console.log('Click position:', { clientX: ev.clientX, clientY: ev.clientY })
      console.log('Canvas size:', { width: canvas.width, height: canvas.height })
      console.log('Mode:', mode, 'Shape:', shape)
      console.log('Camera pos:', overviewPos)
      console.log('Camera forward:', overviewForward)
      console.log('Orbit target:', orbitTarget)

      if (mode === 'highlight' && (shape === 'ellipsoid' || shape === 'plane')) {
        // Convert client coordinates to canvas coordinates
        // IMPORTANT: Use overlayCanvas rect if available, since nodes are drawn on overlay
        const targetCanvas = overlayCanvas || canvas
        const rect = targetCanvas.getBoundingClientRect()
        const canvasX = ((ev.clientX - rect.left) / rect.width) * canvas.width
        const canvasY = ((ev.clientY - rect.top) / rect.height) * canvas.height
        console.log('Coordinate conversion (using overlay):', {
          client: { x: ev.clientX, y: ev.clientY },
          rect: { left: rect.left, top: rect.top, width: rect.width, height: rect.height },
          canvas: { x: canvasX, y: canvasY, width: canvas.width, height: canvas.height },
          usingOverlay: !!overlayCanvas
        })

        const currentSelection = get(highlightSelectionStore)
        let clickedNode: EllipsoidNode = null

        if (currentSelection && currentSelection.shape === 'ellipsoid') {
          const rx = currentSelection.radiusX ?? currentSelection.radius
          const ry = currentSelection.radiusY ?? currentSelection.radius
          const rz = currentSelection.radiusZ ?? currentSelection.radius

          // Calculate if ellipsoid is visible
          const centerWorld = chunkToWorld([
            currentSelection.center[0] + 0.5,
            currentSelection.center[1] + 0.5,
            currentSelection.center[2] + 0.5
          ])
          const isVisible = latestCamera && projectToScreen(centerWorld, latestCamera.viewProjectionMatrix, canvas.width, canvas.height) !== null

          console.log('Current ellipsoid:', currentSelection.center, 'radii:', { rx, ry, rz })
          console.log('⚠️  Ellipsoid visible:', isVisible, '- Center world pos:', centerWorld)

          clickedNode = getClickedNode(canvasX, canvasY, currentSelection.center, rx, ry, rz, 'ellipsoid')
          console.log('Clicked node:', clickedNode)
          console.log('Activated nodes:', Array.from(activatedNodes))

          if (!isVisible) {
            console.warn('⚠️  ELLIPSOID IS OFF-SCREEN! Pan camera to position:', currentSelection.center)
          }
        } else if (currentSelection && currentSelection.shape === 'plane') {
          const size = currentSelection.planeSize ?? 8

          // Calculate if plane is visible
          const centerWorld = chunkToWorld([
            currentSelection.center[0] + 0.5,
            currentSelection.center[1] + 0.5,
            currentSelection.center[2] + 0.5
          ])
          const isVisible = latestCamera && projectToScreen(centerWorld, latestCamera.viewProjectionMatrix, canvas.width, canvas.height) !== null

          console.log('Current plane:', currentSelection.center, 'size:', size)
          console.log('⚠️  Plane visible:', isVisible, '- Center world pos:', centerWorld)

          clickedNode = getClickedNode(canvasX, canvasY, currentSelection.center, size, 0, 0, 'plane')
          console.log('Clicked node:', clickedNode)
          console.log('Activated nodes:', Array.from(activatedNodes))

          if (!isVisible) {
            console.warn('⚠️  PLANE IS OFF-SCREEN! Pan camera to position:', currentSelection.center)
          }
        }

        if (ev.button === 2) {
          ev.preventDefault()

          // Check if right-clicking on a node - show context menu
          if (clickedNode && clickedNode !== 'center') {
            showNodeContextMenu(clickedNode, ev.clientX, ev.clientY)
            ev.stopPropagation()
            return
          }

          // Compute ray direction for click position
          let rayDir: Vec3 = overviewForward
          if (latestCamera) {
            const { forward, right, up, fovYRadians, aspect } = latestCamera
            const tanHalfFov = Math.tan(fovYRadians / 2)
            const ndcX = 2 * (canvasX / canvas.width) - 1
            const ndcY = 1 - 2 * (canvasY / canvas.height)
            const screenX = ndcX * aspect * tanHalfFov
            const screenY = ndcY * tanHalfFov
            const dx = right[0] * screenX + up[0] * screenY + forward[0]
            const dy = right[1] * screenX + up[1] * screenY + forward[1]
            const dz = right[2] * screenX + up[2] * screenY + forward[2]
            const len = Math.hypot(dx, dy, dz)
            if (len > 0) {
              rayDir = [dx / len, dy / len, dz / len]
            }
          }

          const hit = raycast(overviewPos, rayDir)
          console.log(`${shape} mode right-click raycast:`, hit)

          let centerPos: Vec3
          if (hit) {
            centerPos = hit.block
          } else {
            const distance = orbitDistance
            const worldPos: Vec3 = [
              overviewPos[0] + rayDir[0] * distance,
              overviewPos[1] + rayDir[1] * distance,
              overviewPos[2] + rayDir[2] * distance
            ]
            const chunkPos = worldToChunk(worldPos)
            centerPos = [Math.floor(chunkPos[0]), Math.floor(chunkPos[1]), Math.floor(chunkPos[2])]
            console.log(`No raycast hit - creating ${shape} at click position (CHUNK coords):`, centerPos)
          }

          const radius = get(highlightRadius)
          const selection: HighlightSelection = {
            center: centerPos,
            shape,
            radius
          }
          if (shape === 'ellipsoid') {
            selection.radiusX = get(ellipsoidRadiusX)
            selection.radiusY = get(ellipsoidRadiusY)
            selection.radiusZ = get(ellipsoidRadiusZ)
            console.log('Ellipsoid created at (CHUNK coords):', centerPos, 'with radii:', selection.radiusX, selection.radiusY, selection.radiusZ)
          } else if (shape === 'plane') {
            selection.planeSize = get(planeSize)
            console.log('Plane created at (CHUNK coords):', centerPos, 'with size:', selection.planeSize)
          }
          ellipsoidSelectedNode.set(null)
          ellipsoidEditAxis.set(null)
          centerNodeSelected = false
          highlightSelectionStore.set(selection)
          ellipsoidMovementActive = true
          ellipsoidNodeAdjustActive = false
          ellipsoidCenterDragActive = false
          lastCameraPos = [...overviewPos] as Vec3

          ev.stopPropagation()
          return
        }

        if (ev.button === 0 && currentSelection && currentSelection.shape === 'ellipsoid') {
          console.log('LEFT CLICK with existing ellipsoid, clickedNode:', clickedNode)

          if (clickedNode === 'center') {
            console.log('>>> Activating CENTER DRAG mode')
            ev.preventDefault()
            ellipsoidCenterDragActive = true
            centerNodeSelected = true
            ellipsoidMovementActive = false
            ellipsoidNodeAdjustActive = false
            ellipsoidSelectedNode.set('center')
            ellipsoidEditAxis.set(null)
            lastCameraPos = [...overviewPos] as Vec3
            ev.stopPropagation()
            return
          }

          // Left-click on axis node: activate mouse-controlled scaling
          if (clickedNode) {
            console.log(`>>> Activating NODE ${clickedNode} for mouse-controlled scaling`)
            ev.preventDefault()
            ellipsoidNodeAdjustActive = true
            ellipsoidSelectedNode.set(clickedNode)
            ellipsoidEditAxis.set(null)
            ellipsoidMovementActive = false
            ellipsoidCenterDragActive = false
            centerNodeSelected = false
            activatedNodes.add(clickedNode) // Mark as activated

            // Store initial mouse position and current radius
            nodeEditStartX = ev.clientX
            nodeEditStartY = ev.clientY
            const axis = clickedNode[1] as 'x' | 'y' | 'z'
            if (axis === 'x') nodeEditStartRadius = get(ellipsoidRadiusX)
            else if (axis === 'y') nodeEditStartRadius = get(ellipsoidRadiusY)
            else if (axis === 'z') nodeEditStartRadius = get(ellipsoidRadiusZ)

            console.log('Node activation state:', {
              selectedNode: clickedNode,
              startPos: { x: nodeEditStartX, y: nodeEditStartY },
              startRadius: nodeEditStartRadius
            })
            ev.stopPropagation()
            return
          }

          const center = currentSelection.center
          const rx = currentSelection.radiusX ?? currentSelection.radius
          const ry = currentSelection.radiusY ?? currentSelection.radius
          const rz = currentSelection.radiusZ ?? currentSelection.radius

          console.log('Checking ellipse rings with canvas coords:', canvasX, canvasY)

          if (isNearEllipseRing(canvasX, canvasY, center, rx, ry, 'xy')) {
            console.log('>>> Clicked on XY ring (Z axis)')
            ellipsoidEditAxis.set('z')
            ellipsoidSelectedNode.set(null)
            centerNodeSelected = false
            lastCameraPos = [...overviewPos] as Vec3
            ev.stopPropagation()
            return
          }
          if (isNearEllipseRing(canvasX, canvasY, center, rx, rz, 'xz')) {
            console.log('>>> Clicked on XZ ring (Y axis)')
            ellipsoidEditAxis.set('y')
            ellipsoidSelectedNode.set(null)
            centerNodeSelected = false
            lastCameraPos = [...overviewPos] as Vec3
            ev.stopPropagation()
            return
          }
          if (isNearEllipseRing(canvasX, canvasY, center, ry, rz, 'yz')) {
            console.log('>>> Clicked on YZ ring (X axis)')
            ellipsoidEditAxis.set('x')
            ellipsoidSelectedNode.set(null)
            centerNodeSelected = false
            lastCameraPos = [...overviewPos] as Vec3
            ev.stopPropagation()
            return
          }
        }
      }

      // Handle right-click for block placement/highlight selection
      if (ev.button === 2) {
        ev.preventDefault()
        ev.stopPropagation()

        const hit = raycast(overviewPos, overviewForward)
        console.log('Raycast hit:', hit)

        if (hit) {
          if (mode === 'block') {
            // Right click in block mode - place block
            const placePos = hit.previous
            if (isInsideChunk(placePos) && chunk.getBlock(placePos[0], placePos[1], placePos[2]) === BlockType.Air) {
              const selected = getSelectedBlock()
              chunk.setBlock(placePos[0], placePos[1], placePos[2], selected.type)
              meshDirty = true
              console.log('Block placed at', placePos)
            }
          } else if (mode === 'highlight') {
            // Right click in highlight mode - set highlight selection
            const shape = get(highlightShape)
            const radius = get(highlightRadius)
            const selection: HighlightSelection = {
              center: hit.block,
              shape,
              radius
            }

            if (shape === 'ellipsoid') {
              selection.radiusX = get(ellipsoidRadiusX)
              selection.radiusY = get(ellipsoidRadiusY)
              selection.radiusZ = get(ellipsoidRadiusZ)
              ellipsoidEditAxis.set(null)
              ellipsoidSelectedNode.set(null)
            }

            highlightSelectionStore.set(selection)
            console.log('Highlight selection set:', selection)
          }
        } else {
          console.log('No raycast hit')
        }
        return
      }

      // Left and middle mouse for dragging
      console.log('>>> Entering drag mode, button:', ev.button)
      isDragging = true
      dragButton = ev.button
      lastMouseX = ev.clientX
      lastMouseY = ev.clientY
      canvas.style.cursor = ev.button === 1 ? 'move' : 'grab'
    }
  })

  window.addEventListener('mouseup', (ev) => {
    if (cameraMode === 'overview') {
      isDragging = false
      dragButton = -1
      canvas.style.cursor = 'default'
    }

    if (ev.button === 2) {
      if (ellipsoidMovementActive) {
        ellipsoidMovementActive = false
        console.log('Ellipsoid movement mode deactivated')
      }
    }

    if (ev.button === 0) {
      // Left-click release: deactivate node adjustment but keep node marked as activated (red)
      if (ellipsoidNodeAdjustActive) {
        ellipsoidNodeAdjustActive = false
        ellipsoidSelectedNode.set(null)
        lastCameraPos = null
        console.log('Node adjustment deactivated - node will remain red')
      }
      if (ellipsoidCenterDragActive) {
        ellipsoidCenterDragActive = false
        centerNodeSelected = false
        ellipsoidSelectedNode.set(null)
        lastCameraPos = null
      }
      if (get(ellipsoidEditAxis)) {
        ellipsoidEditAxis.set(null)
        lastCameraPos = null
      }
    }
  })

  // Track last camera position for ellipsoid editing
  let lastCameraPos: Vec3 | null = null

  // Track ellipsoid interaction modes
  let ellipsoidMovementActive = false // center translation along view axis while right-click held
  let ellipsoidNodeAdjustActive = false // resizing along axis via node drag
  let ellipsoidCenterDragActive = false // center reposition via left-click drag on center node
  let centerNodeSelected = false

  // Track previously activated nodes (to show as red)
  const activatedNodes = new Set<EllipsoidNode>()

  window.addEventListener('mousemove', (ev) => {
    // Handle ellipsoid node adjustment via mouse drag
    if (cameraMode === 'overview' && ellipsoidNodeAdjustActive) {
      const selectedNode = get(ellipsoidSelectedNode)
      if (selectedNode && selectedNode !== 'center') {
        ev.preventDefault()

        // Calculate mouse delta from start position
        const deltaX = ev.clientX - nodeEditStartX
        const deltaY = ev.clientY - nodeEditStartY

        // Use the primary movement direction (the larger delta)
        const delta = Math.abs(deltaX) > Math.abs(deltaY) ? deltaX : -deltaY

        // Sensitivity: pixels to radius units
        const sensitivity = 0.05
        const newRadius = Math.max(0.5, nodeEditStartRadius + delta * sensitivity)

        // Update the appropriate radius based on which node is selected
        const currentSelection = get(highlightSelectionStore)

        if (currentSelection && currentSelection.shape === 'plane') {
          // For plane, all corner nodes resize the plane uniformly
          planeSize.set(newRadius)
          highlightSelectionStore.update(sel => {
            if (sel && sel.shape === 'plane') {
              return { ...sel, planeSize: newRadius }
            }
            return sel
          })
        } else if (currentSelection && currentSelection.shape === 'ellipsoid') {
          const axis = selectedNode[1] as 'x' | 'y' | 'z'
          if (axis === 'x') {
            ellipsoidRadiusX.set(newRadius)
            highlightSelectionStore.update(sel => {
              if (sel && sel.shape === 'ellipsoid') {
                return { ...sel, radiusX: newRadius }
              }
              return sel
            })
          } else if (axis === 'y') {
            ellipsoidRadiusY.set(newRadius)
            highlightSelectionStore.update(sel => {
              if (sel && sel.shape === 'ellipsoid') {
                return { ...sel, radiusY: newRadius }
              }
              return sel
            })
          } else if (axis === 'z') {
            ellipsoidRadiusZ.set(newRadius)
            highlightSelectionStore.update(sel => {
              if (sel && sel.shape === 'ellipsoid') {
                return { ...sel, radiusZ: newRadius }
              }
              return sel
            })
          }
        }

        // Don't orbit camera while adjusting nodes
        return
      }
    }

    if (cameraMode === 'player' && pointerActive) {
      // Player mode: pointer lock look
      yaw -= ev.movementX * 0.0025
      pitch -= ev.movementY * 0.0025
      pitch = Math.max(-Math.PI / 2 + 0.05, Math.min(Math.PI / 2 - 0.05, pitch))
    } else if (cameraMode === 'overview' && isDragging) {
      // Overview mode: drag to orbit or pan
      const dx = ev.clientX - lastMouseX
      const dy = ev.clientY - lastMouseY
      lastMouseX = ev.clientX
      lastMouseY = ev.clientY

      if (dragButton === 2 || dragButton === 0) {
        // Right or left mouse: orbit
        orbitYaw -= dx * 0.005
        orbitPitch += dy * 0.005
        orbitPitch = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, orbitPitch))
      } else if (dragButton === 1) {
        // Middle mouse: pan
        const right: Vec3 = [Math.cos(orbitYaw), 0, -Math.sin(orbitYaw)]
        const up: Vec3 = [0, 1, 0]
        const panSpeed = orbitDistance * 0.002
        orbitTarget[0] -= right[0] * dx * panSpeed
        orbitTarget[1] -= up[1] * dy * panSpeed
        orbitTarget[2] -= right[2] * dx * panSpeed
      }
    }
  })

  // Mouse wheel handler
  canvas.addEventListener('wheel', (ev) => {
    // Ellipsoid center translation: move ellipsoid along view axis while in placement or center-drag mode
    if (ellipsoidMovementActive || ellipsoidCenterDragActive) {
      ev.preventDefault()
      const currentSelection = get(highlightSelectionStore)
      if (currentSelection && currentSelection.shape === 'ellipsoid') {
        // Get camera forward vector
        const forward = cameraMode === 'player' ? getForwardVector() : normalize([
          orbitTarget[0] - (orbitTarget[0] + orbitDistance * Math.cos(orbitPitch) * Math.sin(orbitYaw)),
          orbitTarget[1] - (orbitTarget[1] + orbitDistance * Math.sin(orbitPitch)),
          orbitTarget[2] - (orbitTarget[2] + orbitDistance * Math.cos(orbitPitch) * Math.cos(orbitYaw))
        ])

        // Move ellipsoid center along view axis
        const moveSpeed = 0.5
        const movement = -ev.deltaY * 0.01 * moveSpeed / worldScale
        const newCenter: [number, number, number] = [
          currentSelection.center[0] + forward[0] * movement,
          currentSelection.center[1] + forward[1] * movement,
          currentSelection.center[2] + forward[2] * movement
        ]

        highlightSelectionStore.update(sel => {
          if (sel && sel.shape === 'ellipsoid') {
            return { ...sel, center: newCenter }
          }
          return sel
        })
      }
      return
    }

    // Overview mode: scroll to zoom
    if (cameraMode === 'overview') {
      ev.preventDefault()
      const zoomSpeed = orbitDistance * 0.1
      orbitDistance += ev.deltaY * 0.01 * zoomSpeed
      orbitDistance = Math.max(5, Math.min(500, orbitDistance))
    }
  }, { passive: false })

  // Helper: check if a screen point is near an ellipse ring
  function isNearEllipseRing(clickX: number, clickY: number, center: Vec3, r1: number, r2: number, axis: 'xy' | 'xz' | 'yz'): boolean {
    const segments = 32
    const threshold = 15 // pixels

    for (let i = 0; i <= segments; i++) {
      const angle = (i / segments) * Math.PI * 2
      let point: Vec3

      if (axis === 'xy') {
        point = [center[0] + r1 * Math.cos(angle), center[1] + r2 * Math.sin(angle), center[2]]
      } else if (axis === 'xz') {
        point = [center[0] + r1 * Math.cos(angle), center[1], center[2] + r2 * Math.sin(angle)]
      } else {
        point = [center[0], center[1] + r1 * Math.cos(angle), center[2] + r2 * Math.sin(angle)]
      }

      const projected = projectToScreen(chunkToWorld(point), latestCamera!.viewProjectionMatrix, canvas.width, canvas.height)
      if (projected) {
        const dist = Math.hypot(projected[0] - clickX, projected[1] - clickY)
        if (dist < threshold) return true
      }
    }

    return false
  }

  // Helper: show context menu for node
  function showNodeContextMenu(node: EllipsoidNode, x: number, y: number) {
    if (!node) return

    // Remove any existing menu
    const existingMenu = document.querySelector('.ellipsoid-context-menu')
    if (existingMenu) existingMenu.remove()

    // Create menu container
    const menu = document.createElement('div')
    menu.className = 'ellipsoid-context-menu'
    menu.style.position = 'fixed'
    menu.style.left = x + 'px'
    menu.style.top = y + 'px'
    menu.style.background = 'rgba(30, 30, 40, 0.95)'
    menu.style.border = '1px solid rgba(255, 255, 255, 0.2)'
    menu.style.borderRadius = '4px'
    menu.style.padding = '4px 0'
    menu.style.zIndex = '10000'
    menu.style.minWidth = '120px'
    menu.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.3)'

    // Move option
    const moveOption = document.createElement('div')
    moveOption.textContent = 'Move'
    moveOption.style.padding = '8px 16px'
    moveOption.style.cursor = 'pointer'
    moveOption.style.color = '#fff'
    moveOption.style.fontSize = '14px'
    moveOption.style.transition = 'background 0.1s'
    moveOption.onmouseenter = () => moveOption.style.background = 'rgba(255, 255, 255, 0.1)'
    moveOption.onmouseleave = () => moveOption.style.background = 'transparent'
    moveOption.onclick = () => {
      menu.remove()
      // Start move/drag mode
      ellipsoidSelectedNode.set(node)
      ellipsoidNodeAdjustActive = true
      ellipsoidEditAxis.set(null)
      ellipsoidMovementActive = false
      ellipsoidCenterDragActive = false
      centerNodeSelected = false
      activatedNodes.add(node)

      nodeEditStartX = x
      nodeEditStartY = y

      // Check if we're working with plane or ellipsoid
      const currentSelection = get(highlightSelectionStore)
      if (currentSelection && currentSelection.shape === 'plane') {
        nodeEditStartRadius = get(planeSize)
      } else {
        const axis = node[1] as 'x' | 'y' | 'z'
        if (axis === 'x') nodeEditStartRadius = get(ellipsoidRadiusX)
        else if (axis === 'y') nodeEditStartRadius = get(ellipsoidRadiusY)
        else if (axis === 'z') nodeEditStartRadius = get(ellipsoidRadiusZ)
      }

      console.log('Context menu: Starting move mode for node', node)
    }

    menu.appendChild(moveOption)

    // Generate Terrain (LLM) option
    const llmOption = document.createElement('div')
    llmOption.textContent = 'Generate Terrain (LLM)'
    llmOption.style.padding = '8px 16px'
    llmOption.style.cursor = 'pointer'
    llmOption.style.color = '#fff'
    llmOption.style.fontSize = '14px'
    llmOption.style.transition = 'background 0.1s'
    llmOption.onmouseenter = () => llmOption.style.background = 'rgba(255, 255, 255, 0.1)'
    llmOption.onmouseleave = () => llmOption.style.background = 'transparent'
    llmOption.onclick = () => {
      menu.remove()
      showTerrainGenerationDialog()
    }

    menu.appendChild(llmOption)
    document.body.appendChild(menu)

    // Close menu when clicking elsewhere
    const closeMenu = (e: MouseEvent) => {
      if (!menu.contains(e.target as Node)) {
        menu.remove()
        document.removeEventListener('mousedown', closeMenu)
      }
    }
    setTimeout(() => document.addEventListener('mousedown', closeMenu), 0)
  }

  // Helper: show terrain generation dialog
  function showTerrainGenerationDialog() {
    // Remove any existing dialog
    const existingDialog = document.querySelector('.terrain-generation-dialog')
    if (existingDialog) existingDialog.remove()

    // Create modal overlay
    const overlay = document.createElement('div')
    overlay.className = 'terrain-generation-dialog'
    overlay.style.position = 'fixed'
    overlay.style.top = '0'
    overlay.style.left = '0'
    overlay.style.width = '100%'
    overlay.style.height = '100%'
    overlay.style.background = 'rgba(0, 0, 0, 0.7)'
    overlay.style.display = 'flex'
    overlay.style.alignItems = 'center'
    overlay.style.justifyContent = 'center'
    overlay.style.zIndex = '20000'

    // Create dialog box
    const dialog = document.createElement('div')
    dialog.style.background = 'rgba(30, 30, 40, 0.98)'
    dialog.style.border = '1px solid rgba(255, 255, 255, 0.2)'
    dialog.style.borderRadius = '8px'
    dialog.style.padding = '24px'
    dialog.style.minWidth = '400px'
    dialog.style.maxWidth = '600px'
    dialog.style.boxShadow = '0 4px 16px rgba(0, 0, 0, 0.5)'

    // Title
    const title = document.createElement('h3')
    title.textContent = 'Generate Terrain with LLM'
    title.style.margin = '0 0 16px 0'
    title.style.color = '#fff'
    title.style.fontSize = '18px'
    title.style.fontWeight = '600'

    // Description
    const desc = document.createElement('p')
    desc.textContent = 'Describe the terrain you want to generate:'
    desc.style.margin = '0 0 12px 0'
    desc.style.color = 'rgba(255, 255, 255, 0.8)'
    desc.style.fontSize = '14px'

    // Text input
    const input = document.createElement('textarea')
    input.placeholder = 'e.g., "rolling hills with mountains in the distance"'
    input.style.width = '100%'
    input.style.height = '100px'
    input.style.padding = '12px'
    input.style.background = 'rgba(0, 0, 0, 0.3)'
    input.style.border = '1px solid rgba(255, 255, 255, 0.2)'
    input.style.borderRadius = '4px'
    input.style.color = '#fff'
    input.style.fontSize = '14px'
    input.style.fontFamily = 'inherit'
    input.style.resize = 'vertical'
    input.style.marginBottom = '16px'
    input.style.boxSizing = 'border-box'

    // Buttons container
    const buttons = document.createElement('div')
    buttons.style.display = 'flex'
    buttons.style.gap = '12px'
    buttons.style.justifyContent = 'flex-end'

    // Cancel button
    const cancelBtn = document.createElement('button')
    cancelBtn.textContent = 'Cancel'
    cancelBtn.style.padding = '8px 16px'
    cancelBtn.style.background = 'rgba(255, 255, 255, 0.1)'
    cancelBtn.style.border = '1px solid rgba(255, 255, 255, 0.2)'
    cancelBtn.style.borderRadius = '4px'
    cancelBtn.style.color = '#fff'
    cancelBtn.style.fontSize = '14px'
    cancelBtn.style.cursor = 'pointer'
    cancelBtn.style.transition = 'background 0.1s'
    cancelBtn.onmouseenter = () => cancelBtn.style.background = 'rgba(255, 255, 255, 0.15)'
    cancelBtn.onmouseleave = () => cancelBtn.style.background = 'rgba(255, 255, 255, 0.1)'
    cancelBtn.onclick = () => overlay.remove()

    // Generate button
    const generateBtn = document.createElement('button')
    generateBtn.textContent = 'Generate'
    generateBtn.style.padding = '8px 16px'
    generateBtn.style.background = 'rgba(80, 140, 255, 0.9)'
    generateBtn.style.border = '1px solid rgba(100, 160, 255, 0.9)'
    generateBtn.style.borderRadius = '4px'
    generateBtn.style.color = '#fff'
    generateBtn.style.fontSize = '14px'
    generateBtn.style.fontWeight = '600'
    generateBtn.style.cursor = 'pointer'
    generateBtn.style.transition = 'background 0.1s'
    generateBtn.onmouseenter = () => generateBtn.style.background = 'rgba(100, 160, 255, 0.9)'
    generateBtn.onmouseleave = () => generateBtn.style.background = 'rgba(80, 140, 255, 0.9)'
    generateBtn.onclick = async () => {
      const description = input.value.trim()
      if (!description) {
        alert('Please enter a terrain description')
        return
      }

      generateBtn.disabled = true
      generateBtn.textContent = 'Generating...'
      generateBtn.style.opacity = '0.6'

      try {
        await generateTerrainWithLLM(description)
        overlay.remove()
      } catch (error) {
        alert('Error generating terrain: ' + (error as Error).message)
        generateBtn.disabled = false
        generateBtn.textContent = 'Generate'
        generateBtn.style.opacity = '1'
      }
    }

    buttons.appendChild(cancelBtn)
    buttons.appendChild(generateBtn)

    dialog.appendChild(title)
    dialog.appendChild(desc)
    dialog.appendChild(input)
    dialog.appendChild(buttons)
    overlay.appendChild(dialog)
    document.body.appendChild(overlay)

    // Auto-focus input
    setTimeout(() => input.focus(), 0)

    // Close on overlay click
    overlay.addEventListener('mousedown', (e) => {
      if (e.target === overlay) overlay.remove()
    })

    // Close on Escape key
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        overlay.remove()
        document.removeEventListener('keydown', handleEscape)
      }
    }
    document.addEventListener('keydown', handleEscape)
  }

  // Helper: generate terrain using OpenAI LLM
  async function generateTerrainWithLLM(description: string) {
    console.log('=== OPENAI API CALL TRIGGERED ===')
    console.log('[LLM] Description:', description)
    console.trace('Call stack:')

    // Check if there's a current ellipsoid/plane selection
    const currentSelection = get(highlightSelectionStore)
    if (!currentSelection) {
      alert('Please create an ellipsoid or plane selection first (right-click to create)')
      return
    }

    const apiKey = import.meta.env.VITE_OPENAI_API_KEY
    if (!apiKey) {
      console.error('[LLM] No API key found')
      throw new Error('VITE_OPENAI_API_KEY not found in environment variables. Please set it in your .env file.')
    }

    console.log('[LLM] Generating terrain for description:', description)
    console.log('[LLM] Using selection:', currentSelection)

    // Prepare the prompt for the LLM
    const systemPrompt = `You are a terrain generation assistant. Based on the user's description, respond with ONLY a JSON object (no explanation, no markdown) with these fields:

{
  "profile": "rolling_hills" | "mountain" | "hybrid",
  "amplitude": number (8-18, controls terrain height variation),
  "roughness": number (2.0-3.0, controls terrain detail/jaggedness),
  "elevation": number (0.3-0.6, controls base height)
}

Guidelines:
- rolling_hills: gentle, smooth terrain (amplitude: 8-12, roughness: 2.0-2.4)
- mountain: dramatic peaks and valleys (amplitude: 14-18, roughness: 2.6-3.0)
- hybrid: mix of both (amplitude: 10-14, roughness: 2.2-2.6)

Examples:
- "gentle rolling hills" → {"profile": "rolling_hills", "amplitude": 10, "roughness": 2.2, "elevation": 0.4}
- "dramatic mountains" → {"profile": "mountain", "amplitude": 16, "roughness": 2.8, "elevation": 0.45}
- "varied landscape" → {"profile": "hybrid", "amplitude": 12, "roughness": 2.5, "elevation": 0.42}`

    const userPrompt = `Terrain description: ${description}`

    try {
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({
          model: 'gpt-4',
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userPrompt }
          ],
          temperature: 0.7,
          max_tokens: 150
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(`OpenAI API error: ${errorData.error?.message || response.statusText}`)
      }

      const data = await response.json()
      const llmResponse = data.choices[0]?.message?.content?.trim()

      if (!llmResponse) {
        throw new Error('Empty response from LLM')
      }

      console.log('[LLM] Raw response:', llmResponse)

      // Parse JSON response
      const params = JSON.parse(llmResponse)
      console.log('[LLM] Parsed params:', params)

      // Now use the gpuHooks.generateTerrain with the current selection
      // This will use the ellipsoid/plane you created
      const { generateRegion, createTerrainGeneratorState } = await import('./procedural/terrainGenerator')

      // Get world coordinates from the current selection
      const worldCenter = chunkToWorld([
        currentSelection.center[0] + 0.5,
        currentSelection.center[1] + 0.5,
        currentSelection.center[2] + 0.5
      ])

      let region: { min: Vec3; max: Vec3 }
      let ellipsoidMask = null

      if (currentSelection.shape === 'ellipsoid') {
        const rx = (currentSelection.radiusX ?? currentSelection.radius) * worldScale
        const ry = (currentSelection.radiusY ?? currentSelection.radius) * worldScale
        const rz = (currentSelection.radiusZ ?? currentSelection.radius) * worldScale

        region = {
          min: [
            Math.floor(worldCenter[0] - rx),
            Math.floor(worldCenter[1] - ry),
            Math.floor(worldCenter[2] - rz)
          ],
          max: [
            Math.floor(worldCenter[0] + rx),
            Math.floor(worldCenter[1] + ry),
            Math.floor(worldCenter[2] + rz)
          ]
        }

        ellipsoidMask = {
          center: worldCenter,
          radiusX: rx,
          radiusY: ry,
          radiusZ: rz
        }
      } else if (currentSelection.shape === 'plane') {
        const size = (currentSelection.planeSize ?? 8) * worldScale
        region = {
          min: [
            Math.floor(worldCenter[0] - size),
            Math.floor(worldCenter[1]),
            Math.floor(worldCenter[2] - size)
          ],
          max: [
            Math.floor(worldCenter[0] + size),
            Math.floor(worldCenter[1] + 64),
            Math.floor(worldCenter[2] + size)
          ]
        }
      }

      console.log('[LLM] Generating terrain with params:', params)
      console.log('[LLM] Region:', region)
      console.log('[LLM] Ellipsoid mask:', ellipsoidMask)

      // Convert world coordinates back to chunk coordinates for terrain generation
      const chunkMin = worldToChunk(region.min)
      const chunkMax = worldToChunk(region.max)

      const chunkRegion = {
        min: [
          Math.max(0, Math.floor(chunkMin[0])),
          Math.max(0, Math.floor(chunkMin[1])),
          Math.max(0, Math.floor(chunkMin[2]))
        ] as Vec3,
        max: [
          Math.min(chunk.size.x - 1, Math.floor(chunkMax[0])),
          Math.min(chunk.size.y - 1, Math.floor(chunkMax[1])),
          Math.min(chunk.size.z - 1, Math.floor(chunkMax[2]))
        ] as Vec3
      }

      console.log('[LLM] Chunk region:', chunkRegion)

      const terrainState = createTerrainGeneratorState(params.profile, {
        seed: Math.floor(Math.random() * 1000000),
        amplitude: params.amplitude,
        roughness: params.roughness,
        elevation: params.elevation
      })

      generateRegion(chunk, chunkRegion, terrainState)

      // Handle ellipsoid masking if needed
      if (ellipsoidMask) {
        const isInsideEllipsoid = (chunkX: number, chunkY: number, chunkZ: number): boolean => {
          const worldX = chunkX * worldScale + chunkOriginOffset[0]
          const worldY = chunkY * worldScale + chunkOriginOffset[1]
          const worldZ = chunkZ * worldScale + chunkOriginOffset[2]

          const dx = (worldX - ellipsoidMask.center[0]) / ellipsoidMask.radiusX
          const dy = (worldY - ellipsoidMask.center[1]) / ellipsoidMask.radiusY
          const dz = (worldZ - ellipsoidMask.center[2]) / ellipsoidMask.radiusZ

          return (dx * dx + dy * dy + dz * dz) <= 1
        }

        // Clear blocks outside ellipsoid
        for (let x = chunkRegion.min[0]; x <= chunkRegion.max[0]; x++) {
          for (let y = chunkRegion.min[1]; y <= chunkRegion.max[1]; y++) {
            for (let z = chunkRegion.min[2]; z <= chunkRegion.max[2]; z++) {
              if (!isInsideEllipsoid(x, y, z)) {
                chunk.setBlock(x, y, z, 0) // BlockType.Air
              }
            }
          }
        }
      }

      console.log('[LLM] Marking mesh dirty')
      meshDirty = true

      alert(`Terrain generated with ${params.profile} profile!`)
    } catch (error) {
      console.error('[LLM] Error:', error)
      throw error
    }
  }

  // Helper: execute terrain-dsl command
  async function executeTerrainCommand(command: string, chunkManager: ChunkManager) {
    // Remove "generate " prefix if present
    const cleanCommand = command.replace(/^generate\s+/, '').trim()
    const args = cleanCommand.split(/\s+/)

    if (args.length < 5) {
      throw new Error('Invalid terrain command format')
    }

    const cx1 = parseInt(args[0]!)
    const cz1 = parseInt(args[1]!)
    const cx2 = parseInt(args[2]!)
    const cz2 = parseInt(args[3]!)
    const profile = args[4] as 'rolling_hills' | 'mountain' | 'hybrid'

    if (!['rolling_hills', 'mountain', 'hybrid'].includes(profile)) {
      throw new Error(`Invalid profile: ${profile}`)
    }

    const seed = args[5] ? parseInt(args[5]) : undefined
    const amplitude = args[6] ? parseFloat(args[6]) : undefined
    const roughness = args[7] ? parseFloat(args[7]) : undefined
    const elevation = args[8] ? parseFloat(args[8]) : undefined

    console.log('[Terrain] Executing command:', {
      region: { cx1, cz1, cx2, cz2 },
      profile,
      params: { seed, amplitude, roughness, elevation }
    })

    // Import and execute terrain generation
    try {
      const { generateRegion, createTerrainGeneratorState } = await import('./procedural/terrainGenerator')

      // Convert chunk coordinates to block coordinates
      const chunkSize = 32 // Standard chunk size
      const minX = cx1 * chunkSize
      const minZ = cz1 * chunkSize
      const maxX = (cx2 + 1) * chunkSize - 1
      const maxZ = (cz2 + 1) * chunkSize - 1

      const region = {
        min: [minX, 0, minZ] as Vec3,
        max: [maxX, chunkSize - 1, maxZ] as Vec3
      }

      const state = createTerrainGeneratorState(profile, {
        seed,
        amplitude,
        roughness,
        elevation
      })

      generateRegion(chunkManager, region, state)
      console.log('[Terrain] Generation complete!')
      alert(`Terrain generated successfully!\nRegion: [${cx1},${cz1}] to [${cx2},${cz2}]\nProfile: ${profile}`)
    } catch (error) {
      console.error('[Terrain] Execution error:', error)
      throw error
    }
  }

  // Helper: check if a click is near an ellipsoid or plane node
  function getClickedNode(clickX: number, clickY: number, center: Vec3, rx: number, ry: number, rz: number, shape: 'ellipsoid' | 'plane' = 'ellipsoid'): EllipsoidNode {
    const halfStep = 0.5
    const baseCenter: Vec3 = [center[0] + halfStep, center[1] + halfStep, center[2] + halfStep]
    const threshold = 20 // pixels - increased for easier clicking

    console.log('getClickedNode called:', { clickX, clickY, center, rx, ry, rz, shape })
    console.log('Canvas dimensions:', { width: canvas.width, height: canvas.height })
    console.log('Canvas client rect:', canvas.getBoundingClientRect())

    // Check center node first for easier selection
    const centerScreen = projectToScreen(chunkToWorld(baseCenter), latestCamera!.viewProjectionMatrix, canvas.width, canvas.height)
    console.log('Center node screen pos:', centerScreen)
    if (centerScreen) {
      const centerDist = Math.hypot(centerScreen[0] - clickX, centerScreen[1] - clickY)
      console.log('Center node distance:', centerDist, 'threshold:', threshold)
      if (centerDist < threshold) {
        console.log('>>> CENTER NODE CLICKED')
        return 'center'
      }
    }

    let nodes: Array<{ pos: Vec3; id: EllipsoidNode }>

    if (shape === 'plane') {
      // For plane, use corner nodes at the same Y level
      const size = rx // For plane, rx is actually the planeSize
      const y = baseCenter[1]
      nodes = [
        { pos: [baseCenter[0] + size, y, baseCenter[2] + size], id: '+x' },  // +X+Z corner
        { pos: [baseCenter[0] - size, y, baseCenter[2] + size], id: '-x' },  // -X+Z corner
        { pos: [baseCenter[0] + size, y, baseCenter[2] - size], id: '+z' },  // +X-Z corner
        { pos: [baseCenter[0] - size, y, baseCenter[2] - size], id: '-z' }   // -X-Z corner
      ]
    } else {
      // Ellipsoid nodes
      nodes = [
        { pos: [baseCenter[0] + rx, baseCenter[1], baseCenter[2]], id: '+x' },
        { pos: [baseCenter[0] - rx, baseCenter[1], baseCenter[2]], id: '-x' },
        { pos: [baseCenter[0], baseCenter[1] + ry, baseCenter[2]], id: '+y' },
        { pos: [baseCenter[0], baseCenter[1] - ry, baseCenter[2]], id: '-y' },
        { pos: [baseCenter[0], baseCenter[1], baseCenter[2] + rz], id: '+z' },
        { pos: [baseCenter[0], baseCenter[1], baseCenter[2] - rz], id: '-z' }
      ]
    }

    console.log('Checking nodes:')
    for (const node of nodes) {
      const screen = projectToScreen(chunkToWorld(node.pos), latestCamera!.viewProjectionMatrix, canvas.width, canvas.height)
      if (screen) {
        const dist = Math.hypot(screen[0] - clickX, screen[1] - clickY)
        console.log(`  ${node.id}: screen=${screen[0].toFixed(1)},${screen[1].toFixed(1)} dist=${dist.toFixed(1)} threshold=${threshold}`)
        if (dist < threshold) {
          console.log(`>>> NODE ${node.id} CLICKED`)
          return node.id
        }
      } else {
        console.log(`  ${node.id}: not visible (off screen)`)
      }
    }

    console.log('>>> NO NODE CLICKED')
    return null
  }

  window.addEventListener('mousedown', (ev) => {
    const path = ev.composedPath() as EventTarget[]
    const isCanvasEvent = canvas ? path.includes(canvas) : false
    const isOverlayEvent = overlayCanvas ? path.includes(overlayCanvas) : false

    if (!pointerActive && !isCanvasEvent && !isOverlayEvent) return

    const mode = get(interactionMode)

    let handledEllipsoidInteraction = false
    const currentShape = get(highlightShape)
    if ((isCanvasEvent || isOverlayEvent) && mode === 'highlight' && (currentShape === 'ellipsoid' || currentShape === 'plane')) {
      // Convert client coordinates to canvas coordinates
      // IMPORTANT: Use overlayCanvas rect if available, since nodes are drawn on overlay
      const targetCanvas = overlayCanvas || canvas
      const rect = targetCanvas.getBoundingClientRect()
      const canvasX = ((ev.clientX - rect.left) / rect.width) * canvas.width
      const canvasY = ((ev.clientY - rect.top) / rect.height) * canvas.height

      const currentSelection = get(highlightSelectionStore)
      const shape = get(highlightShape)
      const radiusDefault = get(highlightRadius)
      let clickedNode: EllipsoidNode = null

      if (currentSelection && currentSelection.shape === 'ellipsoid') {
        const rx = currentSelection.radiusX ?? currentSelection.radius
        const ry = currentSelection.radiusY ?? currentSelection.radius
        const rz = currentSelection.radiusZ ?? currentSelection.radius

        clickedNode = getClickedNode(canvasX, canvasY, currentSelection.center, rx, ry, rz, 'ellipsoid')
      } else if (currentSelection && currentSelection.shape === 'plane') {
        const size = currentSelection.planeSize ?? 8

        clickedNode = getClickedNode(canvasX, canvasY, currentSelection.center, size, 0, 0, 'plane')
      }

      if (ev.button === 2) {
        ev.preventDefault()

        // Check if right-clicking on a node - show context menu
        if (clickedNode && clickedNode !== 'center') {
          showNodeContextMenu(clickedNode, ev.clientX, ev.clientY)
          handledEllipsoidInteraction = true
          return
        }

        // Right-click in player mode: create/position ellipsoid/plane at raycast location
        const hit = raycast(cameraPos, getForwardVector())
        if (hit) {
          const selection: HighlightSelection = {
            center: hit.block,
            shape,
            radius: radiusDefault
          }
          if (shape === 'ellipsoid') {
            selection.radiusX = get(ellipsoidRadiusX)
            selection.radiusY = get(ellipsoidRadiusY)
            selection.radiusZ = get(ellipsoidRadiusZ)
          } else if (shape === 'plane') {
            selection.planeSize = get(planeSize)
          }
          ellipsoidSelectedNode.set(null)
          ellipsoidEditAxis.set(null)
          centerNodeSelected = false
          highlightSelectionStore.set(selection)
          ellipsoidMovementActive = true
          ellipsoidNodeAdjustActive = false
          ellipsoidCenterDragActive = false
          lastCameraPos = getCameraWorldPosition()
          handledEllipsoidInteraction = true
        } else {
          ellipsoidMovementActive = false
        }
      } else if (ev.button === 0 && currentSelection && (currentSelection.shape === 'ellipsoid' || currentSelection.shape === 'plane')) {
        if (clickedNode === 'center') {
          ev.preventDefault()
          ellipsoidCenterDragActive = true
          centerNodeSelected = true
          ellipsoidMovementActive = false
          ellipsoidNodeAdjustActive = false
          ellipsoidSelectedNode.set('center')
          ellipsoidEditAxis.set(null)
          lastCameraPos = getCameraWorldPosition()
          handledEllipsoidInteraction = true
        } else if (clickedNode) {
          // Left-click on axis/corner node: activate mouse-controlled scaling
          ev.preventDefault()
          ellipsoidNodeAdjustActive = true
          ellipsoidSelectedNode.set(clickedNode)
          ellipsoidEditAxis.set(null)
          ellipsoidMovementActive = false
          ellipsoidCenterDragActive = false
          centerNodeSelected = false
          activatedNodes.add(clickedNode) // Mark as activated

          // Store initial mouse position and current radius
          nodeEditStartX = ev.clientX
          nodeEditStartY = ev.clientY

          if (currentSelection.shape === 'plane') {
            // For plane, all corner nodes adjust the same planeSize
            nodeEditStartRadius = get(planeSize)
          } else {
            const axis = clickedNode[1] as 'x' | 'y' | 'z'
            if (axis === 'x') nodeEditStartRadius = get(ellipsoidRadiusX)
            else if (axis === 'y') nodeEditStartRadius = get(ellipsoidRadiusY)
            else if (axis === 'z') nodeEditStartRadius = get(ellipsoidRadiusZ)
          }

          handledEllipsoidInteraction = true
        } else {
          const center = currentSelection.center
          const rx = currentSelection.radiusX ?? currentSelection.radius
          const ry = currentSelection.radiusY ?? currentSelection.radius
          const rz = currentSelection.radiusZ ?? currentSelection.radius

          if (isNearEllipseRing(canvasX, canvasY, center, rx, ry, 'xy')) {
            ellipsoidEditAxis.set('z')
            ellipsoidSelectedNode.set(null)
            centerNodeSelected = false
            lastCameraPos = getCameraWorldPosition()
            handledEllipsoidInteraction = true
          } else if (isNearEllipseRing(canvasX, canvasY, center, rx, rz, 'xz')) {
            ellipsoidEditAxis.set('y')
            ellipsoidSelectedNode.set(null)
            centerNodeSelected = false
            lastCameraPos = getCameraWorldPosition()
            handledEllipsoidInteraction = true
          } else if (isNearEllipseRing(canvasX, canvasY, center, ry, rz, 'yz')) {
            ellipsoidEditAxis.set('x')
            ellipsoidSelectedNode.set(null)
            centerNodeSelected = false
            lastCameraPos = getCameraWorldPosition()
            handledEllipsoidInteraction = true
          }
        }
      }
    }

    if (handledEllipsoidInteraction) {
      return
    }

    if (!pointerActive && cameraMode === 'overview') {
      return
    }

    const hit = raycast(cameraPos, getForwardVector())
    if (!hit) return

    if (mode === 'block') {
      if (ev.button === 0) { // Left click - remove block
        chunk.setBlock(hit.block[0], hit.block[1], hit.block[2], BlockType.Air)
        meshDirty = true
      } else if (ev.button === 2) { // Right click - place block
        const placePos = hit.previous
        if (isInsideChunk(placePos) && chunk.getBlock(placePos[0], placePos[1], placePos[2]) === BlockType.Air) {
          const selected = getSelectedBlock()
          chunk.setBlock(placePos[0], placePos[1], placePos[2], selected.type)
          meshDirty = true
        }
      }
    } else if (mode === 'highlight') {
      const shape = get(highlightShape)
      const radius = get(highlightRadius)
      const selection: HighlightSelection = {
        center: hit.block,
        shape,
        radius
      }

      if (shape === 'ellipsoid') {
        selection.radiusX = get(ellipsoidRadiusX)
        selection.radiusY = get(ellipsoidRadiusY)
        selection.radiusZ = get(ellipsoidRadiusZ)
        ellipsoidEditAxis.set(null) // Reset editing mode
        ellipsoidSelectedNode.set(null) // Reset node selection
      }

      highlightSelectionStore.set(selection)
    }
  })

  // Prevent context menu on right click
  canvas.addEventListener('contextmenu', (ev) => {
    ev.preventDefault()
  })

  function getForwardVector(): Vec3 {
    return normalize([Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch), Math.cos(pitch) * Math.cos(yaw)])
  }

  function getCameraWorldPosition(): Vec3 {
    if (cameraMode === 'player') {
      return [...cameraPos] as Vec3
    }

    return [
      orbitTarget[0] + orbitDistance * Math.cos(orbitPitch) * Math.sin(orbitYaw),
      orbitTarget[1] + orbitDistance * Math.sin(orbitPitch),
      orbitTarget[2] + orbitDistance * Math.cos(orbitPitch) * Math.cos(orbitYaw)
    ] as Vec3
  }

  interface RaycastHit {
    block: Vec3
    normal: Vec3
    previous: Vec3
  }

  function worldToChunk(worldPos: Vec3): Vec3 {
    return [
      (worldPos[0] - chunkOriginOffset[0]) / worldScale,
      (worldPos[1] - chunkOriginOffset[1]) / worldScale,
      (worldPos[2] - chunkOriginOffset[2]) / worldScale
    ]
  }

  function isInsideChunk([x, y, z]: Vec3): boolean {
    return x >= 0 && y >= 0 && z >= 0 && x < chunk.size.x && y < chunk.size.y && z < chunk.size.z
  }

  // DDA (Digital Differential Analyzer) voxel traversal algorithm
  function raycast(origin: Vec3, direction: Vec3, maxDistance = 100): RaycastHit | null {
    const dir = normalize(direction)
    if (Math.abs(dir[0]) < 1e-5 && Math.abs(dir[1]) < 1e-5 && Math.abs(dir[2]) < 1e-5) return null

    const pos = worldToChunk(origin)
    let voxel: Vec3 = [Math.floor(pos[0]), Math.floor(pos[1]), Math.floor(pos[2])]
    let prevVoxel: Vec3 = [...voxel] as Vec3

    const step: Vec3 = [
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
      if (distance > maxDistance / worldScale) break

      if (isInsideChunk(voxel)) {
        const block = chunk.getBlock(voxel[0], voxel[1], voxel[2])
        if (block !== BlockType.Air) {
          const hitNormal = (Math.abs(normal[0]) + Math.abs(normal[1]) + Math.abs(normal[2]) > 0)
            ? normal
            : [-Math.sign(dir[0]), -Math.sign(dir[1]), -Math.sign(dir[2])] as Vec3
          return {
            block: [...voxel] as Vec3,
            previous: [...prevVoxel] as Vec3,
            normal: hitNormal
          }
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

      prevVoxel = [...voxel] as Vec3
      if (axis === 0) {
        voxel = [voxel[0] + step[0], voxel[1], voxel[2]]
        distance = tMaxX
        tMaxX += tDelta[0]
        normal = [-step[0], 0, 0] as Vec3
      } else if (axis === 1) {
        voxel = [voxel[0], voxel[1] + step[1], voxel[2]]
        distance = tMaxY
        tMaxY += tDelta[1]
        normal = [0, -step[1], 0] as Vec3
      } else {
        voxel = [voxel[0], voxel[1], voxel[2] + step[2]]
        distance = tMaxZ
        tMaxZ += tDelta[2]
        normal = [0, 0, -step[2]] as Vec3
      }
    }
    return null
  }

  function updateCamera(dt: number) {
    const aspect = canvas.width / Math.max(1, canvas.height)
    const fovYRad = (60 * Math.PI) / 180
    const near = 0.1, far = 500.0
    const proj = createPerspective(fovYRad, aspect, near, far)

    let view: Mat4
    let forward: Vec3
    let right: Vec3
    let upVec: Vec3
    let position: Vec3

    if (cameraMode === 'player') {
      // Player mode: first-person camera
      forward = getForwardVector()
      right = cross(forward, worldUp)
      if (Math.hypot(right[0], right[1], right[2]) < 1e-4) right = [1, 0, 0]
      right = normalize(right)
      upVec = normalize(cross(right, forward))

      const speedBase = (pressedKeys.has('ShiftLeft') || pressedKeys.has('ShiftRight')) ? 20 : 10
      const speed = paused ? 0 : speedBase * worldScale
      const move = speed * dt
      if (move !== 0) {
        if (pressedKeys.has('KeyW')) addScaled(cameraPos, forward, move)
        if (pressedKeys.has('KeyS')) addScaled(cameraPos, forward, -move)
        if (pressedKeys.has('KeyA')) addScaled(cameraPos, right, -move)
        if (pressedKeys.has('KeyD')) addScaled(cameraPos, right, move)
        if (pressedKeys.has('KeyE') || pressedKeys.has('Space')) addScaled(cameraPos, upVec, move)
        if (pressedKeys.has('KeyQ') || pressedKeys.has('ShiftLeft')) addScaled(cameraPos, upVec, -move)
      }

      position = [...cameraPos] as Vec3
      const target: Vec3 = [cameraPos[0] + forward[0], cameraPos[1] + forward[1], cameraPos[2] + forward[2]]
      view = lookAt(cameraPos, target, upVec)
    } else {
      // Overview mode: orbital camera
      const speedBase = (pressedKeys.has('ShiftLeft') || pressedKeys.has('ShiftRight')) ? 40 : 20
      const speed = speedBase * worldScale
      const move = speed * dt

      // WASD moves the orbit target (fast panning)
      if (move !== 0) {
        const panRight: Vec3 = [Math.cos(orbitYaw), 0, -Math.sin(orbitYaw)]
        const panForward: Vec3 = [-Math.sin(orbitYaw), 0, -Math.cos(orbitYaw)]

        if (pressedKeys.has('KeyW')) {
          orbitTarget[0] += panForward[0] * move
          orbitTarget[2] += panForward[2] * move
        }
        if (pressedKeys.has('KeyS')) {
          orbitTarget[0] -= panForward[0] * move
          orbitTarget[2] -= panForward[2] * move
        }
        if (pressedKeys.has('KeyA')) {
          orbitTarget[0] -= panRight[0] * move
          orbitTarget[2] -= panRight[2] * move
        }
        if (pressedKeys.has('KeyD')) {
          orbitTarget[0] += panRight[0] * move
          orbitTarget[2] += panRight[2] * move
        }
        if (pressedKeys.has('KeyE') || pressedKeys.has('Space')) {
          orbitTarget[1] += move
        }
        if (pressedKeys.has('KeyQ') || pressedKeys.has('ShiftLeft')) {
          orbitTarget[1] -= move
        }
      }

      // Calculate camera position from orbit parameters
      const cosPitch = Math.cos(orbitPitch)
      const sinPitch = Math.sin(orbitPitch)
      const cosYaw = Math.cos(orbitYaw)
      const sinYaw = Math.sin(orbitYaw)

      position = [
        orbitTarget[0] + orbitDistance * cosPitch * sinYaw,
        orbitTarget[1] + orbitDistance * sinPitch,
        orbitTarget[2] + orbitDistance * cosPitch * cosYaw
      ] as Vec3

      view = lookAt(position, orbitTarget, worldUp)

      // Calculate vectors for camera snapshot
      forward = normalize([
        orbitTarget[0] - position[0],
        orbitTarget[1] - position[1],
        orbitTarget[2] - position[2]
      ])
      right = cross(forward, worldUp)
      if (Math.hypot(right[0], right[1], right[2]) < 1e-4) right = [1, 0, 0]
      right = normalize(right)
      upVec = normalize(cross(right, forward))
    }

    // Handle ellipsoid center drag (moved by camera translation)
    const editAxis = get(ellipsoidEditAxis)

    if (ellipsoidCenterDragActive && lastCameraPos) {
      // Center drag: move ellipsoid/plane center based on camera translation
      const currentSelection = get(highlightSelectionStore)

      if (currentSelection && currentSelection.shape === 'plane') {
        // For plane, only move along Y axis (vertical)
        const deltaY = position[1] - lastCameraPos[1]
        const sensitivity = 0.5

        highlightSelectionStore.update(sel => {
          if (sel && sel.shape === 'plane') {
            const nextCenter: Vec3 = [
              sel.center[0],
              sel.center[1] + deltaY * sensitivity / worldScale,
              sel.center[2]
            ]
            return { ...sel, center: nextCenter }
          }
          return sel
        })

        lastCameraPos = [...position] as Vec3
      } else if (currentSelection && currentSelection.shape === 'ellipsoid') {
        const deltaX = position[0] - lastCameraPos[0]
        const deltaY = position[1] - lastCameraPos[1]
        const deltaZ = position[2] - lastCameraPos[2]
        const sensitivity = 0.5

        highlightSelectionStore.update(sel => {
          if (sel && sel.shape === 'ellipsoid') {
            const nextCenter: Vec3 = [
              sel.center[0] + deltaX * sensitivity / worldScale,
              sel.center[1] + deltaY * sensitivity / worldScale,
              sel.center[2] + deltaZ * sensitivity / worldScale
            ]
            return { ...sel, center: nextCenter }
          }
          return sel
        })

        lastCameraPos = [...position] as Vec3
      }
    } else if (editAxis && lastCameraPos) {
      // Axis-based editing: drag to expand/contract both ends symmetrically
      const currentSelection = get(highlightSelectionStore)

      if (currentSelection && currentSelection.shape === 'ellipsoid') {
        // Calculate camera position delta
        const deltaX = position[0] - lastCameraPos[0]
        const deltaY = position[1] - lastCameraPos[1]
        const deltaZ = position[2] - lastCameraPos[2]

        // Map camera movement to radius changes based on selected axis
        const sensitivity = 0.5 // Controls how much camera movement affects radius

        if (editAxis === 'x') {
          // Editing X radius: use X camera movement
          const currentRadius = get(ellipsoidRadiusX)
          const newRadius = Math.max(0.5, currentRadius + deltaX * sensitivity / worldScale)
          ellipsoidRadiusX.set(newRadius)

          // Update selection
          highlightSelectionStore.update(sel => {
            if (sel && sel.shape === 'ellipsoid') {
              return { ...sel, radiusX: newRadius }
            }
            return sel
          })
        } else if (editAxis === 'y') {
          // Editing Y radius: use Y camera movement
          const currentRadius = get(ellipsoidRadiusY)
          const newRadius = Math.max(0.5, currentRadius + deltaY * sensitivity / worldScale)
          ellipsoidRadiusY.set(newRadius)

          // Update selection
          highlightSelectionStore.update(sel => {
            if (sel && sel.shape === 'ellipsoid') {
              return { ...sel, radiusY: newRadius }
            }
            return sel
          })
        } else if (editAxis === 'z') {
          // Editing Z radius: use Z camera movement
          const currentRadius = get(ellipsoidRadiusZ)
          const newRadius = Math.max(0.5, currentRadius + deltaZ * sensitivity / worldScale)
          ellipsoidRadiusZ.set(newRadius)

          // Update selection
          highlightSelectionStore.update(sel => {
            if (sel && sel.shape === 'ellipsoid') {
              return { ...sel, radiusZ: newRadius }
            }
            return sel
          })
        }

        // Update last camera position for next frame
        lastCameraPos = [...position] as Vec3
      }
    }

    const viewProj = multiplyMat4(proj, view)
    device.queue.writeBuffer(cameraBuffer, 0, viewProj)

    latestCamera = {
      position,
      forward,
      up: upVec,
      right,
      viewMatrix: new Float32Array(view),
      projectionMatrix: new Float32Array(proj),
      viewProjectionMatrix: new Float32Array(viewProj),
      fovYRadians: fovYRad,
      aspect,
      near,
      far
    }
  }

  function frame(now: number) {
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
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, renderBindGroup)
      pass.setVertexBuffer(0, vertexBuffer)
      pass.draw(vertexCount, 1, 0, 0)
    }
    pass.end()
    device.queue.submit([encoder.finish()])
    renderOverlay()
    rafHandle = requestAnimationFrame(frame)
  }

  window.addEventListener('keydown', (ev) => {
    // Tab key: toggle camera mode
    if (ev.code === 'Tab') {
      ev.preventDefault()
      const newMode = cameraMode === 'player' ? 'overview' : 'player'
      cameraModeStore.set(newMode)
      console.log(`Camera mode: ${newMode}`)
      return
    }

    if (pointerActive && ev.code !== 'Escape') ev.preventDefault()
    pressedKeys.add(ev.code)
  })

  window.addEventListener('keyup', (ev) => {
    if (pointerActive && ev.code !== 'Escape') ev.preventDefault()
    pressedKeys.delete(ev.code)
  })

  window.addEventListener('blur', () => pressedKeys.clear())

  rafHandle = requestAnimationFrame(frame)

  return {
    device,
    getCamera: () => latestCamera,
    getCameraMode: () => cameraMode,
    setCameraMode: (mode: 'player' | 'overview') => {
      if (mode === cameraMode) return

      cameraMode = mode
      if (cameraMode === 'overview') {
        if (pointerActive) document.exitPointerLock()
        paused = false
        orbitTarget = [...cameraPos] as Vec3
        orbitDistance = 50
        orbitYaw = yaw
        orbitPitch = pitch
      } else {
        paused = true
      }
      console.log(`Camera mode: ${cameraMode}`)
    },
    markMeshDirty: () => { meshDirty = true },
    applyCustomBlockTextures,
    updateCustomTextures,
    setHighlightSelection: (sel: HighlightSelection | null) => { highlightSelection = sel },
    setOverlayViews: (views: Array<{ position: Vec3; id: string }>) => { overlayViews = views },
    focusCameraOnBlocks: (blocks: Array<{ position: [number, number, number] }> | undefined) => {
      const fixedSpawnDistance = 12 // Fixed distance from block center

      if (!blocks || blocks.length === 0) {
        cameraPos[0] = 0
        cameraPos[1] = chunk.size.y * worldScale * 0.55
        cameraPos[2] = chunk.size.z * worldScale * 0.45
        yaw = 0
        pitch = 0
      } else {
        // Find center of all blocks
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity
        for (const b of blocks) {
          const [x, y, z] = b.position
          if (x < minX) minX = x; if (x > maxX) maxX = x
          if (y < minY) minY = y; if (y > maxY) maxY = y
          if (z < minZ) minZ = z; if (z > maxZ) maxZ = z
        }
        const centerChunk: Vec3 = [(minX + maxX + 1) / 2, (minY + maxY + 1) / 2, (minZ + maxZ + 1) / 2]
        const target = chunkToWorld(centerChunk)

        // Position camera high above the origin, looking straight down
        cameraPos[0] = 0
        cameraPos[1] = 80 // High above
        cameraPos[2] = 0

        // Look straight down (negative pitch in this coordinate system)
        yaw = 0
        pitch = -(Math.PI / 2 - 0.1) // Almost straight down
      }

      meshDirty = true
    },
    destroy: () => {
      if (rafHandle !== null) cancelAnimationFrame(rafHandle)
    }
  }
}
