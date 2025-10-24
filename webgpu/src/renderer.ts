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
  highlightSelection as highlightSelectionStore,
  type CustomBlock,
  type Vec3,
  type Mat4,
  type BlockFaceKey,
  type HighlightSelection
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

  const cameraPos: Vec3 = [0, chunk.size.y * worldScale * 0.55, chunk.size.z * worldScale * 0.45]
  const worldUp: Vec3 = [0, 1, 0]
  let yaw = 0, pitch = 0
  const pressedKeys = new Set<string>()
  let pointerActive = false
  let paused = true
  let rafHandle: number | null = null
  let lastFrameTime = performance.now()

  canvas.addEventListener('click', () => {
    if (!pointerActive) canvas.requestPointerLock()
  })

  document.addEventListener('pointerlockchange', () => {
    pointerActive = document.pointerLockElement === canvas
    paused = !pointerActive
    pressedKeys.clear()
    lastFrameTime = performance.now()
  })

  window.addEventListener('mousemove', (ev) => {
    if (!pointerActive) return
    yaw -= ev.movementX * 0.0025
    pitch -= ev.movementY * 0.0025
    pitch = Math.max(-Math.PI / 2 + 0.05, Math.min(Math.PI / 2 - 0.05, pitch))
  })

  window.addEventListener('mousedown', (ev) => {
    if (!pointerActive) return

    const hit = raycast(cameraPos, getForwardVector())
    if (!hit) return

    const mode = get(interactionMode)

    if (mode === 'block') {
      if (ev.button === 0) { // Left click - place block
        const selected = getSelectedBlock()
        const placePos: Vec3 = [
          hit.block[0] + hit.normal[0],
          hit.block[1] + hit.normal[1],
          hit.block[2] + hit.normal[2]
        ]
        if (placePos[0] >= 0 && placePos[0] < chunk.size.x &&
            placePos[1] >= 0 && placePos[1] < chunk.size.y &&
            placePos[2] >= 0 && placePos[2] < chunk.size.z) {
          chunk.setBlock(placePos[0], placePos[1], placePos[2], selected.type)
          meshDirty = true
        }
      } else if (ev.button === 2) { // Right click - remove block
        chunk.setBlock(hit.block[0], hit.block[1], hit.block[2], BlockType.Air)
        meshDirty = true
      }
    } else if (mode === 'highlight') {
      const shape = get(highlightShape)
      const radius = get(highlightRadius)
      highlightSelectionStore.set({
        center: hit.block,
        shape,
        radius
      })
    }
  })

  // Prevent context menu on right click
  canvas.addEventListener('contextmenu', (ev) => {
    if (pointerActive) ev.preventDefault()
  })

  function getForwardVector(): Vec3 {
    return normalize([Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch), Math.cos(pitch) * Math.cos(yaw)])
  }

  interface RaycastHit {
    block: Vec3
    normal: Vec3
    distance: number
  }

  function raycast(origin: Vec3, direction: Vec3, maxDistance = 100): RaycastHit | null {
    const step = 0.1
    let t = 0
    let lastBlock: Vec3 | null = null

    while (t < maxDistance) {
      const x = origin[0] + direction[0] * t
      const y = origin[1] + direction[1] * t
      const z = origin[2] + direction[2] * t

      // Convert world coordinates to chunk coordinates
      const chunkX = Math.floor((x - chunkOriginOffset[0]) / worldScale)
      const chunkY = Math.floor((y - chunkOriginOffset[1]) / worldScale)
      const chunkZ = Math.floor((z - chunkOriginOffset[2]) / worldScale)

      if (chunkX >= 0 && chunkX < chunk.size.x && chunkY >= 0 && chunkY < chunk.size.y && chunkZ >= 0 && chunkZ < chunk.size.z) {
        const block = chunk.getBlock(chunkX, chunkY, chunkZ)
        if (block !== BlockType.Air) {
          // Calculate normal based on previous position
          let normal: Vec3 = [0, 1, 0]
          if (lastBlock) {
            normal = [
              chunkX - lastBlock[0],
              chunkY - lastBlock[1],
              chunkZ - lastBlock[2]
            ] as Vec3
          }
          return {
            block: [chunkX, chunkY, chunkZ],
            normal,
            distance: t
          }
        }
      }

      lastBlock = [chunkX, chunkY, chunkZ]
      t += step
    }
    return null
  }

  function updateCamera(dt: number) {
    const aspect = canvas.width / Math.max(1, canvas.height)
    const fovYRad = (60 * Math.PI) / 180
    const near = 0.1, far = 500.0
    const proj = createPerspective(fovYRad, aspect, near, far)

    const forward = getForwardVector()
    let right = cross(forward, worldUp)
    if (Math.hypot(right[0], right[1], right[2]) < 1e-4) right = [1, 0, 0]
    right = normalize(right)
    const upVec = normalize(cross(right, forward))

    const speedBase = (pressedKeys.has('ShiftLeft') || pressedKeys.has('ShiftRight')) ? 20 : 10
    const speed = paused ? 0 : speedBase * worldScale
    const move = speed * dt
    if (move !== 0) {
      if (pressedKeys.has('KeyW')) addScaled(cameraPos, forward, move)
      if (pressedKeys.has('KeyS')) addScaled(cameraPos, forward, -move)
      if (pressedKeys.has('KeyA')) addScaled(cameraPos, right, -move)
      if (pressedKeys.has('KeyD')) addScaled(cameraPos, right, move)
      if (pressedKeys.has('KeyE') || pressedKeys.has('Space')) addScaled(cameraPos, upVec, move)
      if (pressedKeys.has('KeyQ') || pressedKeys.has('ControlLeft')) addScaled(cameraPos, upVec, -move)
    }

    const target: Vec3 = [cameraPos[0] + forward[0], cameraPos[1] + forward[1], cameraPos[2] + forward[2]]
    const view = lookAt(cameraPos, target, upVec)
    const viewProj = multiplyMat4(proj, view)
    device.queue.writeBuffer(cameraBuffer, 0, viewProj)

    latestCamera = {
      position: [...cameraPos] as Vec3,
      forward: [...forward] as Vec3,
      up: [...upVec] as Vec3,
      right: [...right] as Vec3,
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
