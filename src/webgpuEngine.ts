// WebGPU Rendering Engine
import terrainWGSL from './pipelines/render/terrain.wgsl?raw'
import { createPerspective, lookAt, multiplyMat4 } from './camera'
import { ChunkManager, BlockType, type BlockFaceKey, buildChunkMesh, setBlockTextureIndices } from './chunks'
import { API_BASE_URL, blockFaceOrder, TILE_BASE_URL } from './blockUtils'
import type { CustomBlock, FaceTileInfo } from './stores'
import { get } from 'svelte/store'
import { customBlocks as customBlocksStore, gpuHooks } from './stores'

type Vec3 = [number, number, number]

type CameraSnapshot = {
  position: Vec3
  forward: Vec3
  up: Vec3
  right: Vec3
  viewMatrix: Float32Array
  projectionMatrix: Float32Array
  viewProjectionMatrix: Float32Array
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

type DatasetExportResult = {
  status: string
  datasetSequence: number
  datasetDir: string
  metadataFile: string
  imageCount: number
  captureId: string
}

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

function cloneVec3(vec: Vec3): Vec3 {
  return [vec[0], vec[1], vec[2]]
}

function cloneMat4(mat: Float32Array): Float32Array {
  return new Float32Array(mat)
}

function snapshotCamera(state: CameraSnapshot): CameraSnapshot {
  return {
    position: cloneVec3(state.position),
    forward: cloneVec3(state.forward),
    up: cloneVec3(state.up),
    right: cloneVec3(state.right),
    viewMatrix: cloneMat4(state.viewMatrix),
    projectionMatrix: cloneMat4(state.projectionMatrix),
    viewProjectionMatrix: cloneMat4(state.viewProjectionMatrix),
    fovYRadians: state.fovYRadians,
    aspect: state.aspect,
    near: state.near,
    far: state.far
  }
}

function projectToScreen(point: Vec3, viewProj: Float32Array, width: number, height: number): [number, number] | null {
  const x = point[0]
  const y = point[1]
  const z = point[2]
  const clipX = viewProj[0]! * x + viewProj[4]! * y + viewProj[8]! * z + viewProj[12]!
  const clipY = viewProj[1]! * x + viewProj[5]! * y + viewProj[9]! * z + viewProj[13]!
  const clipZ = viewProj[2]! * x + viewProj[6]! * y + viewProj[10]! * z + viewProj[14]!
  const clipW = viewProj[3]! * x + viewProj[7]! * y + viewProj[11]! * z + viewProj[15]!
  if (clipW <= 0) return null
  const ndcX = clipX / clipW
  const ndcY = clipY / clipW
  const ndcZ = clipZ / clipW
  if (ndcX < -1 || ndcX > 1 || ndcY < -1 || ndcY > 1 || ndcZ < -1 || ndcZ > 1) return null
  const screenX = (ndcX * 0.5 + 0.5) * width
  const screenY = (-ndcY * 0.5 + 0.5) * height
  return [screenX, screenY]
}

export interface WebGPUEngineOptions {
  canvas: HTMLCanvasElement
  overlayCanvas?: HTMLCanvasElement | null
  onBlockSelect: (blockType: BlockType) => void
  getSelectedBlock: () => { type: BlockType; custom: CustomBlock | null }
}

const faceTileCoordinates: Record<BlockFaceKey, [number, number]> = {
  top: [1, 1],
  bottom: [1, 3],
  north: [1, 0],
  south: [1, 2],
  east: [2, 1],
  west: [0, 1]
}

const MAX_CUSTOM_BLOCKS = 8
const MAP_MODE_READ = 0x0001

export async function initWebGPUEngine(options: WebGPUEngineOptions) {
  const { canvas, getSelectedBlock } = options
  const overlayCanvas = options.overlayCanvas ?? null
  const overlayCtx = overlayCanvas ? overlayCanvas.getContext('2d') : null

  if (!('gpu' in navigator)) throw new Error('WebGPU not supported')
  const gpu = (navigator as any).gpu as any
  const adapter = await gpu.requestAdapter()
  if (!adapter) throw new Error('No adapter')
  const device = await adapter.requestDevice()

  try {
    ;(device as any).addEventListener && (device as any).addEventListener('uncapturederror', (ev: any) => {
      console.error(`device uncapturederror: ${ev?.error?.message || ev}`)
    })
  } catch {}

  const context = canvas.getContext('webgpu') as unknown as GPUCanvasContext
  const format = gpu.getPreferredCanvasFormat()
  context.configure({ device, format, alphaMode: 'opaque' })

  const capturedViews: CapturedView[] = []
  let latestCamera: CameraSnapshot | null = null
  let exportingDataset = false
  let rafHandle: number | null = null

  const generateCaptureSessionId = () => {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
      return crypto.randomUUID()
    }
    return `capture_${Math.random().toString(36).slice(2)}_${Date.now().toString(36)}`
  }
  let captureSessionId = generateCaptureSessionId()

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
    renderCaptureOverlay()
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
    for (const m of shaderInfo.messages) console.error(`terrain.wgsl: ${m.lineNum}:${m.linePos} ${m.message}`)
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
  await device.popErrorScope().catch((e: unknown) => console.error(`Render pipeline error: ${String(e)}`))

  const worldConfig = createSimpleWorldConfig(Math.floor(Math.random() * 1000000))
  const chunk = new ChunkManager(worldConfig.dimensions)
  const worldScale = 2
  setBlockTextureIndices(BlockType.Plank, null)

  console.log('Generating rolling hills terrain...')
  generateRollingHills(chunk, worldConfig)
  console.log('Terrain generation complete')
  const chunkOriginOffset: Vec3 = [-chunk.size.x * worldScale / 2, 0, -chunk.size.z * worldScale / 2]
  let meshDirty = true
  let vertexBuffer: GPUBuffer | null = null
  let vertexBufferSize = 0
  let vertexCount = 0

  type BlockPosition = [number, number, number]

  type PlaceBlockParams = {
    position: BlockPosition
    blockType: BlockType
    customBlockId?: number
    source?: string
  }

  type RemoveBlockParams = {
    position: BlockPosition
    source?: string
  }

  type WorldAction =
    | { type: 'place_block'; params: PlaceBlockParams }
    | { type: 'remove_block'; params: RemoveBlockParams }

  function serializeBlockType(type: BlockType | undefined) {
    if (type === undefined) return undefined
    return BlockType[type] ?? type
  }

  function serializePosition(position: BlockPosition) {
    return [position[0], position[1], position[2]]
  }

  function prepareParams(action: string, params: PlaceBlockParams | RemoveBlockParams) {
    const base: Record<string, unknown> = {
      ...params
    }
    if (Array.isArray(params.position) && params.position.length === 3) {
      base.position = serializePosition(params.position)
    } else {
      base.position = [0, 0, 0]
    }
    if ('blockType' in base && typeof base.blockType === 'number') {
      base.blockTypeName = serializeBlockType(base.blockType as BlockType)
    }
    return base
  }

  function prepareResult(result: Record<string, unknown>) {
    const payload: Record<string, unknown> = { ...result }
    if ('blockType' in payload && typeof payload.blockType === 'number') {
      payload.blockTypeName = serializeBlockType(payload.blockType as BlockType)
    }
    if ('previousBlock' in payload && typeof payload.previousBlock === 'number') {
      payload.previousBlockName = serializeBlockType(payload.previousBlock as BlockType)
    }
    return payload
  }

  function logDSLAction(action: string, params: PlaceBlockParams | RemoveBlockParams, result: Record<string, unknown>) {
    try {
      const preparedParams = prepareParams(action, params)
      const preparedResult = prepareResult(result)
      const payload = {
        action,
        params: preparedParams,
        result: preparedResult,
        source: (params as { source?: string }).source ?? 'unknown',
        timestamp: new Date().toISOString()
      }
      void fetch(`${API_BASE_URL}/api/log-dsl`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      }).catch(() => {})
    } catch (err) {
      console.warn('[dsl] Failed to log action', err)
    }
  }

  function positionInBounds([x, y, z]: BlockPosition) {
    return x >= 0 && y >= 0 && z >= 0 && x < chunk.size.x && y < chunk.size.y && z < chunk.size.z
  }

  function buildTextureIndicesForLayer(layer: number): Record<BlockFaceKey, number> {
    const baseLayer = tileLayerCount + (layer * blockFaceOrder.length)
    const indices: Record<BlockFaceKey, number> = {} as Record<BlockFaceKey, number>
    blockFaceOrder.forEach((face, index) => {
      indices[face] = baseLayer + index
    })
    return indices
  }

  const worldDSL = {
    placeBlock(params: PlaceBlockParams) {
      const { position, blockType, customBlockId } = params
      let result: Record<string, unknown>
      if (!positionInBounds(position)) {
        result = { success: false as const, reason: 'out_of_bounds' }
        logDSLAction('place_block', params, result)
        return result
      }
      if (blockType === BlockType.Air) {
        result = { success: false as const, reason: 'invalid_block' }
        logDSLAction('place_block', params, result)
        return result
      }

      const [x, y, z] = position

      if (customBlockId !== undefined) {
        const customBlock = get(customBlocksStore).find(block => block.id === customBlockId)
        if (!customBlock || customBlock.textureLayer === undefined) {
          result = { success: false as const, reason: 'custom_block_unavailable' }
          logDSLAction('place_block', params, result)
          return result
        }
        chunk.setBlock(x, y, z, BlockType.Plank)
        const indices = buildTextureIndicesForLayer(customBlock.textureLayer)
        setBlockTextureIndices(BlockType.Plank, indices)
        meshDirty = true
        result = { success: true as const, blockType: BlockType.Plank, customBlockId }
        console.log(`[dsl] Placed custom block: ${customBlock.name} (layer ${customBlock.textureLayer}) at`, position)
        logDSLAction('place_block', params, result)
        return result
      }

      chunk.setBlock(x, y, z, blockType)
      setBlockTextureIndices(blockType, null)
      meshDirty = true
      result = { success: true as const, blockType }
      console.log(`[dsl] Placed block: ${BlockType[blockType] ?? blockType} at`, position)
      logDSLAction('place_block', params, result)
      return result
    },

    removeBlock(params: RemoveBlockParams) {
      const { position } = params
      let result: Record<string, unknown>
      if (!positionInBounds(position)) {
        result = { success: false as const, reason: 'out_of_bounds' }
        logDSLAction('remove_block', params, result)
        return result
      }
      const [x, y, z] = position
      const existing = chunk.getBlock(x, y, z)
      if (existing === BlockType.Air) {
        result = { success: false as const, reason: 'already_empty' }
        logDSLAction('remove_block', params, result)
        return result
      }
      chunk.setBlock(x, y, z, BlockType.Air)
      meshDirty = true
      result = { success: true as const, previousBlock: existing }
      logDSLAction('remove_block', params, result)
      return result
    },

    execute(action: WorldAction) {
      if (action.type === 'place_block') {
        return this.placeBlock(action.params)
      }
      if (action.type === 'remove_block') {
        return this.removeBlock(action.params)
      }
      const unknownAction = action as any
      const result = { success: false as const, reason: 'unknown_action' }
      const params = unknownAction?.params ?? { position: [0, 0, 0] }
      logDSLAction(String(unknownAction?.type ?? 'unknown'), params as PlaceBlockParams | RemoveBlockParams, result)
      return result
    }
  }

  if (typeof window !== 'undefined') {
    ;(window as any).worldDSL = worldDSL
  }

  function rebuildMesh() {
    const mesh = buildChunkMesh(chunk, worldScale)
    vertexCount = mesh.vertexCount
    if (vertexCount === 0) return
    const byteLength = alignTo(mesh.vertexData.byteLength, 4)
    if (!vertexBuffer || byteLength > vertexBufferSize) {
      ;(vertexBuffer as any)?.destroy?.()
      vertexBuffer = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST })
      vertexBufferSize = byteLength
    }
    device.queue.writeBuffer(vertexBuffer, 0, mesh.vertexData.buffer, mesh.vertexData.byteOffset, mesh.vertexData.byteLength)
  }

  async function fetchFaceBitmaps(tiles: Partial<Record<BlockFaceKey, FaceTileInfo>>) {
    const result: Record<BlockFaceKey, ImageBitmap> = {} as Record<BlockFaceKey, ImageBitmap>
    await Promise.all(blockFaceOrder.map(async (face) => {
      const tile = tiles[face]
      if (!tile) return
      const url = tile.url?.startsWith('http') ? tile.url : `${TILE_BASE_URL}/${tile.path.replace(/^\/+/, '')}`
      try {
        const response = await fetch(url, { mode: 'cors', credentials: 'omit' })
        if (!response.ok) throw new Error(`Failed to load face tile ${face} - HTTP ${response.status}`)
        const blob = await response.blob()
        if (!blob.type.startsWith('image/')) throw new Error(`Invalid response type for ${face}: ${blob.type}`)
        const bitmap = await createImageBitmap(blob, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' })
        result[face] = bitmap
      } catch (err) {
        console.error(`Failed to fetch tile for face ${face}:`, err)
        throw err
      }
    }))
    return result
  }

  function applyFaceBitmapsToGPU(bitmaps: Record<BlockFaceKey, ImageBitmap>, customBlock: CustomBlock) {
    const sampleFace = blockFaceOrder.find(face => Boolean(bitmaps[face]))
    if (!sampleFace) return
    const sampleBitmap = bitmaps[sampleFace]!
    const size = sampleBitmap.width

    const currentCustomBlocks = get(customBlocksStore)

    if (customBlock.textureLayer === undefined) {
      const nextLayer = findNextAvailableLayer(currentCustomBlocks)
      if (nextLayer === null) {
        console.warn(`Maximum custom blocks (${MAX_CUSTOM_BLOCKS}) reached. Cannot add more custom textures.`)
        return
      }
      customBlock.textureLayer = nextLayer
      customBlocksStore.update(blocks => blocks)
    }

    const usedLayers = collectUsedLayers(currentCustomBlocks)
    usedLayers.add(customBlock.textureLayer!)
    const totalLayers = tileLayerCount + usedLayers.size * blockFaceOrder.length

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

    updateAllCustomTextureIndices()
    meshDirty = true
  }

  function updateAllCustomTextureIndices() {
    const currentCustomBlocks = get(customBlocksStore)
    currentCustomBlocks.forEach(block => {
      if (block.textureLayer !== undefined) {
        const baseLayer = tileLayerCount + (block.textureLayer * blockFaceOrder.length)
        const textureIndices: Record<BlockFaceKey, number> = {} as Record<BlockFaceKey, number>
        blockFaceOrder.forEach((face, index) => {
          textureIndices[face] = baseLayer + index
        })
        setBlockTextureIndices(BlockType.Plank, textureIndices)
      }
    })
  }

  function renderCaptureOverlay() {
    if (!overlayCtx || !overlayCanvas) return
    const width = overlayCanvas.width || canvas.width
    const height = overlayCanvas.height || canvas.height
    overlayCtx.clearRect(0, 0, width, height)
    const camera = latestCamera
    if (!camera || capturedViews.length === 0) return

    overlayCtx.save()
    overlayCtx.lineWidth = 1
    overlayCtx.textAlign = 'left'
    overlayCtx.textBaseline = 'middle'
    overlayCtx.font = '12px sans-serif'

    capturedViews.forEach((view, index) => {
      const screen = projectToScreen(view.snapshot.position, camera.viewProjectionMatrix, width, height)
      if (!screen) return
      overlayCtx.fillStyle = 'rgba(255, 64, 64, 0.9)'
      overlayCtx.strokeStyle = 'rgba(0, 0, 0, 0.6)'
      overlayCtx.beginPath()
      overlayCtx.arc(screen[0], screen[1], 4, 0, Math.PI * 2)
      overlayCtx.fill()
      overlayCtx.stroke()

      overlayCtx.fillStyle = 'rgba(255, 255, 255, 0.9)'
      overlayCtx.fillText(String(index + 1), screen[0] + 6, screen[1])
    })

    overlayCtx.restore()
  }

  function captureCurrentView() {
    if (!latestCamera) {
      console.warn('Capture requested before camera state initialized.')
      return
    }
    const snapshot = snapshotCamera(latestCamera)
    const id = `view_${String(capturedViews.length + 1).padStart(3, '0')}`
    capturedViews.push({
      id,
      createdAt: new Date().toISOString(),
      snapshot
    })
    console.log(`[capture] Stored camera view ${id} at position`, snapshot.position)
    renderCaptureOverlay()
  }

  function clearCapturedViews() {
    if (capturedViews.length === 0) {
      console.log('[capture] No stored views to clear.')
    } else {
      capturedViews.length = 0
      console.log('[capture] Cleared all stored camera views.')
    }
    renderCaptureOverlay()
    captureSessionId = generateCaptureSessionId()
  }

  function dataUrlToBase64(dataUrl: string): string {
    const prefix = 'data:image/png;base64,'
    return dataUrl.startsWith(prefix) ? dataUrl.slice(prefix.length) : dataUrl
  }

  async function renderSnapshotToBase64(snapshot: CameraSnapshot, width: number, height: number): Promise<string> {
    if (!vertexBuffer || vertexCount === 0) {
      throw new Error('No geometry available to render for capture.')
    }

    device.queue.writeBuffer(cameraBuffer, 0, snapshot.viewProjectionMatrix)

    const colorTexture = device.createTexture({
      size: { width, height, depthOrArrayLayers: 1 },
      format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
    })
    const captureDepth = device.createTexture({
      size: { width, height, depthOrArrayLayers: 1 },
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT
    })

    const encoder = device.createCommandEncoder()
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: colorTexture.createView(),
        clearValue: { r: 0.53, g: 0.81, b: 0.92, a: 1 },
        loadOp: 'clear',
        storeOp: 'store'
      }],
      depthStencilAttachment: {
        view: captureDepth.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store'
      }
    })

    pass.setPipeline(pipeline)
    pass.setBindGroup(0, renderBindGroup)
    pass.setVertexBuffer(0, vertexBuffer)
    pass.draw(vertexCount, 1, 0, 0)
    pass.end()

    device.queue.submit([encoder.finish()])
    await device.queue.onSubmittedWorkDone()

    const bytesPerPixel = 4
    const bytesPerRow = alignTo(width * bytesPerPixel, 256)
    const outputBuffer = device.createBuffer({
      size: bytesPerRow * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    })

    const copyEncoder = device.createCommandEncoder()
    copyEncoder.copyTextureToBuffer(
      { texture: colorTexture },
      { buffer: outputBuffer, bytesPerRow, rowsPerImage: height },
      { width, height, depthOrArrayLayers: 1 }
    )
    device.queue.submit([copyEncoder.finish()])
    await device.queue.onSubmittedWorkDone()

    await outputBuffer.mapAsync(MAP_MODE_READ)
    const mapped = outputBuffer.getMappedRange()
    const source = new Uint8Array(mapped)
    const imageBytes = new Uint8ClampedArray(width * height * bytesPerPixel)
    for (let row = 0; row < height; row++) {
      const srcOffset = row * bytesPerRow
      const dstOffset = row * width * bytesPerPixel
      imageBytes.set(source.subarray(srcOffset, srcOffset + width * bytesPerPixel), dstOffset)
    }

    if (format === 'bgra8unorm' || format === 'bgra8unorm-srgb') {
      for (let i = 0; i < imageBytes.length; i += 4) {
        const r = imageBytes[i] ?? 0
        const g = imageBytes[i + 1] ?? 0
        const b = imageBytes[i + 2] ?? 0
        const a = imageBytes[i + 3] ?? 255
        imageBytes[i] = b
        imageBytes[i + 1] = g
        imageBytes[i + 2] = r
        imageBytes[i + 3] = a
      }
    }
    outputBuffer.unmap()
    outputBuffer.destroy()
    colorTexture.destroy()
    captureDepth.destroy()

    const exportCanvas = document.createElement('canvas')
    exportCanvas.width = width
    exportCanvas.height = height
    const ctx = exportCanvas.getContext('2d')
    if (!ctx) throw new Error('Failed to create 2D context for dataset export')
    const imageData = new ImageData(imageBytes, width, height)
    ctx.putImageData(imageData, 0, 0)
    const dataUrl = exportCanvas.toDataURL('image/png')
    exportCanvas.remove()
    return dataUrlToBase64(dataUrl)
  }

  async function exportCapturedDataset(): Promise<DatasetExportResult | null> {
    if (exportingDataset) {
      console.warn('Dataset export already in progress; ignoring duplicate request.')
      return null
    }
    if (capturedViews.length === 0) {
      console.warn('No captured views to export.')
      return null
    }
    if (!latestCamera) {
      console.warn('Camera state unavailable; cannot export dataset yet.')
      return null
    }

    exportingDataset = true
    try {
      if (rafHandle !== null) {
        cancelAnimationFrame(rafHandle)
        rafHandle = null
      }

      await device.queue.onSubmittedWorkDone()
      if (meshDirty) {
        rebuildMesh()
        meshDirty = false
      }

      const originalCamera = snapshotCamera(latestCamera)
      const width = canvas.width
      const height = canvas.height
      const exportedAt = new Date().toISOString()
      const captureId = captureSessionId
      const payload = {
        formatVersion: '1.0',
        exportedAt,
        imageSize: { width, height },
        viewCount: capturedViews.length,
        captureId,
        views: [] as Array<{
          id: string
          index: number
          capturedAt: string
          position: Vec3
          forward: Vec3
          up: Vec3
          right: Vec3
          intrinsics: {
            fovYDegrees: number
            aspect: number
            near: number
            far: number
          }
          viewMatrix: number[]
          projectionMatrix: number[]
          viewProjectionMatrix: number[]
          rgbBase64: string
          depthBase64: string | null
          normalBase64: string | null
        }>
      }

      for (let i = 0; i < capturedViews.length; i++) {
        const view = capturedViews[i]!
        const rgbBase64 = await renderSnapshotToBase64(view.snapshot, width, height)
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

      device.queue.writeBuffer(cameraBuffer, 0, originalCamera.viewProjectionMatrix)
      latestCamera = originalCamera

      const response = await fetch(`${API_BASE_URL}/api/export-dataset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })

      if (!response.ok) {
        const detail = await response.text()
        throw new Error(`Export failed (${response.status}): ${detail}`)
      }

      const result = await response.json().catch(() => null)
      console.log('[capture] Dataset saved', result || 'OK')
      capturedViews.length = 0
      renderCaptureOverlay()
      captureSessionId = generateCaptureSessionId()
      return result as DatasetExportResult
    } catch (err) {
      console.error('Failed to export captured dataset', err)
      return null
    } finally {
      exportingDataset = false
      lastFrameTime = performance.now()
      if (rafHandle === null) {
        rafHandle = requestAnimationFrame(frame)
      }
    }
  }

  async function exportDatasetAndPrompt() {
    if (exportingDataset) {
      console.warn('Dataset export in progress; skipping prompt trigger.')
      return
    }
    if (capturedViews.length === 0) {
      console.warn('[capture] No stored views to export for reconstruction prompt.')
      return
    }

    const exportResult = await exportCapturedDataset()
    if (!exportResult) {
      console.warn('[capture] Export failed; aborting reconstruction prompt.')
      return
    }

    const defaultPrompt = 'Describe how to reconstruct this captured area using DSL commands.'
    const promptText = window.prompt('Enter reconstruction prompt for the LLM', defaultPrompt)
    if (!promptText || !promptText.trim()) {
      console.log('[llm] Reconstruction prompt cancelled by user.')
      return
    }

    try {
      const response = await fetch(`${API_BASE_URL}/api/reconstruct-dataset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          datasetSequence: exportResult.datasetSequence,
          metadataFile: exportResult.metadataFile,
          captureId: exportResult.captureId,
          prompt: promptText
        })
      })

      if (!response.ok) {
        const detail = await response.text()
        throw new Error(`LLM reconstruction request failed (${response.status}): ${detail}`)
      }

      const data = await response.json()
      const message = typeof data.output === 'string' && data.output.trim().length > 0
        ? data.output
        : 'No reconstruction guidance returned.'
      console.log('[llm] Reconstruction guidance received:', message)
      alert(`Reconstruction guidance:\n\n${message}`)
    } catch (err) {
      console.error('[llm] Reconstruction request failed', err)
      alert(`Reconstruction request failed: ${err instanceof Error ? err.message : String(err)}`)
    }
  }

  // Set GPU hooks
  gpuHooks.set({
    requestFaceBitmaps: fetchFaceBitmaps,
    uploadFaceBitmapsToGPU: applyFaceBitmapsToGPU
  })

  const pressedKeys = new Set<string>()
  let pointerActive = false
  let paused = true
  function handleKeyDown(ev: KeyboardEvent) {
    if (pointerActive && ev.code !== 'Escape') ev.preventDefault()
    if (ev.repeat) {
      pressedKeys.add(ev.code)
      return
    }

    if (ev.code === 'KeyR') {
      captureCurrentView()
      return
    }
    if (ev.code === 'KeyP') {
      clearCapturedViews()
      return
    }
    if (ev.code === 'KeyG') {
      void exportCapturedDataset()
      return
    }
    if (ev.code === 'KeyV') {
      void exportDatasetAndPrompt()
      return
    }

    pressedKeys.add(ev.code)
  }

  function handleKeyUp(ev: KeyboardEvent) {
    if (pointerActive && ev.code !== 'Escape') ev.preventDefault()
    pressedKeys.delete(ev.code)
  }

  window.addEventListener('keydown', handleKeyDown)
  window.addEventListener('keyup', handleKeyUp)
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
  canvas.addEventListener('click', () => canvas.requestPointerLock())
  canvas.addEventListener('contextmenu', (ev) => ev.preventDefault())
  document.addEventListener('pointerlockchange', () => {
    pointerActive = document.pointerLockElement === canvas
    paused = !pointerActive
    pressedKeys.clear()
    lastFrameTime = performance.now()
  })
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
    if (paused || !pointerActive) return
    ev.preventDefault()
    const hit = raycast(cameraPos, getForwardVector())
    if (!hit) return
    if (ev.button === 0) {
      const result = worldDSL.removeBlock({ position: hit.block, source: 'player' })
      if (!result.success && result.reason !== 'already_empty') {
        console.warn('[dsl] Failed to remove block:', result)
      }
    } else if (ev.button === 2) {
      const target = hit.previous
      if (isInsideChunk(target) && chunk.getBlock(target[0], target[1], target[2]) === BlockType.Air) {
        const selected = getSelectedBlock()
        const params: PlaceBlockParams = {
          position: target,
          blockType: selected.type,
          source: 'player'
        }
        if (selected.custom) {
          params.customBlockId = selected.custom.id
        }
        const result = worldDSL.placeBlock(params)
        if (!result.success) {
          console.warn('[dsl] Failed to place block:', result)
        }
      }
    }
  }

  window.addEventListener('mousedown', handleMouseDown)

  function updateCamera(dt: number) {
    const aspect = canvas.width / Math.max(1, canvas.height)
    const fovYRadians = (60 * Math.PI) / 180
    const near = 0.1
    const far = 500.0
    const proj = createPerspective(fovYRadians, aspect, near, far)

    const forward = getForwardVector()

    let right = cross(forward, worldUp)
    if (Math.hypot(right[0], right[1], right[2]) < 1e-4) {
      right = [1, 0, 0]
    }
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

    const target: Vec3 = [
      cameraPos[0] + forward[0],
      cameraPos[1] + forward[1],
      cameraPos[2] + forward[2]
    ]
    const view = lookAt(cameraPos, target, upVec)
    const viewProj = multiplyMat4(proj, view)
    device.queue.writeBuffer(cameraBuffer, 0, viewProj)

    latestCamera = {
      position: cloneVec3(cameraPos),
      forward: cloneVec3(forward),
      up: cloneVec3(upVec),
      right: cloneVec3(right),
      viewMatrix: cloneMat4(view),
      projectionMatrix: cloneMat4(proj),
      viewProjectionMatrix: cloneMat4(viewProj),
      fovYRadians,
      aspect,
      near,
      far
    }
  }

  let lastFrameTime = performance.now()

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
    renderCaptureOverlay()
    rafHandle = requestAnimationFrame(frame)
  }

  rafHandle = requestAnimationFrame(frame)
}

function createSimpleWorldConfig(seed: number = Date.now()) {
  return {
    seed,
    dimensions: { x: 64, y: 48, z: 64 }
  }
}

function generateRollingHills(chunk: ChunkManager, config: { seed: number; dimensions: { x: number; y: number; z: number } }) {
  const { x: sx, y: sy, z: sz } = chunk.size
  if (sx !== config.dimensions.x || sy !== config.dimensions.y || sz !== config.dimensions.z) {
    throw new Error('Chunk dimensions must match config dimensions')
  }

  const baseFreq = 1 / 32
  const minHeight = Math.floor(sy * 0.2)
  const maxHeight = Math.floor(sy * 0.6)
  const heightRange = maxHeight - minHeight

  for (let x = 0; x < sx; x++) {
    for (let z = 0; z < sz; z++) {
      const elevation = fbmNoise2D(x * baseFreq, z * baseFreq, config.seed, 4, 2.0, 0.5)
      const normalizedHeight = (elevation + 1) / 2
      const columnHeight = Math.floor(minHeight + normalizedHeight * heightRange)

      for (let y = 0; y < sy; y++) {
        if (y > columnHeight) {
          chunk.setBlock(x, y, z, BlockType.Air)
          continue
        }
        if (y === columnHeight) {
          chunk.setBlock(x, y, z, BlockType.Grass)
        } else if (y >= columnHeight - 3) {
          chunk.setBlock(x, y, z, BlockType.Dirt)
        } else {
          chunk.setBlock(x, y, z, BlockType.Stone)
        }
      }
    }
  }
}

function fbmNoise2D(x: number, z: number, seed: number, octaves = 4, lacunarity = 2, gain = 0.5) {
  let freq = 1
  let amp = 1
  let sum = 0
  let max = 0
  for (let i = 0; i < octaves; i++) {
    sum += valueNoise2D(x * freq, z * freq, seed + i * 131) * amp
    max += amp
    freq *= lacunarity
    amp *= gain
  }
  return max > 0 ? sum / max : 0
}

function valueNoise2D(x: number, z: number, seed: number) {
  const xi = Math.floor(x)
  const zi = Math.floor(z)
  const xf = x - xi
  const zf = z - zi

  const h00 = hash2D(xi, zi, seed)
  const h10 = hash2D(xi + 1, zi, seed)
  const h01 = hash2D(xi, zi + 1, seed)
  const h11 = hash2D(xi + 1, zi + 1, seed)

  const u = smoothstep(xf)
  const v = smoothstep(zf)

  const x1 = lerp(h00, h10, u)
  const x2 = lerp(h01, h11, u)
  return lerp(x1, x2, v)
}

function lerp(a: number, b: number, t: number) {
  return a * (1 - t) + b * t
}

function smoothstep(t: number) {
  return t * t * (3 - 2 * t)
}

function hash2D(x: number, z: number, seed: number) {
  let h = seed >>> 0
  h ^= Math.imul(0x27d4eb2d, x)
  h = (h ^ (h >>> 15)) >>> 0
  h ^= Math.imul(0x165667b1, z)
  h = (h ^ (h >>> 13)) >>> 0
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296
}

function findNextAvailableLayer(blocks: CustomBlock[]) {
  const used = collectUsedLayers(blocks)
  for (let i = 0; i < MAX_CUSTOM_BLOCKS; i++) {
    if (!used.has(i)) return i
  }
  return null
}

function collectUsedLayers(blocks: CustomBlock[]) {
  const used = new Set<number>()
  blocks.forEach(block => {
    if (typeof block.textureLayer === 'number' && block.textureLayer >= 0 && block.textureLayer < MAX_CUSTOM_BLOCKS) {
      used.add(block.textureLayer)
    }
  })
  return used
}
