// WebGPU Rendering Engine
import terrainWGSL from './pipelines/render/terrain.wgsl?raw'
import { createPerspective, lookAt, multiplyMat4 } from './camera'
import { ChunkManager, BlockType, type BlockFaceKey, buildChunkMesh, setBlockTextureIndices } from './chunks'
import { blockFaceOrder, TILE_BASE_URL } from './blockUtils'
import type { CustomBlock, FaceTileInfo } from './stores'
import { get } from 'svelte/store'
import { customBlocks as customBlocksStore, gpuHooks } from './stores'

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

export interface WebGPUEngineOptions {
  canvas: HTMLCanvasElement
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

export async function initWebGPUEngine(options: WebGPUEngineOptions) {
  const { canvas, getSelectedBlock } = options

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

  // Set GPU hooks
  gpuHooks.set({
    requestFaceBitmaps: fetchFaceBitmaps,
    uploadFaceBitmapsToGPU: applyFaceBitmapsToGPU
  })

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
        const selected = getSelectedBlock()
        if (selected.custom && selected.custom.textureLayer !== undefined) {
          chunk.setBlock(target[0], target[1], target[2], BlockType.Plank)
          const baseLayer = tileLayerCount + (selected.custom.textureLayer * blockFaceOrder.length)
          const textureIndices: Record<BlockFaceKey, number> = {} as Record<BlockFaceKey, number>
          blockFaceOrder.forEach((face, index) => {
            textureIndices[face] = baseLayer + index
          })
          setBlockTextureIndices(BlockType.Plank, textureIndices)
          console.log(`Placed custom block: ${selected.custom.name} with texture layer ${selected.custom.textureLayer}`)
        } else {
          chunk.setBlock(target[0], target[1], target[2], selected.type)
          setBlockTextureIndices(selected.type, null)
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

function createSimpleWorldConfig(seed: number = Date.now()) {
  return {
    seed,
    dimensions: { x: 128, y: 64, z: 128 }
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
