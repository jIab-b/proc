import terrainWGSL from './pipelines/render/terrain.wgsl?raw'
import { createPerspective, lookAt, multiplyMat4 } from './camera'
import { ChunkManager, BlockType, buildChunkMesh } from './chunks'

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
  const renderBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
    ]
  })

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
          arrayStride: 9 * 4,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32x3' },
            { shaderLocation: 2, offset: 24, format: 'float32x3' }
          ]
        }
      ]
    },
    fragment: { module: shaderModule, entryPoint: 'fs_main', targets: [{ format }] },
    primitive: { topology: 'triangle-list', cullMode: 'back' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' }
  })
  await device.popErrorScope().catch((e: unknown) => log(`Render pipeline error: ${String(e)}`))

  const renderBindGroup = device.createBindGroup({
    layout: renderBGL,
    entries: [
      { binding: 0, resource: { buffer: cameraBuffer } }
    ]
  })

  const chunk = new ChunkManager({ x: 36, y: 28, z: 36 })
  chunk.generateDefaultTerrain(7)
  const chunkOriginOffset: Vec3 = [-chunk.size.x / 2, 0, -chunk.size.z / 2]
  const placeBlockType = BlockType.Plank
  let meshDirty = true
  let vertexBuffer: GPUBuffer | null = null
  let vertexBufferSize = 0
  let vertexCount = 0

  function rebuildMesh() {
    const mesh = buildChunkMesh(chunk)
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

  const pressedKeys = new Set<string>()
  window.addEventListener('keydown', (ev) => { if (!ev.repeat) pressedKeys.add(ev.code) })
  window.addEventListener('keyup', (ev) => { pressedKeys.delete(ev.code) })
  window.addEventListener('blur', () => pressedKeys.clear())

  const cameraPos: Vec3 = [0, chunk.size.y * 0.6, chunk.size.z * 1.4]
  let yaw = -Math.PI / 4
  let pitch = -0.28
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
  const maxRayDistance = Math.sqrt(chunk.size.x ** 2 + chunk.size.y ** 2 + chunk.size.z ** 2)

  function getForwardVector(): Vec3 {
    return normalize([
      Math.cos(pitch) * Math.sin(yaw),
      Math.sin(pitch),
      Math.cos(pitch) * Math.cos(yaw)
    ])
  }

  function worldToChunk(pos: Vec3): Vec3 {
    return [pos[0] - chunkOriginOffset[0], pos[1], pos[2] - chunkOriginOffset[2]]
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

    const tDelta: Vec3 = [
      step[0] !== 0 ? Math.abs(1 / dir[0]) : Number.POSITIVE_INFINITY,
      step[1] !== 0 ? Math.abs(1 / dir[1]) : Number.POSITIVE_INFINITY,
      step[2] !== 0 ? Math.abs(1 / dir[2]) : Number.POSITIVE_INFINITY
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
        chunk.setBlock(target[0], target[1], target[2], placeBlockType)
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

    const speed = (pressedKeys.has('ShiftLeft') || pressedKeys.has('ShiftRight')) ? 20 : 10
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
      colorAttachments: [{ view: colorView, clearValue: { r: 0.04, g: 0.05, b: 0.08, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
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
    requestAnimationFrame(frame)
  }

  requestAnimationFrame(frame)
}

init().catch((err) => log(`Startup error: ${String(err)}`))
