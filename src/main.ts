import terrainWGSL from './pipelines/render/terrain.wgsl?raw'
import heightmapWGSL from './pipelines/compute/heightmap.wgsl?raw'
import noiseWGSL from './noise/noise.wgsl?raw'
import { createPerspective, lookAt, multiplyMat4 } from './camera'

const root = document.getElementById('app') as HTMLDivElement
const canvas = document.createElement('canvas')
canvas.width = root.clientWidth
canvas.height = root.clientHeight
root.appendChild(canvas)

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
  wrap.style.position = 'absolute'
  wrap.style.top = '8px'
  wrap.style.left = '8px'
  wrap.style.display = 'flex'
  wrap.style.gap = '8px'
  const btnDir = document.createElement('button')
  const btnSave = document.createElement('button')
  btnDir.textContent = 'Select Log Dir'
  btnSave.textContent = 'Save Log'
  let dirHandle: any = null
  btnDir.onclick = async () => {
    try {
      // @ts-ignore
      dirHandle = await (window as any).showDirectoryPicker()
      push('Selected log directory')
    } catch (e) { push('Dir select canceled') }
  }
  btnSave.onclick = async () => {
    if (dirHandle) {
      try {
        const fileHandle = await dirHandle.getFileHandle('webgpu-log.txt', { create: true })
        const writable = await fileHandle.createWritable()
        await writable.write(logBuffer.join('\n'))
        await writable.close()
        push('Saved log to chosen directory')
        return
      } catch (e) { push('Failed writing to chosen directory, falling back') }
    }
    saveLog()
  }
  wrap.appendChild(btnDir)
  wrap.appendChild(btnSave)
  root.appendChild(wrap)
}

function log(msg: string) { console.error(msg) }

async function init() {
  installLogUI()
  ;(window as any).onerror = (msg: any, src: any, line: any, col: any, err: any) => { push(`window.onerror ${msg} @${src}:${line}:${col} ${err?.stack||''}`) }
  window.addEventListener('unhandledrejection', (ev) => { push(`unhandledrejection ${String((ev as any).reason)}`) })
  if (!('gpu' in navigator)) throw new Error('WebGPU not supported')
  const gpu = (navigator as any).gpu as any
  const adapter = await gpu.requestAdapter()
  if (!adapter) throw new Error('No adapter')
  const device = await adapter.requestDevice()
  try { (device as any).addEventListener && (device as any).addEventListener('uncapturederror', (ev: any) => { push(`device uncapturederror: ${ev?.error?.message||ev}`) }) } catch {}
  const context = canvas.getContext('webgpu') as unknown as GPUCanvasContext
  const format = gpu.getPreferredCanvasFormat()
  context.configure({ device, format, alphaMode: 'opaque' })

  let depthTexture = device.createTexture({
    size: { width: canvas.width, height: canvas.height },
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT
  })

  function resize() {
    const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1))
    const w = Math.floor(root.clientWidth * dpr)
    const h = Math.floor(root.clientHeight * dpr)
    if (w === canvas.width && h === canvas.height) return
    canvas.width = w
    canvas.height = h
    context.configure({ device, format, alphaMode: 'opaque' })
    depthTexture.destroy()
    depthTexture = device.createTexture({ size: { width: w, height: h }, format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT })
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
          arrayStride: 16,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x4' }
          ]
        }
      ]
    },
    fragment: { module: shaderModule, entryPoint: 'fs_main', targets: [{ format }] },
    primitive: { topology: 'point-list' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' }
  })
  await device.popErrorScope().catch((e: unknown) => log(`Render pipeline error: ${String(e)}`))

  const grid = { x: 256, z: 256 }
  const pointCount = grid.x * grid.z
  const positionsSSBO = device.createBuffer({ size: pointCount * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX })
  const indirectSSBO = device.createBuffer({ size: 4 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT })

  const renderBindGroup = device.createBindGroup({
    layout: renderBGL,
    entries: [
      { binding: 0, resource: { buffer: cameraBuffer } }
    ]
  })

  const paramsSize = 64
  const paramsBuf = device.createBuffer({ size: paramsSize, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  const paramsData = new ArrayBuffer(paramsSize)
  const paramsU32 = new Uint32Array(paramsData)
  const paramsF32 = new Float32Array(paramsData)
  const spacing = 0.12
  paramsU32[0] = grid.x
  paramsU32[1] = grid.z
  paramsU32[2] = 0
  paramsU32[3] = 0
  paramsF32[4] = -grid.x * spacing * 0.5
  paramsF32[5] = -grid.z * spacing * 0.5
  paramsF32[6] = spacing
  paramsF32[7] = spacing
  paramsF32[8] = 3.2
  paramsF32[9] = 0.35
  paramsF32[10] = 1.6
  paramsF32[11] = 0
  device.queue.writeBuffer(paramsBuf, 0, paramsData)

  const heightLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
    ]
  })

  const heightModule = device.createShaderModule({ code: noiseWGSL + "\n" + heightmapWGSL })
  try {
    const info = await heightModule.getCompilationInfo()
    for (const m of info.messages || []) push(`heightmap.wgsl: ${m.lineNum}:${m.linePos} ${m.message}`)
  } catch {}

  const heightPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [heightLayout] }),
    compute: { module: heightModule, entryPoint: 'main' }
  })

  const heightBindGroup = device.createBindGroup({
    layout: heightLayout,
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: positionsSSBO } },
      { binding: 2, resource: { buffer: indirectSSBO } }
    ]
  })

  const pressedKeys = new Set<string>()
  window.addEventListener('keydown', (ev) => { if (!ev.repeat) pressedKeys.add(ev.code) })
  window.addEventListener('keyup', (ev) => { pressedKeys.delete(ev.code) })
  window.addEventListener('blur', () => pressedKeys.clear())

  const cameraPos: [number, number, number] = [0, 3, 12]
  let yaw = Math.PI
  let pitch = -0.25
  let pointerActive = false
  canvas.addEventListener('click', () => canvas.requestPointerLock())
  document.addEventListener('pointerlockchange', () => { pointerActive = document.pointerLockElement === canvas })
  window.addEventListener('mousemove', (ev) => {
    if (!pointerActive) return
    const sensitivity = 0.0025
    yaw -= ev.movementX * sensitivity
    pitch -= ev.movementY * sensitivity
    const limit = Math.PI / 2 - 0.05
    pitch = Math.max(-limit, Math.min(limit, pitch))
  })

  type Vec3 = [number, number, number]

  function normalize(v: Vec3): Vec3 {
    const len = Math.hypot(v[0], v[1], v[2])
    if (len < 1e-5) return [0, 0, 0] as Vec3
    return [v[0] / len, v[1] / len, v[2] / len] as Vec3
  }

  function cross(a: Vec3, b: Vec3): Vec3 {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0]
    ] as Vec3
  }

  function addScaled(target: Vec3, dir: Vec3, amt: number) {
    target[0] += dir[0] * amt
    target[1] += dir[1] * amt
    target[2] += dir[2] * amt
  }

  let lastFrameTime = performance.now()
  const worldUp: Vec3 = [0, 1, 0]

  function updateCamera(dt: number) {
    const aspect = canvas.width / Math.max(1, canvas.height)
    const proj = createPerspective((60 * Math.PI) / 180, aspect, 0.1, 400.0)

    const forward = normalize([
      Math.cos(pitch) * Math.sin(yaw),
      Math.sin(pitch),
      Math.cos(pitch) * Math.cos(yaw)
    ] as Vec3)

    let right = cross(forward, worldUp)
    if (Math.hypot(right[0], right[1], right[2]) < 1e-4) {
      right = [1, 0, 0] as Vec3
    }
    right = normalize(right)
    const upVec = normalize(cross(right, forward))

    const speed = (pressedKeys.has('ShiftLeft') || pressedKeys.has('ShiftRight')) ? 18 : 10
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
    const viewProj = multiplyMat4(view, proj)
    device.queue.writeBuffer(cameraBuffer, 0, viewProj)
  }

  async function frame() {
    const now = performance.now()
    const dt = Math.min(0.1, (now - lastFrameTime) / 1000)
    lastFrameTime = now

    updateCamera(dt)
    paramsF32[11] = now * 0.0005
    device.queue.writeBuffer(paramsBuf, 0, paramsData)

    const encoder = device.createCommandEncoder()
    try {
      device.pushErrorScope('validation')
      const pass = encoder.beginComputePass()
      pass.setPipeline(heightPipeline)
      pass.setBindGroup(0, heightBindGroup)
      pass.dispatchWorkgroups(Math.ceil(grid.x / 8), Math.ceil(grid.z / 8), 1)
      pass.end()
      const err = await device.popErrorScope()
      if (err) push(`Compute validation: ${err.message}`)
    } catch (e) { log(`Compute pipeline error: ${String(e)}`) }

    const colorView = context.getCurrentTexture().createView()
    const depthView = depthTexture.createView()
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: colorView, clearValue: { r: 0.02, g: 0.02, b: 0.03, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
      depthStencilAttachment: { view: depthView, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' }
    })
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, renderBindGroup)
    pass.setVertexBuffer(0, positionsSSBO)
    pass.drawIndirect(indirectSSBO, 0)
    pass.end()
    device.queue.submit([encoder.finish()])
    requestAnimationFrame(frame)
  }

  requestAnimationFrame(frame)
}

init()
