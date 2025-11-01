interface GPU {
  requestAdapter(options?: any): Promise<GPUAdapter | null>
  getPreferredCanvasFormat(): any
}
interface GPUAdapter { requestDevice(options?: any): Promise<GPUDevice> }
interface GPUDevice {
  createBuffer(descriptor: any): GPUBuffer
  createShaderModule(descriptor: any): GPUShaderModule
  createBindGroupLayout(descriptor: any): GPUBindGroupLayout
  createBindGroup(descriptor: any): GPUBindGroup
  createPipelineLayout(descriptor: any): GPUPipelineLayout
  createRenderPipeline(descriptor: any): GPURenderPipeline
  createCommandEncoder(): GPUCommandEncoder
  createTexture(descriptor: any): GPUTexture
  queue: { submit(cmds: GPUCommandBuffer[]): void; writeBuffer(buf: GPUBuffer, offset: number, data: BufferSource): void }
}
interface GPUBuffer {}
interface GPUShaderModule { getCompilationInfo(): Promise<any> }
interface GPUBindGroupLayout {}
interface GPUBindGroup {}
interface GPUPipelineLayout {}
interface GPURenderPipeline {}
interface GPUCommandEncoder { beginRenderPass(descriptor: any): GPURenderPassEncoder; finish(): GPUCommandBuffer }
interface GPUCommandBuffer {}
interface GPURenderPassEncoder {
  setPipeline(p: GPURenderPipeline): void
  setBindGroup(i: number, bg: GPUBindGroup): void
  setVertexBuffer(i: number, b: GPUBuffer): void
  setIndexBuffer(b: GPUBuffer, fmt: any): void
  draw(c: number, inst?: number, first?: number, firstInst?: number): void
  drawIndexed(c: number): void
  end(): void
}
interface GPUTexture { createView(): any; destroy(): void }
interface GPUCanvasContext { configure(cfg: any): void; getCurrentTexture(): GPUTexture }
interface Navigator { gpu: GPU }
interface HTMLCanvasElement { getContext(contextId: 'webgpu'): GPUCanvasContext | null }
declare var GPUBufferUsage: any
declare var GPUTextureUsage: any
declare var GPUShaderStage: any

declare const $customBlocks: import('./core').CustomBlock[]
declare const $gpuHooks: import('./core').GPUHooks
declare const $terrainProfile: 'rolling_hills' | 'mountain' | 'hybrid'
declare const $terrainSeed: number
declare const $terrainAmplitude: number
declare const $terrainRoughness: number
declare const $terrainElevation: number
