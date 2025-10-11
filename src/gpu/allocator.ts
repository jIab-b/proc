export class BufferAllocator {
  constructor(private device: GPUDevice) {}
  create(size: number, usage: number, mappedAtCreation = false) {
    return this.device.createBuffer({ size, usage, mappedAtCreation })
  }
}
