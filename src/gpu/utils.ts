export function createBuffer(device: GPUDevice, size: number, usage: number, mappedAtCreation = false) {
  return device.createBuffer({ size: align(size, 4), usage, mappedAtCreation })
}

export function align(n: number, alignTo: number) {
  return Math.ceil(n / alignTo) * alignTo
}

export async function createShaderModule(device: GPUDevice, code: string) {
  const module = device.createShaderModule({ code })
  await module.getCompilationInfo()
  return module
}
