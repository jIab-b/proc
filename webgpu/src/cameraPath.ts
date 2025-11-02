import type { CameraSnapshot } from './engine'

export type CameraKeyframe = { t: number; snapshot: CameraSnapshot }
export type CameraPath = { fps: number; width: number; height: number; frames: Array<{ t: number; snapshot: CameraSnapshot }> }

function nlerp(a: [number, number, number], b: [number, number, number], t: number): [number, number, number] {
  const x = a[0] + (b[0] - a[0]) * t
  const y = a[1] + (b[1] - a[1]) * t
  const z = a[2] + (b[2] - a[2]) * t
  const len = Math.hypot(x, y, z) || 1
  return [x / len, y / len, z / len]
}

function lerp(a: number, b: number, t: number) { return a + (b - a) * t }

export function interpolateSnapshot(a: CameraSnapshot, b: CameraSnapshot, t: number): CameraSnapshot {
  const position: any = [lerp(a.position[0], b.position[0], t), lerp(a.position[1], b.position[1], t), lerp(a.position[2], b.position[2], t)]
  const forward: any = nlerp(a.forward as any, b.forward as any, t)
  const up: any = nlerp(a.up as any, b.up as any, t)
  const right: any = nlerp(a.right as any, b.right as any, t)
  return {
    position, forward, up, right,
    viewMatrix: new Float32Array(a.viewMatrix),
    projectionMatrix: new Float32Array(a.projectionMatrix),
    viewProjectionMatrix: new Float32Array(a.viewProjectionMatrix),
    fovYRadians: lerp(a.fovYRadians, b.fovYRadians, t),
    aspect: lerp(a.aspect, b.aspect, t),
    near: lerp(a.near, b.near, t),
    far: lerp(a.far, b.far, t)
  }
}

export function sampleCameraPath(keyframes: CameraKeyframe[], fps: number, width: number, height: number): CameraPath {
  const frames: Array<{ t: number; snapshot: CameraSnapshot }> = []
  if (keyframes.length === 0) return { fps, width, height, frames }
  keyframes = [...keyframes].sort((x, y) => x.t - y.t)
  const totalT = keyframes[keyframes.length - 1]!.t
  const count = Math.max(1, Math.round(totalT * fps))
  for (let i = 0; i < count; i++) {
    const t = i / fps
    let k = 0
    while (k + 1 < keyframes.length && keyframes[k + 1]!.t < t) k++
    const a = keyframes[Math.min(k, keyframes.length - 1)]!
    const b = keyframes[Math.min(k + 1, keyframes.length - 1)]!
    const seg = b.t - a.t || 1
    const lt = Math.max(0, Math.min(1, (t - a.t) / seg))
    const snap = interpolateSnapshot(a.snapshot, b.snapshot, lt)
    frames.push({ t, snapshot: snap })
  }
  return { fps, width, height, frames }
}

export function serializeCameraPath(path: CameraPath) {
  return JSON.stringify(path)
}


