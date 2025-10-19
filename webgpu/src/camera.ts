export type Mat4 = Float32Array
export type Vec3 = [number, number, number]

export function createPerspective(fovYRad: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1.0 / Math.tan(fovYRad / 2)
  const nf = 1 / (near - far)
  const out = new Float32Array(16)
  out[0] = f / aspect
  out[5] = f
  out[10] = (far + near) * nf
  out[11] = -1
  out[14] = (2 * far * near) * nf
  return out
}

export function lookAt(eye: Vec3, center: Vec3, up: Vec3): Mat4 {
  const [ex, ey, ez] = eye
  const [cx, cy, cz] = center
  let zx = ex - cx, zy = ey - cy, zz = ez - cz
  let len = Math.hypot(zx, zy, zz)
  zx /= len; zy /= len; zz /= len
  let xx = up[1] * zz - up[2] * zy
  let xy = up[2] * zx - up[0] * zz
  let xz = up[0] * zy - up[1] * zx
  len = Math.hypot(xx, xy, xz)
  xx /= len; xy /= len; xz /= len
  const yx = zy * xz - zz * xy
  const yy = zz * xx - zx * xz
  const yz = zx * xy - zy * xx
  const out = new Float32Array(16)
  out[0] = xx; out[1] = yx; out[2] = zx; out[3] = 0
  out[4] = xy; out[5] = yy; out[6] = zy; out[7] = 0
  out[8] = xz; out[9] = yz; out[10] = zz; out[11] = 0
  out[12] = -(xx * ex + xy * ey + xz * ez)
  out[13] = -(yx * ex + yy * ey + yz * ez)
  out[14] = -(zx * ex + zy * ey + zz * ez)
  out[15] = 1
  return out
}

export function multiplyMat4(a: Mat4, b: Mat4): Mat4 {
  const out = new Float32Array(16)
  for (let i = 0; i < 4; i++) {
    const ai0 = a[i]!
    const ai1 = a[i + 4]!
    const ai2 = a[i + 8]!
    const ai3 = a[i + 12]!
    out[i] = ai0 * b[0]! + ai1 * b[1]! + ai2 * b[2]! + ai3 * b[3]!
    out[i + 4] = ai0 * b[4]! + ai1 * b[5]! + ai2 * b[6]! + ai3 * b[7]!
    out[i + 8] = ai0 * b[8]! + ai1 * b[9]! + ai2 * b[10]! + ai3 * b[11]!
    out[i + 12] = ai0 * b[12]! + ai1 * b[13]! + ai2 * b[14]! + ai3 * b[15]!
  }
  return out
}
