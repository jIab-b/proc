import { BlockType, ChunkManager, type Vec3 } from '../core'

export type TerrainRegion = {
  min: Vec3
  max: Vec3
}

export interface TerrainParams {
  seed?: number
  amplitude?: number
  roughness?: number
  elevation?: number
}

export interface TerrainGeneratorState {
  params: Required<TerrainParams>
}

export function validateParams(p: TerrainParams): string[] {
  const errors: string[] = []
  if (p.amplitude && (p.amplitude < 8 || p.amplitude > 18)) {
    errors.push(`amplitude ${p.amplitude} not in [8, 18]`)
  }
  if (p.roughness && (p.roughness < 2.2 || p.roughness > 2.8)) {
    errors.push(`roughness ${p.roughness} not in [2.2, 2.8]`)
  }
  if (p.elevation && (p.elevation < 0.35 || p.elevation > 0.50)) {
    errors.push(`elevation ${p.elevation} not in [0.35, 0.50]`)
  }
  return errors
}

export function createTerrainGeneratorState(params: Required<TerrainParams>): TerrainGeneratorState {
  const errors = validateParams(params)
  if (errors.length) throw new Error(errors.join('; '))
  return { params }
}

export function generateRegion(chunk: ChunkManager, region: TerrainRegion, state: TerrainGeneratorState, floorHeightFn?: (x: number, z: number) => number) {
  const { params } = state
  const { min, max } = normalizeRegion(region)
  const heights: number[][] = []
  const terrainSeed = params.seed >>> 0

  for (let x = min[0]; x <= max[0]; x++) {
    const row: number[] = []
    for (let z = min[2]; z <= max[2]; z++) {
      const globalX = x
      const globalZ = z
      const baseHeight = sampleHeight(globalX, globalZ, params, terrainSeed)
      const floorHeight = floorHeightFn ? floorHeightFn(globalX, globalZ) : 0
      // Terrain height is relative to the floor
      const height = Math.max(0, baseHeight - floorHeight)
      row.push(height)
    }
    heights.push(row)
  }

  const sy = chunk.size.y
  for (let xi = 0; xi < heights.length; xi++) {
    for (let zi = 0; zi < heights[xi]!.length; zi++) {
      const worldX = min[0] + xi
      const worldZ = min[2] + zi
      const floorHeight = floorHeightFn ? floorHeightFn(worldX, worldZ) : 0
      const columnHeight = clamp(Math.floor(floorHeight + heights[xi]![zi]!), 0, sy - 1)

      for (let y = 0; y < sy; y++) {
        const globalY = y
        if (globalY > columnHeight) {
          chunk.setBlock(worldX, y, worldZ, BlockType.Air)
          continue
        }

        const block = selectBlock(globalY, columnHeight, params)
        chunk.setBlock(worldX, y, worldZ, block)
      }
    }
  }
}

export function sampleHeight(x: number, z: number, params: Required<TerrainParams>, seed: number) {
  const scale = 1 / 42
  const elevation = params.elevation
  const amplitude = params.amplitude
  const roughness = params.roughness

  const base = fbm((x + seed) * scale, (z + seed * 0.5) * scale, seed, 5, roughness, 0.45)
  const ridge = ridgeNoise((x + seed * 2) * scale * 0.8, (z - seed) * scale * 0.8, seed)
  const valley = fbm((x - seed) * scale * 0.6, (z + seed) * scale * 0.6, seed ^ 0x9e3779b9, 3, roughness - 0.2, 0.5)

  // Blend style determined by amplitude range
  if (amplitude < 11) {
    // Gentle: blend base + valley
    return amplitude * ((base + 1) * 0.5 * 0.6 + (valley + 1) * 0.5 * 0.4) + elevation * chunkHeight()
  } else if (amplitude < 16) {
    // Balanced: blend all three
    const peak = Math.pow(Math.max(0, ridge), 1.2)
    return amplitude * (0.5 * ((base + 1) * 0.5) + 0.5 * peak) + elevation * chunkHeight()
  } else {
    // Dramatic: emphasize peaks
    const peak = Math.pow(Math.max(0, ridge), 1.4)
    return amplitude * (0.65 * peak + 0.25 * ((base + 1) * 0.5) + 0.1 * ((valley + 1) * 0.5)) + elevation * chunkHeight()
  }
}

function chunkHeight() {
  return 48
}

function selectBlock(y: number, columnHeight: number, params: Required<TerrainParams>) {
  const surfaceThreshold = columnHeight
  const amplitude = params.amplitude

  if (y === surfaceThreshold) {
    if (amplitude >= 16 && columnHeight > 28) return BlockType.AlpineRock
    if (amplitude >= 16 && columnHeight > 34) return BlockType.GlacierIce
    if (amplitude >= 11 && amplitude < 16 && columnHeight > 30) return BlockType.AlpineGrass
    if (amplitude < 11 && columnHeight > 18) return BlockType.Grass
    return BlockType.AlpineGrass
  }

  if (y >= surfaceThreshold - 2) {
    if (amplitude < 11) return BlockType.Dirt
    if (amplitude >= 16) return BlockType.Gravel
    return BlockType.Dirt
  }

  if (amplitude >= 16) return BlockType.Stone
  return BlockType.Stone
}

function normalizeRegion(region: TerrainRegion): TerrainRegion {
  const min: Vec3 = [
    Math.min(region.min[0], region.max[0]),
    Math.min(region.min[1], region.max[1]),
    Math.min(region.min[2], region.max[2])
  ]
  const max: Vec3 = [
    Math.max(region.min[0], region.max[0]),
    Math.max(region.min[1], region.max[1]),
    Math.max(region.min[2], region.max[2])
  ]
  return { min, max }
}

function fbm(x: number, z: number, seed: number, octaves: number, lacunarity: number, gain: number) {
  let freq = 1
  let amp = 1
  let sum = 0
  let maxAmp = 0
  for (let i = 0; i < octaves; i++) {
    sum += noise2D(x * freq, z * freq, seed + i * 7919) * amp
    maxAmp += amp
    freq *= lacunarity
    amp *= gain
  }
  return maxAmp > 0 ? sum / maxAmp : 0
}

function ridgeNoise(x: number, z: number, seed: number) {
  const n = fbm(x, z, seed ^ 0x51a8d5, 4, 2.1, 0.47)
  return 1 - Math.abs(n)
}

function noise2D(x: number, z: number, seed: number) {
  const xi = Math.floor(x)
  const zi = Math.floor(z)
  const xf = x - xi
  const zf = z - zi
  const u = smoothstep(xf)
  const v = smoothstep(zf)
  const h00 = hash(xi, zi, seed)
  const h10 = hash(xi + 1, zi, seed)
  const h01 = hash(xi, zi + 1, seed)
  const h11 = hash(xi + 1, zi + 1, seed)
  const x1 = lerp(h00, h10, u)
  const x2 = lerp(h01, h11, u)
  return lerp(x1, x2, v) * 2 - 1
}

function smoothstep(t: number) {
  return t * t * (3 - 2 * t)
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t
}

function hash(x: number, z: number, seed: number) {
  let h = seed >>> 0
  h ^= Math.imul(0x27d4eb2d, x)
  h = (h ^ (h >>> 15)) >>> 0
  h ^= Math.imul(0x165667b1, z)
  h = (h ^ (h >>> 13)) >>> 0
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value))
}

