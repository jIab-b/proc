import { ChunkManager, BlockType } from '../chunks'
import { binToScalar, macroScaleToFrequency, seaLevelToRatio, WorldSpec } from './dsl'
import { StageRandom } from './random'
import { fbmNoise2D } from './noise'
import { TracingContext } from './trace'

type ClimateField = {
  temperature: Float32Array
  moisture: Float32Array
}

type HeightField = Float32Array
type RiverMask = Uint8Array

export interface GenerateChunkOptions {
  chunk: ChunkManager
  spec: WorldSpec
  trace?: TracingContext
}

export function generateChunk(options: GenerateChunkOptions) {
  const { chunk, spec } = options
  const trace = options.trace
  if (
    chunk.size.x !== spec.dimensions.x ||
    chunk.size.y !== spec.dimensions.y ||
    chunk.size.z !== spec.dimensions.z
  ) {
    throw new Error('Chunk dimensions do not match WorldSpec.dimensions')
  }
  const climate = sampleClimate(chunk, spec, trace)
  const heightfield = computeHeightfield(chunk, spec, climate, trace)
  const rivers = carveRivers(chunk, spec, heightfield, trace)
  applySurfaceLayers(chunk, spec, heightfield, rivers, trace)
  carveCaves(chunk, spec, trace)
  placeSurfaceFeatures(chunk, spec, heightfield, rivers, trace)
}

function sampleClimate(chunk: ChunkManager, spec: WorldSpec, trace?: TracingContext): ClimateField {
  const scope = trace?.begin('climate.sample', spec.seeds.climate, {
    biome: spec.biome,
    terrain: spec.terrain
  })
  const { x: sx, z: sz } = chunk.size
  const temperature = new Float32Array(sx * sz)
  const moisture = new Float32Array(sx * sz)
  const rng = new StageRandom(spec.seeds.climate ^ 0x1234abcd)

  const baseFreq = 1 / 92
  const tempBias = rng.nextRange(-0.1, 0.1)
  const moistBias = rng.nextRange(-0.05, 0.05)

  let tempMin = Number.POSITIVE_INFINITY
  let tempMax = Number.NEGATIVE_INFINITY
  let tempSum = 0
  let moistMin = Number.POSITIVE_INFINITY
  let moistMax = Number.NEGATIVE_INFINITY
  let moistSum = 0

  for (let z = 0; z < sz; z++) {
    for (let x = 0; x < sx; x++) {
      const idx = x + sx * z
      const tx = (x + rng.nextRange(-1000, 1000)) * baseFreq
      const tz = (z + rng.nextRange(-1000, 1000)) * baseFreq
      const temp = clamp01(fbmNoise2D(tx, tz, spec.seeds.climate, 4, 2, 0.55) + tempBias)
      const moist = clamp01(fbmNoise2D(tx, tz, spec.seeds.climate + 911, 4, 2, 0.6) + moistBias)
      temperature[idx] = temp
      moisture[idx] = moist
      tempMin = Math.min(tempMin, temp)
      tempMax = Math.max(tempMax, temp)
      tempSum += temp
      moistMin = Math.min(moistMin, moist)
      moistMax = Math.max(moistMax, moist)
      moistSum += moist
    }
  }

  scope?.end({
    stats: {
      temperature: summarise(tempMin, tempMax, tempSum / (sx * sz)),
      moisture: summarise(moistMin, moistMax, moistSum / (sx * sz))
    },
    rng: rng.snapshot()
  })

  return { temperature, moisture }
}

function computeHeightfield(chunk: ChunkManager, spec: WorldSpec, _climate: ClimateField, trace?: TracingContext): HeightField {
  const scope = trace?.begin('terrain.heightfield', spec.seeds.terrain, {
    terrainProfile: spec.terrainProfile
  })

  const { x: sx, z: sz } = chunk.size
  const height = new Float32Array(sx * sz)

  const ridgeScalar = binToScalar(spec.terrainProfile.ridgeFrequency)
  const varianceScalar = binToScalar(spec.terrainProfile.elevationVariance)
  const warpScalar = binToScalar(spec.terrainProfile.warpStrength)
  const macroFreq = macroScaleToFrequency(spec.terrainProfile.macroScale)
  const baseFreq = macroFreq * (0.45 + ridgeScalar)
  const warpFreq = baseFreq * (0.5 + binToScalar(spec.terrainProfile.warpStrength) * 0.5)

  let minH = Number.POSITIVE_INFINITY
  let maxH = Number.NEGATIVE_INFINITY
  let sum = 0

  for (let z = 0; z < sz; z++) {
    for (let x = 0; x < sx; x++) {
      const idx = x + sx * z
      const nx = x * baseFreq
      const nz = z * baseFreq
      const warp = fbmNoise2D(nx, nz, spec.seeds.terrain + 1000, 3, 2.7, 0.5) * warpScalar
      const h = fbmNoise2D(nx + warp, nz + warp, spec.seeds.terrain, 5, 2, 0.52)
      const value = clamp01(h * (0.4 + varianceScalar * 0.6))
      height[idx] = value
      minH = Math.min(minH, value)
      maxH = Math.max(maxH, value)
      sum += value
    }
  }

  scope?.end({
    stats: summarise(minH, maxH, sum / (sx * sz)),
    baseFrequency: baseFreq,
    warpFrequency: warpFreq
  })

  return height
}

function applySurfaceLayers(chunk: ChunkManager, spec: WorldSpec, heightfield: HeightField, rivers: RiverMask, trace?: TracingContext) {
  const scope = trace?.begin('terrain.surface', spec.seeds.surface, {
    biome: spec.biome
  })

  const { x: sx, y: sy, z: sz } = chunk.size
  const seaLevel = Math.floor(seaLevelToRatio(spec.terrainProfile.seaLevel) * sy)

  let minCol = Number.POSITIVE_INFINITY
  let maxCol = Number.NEGATIVE_INFINITY
  const topCount: Record<BlockType, number> = {} as Record<BlockType, number>

  for (let z = 0; z < sz; z++) {
    for (let x = 0; x < sx; x++) {
      const idx = x + sx * z
      const hNorm = heightfield[idx] ?? 0
      const columnHeight = Math.floor(hNorm * (sy - 3)) + 1
      minCol = Math.min(minCol, columnHeight)
      maxCol = Math.max(maxCol, columnHeight)

      const profile = getBiomeSurface(spec.biome)
      for (let y = 0; y < sy; y++) {
        let type = BlockType.Air
        if (rivers[idx] && y >= columnHeight - 1 && y <= seaLevel) {
          type = BlockType.Water
        } else if (y > columnHeight && y <= seaLevel) {
          type = profile.seaFill
        } else if (y === columnHeight) {
          type = rivers[idx] ? BlockType.Water : profile.top
        } else if (y < columnHeight && y >= columnHeight - 3) {
          type = profile.subsurface
        } else if (y < columnHeight) {
          type = profile.underground
        }
        chunk.setBlock(x, y, z, type)
      }
      topCount[chunk.getBlock(x, columnHeight, z)] = (topCount[chunk.getBlock(x, columnHeight, z)] ?? 0) + 1
    }
  }

  scope?.end({
    columnHeights: summarise(minCol, maxCol, (minCol + maxCol) / 2),
    seaLevel,
    topDistribution: topCount
  })
}

function placeSurfaceFeatures(chunk: ChunkManager, spec: WorldSpec, heightfield: HeightField, rivers: RiverMask, trace?: TracingContext) {
  const scope = trace?.begin('features.surface', spec.seeds.features, {
    features: spec.features
  })

  const { x: sx, y: sy, z: sz } = chunk.size
  const rng = new StageRandom(spec.seeds.features ^ 0x9e3779b1)
  const treeThreshold = binToScalar(spec.features.treeDensity) * 0.5
  let trees = 0

  for (let z = 0; z < sz; z++) {
    for (let x = 0; x < sx; x++) {
      const idx = x + sx * z
      const hNorm = heightfield[idx] ?? 0
      const columnHeight = Math.floor(hNorm * (sy - 3))
      if (columnHeight <= 2) continue
      if (rng.next() > treeThreshold) continue
      if (rivers[idx]) continue
      const topBlock = chunk.getBlock(x, columnHeight, z)
      if (!isSoil(topBlock)) continue

      const treeHeight = 3 + rng.nextInt(2)
      for (let i = 0; i < treeHeight && columnHeight + 1 + i < sy; i++) {
        chunk.setBlock(x, columnHeight + 1 + i, z, BlockType.Plank)
      }
      trees += 1
    }
  }

  scope?.end({
    treesPlaced: trees,
    rng: rng.snapshot()
  })
}

function carveRivers(chunk: ChunkManager, spec: WorldSpec, heightfield: HeightField, trace?: TracingContext): RiverMask {
  const scope = trace?.begin('terrain.rivers', spec.seeds.rivers, {
    requested: spec.features.riverCount
  })
  const { x: sx, y: sy, z: sz } = chunk.size
  const mask = new Uint8Array(sx * sz)
  const rng = new StageRandom(spec.seeds.rivers ^ 0x51eb851f)
  const riverCapacity = Math.max(1, Math.round(sx / 80))
  const riverCount = Math.min(spec.features.riverCount, riverCapacity)
  const width = Math.max(1, Math.round((sx / 256) * (1 + binToScalar(spec.terrainProfile.warpStrength))))
  const seaRatio = seaLevelToRatio(spec.terrainProfile.seaLevel)
  let carvedCells = 0

  for (let r = 0; r < riverCount; r++) {
    let x = rng.nextInt(sx)
    const direction = rng.next() > 0.5 ? 1 : -1
    for (let step = 0; step < sz; step++) {
      const z = direction > 0 ? step : sz - 1 - step
      const idx = x + sx * z
      for (let w = -width; w <= width; w++) {
        const px = clampInt(x + w, 0, sx - 1)
        const pIdx = px + sx * z
        if (!mask[pIdx]) carvedCells += 1
        mask[pIdx] = 1
        heightfield[pIdx] = Math.min(heightfield[pIdx] ?? 1, seaRatio)
      }
      const meander = rng.nextRange(-1, 1)
      x = clampInt(Math.round(x + meander), 1, sx - 2)
    }
  }

  scope?.end({
    riversCarved: riverCount,
    width,
    carvedCells,
    rng: rng.snapshot()
  })

  return mask
}

function carveCaves(chunk: ChunkManager, spec: WorldSpec, trace?: TracingContext) {
  const scope = trace?.begin('terrain.caves', spec.seeds.caves, {
    density: spec.features.caveDensity
  })
  const { x: sx, y: sy, z: sz } = chunk.size
  const rng = new StageRandom(spec.seeds.caves ^ 0x7f4a7c15)
  const volumeFactor = (sx * sz) / 4096
  const attempts = Math.max(1, Math.round(binToScalar(spec.features.caveDensity) * 4 * volumeFactor))
  let carvedBlocks = 0

  for (let i = 0; i < attempts; i++) {
    const cx = clampInt(rng.nextInt(sx), 2, sx - 3)
    const cz = clampInt(rng.nextInt(sz), 2, sz - 3)
    const cy = clampInt(3 + rng.nextInt(Math.max(1, sy - 8)), 3, sy - 5)
    const radius = 2 + rng.nextRange(0, 1 + binToScalar(spec.features.caveDensity) * 4)
    const radiusSq = radius * radius

    const minX = clampInt(Math.floor(cx - radius), 1, sx - 2)
    const maxX = clampInt(Math.ceil(cx + radius), 1, sx - 2)
    const minZ = clampInt(Math.floor(cz - radius), 1, sz - 2)
    const maxZ = clampInt(Math.ceil(cz + radius), 1, sz - 2)
    const minY = clampInt(Math.floor(cy - radius), 2, sy - 6)
    const maxY = clampInt(Math.ceil(cy + radius), 2, sy - 6)

    for (let z = minZ; z <= maxZ; z++) {
      for (let x = minX; x <= maxX; x++) {
        const surface = findSurfaceHeight(chunk, x, z)
        for (let y = minY; y <= Math.min(maxY, surface - 2); y++) {
          const dx = x - cx
          const dy = y - cy
          const dz = z - cz
          const distSq = dx * dx + dy * dy + dz * dz
          if (distSq > radiusSq) continue
          if (chunk.getBlock(x, y, z) === BlockType.Air) continue
          chunk.setBlock(x, y, z, BlockType.Air)
          carvedBlocks += 1
        }
      }
    }
  }

  scope?.end({
    attempts,
    carvedBlocks,
    rng: rng.snapshot()
  })
}

function getBiomeSurface(biome: WorldSpec['biome']) {
  switch (biome) {
    case 'temperate':
      return surfaceProfile(BlockType.Grass, BlockType.Dirt, BlockType.Stone, BlockType.Air)
    case 'tundra':
      return surfaceProfile(BlockType.Snow, BlockType.Stone, BlockType.Stone, BlockType.Air)
    case 'desert':
      return surfaceProfile(BlockType.Sand, BlockType.Sand, BlockType.Stone, BlockType.Sand)
    case 'islands':
      return surfaceProfile(BlockType.Grass, BlockType.Dirt, BlockType.Stone, BlockType.Water)
    default:
      return surfaceProfile(BlockType.Grass, BlockType.Dirt, BlockType.Stone, BlockType.Air)
  }
}

function surfaceProfile(top: BlockType, subsurface: BlockType, underground: BlockType, seaFill: BlockType) {
  return { top, subsurface, underground, seaFill }
}

function isSoil(block: BlockType) {
  return block === BlockType.Grass || block === BlockType.Dirt || block === BlockType.Sand || block === BlockType.Snow
}

function summarise(min: number, max: number, mean: number) {
  return { min, max, mean }
}

function clamp01(v: number) {
  return Math.max(0, Math.min(1, v))
}

function clampInt(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value))
}

function findSurfaceHeight(chunk: ChunkManager, x: number, z: number) {
  const { y: sy } = chunk.size
  for (let y = sy - 1; y >= 0; y--) {
    if (chunk.getBlock(x, y, z) !== BlockType.Air && chunk.getBlock(x, y, z) !== BlockType.Water) {
      return y
    }
  }
  return 0
}
