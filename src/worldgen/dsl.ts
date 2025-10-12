export type BinLevel = 'very_low' | 'low' | 'medium' | 'high' | 'very_high'

export type BiomeId = 'temperate' | 'tundra' | 'desert' | 'islands'
export type TerrainForm = 'plains' | 'ridged' | 'valley' | 'mesa'

export type ShaderProfileId = 'default_day' | 'icy_overcast' | 'desert_heat' | 'island_breeze'

export type MacroScale = 'small' | 'medium' | 'large' | 'massive'
export type UnitScale = 1 | 2 | 4 | 8

export interface TerrainProfile {
  elevationVariance: BinLevel
  ridgeFrequency: BinLevel
  warpStrength: BinLevel
  seaLevel: 'low' | 'normal' | 'high'
  macroScale: MacroScale
}

export interface FeatureBudget {
  riverCount: number
  treeDensity: BinLevel
  caveDensity: BinLevel
}

export interface StageSeeds {
  climate: number
  terrain: number
  surface: number
  features: number
  rivers: number
  caves: number
}

export interface WorldSpec {
  seed: number
  label?: string
  biome: BiomeId
  terrain: TerrainForm
  terrainProfile: TerrainProfile
  features: FeatureBudget
  shaders: ShaderProfileId[]
  seeds: StageSeeds
  dimensions: WorldDimensions
  unitsPerBlock: UnitScale
}

export interface CameraRigSpec {
  id: string
  tag: string
  position: [number, number, number]
  rotation: [number, number, number]
  fov: number
}

export interface WorldDimensions {
  x: number
  y: number
  z: number
}

const BIN_TO_SCALAR: Record<BinLevel, number> = {
  very_low: 0.15,
  low: 0.35,
  medium: 0.5,
  high: 0.7,
  very_high: 0.9
}

const SEA_LEVEL_TO_RATIO: Record<TerrainProfile['seaLevel'], number> = {
  low: 0.25,
  normal: 0.35,
  high: 0.45
}

const MACRO_SCALE_TO_FREQ: Record<MacroScale, number> = {
  small: 1 / 128,
  medium: 1 / 256,
  large: 1 / 512,
  massive: 1 / 1024
}

export const UNIT_SCALE_VALUES: UnitScale[] = [1, 2, 4, 8]

export function binToScalar(bin: BinLevel) {
  return BIN_TO_SCALAR[bin]
}

export function seaLevelToRatio(level: TerrainProfile['seaLevel']) {
  return SEA_LEVEL_TO_RATIO[level]
}

export function macroScaleToFrequency(scale: MacroScale) {
  return MACRO_SCALE_TO_FREQ[scale]
}

export function createDefaultWorldSpec(seed = 1337): WorldSpec {
  return {
    seed,
    label: 'temperate_basin',
    biome: 'temperate',
    terrain: 'valley',
    terrainProfile: {
      elevationVariance: 'high',
      ridgeFrequency: 'medium',
      warpStrength: 'low',
      seaLevel: 'normal',
      macroScale: 'massive'
    },
    features: {
      riverCount: 4,
      treeDensity: 'high',
      caveDensity: 'low'
    },
    shaders: ['default_day'],
    seeds: deriveStageSeeds(seed),
    dimensions: { x: 256, y: 96, z: 256 },
    unitsPerBlock: 4
  }
}

export function deriveStageSeeds(seed: number): StageSeeds {
  return {
    climate: deriveSeed(seed, 'climate'),
    terrain: deriveSeed(seed, 'terrain'),
    surface: deriveSeed(seed, 'surface'),
    features: deriveSeed(seed, 'features'),
    rivers: deriveSeed(seed, 'rivers'),
    caves: deriveSeed(seed, 'caves')
  }
}

export function deriveSeed(base: number, label: string) {
  let h = base >>> 0
  for (let i = 0; i < label.length; i++) {
    const c = label.charCodeAt(i)
    h = Math.imul(h ^ c, 0x45d9f3b)
    h = (h ^ (h >>> 13)) >>> 0
  }
  h = Math.imul(h ^ (h >>> 16), 0x9e3779b1)
  return h >>> 0
}
