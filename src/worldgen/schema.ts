import {
  BinLevel,
  BiomeId,
  FeatureBudget,
  MacroScale,
  StageSeeds,
  TerrainForm,
  TerrainProfile,
  UNIT_SCALE_VALUES,
  UnitScale,
  WorldDimensions,
  WorldSpec
} from './dsl'

type ValidationResult = { valid: true; errors: [] } | { valid: false; errors: string[] }

export const WORLD_SPEC_SCHEMA = {
  type: 'object',
  required: ['seed', 'biome', 'terrain', 'terrainProfile', 'features', 'shaders', 'seeds'],
  properties: {
    seed: { type: 'integer' },
    label: { type: ['string', 'null'] },
    biome: { enum: ['temperate', 'tundra', 'desert', 'islands'] satisfies BiomeId[] },
    terrain: { enum: ['plains', 'ridged', 'valley', 'mesa'] satisfies TerrainForm[] },
    terrainProfile: {
      type: 'object',
      required: ['elevationVariance', 'ridgeFrequency', 'warpStrength', 'seaLevel', 'macroScale'],
      properties: {
        elevationVariance: { enum: binLevels() },
        ridgeFrequency: { enum: binLevels() },
        warpStrength: { enum: binLevels() },
        seaLevel: { enum: ['low', 'normal', 'high'] },
        macroScale: { enum: macroScales() }
      }
    },
    features: {
      type: 'object',
      required: ['riverCount', 'treeDensity', 'caveDensity'],
      properties: {
        riverCount: { type: 'integer', minimum: 0, maximum: 8 },
        treeDensity: { enum: binLevels() },
        caveDensity: { enum: binLevels() }
      }
    },
    dimensions: {
      type: 'object',
      required: ['x', 'y', 'z'],
      properties: {
        x: { type: 'integer', minimum: 8, maximum: 2048 },
        y: { type: 'integer', minimum: 16, maximum: 384 },
        z: { type: 'integer', minimum: 8, maximum: 2048 }
      }
    },
    shaders: {
      type: 'array',
      items: { type: 'string' }
    },
    seeds: {
      type: 'object',
      required: ['climate', 'terrain', 'surface', 'features', 'rivers', 'caves'],
      properties: {
        climate: { type: 'integer' },
        terrain: { type: 'integer' },
        surface: { type: 'integer' },
        features: { type: 'integer' },
        rivers: { type: 'integer' },
        caves: { type: 'integer' }
      }
    },
    unitsPerBlock: { enum: UNIT_SCALE_VALUES }
  }
} as const

export function validateWorldSpec(spec: unknown): ValidationResult {
  const errors: string[] = []
  if (!isObject(spec)) {
    return { valid: false, errors: ['WorldSpec must be an object'] }
  }
  const candidate = spec as Partial<WorldSpec>

  if (!isInteger(candidate.seed)) errors.push('seed missing or not integer')
  if (!isEnum(candidate.biome, WORLD_SPEC_SCHEMA.properties.biome.enum)) errors.push('invalid biome')
  if (!isEnum(candidate.terrain, WORLD_SPEC_SCHEMA.properties.terrain.enum)) errors.push('invalid terrain')

  if (!isObject(candidate.terrainProfile)) {
    errors.push('terrainProfile missing')
  } else {
    validateTerrainProfile(candidate.terrainProfile as Partial<TerrainProfile>, errors)
  }

  if (!isObject(candidate.features)) {
    errors.push('features missing')
  } else {
    validateFeatures(candidate.features as Partial<FeatureBudget>, errors)
  }

  if (!Array.isArray(candidate.shaders)) errors.push('shaders must be array')

  if (!isObject(candidate.seeds)) {
    errors.push('seeds missing')
  } else {
    validateSeeds(candidate.seeds as Partial<StageSeeds>, errors)
  }

  if (!isObject(candidate.dimensions)) {
    errors.push('dimensions missing')
  } else {
    validateDimensions(candidate.dimensions as Partial<WorldDimensions>, errors)
  }

  if (!isEnum(candidate.unitsPerBlock, UNIT_SCALE_VALUES)) errors.push('unitsPerBlock invalid')

  if (errors.length) return { valid: false, errors }
  return { valid: true, errors: [] }
}

function validateTerrainProfile(profile: Partial<TerrainProfile>, errors: string[]) {
  if (!isEnum(profile.elevationVariance, binLevels())) errors.push('terrainProfile.elevationVariance invalid')
  if (!isEnum(profile.ridgeFrequency, binLevels())) errors.push('terrainProfile.ridgeFrequency invalid')
  if (!isEnum(profile.warpStrength, binLevels())) errors.push('terrainProfile.warpStrength invalid')
  if (!isEnum(profile.seaLevel, ['low', 'normal', 'high'] as const)) errors.push('terrainProfile.seaLevel invalid')
  if (!isEnum(profile.macroScale, macroScales())) errors.push('terrainProfile.macroScale invalid')
}

function validateFeatures(features: Partial<FeatureBudget>, errors: string[]) {
  if (!isInteger(features.riverCount) || features.riverCount! < 0) errors.push('features.riverCount invalid')
  if (!isEnum(features.treeDensity, binLevels())) errors.push('features.treeDensity invalid')
  if (!isEnum(features.caveDensity, binLevels())) errors.push('features.caveDensity invalid')
}

function validateSeeds(seeds: Partial<StageSeeds>, errors: string[]) {
  const keys: Array<keyof StageSeeds> = ['climate', 'terrain', 'surface', 'features', 'rivers', 'caves']
  for (const key of keys) {
    if (!isInteger(seeds[key])) errors.push(`seeds.${String(key)} invalid`)
  }
}

function binLevels(): readonly BinLevel[] {
  return ['very_low', 'low', 'medium', 'high', 'very_high']
}

function macroScales(): readonly MacroScale[] {
  return ['small', 'medium', 'large', 'massive']
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function isInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value)
}

function isEnum<T extends readonly unknown[]>(value: unknown, values: T): value is T[number] {
  return values.includes(value as T[number])
}

function validateDimensions(dimensions: Partial<WorldDimensions>, errors: string[]) {
  if (!isInteger(dimensions.x) || dimensions.x! < 8) errors.push('dimensions.x invalid')
  if (!isInteger(dimensions.y) || dimensions.y! < 16) errors.push('dimensions.y invalid')
  if (!isInteger(dimensions.z) || dimensions.z! < 8) errors.push('dimensions.z invalid')
}

export function assertWorldSpec(spec: unknown): asserts spec is WorldSpec {
  const result = validateWorldSpec(spec)
  if (!result.valid) {
    throw new Error(`Invalid WorldSpec:\n- ${result.errors.join('\n- ')}`)
  }
}
