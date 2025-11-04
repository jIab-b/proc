import type { HighlightSelection, TerrainGenerateParams, Vec3 } from '../core'

export const DSL_VERSION = 2 as const

export type DSLVersion = typeof DSL_VERSION

export type DSLSource = string | undefined

export type VoxelEdit = {
  position: Vec3
  blockType: number
}

interface BaseCommand<T extends string> {
  v: DSLVersion
  type: T
  source?: string
}

export type SetBlockCommand = BaseCommand<'set_block'> & {
  edit: VoxelEdit
}

export type SetBlocksCommand = BaseCommand<'set_blocks'> & {
  edits: VoxelEdit[]
}

export type ClearAllCommand = BaseCommand<'clear_all'>

export type LoadSnapshotCommand = BaseCommand<'load_snapshot'> & {
  blocks: VoxelEdit[]
  worldScale?: number
  clear?: boolean
}

export type HighlightSetCommand = BaseCommand<'highlight_set'> & {
  selection: HighlightSelection
}

export type HighlightClearCommand = BaseCommand<'highlight_clear'>

export type TerrainRegionCommand = BaseCommand<'terrain_region'> & {
  params: TerrainGenerateParams
}

// === NEW V2 COMMANDS ===

// Materials
export type SetMaterialCommand = BaseCommand<'set_material'> & {
  blockType: number
  material: MaterialParams
}

export interface MaterialParams {
  albedo?: Vec3
  roughness: number        // [0, 1]
  metallic: number         // [0, 1]
  emissive?: Vec3
  emissiveStrength?: number // [0, 100]
  ao?: number              // [0, 1] ambient occlusion
}

// Lighting
export type SetLightingCommand = BaseCommand<'set_lighting'> & {
  params: LightingParams
}

export interface LightingParams {
  sun: {
    direction: Vec3
    color: Vec3
    intensity: number      // [0, 10]
  }
  sky: {
    zenithColor: Vec3
    horizonColor: Vec3
    groundColor: Vec3
    intensity: number      // [0, 2]
  }
  ambient: {
    color: Vec3
    intensity: number      // [0, 2]
  }
  timeOfDay?: number       // [0, 24] auto-compute sun
}

export type AddPointLightCommand = BaseCommand<'add_point_light'> & {
  id: string
  light: PointLight
}

export interface PointLight {
  position: Vec3
  color: Vec3
  intensity: number        // [0, 100]
  radius: number           // [0.1, 50]
}

export type RemovePointLightCommand = BaseCommand<'remove_point_light'> & {
  id: string
}

// Procedural Generation
export type GenerateStructureCommand = BaseCommand<'generate_structure'> & {
  generator: StructureGenerator
}

export interface BoundingBox {
  min: Vec3
  max: Vec3
}

export interface StructureGenerator {
  type: 'l-system' | 'cellular_automata' | 'noise_sculpt'
  region: BoundingBox
  seed?: number

  lSystem?: LSystemParams
  cellularAutomata?: CellularAutomataParams
  noiseSculpt?: NoiseSculptParams
}

export interface LSystemParams {
  axiom: string
  rules: Record<string, string>
  iterations: number       // [1, 8]
  angle: number            // [10, 90] degrees
  thickness: number        // [0.5, 3]
  taper: number           // [0.7, 0.95]
  blockType: number
  leafBlockType?: number
  leafProbability?: number // [0, 1]
}

export interface CellularAutomataParams {
  fillProbability: number  // [0, 1]
  birthLimit: number       // [3, 8]
  deathLimit: number       // [3, 8]
  iterations: number       // [1, 20]
  fillBlockType: number
  emptyBlockType: number
}

export interface NoiseSculptParams {
  noiseType: 'perlin' | 'simplex' | 'worley'
  frequency: number        // [0.01, 1]
  octaves: number          // [1, 6]
  threshold: number        // [0, 1] density cutoff
  invert: boolean
  blockType: number
}

export type DSLCommand =
  | SetBlockCommand
  | SetBlocksCommand
  | ClearAllCommand
  | LoadSnapshotCommand
  | HighlightSetCommand
  | HighlightClearCommand
  | TerrainRegionCommand
  // V2 commands
  | SetMaterialCommand
  | SetLightingCommand
  | AddPointLightCommand
  | RemovePointLightCommand
  | GenerateStructureCommand

export function withVersion<T extends Omit<BaseCommand<string>, 'v'>>(command: T): T & { v: DSLVersion } {
  return { v: DSL_VERSION, ...command }
}

export function isHighlightCommand(cmd: DSLCommand): cmd is HighlightSetCommand | HighlightClearCommand {
  return cmd.type === 'highlight_set' || cmd.type === 'highlight_clear'
}
