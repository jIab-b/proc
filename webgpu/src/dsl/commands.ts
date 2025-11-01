import type { HighlightSelection, TerrainGenerateParams, Vec3 } from '../core'

export const DSL_VERSION = 1 as const

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

export type DSLCommand =
  | SetBlockCommand
  | SetBlocksCommand
  | ClearAllCommand
  | LoadSnapshotCommand
  | HighlightSetCommand
  | HighlightClearCommand
  | TerrainRegionCommand

export function withVersion<T extends Omit<BaseCommand<string>, 'v'>>(command: T): T & { v: DSLVersion } {
  return { v: DSL_VERSION, ...command }
}

export function isHighlightCommand(cmd: DSLCommand): cmd is HighlightSetCommand | HighlightClearCommand {
  return cmd.type === 'highlight_set' || cmd.type === 'highlight_clear'
}
