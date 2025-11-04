import type { DSLCommand, VoxelEdit, MaterialParams, LightingParams, PointLight, StructureGenerator } from './commands'
import { withVersion } from './commands'
import type { TerrainGenerateParams, HighlightSelection, Vec3 } from '../core'
import type { WorldState } from '../world'
import { highlightSelection as highlightSelectionStore, blockMaterials, sceneLighting, pointLights } from '../core'

export type CommandDispatcher = (command: DSLCommand) => void

export interface CommandEmitterDependencies {
  world: WorldState
  fallbackHighlightSet?: (selection: HighlightSelection | null) => void
}

function cloneVector(vec: Vec3): Vec3 {
  return [...vec] as Vec3
}

function cloneSelection(selection: HighlightSelection): HighlightSelection {
  const copy: HighlightSelection = {
    center: cloneVector(selection.center),
    radius: selection.radius,
    shape: selection.shape
  }
  if (selection.radiusX !== undefined) copy.radiusX = selection.radiusX
  if (selection.radiusY !== undefined) copy.radiusY = selection.radiusY
  if (selection.radiusZ !== undefined) copy.radiusZ = selection.radiusZ
  if (selection.planeSizeX !== undefined) copy.planeSizeX = selection.planeSizeX
  if (selection.planeSizeZ !== undefined) copy.planeSizeZ = selection.planeSizeZ
  return copy
}

function cloneEdit(edit: VoxelEdit): VoxelEdit {
  return {
    position: cloneVector(edit.position),
    blockType: edit.blockType
  }
}

function cloneEdits(edits: VoxelEdit[]): VoxelEdit[] {
  return edits.map(cloneEdit)
}

export function createCommandEmitter(
  deps: CommandEmitterDependencies,
  dispatchCommand?: CommandDispatcher
) {
  const dispatch = dispatchCommand ?? ((command: DSLCommand) => {
    switch (command.type) {
      case 'set_block':
        deps.world.apply({ type: 'set_block', edit: command.edit, source: command.source })
        break
      case 'set_blocks':
        deps.world.apply({ type: 'set_blocks', edits: command.edits, source: command.source })
        break
      case 'clear_all':
        deps.world.apply({ type: 'clear_all', source: command.source })
        break
      case 'load_snapshot':
        deps.world.apply({
          type: 'load_snapshot',
          blocks: command.blocks,
          worldScale: command.worldScale,
          clear: command.clear,
          source: command.source
        })
        break
      case 'terrain_region':
        deps.world.apply({ type: 'terrain_region', params: command.params, source: command.source })
        break
      case 'highlight_set':
        (deps.fallbackHighlightSet ?? highlightSelectionStore.set)(cloneSelection(command.selection))
        break
      case 'highlight_clear':
        (deps.fallbackHighlightSet ?? highlightSelectionStore.set)(null)
        break
      case 'set_material':
        blockMaterials.update(m => {
          m.set(command.blockType, command.material)
          return m
        })
        break
      case 'set_lighting':
        sceneLighting.set(command.params)
        break
      case 'add_point_light':
        pointLights.update(lights => {
          lights.set(command.id, command.light)
          return lights
        })
        break
      case 'remove_point_light':
        pointLights.update(lights => {
          lights.delete(command.id)
          return lights
        })
        break
      case 'generate_structure':
        deps.world.apply({
          type: 'generate_structure',
          generator: command.generator,
          source: command.source
        })
        break
      default:
        throw new Error(`Unhandled fallback DSL command ${(command as DSLCommand).type}`)
    }
  })

  return {
    dispatch,
    setBlock(edit: VoxelEdit, source: string) {
      dispatch(withVersion({ type: 'set_block', edit: cloneEdit(edit), source }))
    },
    setBlocks(edits: VoxelEdit[], source: string) {
      if (!edits.length) return
      dispatch(withVersion({ type: 'set_blocks', edits: cloneEdits(edits), source }))
    },
    terrainRegion(params: TerrainGenerateParams, source: string) {
      const payload: TerrainGenerateParams = JSON.parse(JSON.stringify(params))
      dispatch(withVersion({ type: 'terrain_region', params: payload, source }))
    },
    highlight(selection: HighlightSelection, source: string) {
      dispatch(withVersion({ type: 'highlight_set', selection: cloneSelection(selection), source }))
    },
    clearHighlight(source: string) {
      dispatch(withVersion({ type: 'highlight_clear', source }))
    },
    setMaterial(blockType: number, material: MaterialParams, source: string) {
      dispatch(withVersion({ type: 'set_material', blockType, material, source }))
    },
    setLighting(params: LightingParams, source: string) {
      dispatch(withVersion({ type: 'set_lighting', params, source }))
    },
    addPointLight(id: string, light: PointLight, source: string) {
      dispatch(withVersion({ type: 'add_point_light', id, light, source }))
    },
    removePointLight(id: string, source: string) {
      dispatch(withVersion({ type: 'remove_point_light', id, source }))
    },
    generateStructure(generator: StructureGenerator, source: string) {
      dispatch(withVersion({ type: 'generate_structure', generator, source }))
    }
  }
}
