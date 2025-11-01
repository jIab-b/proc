import type { RenderBackend } from '../render/renderBackend'
import type { DSLCommand } from '../dsl/commands'
import type { HighlightSelection, CustomBlock } from '../core'
import { highlightSelection as highlightSelectionStore } from '../core'
import { WorldState, type WorldChange } from '../world'
import type { WorldResult } from '../world'

type Listener = (change: WorldChange) => void

export type EngineApplyResult = WorldResult & { command: DSLCommand }

export interface WorldEngineOptions {
  world: WorldState
  backend: RenderBackend
}

export class WorldEngine {
  private listeners = new Set<Listener>()
  private stopWorldListener: (() => void) | null = null
  private highlight: HighlightSelection | null = null

  constructor(private options: WorldEngineOptions) {
    this.stopWorldListener = this.options.world.onChange(change => {
      this.options.backend.applyWorldChange(change)
      this.options.backend.markWorldDirty()
      this.listeners.forEach(listener => listener(change))
    })
  }

  get world() {
    return this.options.world
  }

  onChange(listener: Listener) {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  apply(command: DSLCommand): EngineApplyResult {
    switch (command.type) {
      case 'set_block':
        return this.emitWorldResult(command, this.options.world.apply({
          type: 'set_block',
          edit: command.edit,
          source: command.source
        }))
      case 'set_blocks':
        return this.emitWorldResult(command, this.options.world.apply({
          type: 'set_blocks',
          edits: command.edits,
          source: command.source
        }))
      case 'clear_all':
        return this.emitWorldResult(command, this.options.world.apply({
          type: 'clear_all',
          source: command.source
        }))
      case 'load_snapshot':
        return this.emitWorldResult(command, this.options.world.apply({
          type: 'load_snapshot',
          blocks: command.blocks,
          worldScale: command.worldScale,
          clear: command.clear,
          source: command.source
        }))
      case 'terrain_region':
        return this.emitWorldResult(command, this.options.world.apply({
          type: 'terrain_region',
          params: command.params,
          source: command.source
        }))
      case 'highlight_set':
        this.setHighlight(command.selection)
        return { success: true, change: undefined, command }
      case 'highlight_clear':
        this.setHighlight(null)
        return { success: true, change: undefined, command }
      default:
        return {
          success: false,
          message: `Unhandled DSL command ${(command as DSLCommand).type}`,
          command
        }
    }
  }

  setHighlight(selection: HighlightSelection | null) {
    this.highlight = selection ? { ...selection } : null
    highlightSelectionStore.set(this.highlight)
    this.options.backend.setHighlight(this.highlight)
  }

  getHighlight() {
    return this.highlight ? { ...this.highlight } : null
  }

  updateCustomBlocks(customBlocks: CustomBlock[]) {
    this.options.backend.updateCustomTextures(customBlocks)
  }

  dispose() {
    this.stopWorldListener?.()
    this.stopWorldListener = null
    this.listeners.clear()
    this.options.backend.setHighlight(null)
  }

  private emitWorldResult(command: DSLCommand, result: WorldResult): EngineApplyResult {
    if (!result.change) {
      return { ...result, command }
    }
    // change propagation handled via world.onChange listener
    return { ...result, command }
  }
}
