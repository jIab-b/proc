import type { CameraSnapshot } from '../engine'
import type { WorldChange, WorldState } from '../world'
import type {
  BlockFaceKey,
  CustomBlock,
  HighlightSelection,
  Vec3
} from '../core'
import type { DSLCommand } from '../dsl/commands'

export type OverlayView = {
  position: Vec3
  id: string
}

export interface SelectedBlock {
  type: number
  custom: CustomBlock | null
}

export interface RenderBackendInitOptions {
  /**
   * HTML canvas that receives the main WebGPU output.
   */
  canvas: HTMLCanvasElement
  /**
   * Optional overlay canvas used for selection/highlight drawing.
   */
  overlayCanvas?: HTMLCanvasElement | null
  /**
   * Read-only access to the authoritative world state.
   */
  world: WorldState
  /**
   * Function that reports the currently selected block and optional custom block payload.
   */
  getSelectedBlock: () => SelectedBlock
  /**
   * Optional callback that consumes DSL commands emitted by the render/input layer.
   */
  dispatchCommand?: (command: DSLCommand) => void
}

export interface RenderBackend {
  /**
   * Perform any async setup and attach to DOM canvases.
   */
  init(options: RenderBackendInitOptions): Promise<void>

  /**
   * Apply a world change emitted by the engine.
   */
  applyWorldChange(change: WorldChange): void

  /**
   * Force the backend to rebuild GPU buffers after out-of-band edits.
   */
  markWorldDirty(): void

  /**
   * Reposition the camera to frame the provided block edits.
   */
  focusCameraOnBlocks(blocks: Array<{ position: [number, number, number] }> | undefined): void

  /**
   * Update highlight metadata used for overlay rendering.
   */
  setHighlight(selection: HighlightSelection | null): void

  /**
   * Update overlay marker information (camera capture helpers, etc).
   */
  setOverlayViews(views: OverlayView[]): void

  /**
   * Keep custom block GPU textures in sync with UI edits.
   */
  applyCustomBlockTextures(
    bitmaps: Record<BlockFaceKey, ImageBitmap>,
    customBlock: CustomBlock,
    customBlocks: CustomBlock[]
  ): void

  /**
   * Refresh any GPU-side metadata derived from custom block definitions.
   */
  updateCustomTextures(customBlocks: CustomBlock[]): void

  /**
   * Switch between camera controller modes.
   */
  setCameraMode(mode: 'player' | 'overview'): void

  /**
   * Fetch the latest camera snapshot for UI consumption.
   */
  getCameraSnapshot(): CameraSnapshot | null

  /**
   * Release GPU resources and detach listeners.
   */
  dispose(): void
}
