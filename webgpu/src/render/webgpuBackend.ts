import { createRenderer } from '../renderer'
import type { RenderBackend, RenderBackendInitOptions, OverlayView } from './renderBackend'
import type { BlockFaceKey, CustomBlock, HighlightSelection } from '../core'
import type { WorldChange } from '../world'

type LegacyRenderer = Awaited<ReturnType<typeof createRenderer>>

class WebGPUBackend implements RenderBackend {
  private renderer: LegacyRenderer | null = null

  async init(options: RenderBackendInitOptions): Promise<void> {
    const { canvas, overlayCanvas, getSelectedBlock, world, dispatchCommand } = options
    this.renderer = await createRenderer(
      {
        canvas,
        overlayCanvas,
        getSelectedBlock,
        dispatchCommand
      },
      world
    )
  }

  applyWorldChange(_change: WorldChange): void {
    // The legacy renderer listens to world.onChange internally;
    // forcing the mesh dirty makes sure GPU buffers refresh immediately.
    this.renderer?.markMeshDirty()
  }

  markWorldDirty(): void {
    this.renderer?.markMeshDirty()
  }

  focusCameraOnBlocks(blocks: Array<{ position: [number, number, number] }> | undefined): void {
    this.renderer?.focusCameraOnBlocks(blocks)
  }

  setHighlight(selection: HighlightSelection | null): void {
    this.renderer?.setHighlightSelection(selection)
  }

  setOverlayViews(views: OverlayView[]): void {
    this.renderer?.setOverlayViews(views)
  }

  applyCustomBlockTextures(
    bitmaps: Record<BlockFaceKey, ImageBitmap>,
    customBlock: CustomBlock,
    customBlocks: CustomBlock[]
  ): void {
    this.renderer?.applyCustomBlockTextures(bitmaps, customBlock, customBlocks)
  }

  updateCustomTextures(customBlocks: CustomBlock[]): void {
    this.renderer?.updateCustomTextures(customBlocks)
  }

  setCameraMode(mode: 'player' | 'overview'): void {
    this.renderer?.setCameraMode(mode)
  }

  getCameraSnapshot() {
    return this.renderer?.getCamera() ?? null
  }

  dispose(): void {
    this.renderer?.destroy()
    this.renderer = null
  }
}

export function createWebGPUBackend(): RenderBackend {
  return new WebGPUBackend()
}
