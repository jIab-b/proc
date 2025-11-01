<script lang="ts">
import { onMount } from 'svelte'
import { get } from 'svelte/store'
import { MapManager, CaptureSystem, DSLEngine, createWorldConfig, type CameraSnapshot } from './engine'
import { WorldState } from './world'
import { WorldEngine } from './engine/worldEngine'
import { createWebGPUBackend } from './render/webgpuBackend'
import type { RenderBackend } from './render/renderBackend'
import { createInputController } from './input/inputController'
import type { InputController } from './input/inputController'
import CanvasContextMenu from './lib/CanvasContextMenu.svelte'
import { withVersion } from './dsl/commands'
  import {
    ChunkManager,
    selectedBlockType,
    selectedCustomBlock,
    interactionMode,
    highlightShape,
    highlightRadius,
    ellipsoidRadiusX,
    ellipsoidRadiusY,
    ellipsoidRadiusZ,
    planeSizeX,
    planeSizeZ,
    highlightSelection,
    gpuHooks,
    BlockType,
    setBlockTextureIndices,
    blockFaceOrder,
    API_BASE_URL,
    customBlocks as customBlocksStore,
    type CustomBlock,
    type Vec3,
    type TerrainGenerateParams
  } from './core'

  let canvasEl: HTMLCanvasElement
  let overlayCanvasEl: HTMLCanvasElement
  let availableMaps: Array<any> = []

  // Context menu
  let showContextMenu = false
  let contextMenuX = 0
  let contextMenuY = 0
  let fileInputRef: HTMLInputElement | null = null

  const worldConfig = createWorldConfig(Math.floor(Math.random() * 1000000))
  const chunk = new ChunkManager(worldConfig.dimensions)
  let worldScale = 2
  const chunkOriginOffset: Vec3 = [-chunk.size.x * worldScale / 2, 0, -chunk.size.z * worldScale / 2]
  const world = new WorldState(chunk, { worldScale, chunkOriginOffset })

  let mapManager: MapManager
  let captureSystem: CaptureSystem
  let dslEngine: DSLEngine
  let renderBackend: RenderBackend | null = null
  let worldEngine: WorldEngine | null = null
  let inputController: InputController | null = null
  let isInGame = false

  async function fetchMaps() {
    try {
      const res = await fetch(`${API_BASE_URL}/api/maps`)
      if (res.ok) {
        const data = await res.json()
        availableMaps = data.maps || []
      }
    } catch {}
  }

  function handleContextMenu(event: MouseEvent) {
    const path = event.composedPath()
    const isCanvasContext = (canvasEl && path.includes(canvasEl)) || (overlayCanvasEl && path.includes(overlayCanvasEl))
    if (isCanvasContext && get(interactionMode) === 'highlight') {
      event.preventDefault()
      return
    }

    if (isInGame) return
    event.preventDefault()
    contextMenuX = event.clientX
    contextMenuY = event.clientY
    showContextMenu = true
  }

  function closeContextMenu() {
    showContextMenu = false
  }

  async function handleSaveMap() {
    closeContextMenu()
    try {
      await mapManager.save(worldConfig, $customBlocksStore, BlockType)
      console.log('Map saved')
    } catch (err) {
      console.error('Save failed:', err)
      alert(`Save failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  async function handleLoadMap(sequence: number) {
    closeContextMenu()
    try {
      const { blocks } = await mapManager.load(sequence, BlockType)
      renderBackend?.markWorldDirty()
      renderBackend?.focusCameraOnBlocks(blocks)
      console.log(`Loaded map ${sequence}`)
    } catch (err) {
      console.error('Load failed:', err)
      alert(`Load failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  async function handleNewMap(copyFrom?: number) {
    closeContextMenu()
    try {
      const blocks = await mapManager.createNew(copyFrom, BlockType)
      renderBackend?.markWorldDirty()
      renderBackend?.focusCameraOnBlocks(blocks)
      console.log('Created new map')
    } catch (err) {
      console.error('Create failed:', err)
      alert(`Create failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  async function handleLoadFromFile() {
    closeContextMenu()
    fileInputRef?.click()
  }

  async function handleFileChange(event: Event) {
    const input = event.target as HTMLInputElement
    const file = input.files?.[0]
    if (!file) return
    try {
      const contents = await file.text()
      const { blocks } = await mapManager.loadFromFile(contents, BlockType)
      renderBackend?.markWorldDirty()
      renderBackend?.focusCameraOnBlocks(blocks)
      console.log('Loaded from file')
      input.value = ''
    } catch (err) {
      console.error('Load from file failed:', err)
      alert(`Load failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
      input.value = ''
    }
  }

  onMount(async () => {
    mapManager = new MapManager(world)
    captureSystem = new CaptureSystem()
    dslEngine = new DSLEngine()

    try {
      const backend = createWebGPUBackend()
      const engineInstance = new WorldEngine({ world, backend })
      const controller = createInputController(engineInstance)

      await backend.init({
        canvas: canvasEl,
        overlayCanvas: overlayCanvasEl,
        getSelectedBlock: () => ({ type: $selectedBlockType, custom: $selectedCustomBlock }),
        world,
        dispatchCommand: (command) => controller.dispatch(command)
      })
      renderBackend = backend
      worldEngine = engineInstance
      inputController = controller

      gpuHooks.set({
        requestFaceBitmaps: async (tiles) => {
          const bitmaps: Record<any, ImageBitmap> = {} as any
          for (const face of blockFaceOrder) {
            const tile = tiles[face]
            if (!tile) continue
            try {
              const res = await fetch(tile.url, { mode: 'cors', credentials: 'omit' })
              if (!res.ok) throw new Error(`HTTP ${res.status}`)
              const blob = await res.blob()
              bitmaps[face] = await createImageBitmap(blob, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' })
            } catch (err) {
              console.error(`Failed to fetch ${face}:`, err)
            }
          }
          return bitmaps
        },
        uploadFaceBitmapsToGPU: (bitmaps, customBlock) => {
          renderBackend?.applyCustomBlockTextures(bitmaps, customBlock, $customBlocksStore)
        },
        getCameraPosition: () => {
          const camera = renderBackend?.getCameraSnapshot()
          return camera ? [...camera.position] as [number, number, number] : null
        },
        getWorldScale: () => world.getWorldScale(),
        chunkToWorld: (chunkCoord: Vec3): Vec3 => {
          const scale = world.getWorldScale()
          return [
            chunkCoord[0] * scale + chunkOriginOffset[0],
            chunkCoord[1] * scale + chunkOriginOffset[1],
            chunkCoord[2] * scale + chunkOriginOffset[2]
          ]
        },
        worldToChunk: (worldCoord: Vec3): Vec3 => {
          const scale = world.getWorldScale()
          return [
            (worldCoord[0] - chunkOriginOffset[0]) / scale,
            (worldCoord[1] - chunkOriginOffset[1]) / scale,
            (worldCoord[2] - chunkOriginOffset[2]) / scale
          ]
        },
        generateTerrain: (params: TerrainGenerateParams) => {
          if (worldEngine) {
            worldEngine.apply(withVersion({ type: 'terrain_region', params, source: 'gpuHooks.generateTerrain' }))
          } else {
            world.apply({ type: 'terrain_region', params, source: 'gpuHooks.generateTerrain' })
          }
        }
      })

      const loadedData = await mapManager.loadFirstAvailable(BlockType, worldConfig)
      renderBackend?.markWorldDirty()
      // Focus camera on loaded blocks, or reset to default if terrain was generated
      if (loadedData && loadedData.blocks) {
        renderBackend?.focusCameraOnBlocks(loadedData.blocks)
      } else {
        // For generated terrain, focus on the center of the chunk
        const centerX = Math.floor(chunk.size.x / 2)
        const centerY = Math.floor(chunk.size.y / 2)
        const centerZ = Math.floor(chunk.size.z / 2)
        renderBackend?.focusCameraOnBlocks([{ position: [centerX, centerY, centerZ] }])
      }

      await fetchMaps()

      document.addEventListener('click', closeContextMenu)
      const handlePointerLock = () => {
        isInGame = document.pointerLockElement === canvasEl
      }
      document.addEventListener('pointerlockchange', handlePointerLock)

      return () => {
        renderBackend?.dispose()
        renderBackend = null
        inputController?.dispose()
        inputController = null
        worldEngine?.dispose()
        worldEngine = null
        document.removeEventListener('click', closeContextMenu)
        document.removeEventListener('pointerlockchange', handlePointerLock)
      }
    } catch (err) {
      console.error('Init failed:', err)
      alert(`WebGPU init failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  })
</script>

<svelte:window on:contextmenu={handleContextMenu} />

<div class="canvas-container">
  <div class="canvas-shell">
    <canvas bind:this={canvasEl}></canvas>
    <canvas class="capture-overlay" bind:this={overlayCanvasEl}></canvas>
  </div>
</div>

<input type="file" accept=".json" bind:this={fileInputRef} on:change={handleFileChange} style="display: none;" />

<CanvasContextMenu
  visible={showContextMenu}
  x={contextMenuX}
  y={contextMenuY}
  {availableMaps}
  on:save={handleSaveMap}
  on:load={(event) => handleLoadMap(event.detail.sequence)}
  on:new={(event) => handleNewMap(event.detail?.copyFrom)}
  on:loadFile={handleLoadFromFile}
  on:refreshMaps={fetchMaps}
/>

<style>
  .canvas-container {
    position: relative;
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 12px;
  }

  .canvas-shell {
    position: relative;
    width: 100%;
    height: 100%;
    aspect-ratio: 16 / 9;
    background: rgba(12, 22, 32, 0.6);
    border: 1px solid rgba(210, 223, 244, 0.1);
    border-radius: 12px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 12px;
  }

  canvas {
    width: 100%;
    height: 100%;
    border-radius: 10px;
    background: #10161d;
  }

  .capture-overlay {
    position: absolute;
    inset: 12px;
    width: calc(100% - 24px);
    height: calc(100% - 24px);
    pointer-events: none;
    background: transparent;
  }

</style>
