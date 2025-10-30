<script lang="ts">
import { onMount } from 'svelte'
import { get } from 'svelte/store'
import { createRenderer } from './renderer'
import { MapManager, CaptureSystem, DSLEngine, createWorldConfig, type CameraSnapshot } from './engine'
import { WorldState } from './world'
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
  let showLoadSubmenu = false
  let showNewMapSubmenu = false
  let fileInputRef: HTMLInputElement | null = null

  const worldConfig = createWorldConfig(Math.floor(Math.random() * 1000000))
  const chunk = new ChunkManager(worldConfig.dimensions)
  let worldScale = 2
  const chunkOriginOffset: Vec3 = [-chunk.size.x * worldScale / 2, 0, -chunk.size.z * worldScale / 2]
  const world = new WorldState(chunk, { worldScale, chunkOriginOffset })

  let mapManager: MapManager
  let captureSystem: CaptureSystem
  let dslEngine: DSLEngine
  let renderer: any = null
  let isInGame = false
  let pointerActive = false

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
    showLoadSubmenu = false
    showNewMapSubmenu = false
  }

  function closeContextMenu() {
    showContextMenu = false
    showLoadSubmenu = false
    showNewMapSubmenu = false
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
      renderer?.markMeshDirty()
      renderer?.focusCameraOnBlocks(blocks)
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
      renderer?.markMeshDirty()
      renderer?.focusCameraOnBlocks(blocks)
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
      renderer?.markMeshDirty()
      renderer?.focusCameraOnBlocks(blocks)
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
      renderer = await createRenderer(
        {
          canvas: canvasEl,
          overlayCanvas: overlayCanvasEl,
          getSelectedBlock: () => ({ type: $selectedBlockType, custom: $selectedCustomBlock })
        },
        world
      )

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
          renderer?.applyCustomBlockTextures(bitmaps, customBlock, $customBlocksStore)
        },
        getCameraPosition: () => {
          const camera = renderer?.getCamera()
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
          world.apply({ type: 'terrain_region', params, source: 'gpuHooks.generateTerrain' })
        }
      })

      const loadedData = await mapManager.loadFirstAvailable(BlockType, worldConfig)
      renderer.markMeshDirty()
      // Focus camera on loaded blocks, or reset to default if terrain was generated
      if (loadedData && loadedData.blocks) {
        renderer.focusCameraOnBlocks(loadedData.blocks)
      } else {
        // For generated terrain, focus on the center of the chunk
        const centerX = Math.floor(chunk.size.x / 2)
        const centerY = Math.floor(chunk.size.y / 2)
        const centerZ = Math.floor(chunk.size.z / 2)
        renderer.focusCameraOnBlocks([{ position: [centerX, centerY, centerZ] }])
      }

      await fetchMaps()

      document.addEventListener('click', closeContextMenu)
      const handlePointerLock = () => {
        isInGame = document.pointerLockElement === canvasEl
        pointerActive = isInGame
      }
      document.addEventListener('pointerlockchange', handlePointerLock)

      highlightSelection.subscribe(sel => renderer?.setHighlightSelection(sel))

      return () => {
        renderer?.destroy()
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

{#if showContextMenu}
  <div class="context-menu" style="left: {contextMenuX}px; top: {contextMenuY}px;" on:click|stopPropagation>
    <button class="context-menu-item" on:click={handleSaveMap}>Save Map</button>
    <button class="context-menu-item" on:click={() => { showLoadSubmenu = !showLoadSubmenu; showNewMapSubmenu = false; fetchMaps(); }}>
      Load Map {showLoadSubmenu ? '▼' : '▶'}
    </button>
    {#if showLoadSubmenu}
      <div class="submenu">
        {#if availableMaps.length === 0}
          <div class="submenu-item disabled">No maps available</div>
        {:else}
          {#each availableMaps as map}
            <button class="submenu-item" on:click={() => handleLoadMap(map.sequence)}>
              Map {map.sequence} ({map.blockCount} blocks)
            </button>
          {/each}
        {/if}
      </div>
    {/if}
    <button class="context-menu-item" on:click={() => { showNewMapSubmenu = !showNewMapSubmenu; showLoadSubmenu = false; fetchMaps(); }}>
      New Map {showNewMapSubmenu ? '▼' : '▶'}
    </button>
    {#if showNewMapSubmenu}
      <div class="submenu">
        <button class="submenu-item" on:click={() => handleNewMap()}>Create Empty Map</button>
        {#if availableMaps.length > 0}
          <div class="submenu-divider"></div>
          <div class="submenu-label">Copy From:</div>
          {#each availableMaps as map}
            <button class="submenu-item" on:click={() => handleNewMap(map.sequence)}>
              Map {map.sequence} ({map.blockCount} blocks)
            </button>
          {/each}
        {/if}
      </div>
    {/if}
    <button class="context-menu-item" on:click={handleLoadFromFile}>Load From File</button>
    <div class="context-divider"></div>
    <div class="context-section">
      <label>Interaction Mode</label>
      <select bind:value={$interactionMode}>
        <option value="block">Block Placement</option>
        <option value="highlight">Highlight Select</option>
      </select>
    </div>
    {#if $interactionMode === 'highlight'}
      <div class="context-section">
        <label>Highlight Shape</label>
        <select bind:value={$highlightShape}>
          <option value="cube">Cube</option>
          <option value="sphere">Sphere</option>
          <option value="ellipsoid">Ellipsoid</option>
          <option value="plane">Plane (Terrain Base)</option>
        </select>
      </div>
      {#if $highlightShape === 'ellipsoid'}
        <div class="context-section">
          <label>Radius X: {$ellipsoidRadiusX}</label>
          <input type="range" min="1" max="32" step="0.5" bind:value={$ellipsoidRadiusX} />
        </div>
        <div class="context-section">
          <label>Radius Y: {$ellipsoidRadiusY}</label>
          <input type="range" min="1" max="32" step="0.5" bind:value={$ellipsoidRadiusY} />
        </div>
        <div class="context-section">
          <label>Radius Z: {$ellipsoidRadiusZ}</label>
          <input type="range" min="1" max="32" step="0.5" bind:value={$ellipsoidRadiusZ} />
        </div>
      {:else if $highlightShape === 'plane'}
        <div class="context-section">
          <label>Plane Size X: {$planeSizeX}</label>
          <input type="range" min="4" max="32" step="1" bind:value={$planeSizeX} />
        </div>
        <div class="context-section">
          <label>Plane Size Z: {$planeSizeZ}</label>
          <input type="range" min="4" max="32" step="1" bind:value={$planeSizeZ} />
        </div>
      {:else}
        <div class="context-section">
          <label>Radius: {$highlightRadius}</label>
          <input type="range" min="1" max="16" step="1" bind:value={$highlightRadius} />
        </div>
      {/if}
    {/if}
  </div>
{/if}

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

  .context-menu {
    position: fixed;
    background: rgba(20, 30, 45, 0.98);
    border: 1px solid rgba(210, 223, 244, 0.3);
    border-radius: 8px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
    padding: 4px;
    min-width: clamp(160px, 15vw, 180px);
    z-index: 1000;
    backdrop-filter: blur(8px);
  }

  .context-menu-item {
    display: block;
    width: 100%;
    background: transparent;
    border: none;
    color: #e3ebf7;
    padding: clamp(6px, 1.5vw, 8px) clamp(10px, 2vw, 12px);
    text-align: left;
    cursor: pointer;
    font-size: clamp(11px, 2.5vw, 13px);
    border-radius: 4px;
    transition: background 0.15s;
  }

  .context-menu-item:hover {
    background: rgba(48, 66, 88, 0.6);
  }

  .submenu {
    margin-left: clamp(6px, 1.5vw, 8px);
    margin-top: clamp(2px, 0.5vw, 4px);
    padding-left: clamp(6px, 1.5vw, 8px);
    border-left: 2px solid rgba(210, 223, 244, 0.2);
  }

  .submenu-item {
    display: block;
    width: 100%;
    background: transparent;
    border: none;
    color: #c8d5e8;
    padding: clamp(4px, 1vw, 6px) clamp(8px, 2vw, 12px);
    text-align: left;
    cursor: pointer;
    font-size: clamp(10px, 2.2vw, 12px);
    border-radius: 4px;
    transition: background 0.15s;
  }

  .submenu-item:hover:not(.disabled) {
    background: rgba(48, 66, 88, 0.4);
  }

  .submenu-item.disabled {
    color: #6b7785;
    cursor: default;
  }

  .submenu-divider {
    height: 1px;
    background: rgba(210, 223, 244, 0.2);
    margin: 4px 0;
  }

  .submenu-label {
    color: #8a98ab;
    padding: clamp(2px, 0.5vw, 4px) clamp(8px, 2vw, 12px);
    font-size: clamp(9px, 2vw, 11px);
    font-weight: 500;
    text-transform: uppercase;
  }

  .context-divider {
    height: 1px;
    background: rgba(210, 223, 244, 0.2);
    margin: clamp(6px, 1.5vw, 8px) 0;
  }

  .context-section {
    padding: clamp(2px, 0.5vw, 4px) clamp(6px, 1.5vw, 8px) clamp(4px, 1vw, 6px);
    display: flex;
    flex-direction: column;
    gap: clamp(2px, 0.5vw, 4px);
  }

  .context-section label {
    font-size: clamp(9px, 2vw, 11px);
    text-transform: uppercase;
    color: #8fa0b8;
  }

  .context-section select,
  .context-section input[type='range'] {
    width: 100%;
    background: rgba(34, 50, 68, 0.6);
    border: 1px solid rgba(190, 210, 230, 0.25);
    border-radius: 4px;
    padding: clamp(4px, 1vw, 6px);
    color: #e3ebf7;
    font-size: clamp(10px, 2.2vw, 12px);
  }

  .context-section input[type='range'] {
    padding: 0;
  }
</style>
