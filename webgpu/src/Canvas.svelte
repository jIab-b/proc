<script lang="ts">
import { onMount } from 'svelte'
import { get } from 'svelte/store'
import { createRenderer } from './renderer'
import { MapManager, CaptureSystem, DSLEngine, createWorldConfig, generateTerrain, type CameraSnapshot } from './engine'
import { generateRegion, createTerrainGeneratorState } from './procedural/terrainGenerator'
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
    planeSize,
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
    mapManager = new MapManager(chunk, worldScale)
    captureSystem = new CaptureSystem()
    dslEngine = new DSLEngine()

    try {
      renderer = await createRenderer(
        {
          canvas: canvasEl,
          overlayCanvas: overlayCanvasEl,
          getSelectedBlock: () => ({ type: $selectedBlockType, custom: $selectedCustomBlock })
        },
        chunk,
        worldScale,
        chunkOriginOffset
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
        getWorldScale: () => worldScale,
        chunkToWorld: (chunkCoord: Vec3): Vec3 => [
          chunkCoord[0] * worldScale + chunkOriginOffset[0],
          chunkCoord[1] * worldScale + chunkOriginOffset[1],
          chunkCoord[2] * worldScale + chunkOriginOffset[2]
        ],
        worldToChunk: (worldCoord: Vec3): Vec3 => [
          (worldCoord[0] - chunkOriginOffset[0]) / worldScale,
          (worldCoord[1] - chunkOriginOffset[1]) / worldScale,
          (worldCoord[2] - chunkOriginOffset[2]) / worldScale
        ],
        generateTerrain: (params: TerrainGenerateParams) => {
          console.log('=== GENERATE TERRAIN HOOK CALLED ===')
          console.log('Params (WORLD coords):', params)
          console.log('chunkOriginOffset:', chunkOriginOffset, 'worldScale:', worldScale)
          console.log('Region min (WORLD):', params.region.min)
          console.log('Region max (WORLD):', params.region.max)
          console.log('Ellipsoid mask:', params.ellipsoidMask)

          // Convert world coordinates to chunk coordinates
          const worldToChunk = (worldCoord: Vec3): Vec3 => [
            (worldCoord[0] - chunkOriginOffset[0]) / worldScale,
            (worldCoord[1] - chunkOriginOffset[1]) / worldScale,
            (worldCoord[2] - chunkOriginOffset[2]) / worldScale
          ]

          const chunkMin = worldToChunk(params.region.min)
          const chunkMax = worldToChunk(params.region.max)

          console.log('After conversion to CHUNK coords:')
          console.log('  chunkMin:', chunkMin)
          console.log('  chunkMax:', chunkMax)

          // Clamp to chunk bounds
          const chunkRegion = {
            min: [
              Math.max(0, Math.floor(chunkMin[0])),
              Math.max(0, Math.floor(chunkMin[1])),
              Math.max(0, Math.floor(chunkMin[2]))
            ] as [number, number, number],
            max: [
              Math.min(chunk.size.x - 1, Math.floor(chunkMax[0])),
              Math.min(chunk.size.y - 1, Math.floor(chunkMax[1])),
              Math.min(chunk.size.z - 1, Math.floor(chunkMax[2]))
            ] as [number, number, number]
          }

          console.log('World region min:', params.region.min, 'max:', params.region.max)
          console.log('World to chunk conversion:')
          console.log('  chunkMin:', chunkMin)
          console.log('  chunkMax:', chunkMax)
          console.log('Converted to chunk coords (floored):', chunkRegion)
          console.log('Chunk size:', chunk.size)
          console.log('Region within bounds?',
            chunkRegion.min[0] >= 0 && chunkRegion.max[0] < chunk.size.x &&
            chunkRegion.min[1] >= 0 && chunkRegion.max[1] < chunk.size.y &&
            chunkRegion.min[2] >= 0 && chunkRegion.max[2] < chunk.size.z
          )
          console.log('Bounds check details:')
          console.log('  X: [', chunkRegion.min[0], ',', chunkRegion.max[0], '] vs chunk [0,', chunk.size.x - 1, ']')
          console.log('  Y: [', chunkRegion.min[1], ',', chunkRegion.max[1], '] vs chunk [0,', chunk.size.y - 1, ']')
          console.log('  Z: [', chunkRegion.min[2], ',', chunkRegion.max[2], '] vs chunk [0,', chunk.size.z - 1, ']')

          const terrainState = createTerrainGeneratorState(params.profile, params.params)

          // Helper function to check if a point is inside the ellipsoid
          // Takes CHUNK coordinates and converts to WORLD coordinates for comparison
          const isInsideEllipsoid = (chunkX: number, chunkY: number, chunkZ: number): boolean => {
            if (!params.ellipsoidMask) return true

            // Convert chunk coordinates to world coordinates
            const worldX = chunkX * worldScale + chunkOriginOffset[0]
            const worldY = chunkY * worldScale + chunkOriginOffset[1]
            const worldZ = chunkZ * worldScale + chunkOriginOffset[2]

            const mask = params.ellipsoidMask
            const dx = (worldX - mask.center[0]) / mask.radiusX
            const dy = (worldY - mask.center[1]) / mask.radiusY
            const dz = (worldZ - mask.center[2]) / mask.radiusZ

            return (dx * dx + dy * dy + dz * dz) <= 1
          }

          if (params.action === 'clear') {
            console.log('Clearing region')
            const min = chunkRegion.min
            const max = chunkRegion.max
            for (let x = min[0]; x <= max[0]; x++) {
              for (let y = min[1]; y <= max[1]; y++) {
                for (let z = min[2]; z <= max[2]; z++) {
                  if (x >= 0 && x < chunk.size.x && y >= 0 && y < chunk.size.y && z >= 0 && z < chunk.size.z) {
                    if (isInsideEllipsoid(x, y, z)) {
                      chunk.setBlock(x, y, z, BlockType.Air)
                    }
                  }
                }
              }
            }
          } else {
            console.log('Generating terrain in chunk region:', chunkRegion)

            // Generate terrain with ellipsoid masking
            if (params.ellipsoidMask) {
              console.log('Using ellipsoid mask:', params.ellipsoidMask)

              // Generate terrain only inside the ellipsoid using the procedural generator
              const min = chunkRegion.min
              const max = chunkRegion.max

              console.log('Generating terrain in region:', { min, max })
              console.log('Region size:', (max[0] - min[0] + 1), 'x', (max[1] - min[1] + 1), 'x', (max[2] - min[2] + 1))

              // First, generate the full terrain region
              console.log('Calling generateRegion...')
              generateRegion(chunk, chunkRegion, terrainState)
              console.log('generateRegion completed')

              // Then, clear blocks outside the ellipsoid
              let clearedCount = 0
              for (let x = min[0]; x <= max[0]; x++) {
                for (let y = min[1]; y <= max[1]; y++) {
                  for (let z = min[2]; z <= max[2]; z++) {
                    if (!isInsideEllipsoid(x, y, z)) {
                      chunk.setBlock(x, y, z, BlockType.Air)
                      clearedCount++
                    }
                  }
                }
              }
              console.log('Cleared', clearedCount, 'blocks outside ellipsoid')
            } else {
              console.log('No ellipsoid mask, generating full region:', chunkRegion)
              console.log('Region size:',
                (chunkRegion.max[0] - chunkRegion.min[0] + 1), 'x',
                (chunkRegion.max[1] - chunkRegion.min[1] + 1), 'x',
                (chunkRegion.max[2] - chunkRegion.min[2] + 1))
              generateRegion(chunk, chunkRegion, terrainState)
              console.log('generateRegion completed')
            }

            // Count non-air blocks to verify terrain was generated
            let nonAirCount = 0
            for (let x = chunkRegion.min[0]; x <= chunkRegion.max[0]; x++) {
              for (let y = chunkRegion.min[1]; y <= chunkRegion.max[1]; y++) {
                for (let z = chunkRegion.min[2]; z <= chunkRegion.max[2]; z++) {
                  if (x >= 0 && x < chunk.size.x && y >= 0 && y < chunk.size.y && z >= 0 && z < chunk.size.z) {
                    const block = chunk.getBlock(x, y, z)
                    if (block !== BlockType.Air) {
                      nonAirCount++
                    }
                  }
                }
              }
            }
            console.log('Terrain generation complete - Non-air blocks in region:', nonAirCount)
          }

          console.log('Marking mesh dirty')
          renderer?.markMeshDirty()
          console.log('Done')
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
          <label>Plane Size: {$planeSize}</label>
          <input type="range" min="4" max="32" step="1" bind:value={$planeSize} />
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
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: clamp(8px, 2vw, 20px);
  }

  .canvas-shell {
    position: relative;
    width: 100%;
    height: 100%;
    max-width: calc(100vw - var(--sidebar-width) - 2 * clamp(8px, 2vw, 20px));
    aspect-ratio: 16 / 9;
    background: rgba(12, 22, 32, 0.6);
    border: 1px solid rgba(210, 223, 244, 0.1);
    border-radius: 18px;
    box-shadow: 0 24px 48px rgba(0, 0, 0, 0.45);
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
