<script lang="ts">
  import { onMount } from 'svelte'
  import { get } from 'svelte/store'
  import { initWebGPUEngine } from '../webgpuEngine'
  import {
    selectedBlockType,
    selectedCustomBlock,
    interactionMode,
    highlightShape,
    highlightRadius
  } from '../stores'
  import { BlockType } from '../chunks'
  import { API_BASE_URL } from '../blockUtils'

  let canvasEl: HTMLCanvasElement
  let overlayCanvasEl: HTMLCanvasElement
  let containerEl: HTMLDivElement
  let availableMaps: Array<{sequence: number, lastUpdated: string, captureId: string, blockCount: number, customBlockCount: number}> = []
  let loadMapCallback: ((mapData: any) => Promise<void>) | null = null
  let saveMapCallback: (() => Promise<void>) | null = null
  let newMapCallback: ((copyFromSequence?: number) => Promise<void>) | null = null
  let isInGame = false

  // Context menu state
  let showContextMenu = false
  let contextMenuX = 0
  let contextMenuY = 0
  let showLoadSubmenu = false
  let showNewMapSubmenu = false
  let loadMapFileCallback: ((mapData: any) => Promise<void>) | null = null
  let fileInputRef: HTMLInputElement | null = null

  async function fetchAvailableMaps() {
    try {
      const response = await fetch(`${API_BASE_URL}/api/maps`)
      if (!response.ok) {
        console.warn('Failed to fetch available maps')
        return
      }
      const data = await response.json()
      availableMaps = data.maps || []
    } catch (err) {
      console.error('Error fetching maps:', err)
    }
  }

  function handleContextMenu(event: MouseEvent) {
    const path = event.composedPath()
    const isCanvasContext = (canvasEl && path.includes(canvasEl)) || (overlayCanvasEl && path.includes(overlayCanvasEl))
    if (isCanvasContext && get(interactionMode) === 'highlight') {
      event.preventDefault()
      return
    }

    // Don't show context menu if we're in game (pointer locked)
    if (isInGame) {
      return
    }

    event.preventDefault()
    contextMenuX = event.clientX
    contextMenuY = event.clientY
    showContextMenu = true
    showLoadSubmenu = false
  }

  function closeContextMenu() {
    showContextMenu = false
    showLoadSubmenu = false
    showNewMapSubmenu = false
  }

  async function handleSaveMap() {
    closeContextMenu()
    if (saveMapCallback) {
      try {
        await saveMapCallback()
        console.log('Map saved successfully')
      } catch (err) {
        console.error('Failed to save map:', err)
        alert(`Failed to save map: ${err instanceof Error ? err.message : 'Unknown error'}`)
      }
    }
  }

  function toggleLoadSubmenu() {
    showLoadSubmenu = !showLoadSubmenu
    showNewMapSubmenu = false
    fetchAvailableMaps()
  }

  function toggleNewMapSubmenu() {
    showNewMapSubmenu = !showNewMapSubmenu
    showLoadSubmenu = false
    fetchAvailableMaps()
  }

  async function handleLoadMap(sequence: number) {
    closeContextMenu()
    if (!loadMapCallback) return

    try {
      await loadMapCallback(sequence)
      console.log(`Loaded map ${sequence}`)
    } catch (err) {
      console.error('Failed to load map:', err)
      alert(`Failed to load map: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  async function handleNewEmptyMap() {
    closeContextMenu()
    if (!newMapCallback) return

    try {
      await newMapCallback()
      console.log('Created new empty map')
    } catch (err) {
      console.error('Failed to create new map:', err)
      alert(`Failed to create new map: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  async function handleCopyFromMap(sequence: number) {
    closeContextMenu()
    if (!newMapCallback) return

    try {
      await newMapCallback(sequence)
      console.log(`Created new map copied from map ${sequence}`)
    } catch (err) {
      console.error('Failed to copy map:', err)
      alert(`Failed to copy map: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }

  function handleDownloadLog() {
    closeContextMenu()
    console.log('Download log feature pending')
  }

  async function handleLoadMapFromFile() {
    closeContextMenu()
    fileInputRef?.click()
  }

  async function handleFileInputChange(event: Event) {
    const input = event.target as HTMLInputElement
    const file = input.files?.[0]
    if (!file) return

    try {
      const contents = await file.text()
      if (!loadMapFileCallback) return

      await loadMapFileCallback(contents)
      console.log(`Loaded map from file: ${file.name}`)
      input.value = ''
    } catch (err) {
      console.error('Failed to load map from file:', err)
      alert(`Failed to load map: ${err instanceof Error ? err.message : 'Unknown error'}`)
      input.value = ''
    }
  }

  onMount(async () => {
    try {
      const engine = await initWebGPUEngine({
        canvas: canvasEl,
        overlayCanvas: overlayCanvasEl,
        onBlockSelect: (blockType) => {
          selectedBlockType.set(blockType)
        },
        getSelectedBlock: () => ({
          type: $selectedBlockType,
          custom: $selectedCustomBlock
        })
      })
      console.log('WebGPU engine initialized')

      // Store the callbacks
      if (engine && typeof engine.loadMap === 'function') {
        loadMapCallback = engine.loadMap
      }
      if (engine && typeof engine.saveMap === 'function') {
        saveMapCallback = engine.saveMap
      }
      if (engine && typeof engine.newMap === 'function') {
        newMapCallback = engine.newMap
      }
      if (engine && typeof engine.loadMapFromFile === 'function') {
        loadMapFileCallback = engine.loadMapFromFile
      }

      // Fetch available maps
      await fetchAvailableMaps()

      // Add click listener to close context menu
      document.addEventListener('click', closeContextMenu)

      // Track pointer lock state to know if we're in game
      const handlePointerLockChange = () => {
        isInGame = document.pointerLockElement === canvasEl
      }
      document.addEventListener('pointerlockchange', handlePointerLockChange)

      return () => {
        if (engine && typeof engine.destroy === 'function') {
          engine.destroy()
        }
        document.removeEventListener('click', closeContextMenu)
        document.removeEventListener('pointerlockchange', handlePointerLockChange)
      }
    } catch (err) {
      console.error('Failed to initialize WebGPU:', err)
      alert(`WebGPU initialization failed: ${err instanceof Error ? err.message : 'Unknown error'}\n\nMake sure your browser supports WebGPU.`)
    }
  })
</script>

<svelte:window on:contextmenu={handleContextMenu} />

<div class="canvas-container" bind:this={containerEl}>
  <div class="canvas-shell">
    <canvas bind:this={canvasEl}></canvas>
    <canvas class="capture-overlay" bind:this={overlayCanvasEl}></canvas>
  </div>
</div>

<input
  type="file"
  accept=".json"
  bind:this={fileInputRef}
  on:change={handleFileInputChange}
  style="display: none;"
/>

{#if showContextMenu}
  <div
    class="context-menu"
    style="left: {contextMenuX}px; top: {contextMenuY}px;"
    on:click|stopPropagation
  >
    <button class="context-menu-item" on:click={handleSaveMap}>
      Save Map
    </button>
    <button class="context-menu-item" on:click={toggleLoadSubmenu}>
      Load Map {showLoadSubmenu ? '▼' : '▶'}
    </button>
    {#if showLoadSubmenu}
      <div class="submenu">
        {#if availableMaps.length === 0}
          <div class="submenu-item disabled">No maps available</div>
        {:else}
          {#each availableMaps as map}
            <button class="submenu-item" on:click={() => handleLoadMap(map.sequence)}>
              {map.isTrained ? '🤖 ' : ''}Map {map.sequence} ({map.blockCount} blocks)
            </button>
          {/each}
        {/if}
      </div>
    {/if}
    <button class="context-menu-item" on:click={toggleNewMapSubmenu}>
      New Map {showNewMapSubmenu ? '▼' : '▶'}
    </button>
    {#if showNewMapSubmenu}
      <div class="submenu">
        <button class="submenu-item" on:click={handleNewEmptyMap}>
          Create Empty Map
        </button>
        {#if availableMaps.length > 0}
          <div class="submenu-divider"></div>
          <div class="submenu-label">Copy From:</div>
          {#each availableMaps as map}
            <button class="submenu-item" on:click={() => handleCopyFromMap(map.sequence)}>
              {map.isTrained ? '🤖 ' : ''}Map {map.sequence} ({map.blockCount} blocks)
            </button>
          {/each}
        {/if}
      </div>
    {/if}
    <button class="context-menu-item" on:click={handleDownloadLog}>
      Download Log
    </button>
    <button class="context-menu-item" on:click={handleLoadMapFromFile}>
      Load Map From File
    </button>
    <div class="context-divider"></div>
    <div class="context-section">
      <label for="interaction-mode">Interaction Mode</label>
      <select id="interaction-mode" bind:value={$interactionMode}>
        <option value="block">Block Placement</option>
        <option value="highlight">Highlight Select</option>
      </select>
    </div>
    {#if $interactionMode === 'highlight'}
      <div class="context-section">
        <label for="highlight-shape">Highlight Shape</label>
        <select id="highlight-shape" bind:value={$highlightShape}>
          <option value="cube">Cube</option>
          <option value="sphere">Sphere</option>
        </select>
      </div>
      <div class="context-section">
        <label for="highlight-radius">Highlight Radius: {$highlightRadius}</label>
        <input
          id="highlight-radius"
          type="range"
          min="1"
          max="16"
          step="1"
          bind:value={$highlightRadius}
        />
      </div>
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
    padding: 20px;
  }

  .canvas-shell {
    position: relative;
    width: 100%;
    height: 100%;
    max-width: 1120px;
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
    display: block;
  }

  .context-menu {
    position: fixed;
    background: rgba(20, 30, 45, 0.98);
    border: 1px solid rgba(210, 223, 244, 0.3);
    border-radius: 8px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
    padding: 4px;
    min-width: 180px;
    z-index: 1000;
    backdrop-filter: blur(8px);
  }

  .context-menu-item {
    display: block;
    width: 100%;
    background: transparent;
    border: none;
    color: #e3ebf7;
    padding: 8px 12px;
    text-align: left;
    cursor: pointer;
    font-size: 13px;
    border-radius: 4px;
    transition: background 0.15s;
  }

  .context-menu-item:hover {
    background: rgba(48, 66, 88, 0.6);
  }

  .submenu {
    margin-left: 8px;
    margin-top: 4px;
    padding-left: 8px;
    border-left: 2px solid rgba(210, 223, 244, 0.2);
  }

  .submenu-item {
    display: block;
    width: 100%;
    background: transparent;
    border: none;
    color: #c8d5e8;
    padding: 6px 12px;
    text-align: left;
    cursor: pointer;
    font-size: 12px;
    border-radius: 4px;
    transition: background 0.15s;
  }

  .context-divider {
    height: 1px;
    background: rgba(210, 223, 244, 0.2);
    margin: 8px 0;
  }

  .context-section {
    padding: 4px 8px 6px 8px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .context-section label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #8fa0b8;
  }

  .context-section select,
  .context-section input[type='range'] {
    width: 100%;
    background: rgba(34, 50, 68, 0.6);
    border: 1px solid rgba(190, 210, 230, 0.25);
    border-radius: 4px;
    padding: 6px;
    color: #e3ebf7;
    font-size: 12px;
  }

  .context-section input[type='range'] {
    padding: 0;
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
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
</style>
