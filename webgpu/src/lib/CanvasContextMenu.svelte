<script lang="ts">
  import { createEventDispatcher } from 'svelte'
  import {
    interactionMode,
    highlightShape,
    highlightRadius,
    ellipsoidRadiusX,
    ellipsoidRadiusY,
    ellipsoidRadiusZ,
    planeSizeX,
    planeSizeZ
  } from '../core'

  const dispatch = createEventDispatcher<{
    save: void
    load: { sequence: number }
    new: { copyFrom?: number }
    loadFile: void
    refreshMaps: void
  }>()

  export let visible = false
  export let x = 0
  export let y = 0
  export let availableMaps: Array<{ sequence: number; blockCount: number }> = []

  let loadOpen = false
  let newOpen = false

  function toggleLoad() {
    loadOpen = !loadOpen
    if (loadOpen) {
      newOpen = false
      dispatch('refreshMaps')
    }
  }

  function toggleNew() {
    newOpen = !newOpen
    if (newOpen) {
      loadOpen = false
      dispatch('refreshMaps')
    }
  }

  $: if (!visible) {
    loadOpen = false
    newOpen = false
  }
</script>

{#if visible}
  <div class="context-menu" style={`left:${x}px;top:${y}px;`} on:click|stopPropagation>
    <button class="context-menu-item" on:click={() => dispatch('save')}>Save Map</button>

    <button class="context-menu-item" on:click={toggleLoad}>
      Load Map {loadOpen ? '▼' : '▶'}
    </button>
    {#if loadOpen}
      <div class="submenu">
        {#if availableMaps.length === 0}
          <div class="submenu-item disabled">No maps available</div>
        {:else}
          {#each availableMaps as map}
            <button class="submenu-item" on:click={() => dispatch('load', { sequence: map.sequence })}>
              Map {map.sequence} ({map.blockCount} blocks)
            </button>
          {/each}
        {/if}
      </div>
    {/if}

    <button class="context-menu-item" on:click={toggleNew}>
      New Map {newOpen ? '▼' : '▶'}
    </button>
    {#if newOpen}
      <div class="submenu">
        <button class="submenu-item" on:click={() => dispatch('new')}>Create Empty Map</button>
        {#if availableMaps.length > 0}
          <div class="submenu-divider"></div>
          <div class="submenu-label">Copy From:</div>
          {#each availableMaps as map}
            <button class="submenu-item" on:click={() => dispatch('new', { copyFrom: map.sequence })}>
              Map {map.sequence} ({map.blockCount} blocks)
            </button>
          {/each}
        {/if}
      </div>
    {/if}

    <button class="context-menu-item" on:click={() => dispatch('loadFile')}>Load From File</button>

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
  .context-menu {
    position: fixed;
    background: rgba(28, 40, 58, 0.95);
    border: 1px solid rgba(78, 110, 148, 0.6);
    border-radius: 8px;
    box-shadow: 0 14px 40px rgba(0, 0, 0, 0.45);
    padding: 8px;
    min-width: 220px;
    z-index: 1000;
    color: #d5e0f5;
    backdrop-filter: blur(6px);
  }

  .context-menu-item {
    display: block;
    width: 100%;
    background: transparent;
    border: none;
    color: inherit;
    text-align: left;
    padding: 6px 10px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.9rem;
  }

  .context-menu-item:hover {
    background: rgba(255, 255, 255, 0.08);
  }

  .submenu {
    margin: 4px 0 8px;
    padding-left: 8px;
    border-left: 1px solid rgba(110, 150, 200, 0.3);
  }

  .submenu-item {
    display: block;
    width: 100%;
    background: none;
    border: none;
    color: inherit;
    text-align: left;
    padding: 5px 8px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.85rem;
  }

  .submenu-item:hover {
    background: rgba(255, 255, 255, 0.06);
  }

  .submenu-item.disabled {
    opacity: 0.6;
    cursor: default;
  }

  .submenu-divider {
    margin: 6px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  }

  .submenu-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
    color: rgba(213, 224, 245, 0.65);
  }

  .context-divider {
    margin: 8px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  }

  .context-section {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin: 6px 0;
  }

  label {
    font-size: 0.8rem;
    color: rgba(213, 224, 245, 0.8);
  }

  select,
  input[type='range'] {
    width: 100%;
  }
</style>
