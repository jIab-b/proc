<script lang="ts">
  import { onDestroy } from 'svelte'
  import {
    terrainProfile,
    terrainSeed,
    terrainAmplitude,
    terrainRoughness,
    terrainElevation
  } from '../stores'
  import { gpuHooks, API_BASE_URL, highlightSelection } from '../core'

  import { writable } from 'svelte/store'

  const status = writable<string>('Ready')
  let isGenerating = false

  const profileOptions = [
    { value: 'rolling_hills', label: 'Rolling Hills' },
    { value: 'mountain', label: 'Mountain Range' },
    { value: 'hybrid', label: 'Hybrid' }
  ]

  const unsubscribers: Array<() => void> = []

  onDestroy(() => {
    unsubscribers.forEach(fn => fn())
  })

  async function runGeneration(action: 'generate' | 'preview' | 'clear') {
    if (isGenerating) return

    let region: { min: [number, number, number]; max: [number, number, number] }

    // Check if there's a highlight selection (ellipsoid or plane)
    if ($highlightSelection && $highlightSelection.shape === 'ellipsoid') {
      // Use ellipsoid as floor for terrain generation - generate above ellipsoid surface
      // highlightSelection.center is in CHUNK coordinates, need to convert to WORLD coordinates
      const chunkCenter = $highlightSelection.center
      const worldCenter = $gpuHooks.chunkToWorld?.(chunkCenter) || chunkCenter
      const worldScale = $gpuHooks.getWorldScale?.() || 2

      const rx = $highlightSelection.radiusX ?? $highlightSelection.radius
      const ry = $highlightSelection.radiusY ?? $highlightSelection.radius
      const rz = $highlightSelection.radiusZ ?? $highlightSelection.radius

      // Convert radii from chunk coordinates to world coordinates
      const worldRx = rx * worldScale
      const worldRy = ry * worldScale
      const worldRz = rz * worldScale

      // Get camera position to determine max Y
      const cameraPos = $gpuHooks.getCameraPosition?.()
      const maxY = cameraPos ? cameraPos[1] + 32 : worldCenter[1] + 64

      region = {
        min: [
          Math.floor(worldCenter[0] - worldRx),
          Math.floor(worldCenter[1] - worldRy),  // Start from ellipsoid bottom for proper bounds
          Math.floor(worldCenter[2] - worldRz)
        ] as [number, number, number],
        max: [
          Math.floor(worldCenter[0] + worldRx),
          Math.floor(maxY),  // Generate upwards to max height
          Math.floor(worldCenter[2] + worldRz)
        ] as [number, number, number]
      }
    } else if ($highlightSelection && $highlightSelection.shape === 'plane') {
      // Use plane for terrain base generation - plane becomes the floor
      const chunkCenter = $highlightSelection.center
      const worldCenter = $gpuHooks.chunkToWorld?.(chunkCenter) || chunkCenter
      const worldScale = $gpuHooks.getWorldScale?.() || 2
      const sizeX = $highlightSelection.planeSizeX ?? 8
      const sizeZ = $highlightSelection.planeSizeZ ?? 8
      const worldSizeX = sizeX * worldScale
      const worldSizeZ = sizeZ * worldScale

      // Get camera position to determine max Y
      const cameraPos = $gpuHooks.getCameraPosition?.()
      const maxY = cameraPos ? cameraPos[1] + 32 : worldCenter[1] + 64

      region = {
        min: [
          Math.floor(worldCenter[0] - worldSizeX),
          Math.floor(worldCenter[1]),  // Base Y from plane (plane is the floor)
          Math.floor(worldCenter[2] - worldSizeZ)
        ] as [number, number, number],
        max: [
          Math.floor(worldCenter[0] + worldSizeX),
          Math.floor(maxY),  // Generate upwards from plane
          Math.floor(worldCenter[2] + worldSizeZ)
        ] as [number, number, number]
      }
    } else {
      // Default: use camera position
      const cameraPos = $gpuHooks.getCameraPosition?.()
      if (!cameraPos) {
        alert('Camera position not available.')
        return
      }

      const regionSize = 24
      region = {
        min: cameraPos.map(pos => Math.floor(pos - regionSize)) as [number, number, number],
        max: cameraPos.map(pos => Math.floor(pos + regionSize)) as [number, number, number]
      }
    }

    const requestBody: any = {
      action,
      region,
      profile: $terrainProfile,
      selectionType: $highlightSelection?.shape || 'default',
      params: {
        seed: $terrainSeed,
        amplitude: $terrainAmplitude,
        roughness: $terrainRoughness,
        elevation: $terrainElevation
      }
    }

    console.log('=== TERRAIN GENERATION REQUEST ===')
    console.log('Action:', action)
    console.log('Region (WORLD coords):', region)
    console.log('Profile:', $terrainProfile)
    console.log('Highlight selection:', $highlightSelection)
    if ($highlightSelection) {
      console.log('  Shape:', $highlightSelection.shape)
      console.log('  Center (CHUNK coords):', $highlightSelection.center)
      if ($gpuHooks.chunkToWorld) {
        const worldCenter = $gpuHooks.chunkToWorld($highlightSelection.center)
        console.log('  Center (WORLD coords):', worldCenter)
      }
      console.log('  WorldScale:', $gpuHooks.getWorldScale?.())
    }

    // Add ellipsoid mask if using ellipsoid selection
    if ($highlightSelection && $highlightSelection.shape === 'ellipsoid') {
      // Convert center from chunk to world coordinates
      const chunkCenter = $highlightSelection.center
      const worldCenter = $gpuHooks.chunkToWorld?.(chunkCenter) || chunkCenter
      const worldScale = $gpuHooks.getWorldScale?.() || 2

      const rx = $highlightSelection.radiusX ?? $highlightSelection.radius
      const ry = $highlightSelection.radiusY ?? $highlightSelection.radius
      const rz = $highlightSelection.radiusZ ?? $highlightSelection.radius

      requestBody.ellipsoidMask = {
        center: worldCenter,
        radiusX: rx * worldScale,
        radiusY: ry * worldScale,
        radiusZ: rz * worldScale
      }
    }

    try {
      isGenerating = true
      status.set('Working...')

      console.log('Terrain generation request:', requestBody)
      console.log('generateTerrain hook available:', !!$gpuHooks.generateTerrain)

      // Call backend API to validate and log the request
      const res = await fetch(`${API_BASE_URL}/api/terrain/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      })

      if (!res.ok) {
        const detail = await res.text()
        throw new Error(detail || `HTTP ${res.status}`)
      }

      console.log('Backend API responded OK, applying terrain generation...')

      // Apply terrain generation locally
      if (!$gpuHooks.generateTerrain) {
        throw new Error('generateTerrain hook not available')
      }
      $gpuHooks.generateTerrain(requestBody)

      console.log('Terrain generation complete')
      status.set('Done')
      setTimeout(() => status.set('Ready'), 1200)
    } catch (err) {
      console.error('Terrain generation error:', err)
      status.set('Error')
      alert(`Terrain generation failed: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      isGenerating = false
    }
  }

  function randomizeSeed() {
    terrainSeed.set(Math.floor(Math.random() * 1_000_000))
  }

  function randomizeAll() {
    // Randomize profile
    const profiles = ['rolling_hills', 'mountain', 'hybrid'] as const
    terrainProfile.set(profiles[Math.floor(Math.random() * profiles.length)]!)

    // Randomize seed
    terrainSeed.set(Math.floor(Math.random() * 1_000_000))

    // Randomize amplitude (4-32, favor middle range 8-20)
    const amplitude = 8 + Math.random() * 12
    terrainAmplitude.set(amplitude)

    // Randomize roughness (1.2-3.4, favor middle range 2.0-2.8)
    const roughness = 2.0 + Math.random() * 0.8
    terrainRoughness.set(roughness)

    // Randomize elevation (0.2-0.7, favor middle range 0.3-0.5)
    const elevation = 0.3 + Math.random() * 0.2
    terrainElevation.set(elevation)

    console.log('Randomized terrain params:', {
      profile: $terrainProfile,
      seed: $terrainSeed,
      amplitude: amplitude.toFixed(1),
      roughness: roughness.toFixed(2),
      elevation: elevation.toFixed(2)
    })
  }
</script>

<div class="panel">
  <header>
    <h3>Procedural Terrain</h3>
    <span class="status">{$status}</span>
  </header>

  <section>
    <label>Profile</label>
    <select bind:value={$terrainProfile}>
      {#each profileOptions as option}
        <option value={option.value}>{option.label}</option>
      {/each}
    </select>
  </section>

  <section class="grid">
    <label>Seed</label>
    <div class="seed-row">
      <input type="number" bind:value={$terrainSeed} min="0" max="1000000" step="1" />
      <button class="ghost" type="button" on:click={randomizeSeed}>🎲</button>
    </div>

    <label>Amplitude</label>
    <input type="range" min="4" max="32" step="0.5" bind:value={$terrainAmplitude} />
    <span class="value">{$terrainAmplitude.toFixed(1)}</span>

    <label>Roughness</label>
    <input type="range" min="1.2" max="3.4" step="0.05" bind:value={$terrainRoughness} />
    <span class="value">{$terrainRoughness.toFixed(2)}</span>

    <label>Elevation</label>
    <input type="range" min="0.2" max="0.7" step="0.01" bind:value={$terrainElevation} />
    <span class="value">{$terrainElevation.toFixed(2)}</span>
  </section>

  <section class="actions">
    <button type="button" class="primary" on:click={() => { randomizeAll(); runGeneration('generate'); }} disabled={isGenerating}>
      🎲 Generate Random Terrain
    </button>
    <button type="button" on:click={() => runGeneration('generate')} disabled={isGenerating}>
      Generate Around Player
    </button>
    <button type="button" on:click={() => runGeneration('preview')} disabled={isGenerating}>
      Preview Around Player
    </button>
    <button type="button" class="ghost" on:click={() => runGeneration('clear')} disabled={isGenerating}>
      Clear Region
    </button>
  </section>
</div>

<style>
  .panel {
    padding: clamp(12px, 2.5vw, 16px);
    display: flex;
    flex-direction: column;
    gap: clamp(12px, 3vw, 16px);
    border-top: 1px solid rgba(210, 223, 244, 0.1);
  }

  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  h3 {
    margin: 0;
    font-size: clamp(12px, 3vw, 14px);
    font-weight: 600;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    color: #a0b5d0;
  }

  .status {
    font-size: clamp(9px, 2vw, 11px);
    color: #7fa8f5;
  }

  section {
    display: flex;
    flex-direction: column;
    gap: clamp(6px, 1.5vw, 8px);
  }

  .grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: clamp(8px, 2vw, 10px) clamp(10px, 2.5vw, 12px);
  }

  label {
    font-size: clamp(9px, 2vw, 11px);
    color: #7f8ca5;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }

  select,
  input[type='number'],
  input[type='range'] {
    width: 100%;
    border-radius: 8px;
    border: 1px solid rgba(160, 180, 205, 0.3);
    background: rgba(34, 50, 68, 0.6);
    color: #e3ebf7;
    padding: clamp(6px, 1.5vw, 8px);
    font-size: clamp(11px, 2.5vw, 13px);
  }

  input[type='range'] {
    accent-color: #80a9ff;
  }

  .value {
    font-size: clamp(9px, 2vw, 11px);
    color: #a0b5d0;
    text-align: right;
  }

  .actions {
    display: flex;
    flex-direction: column;
    gap: clamp(8px, 2vw, 10px);
  }

  button {
    padding: clamp(8px, 2vw, 10px);
    border-radius: 8px;
    border: 1px solid rgba(120, 150, 190, 0.4);
    background: rgba(60, 80, 108, 0.6);
    color: #d6e3ff;
    font-size: clamp(11px, 2.5vw, 13px);
    cursor: pointer;
    transition: background 0.15s ease, transform 0.15s ease;
  }

  button:hover:not(:disabled) {
    background: rgba(80, 110, 148, 0.7);
    transform: translateY(-1px);
  }

  button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .primary {
    background: linear-gradient(135deg, rgba(99, 153, 235, 0.9), rgba(66, 120, 210, 0.9));
    border: none;
  }

  .ghost {
    background: rgba(28, 40, 58, 0.6);
    border: 1px solid rgba(120, 150, 190, 0.25);
  }

  .seed-row {
    display: grid;
    grid-template-columns: 1fr clamp(32px, 8vw, 40px);
    gap: clamp(6px, 1.5vw, 8px);
    align-items: center;
  }
</style>

