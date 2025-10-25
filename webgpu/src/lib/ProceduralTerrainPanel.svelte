<script lang="ts">
  import { onDestroy } from 'svelte'
  import {
    terrainProfile,
    terrainSeed,
    terrainAmplitude,
    terrainRoughness,
    terrainElevation
  } from '../stores'
  import { gpuHooks, API_BASE_URL } from '../core'

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

    // Get camera position
    const cameraPos = $gpuHooks.getCameraPosition?.()
    if (!cameraPos) {
      alert('Camera position not available.')
      return
    }

    // Define a large region around the player (32x32x32 blocks centered on player)
    const regionSize = 16
    const region = {
      min: cameraPos.map(pos => Math.floor(pos - regionSize)) as [number, number, number],
      max: cameraPos.map(pos => Math.floor(pos + regionSize)) as [number, number, number]
    }

    const requestBody = {
      action,
      region,
      profile: $terrainProfile,
      params: {
        seed: $terrainSeed,
        amplitude: $terrainAmplitude,
        roughness: $terrainRoughness,
        elevation: $terrainElevation
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
    <button type="button" class="primary" on:click={() => runGeneration('generate')} disabled={isGenerating}>
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
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    border-top: 1px solid rgba(210, 223, 244, 0.1);
  }

  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    color: #a0b5d0;
  }

  .status {
    font-size: 11px;
    color: #7fa8f5;
  }

  section {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 10px 12px;
  }

  label {
    font-size: 11px;
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
    padding: 8px;
    font-size: 13px;
  }

  input[type='range'] {
    accent-color: #80a9ff;
  }

  .value {
    font-size: 11px;
    color: #a0b5d0;
    text-align: right;
  }

  .actions {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  button {
    padding: 10px;
    border-radius: 8px;
    border: 1px solid rgba(120, 150, 190, 0.4);
    background: rgba(60, 80, 108, 0.6);
    color: #d6e3ff;
    font-size: 13px;
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
    grid-template-columns: 1fr 40px;
    gap: 8px;
    align-items: center;
  }
</style>

