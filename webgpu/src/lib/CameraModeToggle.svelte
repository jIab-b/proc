<script lang="ts">
  import { cameraMode } from '../core'

  function toggleMode(mode: 'player' | 'overview') {
    cameraMode.set(mode)
  }
</script>

<div class="camera-mode-toggle">
  <label>Camera Mode:</label>
  <div class="toggle-group">
    <button
      class="mode-btn"
      class:active={$cameraMode === 'player'}
      on:click={() => toggleMode('player')}
      title="First-person camera (Click to lock pointer, WASD to move)"
    >
      🎮 Player
    </button>
    <button
      class="mode-btn"
      class:active={$cameraMode === 'overview'}
      on:click={() => toggleMode('overview')}
      title="Orbital camera (Drag to orbit, Scroll to zoom, WASD to pan)"
    >
      🌐 Overview
    </button>
  </div>
  <div class="hint">
    {#if $cameraMode === 'player'}
      <span>Click canvas to lock pointer • WASD to move • Mouse to look • Tab to switch</span>
    {:else}
      <span>Drag to orbit • Scroll to zoom • WASD to pan • Tab to switch</span>
    {/if}
  </div>
</div>

<style>
  .camera-mode-toggle {
    padding: 12px 16px;
    background: rgba(28, 40, 58, 0.8);
    border-bottom: 1px solid rgba(210, 223, 244, 0.1);
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  label {
    font-size: 11px;
    color: #7f8ca5;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }

  .toggle-group {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }

  .mode-btn {
    padding: 8px 12px;
    border-radius: 6px;
    border: 1px solid rgba(120, 150, 190, 0.3);
    background: rgba(60, 80, 108, 0.4);
    color: #d6e3ff;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
  }

  .mode-btn:hover {
    background: rgba(80, 110, 148, 0.6);
    transform: translateY(-1px);
  }

  .mode-btn.active {
    background: linear-gradient(135deg, rgba(99, 153, 235, 0.9), rgba(66, 120, 210, 0.9));
    border-color: rgba(99, 153, 235, 0.8);
    color: white;
    font-weight: 600;
  }

  .hint {
    font-size: 10px;
    color: #7fa8f5;
    text-align: center;
    line-height: 1.4;
  }
</style>
