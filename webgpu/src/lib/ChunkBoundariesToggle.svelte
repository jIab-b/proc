<script lang="ts">
  import { showChunkBoundaries } from '../core'

  function toggleBoundaries() {
    showChunkBoundaries.update(v => !v)
  }
</script>

<div class="chunk-boundaries-toggle">
  <button
    class="toggle-btn"
    class:active={$showChunkBoundaries}
    on:click={toggleBoundaries}
    title={$showChunkBoundaries ? "Hide chunk boundaries" : "Show chunk boundaries"}
  >
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
      <!-- Grid icon -->
      <path
        d="M2 2 L18 2 L18 18 L2 18 Z M2 10 L18 10 M10 2 L10 18"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-dasharray={$showChunkBoundaries ? "none" : "2,2"}
      />
    </svg>
    <span>Grid</span>
  </button>
</div>

<style>
  .chunk-boundaries-toggle {
    padding: 8px 12px;
    border-bottom: 1px solid rgba(210, 223, 244, 0.15);
  }

  .toggle-btn {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: rgba(34, 50, 68, 0.5);
    border: 2px solid rgba(190, 210, 230, 0.2);
    border-radius: 8px;
    color: #c8d5e8;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
  }

  .toggle-btn:hover {
    background: rgba(48, 66, 88, 0.6);
    border-color: rgba(190, 210, 230, 0.4);
  }

  .toggle-btn.active {
    background: rgba(68, 96, 128, 0.7);
    border-color: rgba(120, 180, 240, 0.8);
    color: #d2dff4;
  }

  .toggle-btn svg {
    flex-shrink: 0;
  }

  .toggle-btn span {
    flex: 1;
    text-align: left;
  }
</style>
