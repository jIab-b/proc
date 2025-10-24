<script lang="ts">
  import { selectedFace } from '../stores'
  import { blockFaceOrder } from '../blockUtils'
  import type { BlockFaceKey } from '../chunks'

  const labels = ['Top (+Y)', 'Bottom (-Y)', 'Front (+Z)', 'Back (-Z)', 'Right (+X)', 'Left (-X)']

  function chooseFace(face: BlockFaceKey) {
    selectedFace.set(face)
  }
</script>

<div class="face-viewer">
  <div class="face-grid">
    {#each labels as label, index}
      <div
        class="face-box"
        class:selected={$selectedFace === blockFaceOrder[index]}
        on:click={() => { const face = blockFaceOrder[index]; if (face) chooseFace(face) }}
      >
        <label>{label}</label>
        <slot name={`face-${index}`} />
      </div>
    {/each}
  </div>
</div>

<style>
  .face-viewer {
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding-top: 12px;
  }

  .face-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
  }

  .face-box {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    padding: 8px;
    background: rgba(34, 50, 68, 0.5);
    border: 2px solid rgba(190, 210, 230, 0.2);
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .face-box:hover {
    background: rgba(48, 66, 88, 0.6);
    border-color: rgba(190, 210, 230, 0.4);
  }

  .face-box.selected {
    background: rgba(68, 96, 128, 0.7);
    border-color: rgba(120, 180, 240, 0.8);
  }

  label {
    font-size: 10px;
    color: #8fa0b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
</style>

