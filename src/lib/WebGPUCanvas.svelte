<script lang="ts">
  import { onMount } from 'svelte'
  import { initWebGPUEngine } from '../webgpuEngine'
  import { selectedBlockType, selectedCustomBlock } from '../stores'
  import { BlockType } from '../chunks'

  let canvasEl: HTMLCanvasElement

  onMount(async () => {
    try {
      await initWebGPUEngine({
        canvas: canvasEl,
        onBlockSelect: (blockType) => {
          selectedBlockType.set(blockType)
        },
        getSelectedBlock: () => ({
          type: $selectedBlockType,
          custom: $selectedCustomBlock
        })
      })
      console.log('WebGPU engine initialized')
    } catch (err) {
      console.error('Failed to initialize WebGPU:', err)
      alert(`WebGPU initialization failed: ${err instanceof Error ? err.message : 'Unknown error'}\n\nMake sure your browser supports WebGPU.`)
    }
  })
</script>

<div class="canvas-container">
  <div class="canvas-shell">
    <canvas bind:this={canvasEl}></canvas>
  </div>

  <div class="log-ui">
    <button on:click={() => console.log('Download log feature pending')}>
      Download Log
    </button>
  </div>
</div>

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

  .log-ui {
    position: fixed;
    top: 16px;
    right: 16px;
    display: flex;
    gap: 8px;
    z-index: 10;
  }

  .log-ui button {
    background: rgba(34, 50, 68, 0.88);
    border: 1px solid rgba(190, 210, 230, 0.35);
    color: inherit;
    border-radius: 6px;
    padding: 6px 10px;
    cursor: pointer;
    font-size: 12px;
  }

  .log-ui button:hover {
    background: rgba(48, 66, 88, 0.92);
  }
</style>
