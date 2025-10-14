<script lang="ts">
  import { onMount } from 'svelte'
  import { selectedFace, selectedBlockType, selectedCustomBlock } from '../stores'
  import { blockFaceOrder, blockPalette } from '../blockUtils'
  import type { BlockFaceKey } from '../chunks'

  const faceLabels = ['Top (+Y)', 'Bottom (-Y)', 'Front (+Z)', 'Back (-Z)', 'Right (+X)', 'Left (-X)']
  let faceCanvases: (HTMLCanvasElement | null)[] = []

  function handleFaceClick(face: BlockFaceKey) {
    selectedFace.set(face)
  }

  function updateFacePreviews() {
    if ($selectedCustomBlock) {
      updateFacePreviewsWithTexture($selectedCustomBlock)
    } else {
      const palette = blockPalette[$selectedBlockType]
      if (palette) {
        const faceColors = [palette.top, palette.bottom, palette.side, palette.side, palette.side, palette.side]
        faceCanvases.forEach((canvas, index) => {
          if (!canvas) return
          const ctx = canvas.getContext('2d')
          if (!ctx) return
          const color = faceColors[index]
          if (!color) return
          const rgb = `rgb(${Math.floor((color[0] ?? 0) * 255)}, ${Math.floor((color[1] ?? 0) * 255)}, ${Math.floor((color[2] ?? 0) * 255)})`
          ctx.fillStyle = rgb
          ctx.fillRect(0, 0, canvas.width, canvas.height)
        })
      }
    }
  }

  function updateFacePreviewsWithTexture(customBlock: typeof $selectedCustomBlock) {
    if (!customBlock) return

    faceCanvases.forEach((canvas, index) => {
      if (!canvas) return
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      const face = blockFaceOrder[index]
      if (!face) return
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      const bitmap = customBlock.faceBitmaps?.[face]
      if (bitmap) {
        ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height)
        return
      }

      // Fallback to color
      const palette = customBlock.colors
      const colorKey: 'top' | 'bottom' | 'side' = face === 'top' ? 'top' : face === 'bottom' ? 'bottom' : 'side'
      const color = palette[colorKey]
      if (color) {
        const rgb = `rgb(${Math.floor((color[0] ?? 0) * 255)}, ${Math.floor((color[1] ?? 0) * 255)}, ${Math.floor((color[2] ?? 0) * 255)})`
        ctx.fillStyle = rgb
        ctx.fillRect(0, 0, canvas.width, canvas.height)
      }
    })
  }

  onMount(() => {
    updateFacePreviews()
  })

  $: if (faceCanvases.length > 0) {
    updateFacePreviews()
  }

  $: $selectedBlockType, $selectedCustomBlock, updateFacePreviews()
</script>

<div class="face-viewer">
  <h3>Block Faces</h3>

  <div class="face-grid">
    {#each faceLabels as label, index}
      <div
        class="face-box"
        class:selected={$selectedFace === blockFaceOrder[index]}
        on:click={() => {
          const face = blockFaceOrder[index];
          if (face) handleFaceClick(face);
        }}
        on:mousedown={(e) => e.stopPropagation()}
      >
        <label>{label}</label>
        <canvas
          class="face-preview"
          width="48"
          height="48"
          bind:this={faceCanvases[index]}
        ></canvas>
      </div>
    {/each}
  </div>
</div>

<style>
  .face-viewer {
    padding: 16px;
    background: rgba(15, 25, 38, 0.95);
    max-height: 50vh;
    overflow-y: auto;
  }

  h3 {
    margin: 0 0 12px 0;
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    color: #a0b5d0;
  }

  .face-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    grid-template-rows: repeat(4, 1fr);
    gap: 6px;
    margin-bottom: 16px;
    aspect-ratio: 1;
  }

  .face-box {
    background: rgba(34, 50, 68, 0.6);
    border: 1px solid rgba(190, 210, 230, 0.25);
    border-radius: 6px;
    padding: 4px;
    text-align: center;
    display: flex;
    flex-direction: column;
    cursor: pointer;
    transition: all 0.15s;
  }

  .face-box:hover {
    background: rgba(48, 66, 88, 0.8);
    border-color: rgba(190, 210, 230, 0.45);
  }

  .face-box label {
    display: block;
    font-size: 9px;
    color: #8090a8;
    margin-bottom: 3px;
    text-transform: uppercase;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .face-preview {
    width: 100%;
    flex: 1;
    background: #10161d;
    border-radius: 3px;
    border: 1px solid rgba(190, 210, 230, 0.2);
    image-rendering: pixelated;
  }

  .face-box.selected {
    border-color: rgba(120, 170, 230, 0.7);
    background: rgba(70, 120, 180, 0.2);
    box-shadow: 0 0 0 2px rgba(120, 170, 230, 0.35);
  }

  .face-box.selected .face-preview {
    border-color: rgba(140, 190, 255, 0.9);
    box-shadow: 0 0 0 1px rgba(140, 190, 255, 0.6);
  }

  /* Cross layout for unfolded cube */
  .face-box:nth-child(1) { grid-column: 2; grid-row: 1; }
  .face-box:nth-child(2) { grid-column: 2; grid-row: 3; }
  .face-box:nth-child(3) { grid-column: 2; grid-row: 2; }
  .face-box:nth-child(4) { grid-column: 3; grid-row: 2; }
  .face-box:nth-child(5) { grid-column: 1; grid-row: 2; }
  .face-box:nth-child(6) { grid-column: 2; grid-row: 4; }
</style>
