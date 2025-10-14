<script lang="ts">
  import { onMount } from 'svelte'
  import { selectedBlockType, selectedCustomBlock, customBlocks, selectedFace } from '../stores'
  import { availableBlocks, blockPalette, drawIsometricBlock, drawIsometricBlockWithTexture, BLOCKS_ENDPOINT, blockFaceOrder } from '../blockUtils'
  import { BlockType } from '../chunks'
  import type { CustomBlock } from '../stores'

  let blockGridEl: HTMLDivElement

  function selectDefaultBlock(type: BlockType) {
    selectedBlockType.set(type)
    selectedCustomBlock.set(null)
    selectedFace.set(null)
  }

  function selectCustomBlock(block: CustomBlock) {
    selectedCustomBlock.set(block)
    selectedBlockType.set(BlockType.Plank)
  }

  async function deleteBlock(block: CustomBlock, event: MouseEvent) {
    event.stopPropagation()

    const confirmDelete = confirm(`Delete "${block.name}"? This will also remove the texture files.`)
    if (!confirmDelete) return

    try {
      if (block.remoteId) {
        const response = await fetch(`${BLOCKS_ENDPOINT}/${block.remoteId}`, { method: 'DELETE' })
        if (!response.ok) {
          console.warn(`Failed to delete block on server: ${response.status}`)
        } else {
          console.log(`Deleted block ${block.remoteId} from server`)
        }
      }

      // Clean up face bitmaps
      if (block.faceBitmaps) {
        blockFaceOrder.forEach(face => {
          const bitmap = block.faceBitmaps?.[face]
          if (bitmap && bitmap.close) bitmap.close()
        })
      }

      // Remove from store
      customBlocks.update(blocks => blocks.filter(b => b.id !== block.id))

      // Clear selection if this block was selected
      if ($selectedCustomBlock?.id === block.id) {
        selectedCustomBlock.set(null)
        selectedBlockType.set(BlockType.Plank)
        selectedFace.set(null)
      }

      console.log(`Deleted custom block: ${block.name}`)
    } catch (err) {
      console.error('Error deleting custom block:', err)
      alert('Failed to delete texture. Please try again.')
    }
  }

  function createNewBlock() {
    const blockName = prompt('Enter block name:')
    if (!blockName || !blockName.trim()) return

    const nextId = ($customBlocks.reduce((max, block) => Math.max(max, block.id), 999) + 1)
    const newBlock: CustomBlock = {
      id: nextId,
      name: blockName.trim(),
      colors: { top: [0.5, 0.5, 0.5], bottom: [0.4, 0.4, 0.4], side: [0.45, 0.45, 0.45] },
      faceBitmaps: {},
      faceTiles: {}
    }

    customBlocks.update(blocks => [...blocks, newBlock])
    console.log('Created custom block:', blockName)
  }

  function renderBlockPreview(canvas: HTMLCanvasElement, block: { type?: BlockType, custom?: CustomBlock }) {
    if (block.custom) {
      if (Object.values(block.custom.faceBitmaps || {}).some(Boolean)) {
        drawIsometricBlockWithTexture(canvas, block.custom)
      } else {
        drawIsometricBlock(canvas, block.custom.colors)
      }
    } else if (block.type !== undefined) {
      const palette = blockPalette[block.type]
      if (palette) drawIsometricBlock(canvas, palette)
    }
  }

  onMount(() => {
    // Render all block previews
    const canvases = blockGridEl.querySelectorAll('canvas')
    canvases.forEach((canvas, idx) => {
      if (idx < availableBlocks.length) {
        const block = availableBlocks[idx]
        if (block) renderBlockPreview(canvas as HTMLCanvasElement, { type: block.type })
      } else {
        const customIdx = idx - availableBlocks.length
        const customBlock = $customBlocks[customIdx]
        if (customBlock) renderBlockPreview(canvas as HTMLCanvasElement, { custom: customBlock })
      }
    })
  })

  // Re-render when custom blocks change
  $: if (blockGridEl && $customBlocks) {
    setTimeout(() => {
      const canvases = blockGridEl.querySelectorAll('canvas')
      canvases.forEach((canvas, idx) => {
        if (idx >= availableBlocks.length) {
          const customIdx = idx - availableBlocks.length
          const customBlock = $customBlocks[customIdx]
          if (customBlock) renderBlockPreview(canvas as HTMLCanvasElement, { custom: customBlock })
        }
      })
    }, 0)
  }
</script>

<div class="block-selection">
  <h3>Block Selection</h3>

  <div class="block-grid" bind:this={blockGridEl}>
    <!-- Default blocks -->
    {#each availableBlocks as block}
      <div
        class="block-item"
        class:selected={$selectedBlockType === block.type && !$selectedCustomBlock}
        on:click={() => selectDefaultBlock(block.type)}
        on:mousedown={(e) => e.stopPropagation()}
      >
        <canvas class="block-preview-3d" width="48" height="48"></canvas>
        <span>{block.name}</span>
      </div>
    {/each}

    <!-- Custom blocks -->
    {#each $customBlocks as block (block.id)}
      <div
        class="block-item"
        class:selected={$selectedCustomBlock?.id === block.id}
        on:click={(e) => {
          if (e.shiftKey) {
            deleteBlock(block, e)
          } else {
            selectCustomBlock(block)
          }
        }}
        on:mousedown={(e) => e.stopPropagation()}
      >
        <canvas class="block-preview-3d" width="48" height="48"></canvas>
        <span>{block.name}</span>
      </div>
    {/each}

    <!-- Add new block button -->
    <div
      class="block-item add-block-btn"
      on:click={createNewBlock}
      on:mousedown={(e) => e.stopPropagation()}
    >
      <div class="icon">+</div>
      <span>Add Block</span>
    </div>
  </div>

  <div class="helper-text">
    Click a block face to select it. Shift+Click custom tiles to delete them.
  </div>
</div>

<style>
  .block-selection {
    flex: 1;
    padding: 16px;
    overflow-y: auto;
    border-bottom: 1px solid rgba(210, 223, 244, 0.15);
  }

  h3 {
    margin: 0 0 12px 0;
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    color: #a0b5d0;
  }

  .block-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
  }

  .block-item {
    padding: 8px;
    background: rgba(34, 50, 68, 0.6);
    border: 1px solid rgba(190, 210, 230, 0.25);
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.15s;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
  }

  .block-preview-3d {
    width: 48px;
    height: 48px;
    flex-shrink: 0;
  }

  .block-item span {
    font-size: 11px;
    text-align: center;
    word-break: break-word;
    line-height: 1.2;
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
  }

  .add-block-btn {
    background: rgba(70, 120, 180, 0.4) !important;
    border: 1px dashed rgba(100, 150, 220, 0.6) !important;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 80px;
  }

  .add-block-btn:hover {
    background: rgba(80, 140, 200, 0.5) !important;
  }

  .add-block-btn .icon {
    font-size: 24px;
    margin-bottom: 4px;
  }

  .block-item:hover {
    background: rgba(48, 66, 88, 0.8);
    border-color: rgba(190, 210, 230, 0.45);
  }

  .block-item.selected {
    background: rgba(70, 120, 180, 0.4);
    border-color: rgba(100, 150, 220, 0.6);
  }

  .helper-text {
    font-size: 10px;
    color: #8090a8;
    margin-top: 8px;
    text-align: center;
    font-style: italic;
  }
</style>
