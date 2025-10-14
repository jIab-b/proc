<script lang="ts">
  import { texturePrompt, selectedFace, canGenerate, selectedCustomBlock, customBlocks, nextCustomBlockId, gpuHooks, nextTextureLayer } from '../stores'
  import { GENERATE_TILE_ENDPOINT, blockFaceOrder } from '../blockUtils'
  import { BlockType } from '../chunks'
  import type { CustomBlock, FaceTileInfo } from '../stores'
  import type { BlockFaceKey } from '../chunks'

  let isGenerating = false
  let buttonText = 'Generate Face Tile'

  async function handleGenerate() {
    const prompt = $texturePrompt.trim()
    if (!prompt || !$selectedFace) {
      alert('Select a block face before generating a tile.')
      return
    }

    isGenerating = true
    buttonText = 'Generating...'
    let succeeded = false

    try {
      const isExistingBlock = Boolean($selectedCustomBlock)
      const defaultName = (prompt.split(',')[0] ?? '').trim() || 'Custom Block'
      const targetName = isExistingBlock ? $selectedCustomBlock!.name : defaultName

      const response = await fetch(GENERATE_TILE_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          face: $selectedFace,
          block_id: $selectedCustomBlock?.remoteId ?? null,
          block_name: targetName
        })
      })

      if (!response.ok) {
        const errorPayload = await response.json().catch(async () => ({ detail: await response.text() }))
        const message = (errorPayload && typeof errorPayload.detail === 'string') ? errorPayload.detail : `HTTP ${response.status}`
        throw new Error(message)
      }

      const data = await response.json()
      const tile = data.tile as FaceTileInfo & { face: BlockFaceKey }
      const blockData = data.block as { id: number; name?: string }
      const imageBase64 = data.image_base64 as string

      const binary = Uint8Array.from(atob(imageBase64), char => char.charCodeAt(0))
      const blob = new Blob([binary], { type: 'image/png' })
      const bitmap = await createImageBitmap(blob, {
        premultiplyAlpha: 'none',
        colorSpaceConversion: 'none'
      })

      let targetBlock = $selectedCustomBlock
      if (!targetBlock) {
        const newBlock: CustomBlock = {
          id: $nextCustomBlockId,
          name: blockData?.name || targetName,
          colors: { top: [0.5, 0.5, 0.5], bottom: [0.4, 0.4, 0.4], side: [0.45, 0.45, 0.45] },
          faceBitmaps: {},
          faceTiles: {},
          remoteId: blockData?.id
        }
        nextCustomBlockId.update(n => n + 1)
        customBlocks.update(blocks => [...blocks, newBlock])
        selectedCustomBlock.set(newBlock)
        targetBlock = newBlock
        console.log(`Created new custom block: ${newBlock.name} (remote #${newBlock.remoteId ?? 'local'})`)
      } else {
        if (blockData?.id && targetBlock) {
          customBlocks.update(blocks => {
            const block = blocks.find(b => b.id === targetBlock.id)
            if (block) {
              block.remoteId = blockData.id
              if (blockData.name) block.name = blockData.name
            }
            return blocks
          })
        }
      }

      const faceKey = tile.face
      const previousBitmap = targetBlock.faceBitmaps?.[faceKey]
      if (previousBitmap && typeof previousBitmap.close === 'function') {
        previousBitmap.close()
      }

      // Update the block with new face data
      customBlocks.update(blocks => {
        const block = blocks.find(b => b.id === targetBlock?.id)
        if (block) {
          if (!block.faceBitmaps) block.faceBitmaps = {}
          if (!block.faceTiles) block.faceTiles = {}

          block.faceBitmaps[faceKey] = bitmap
          block.faceTiles[faceKey] = {
            sequence: tile.sequence,
            path: tile.path,
            url: tile.url.startsWith('http') ? tile.url : `http://localhost:8000${tile.url}`,
            prompt
          }
        }
        return blocks
      })

      // Upload to GPU if hooks available
      if ($gpuHooks.uploadFaceBitmapsToGPU && targetBlock) {
        const gpuBitmaps: Record<BlockFaceKey, ImageBitmap> = {} as Record<BlockFaceKey, ImageBitmap>
        blockFaceOrder.forEach(face => {
          const bmp = targetBlock.faceBitmaps?.[face]
          if (bmp) gpuBitmaps[face] = bmp
        })
        $gpuHooks.uploadFaceBitmapsToGPU(gpuBitmaps, targetBlock)
      } else {
        console.warn('GPU upload hook missing; tile will use fallback until upload available')
      }

      selectedFace.set(faceKey)
      succeeded = true
      console.log(`Generated ${faceKey} tile for block ${targetBlock.name} (sequence ${tile.sequence})`)
    } catch (err) {
      console.error('Tile generation failed:', err)
      const message = err instanceof Error ? err.message : 'Unknown error'
      alert(`Tile generation failed: ${message}\n\nVerify the backend is running (python3 aio.py).`)
    } finally {
      isGenerating = false
      buttonText = succeeded ? 'Generate Face Tile' : 'Failed - Retry'
    }
  }
</script>

<div class="texture-gen">
  <input
    type="text"
    bind:value={$texturePrompt}
    placeholder='Describe the selected face (e.g. "mossy stone with vines")'
    on:mousedown={(e) => e.stopPropagation()}
    on:click={(e) => e.stopPropagation()}
  />
  <button
    on:click={handleGenerate}
    on:mousedown={(e) => e.stopPropagation()}
    disabled={!$canGenerate}
  >
    {buttonText}
  </button>
</div>

<style>
  .texture-gen {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid rgba(210, 223, 244, 0.15);
  }

  input {
    width: 100%;
    padding: 8px 10px;
    background: rgba(34, 50, 68, 0.6);
    border: 1px solid rgba(190, 210, 230, 0.25);
    border-radius: 6px;
    color: inherit;
    font-size: 12px;
    margin-bottom: 8px;
  }

  button {
    width: 100%;
    padding: 8px 10px;
    background: rgba(70, 120, 180, 0.6);
    border: 1px solid rgba(100, 150, 220, 0.4);
    color: inherit;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    transition: all 0.15s;
  }

  button:hover:not(:disabled) {
    background: rgba(80, 140, 200, 0.7);
  }

  button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
