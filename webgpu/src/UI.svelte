<script lang="ts">
  import { onMount } from 'svelte'
  import {
    selectedBlockType, selectedCustomBlock, customBlocks, selectedFace, texturePrompt,
    availableBlocks, blockPalette, drawIsometricBlock, fetchTileBitmap,
    TEXTURES_ENDPOINT, GENERATE_TILE_ENDPOINT, BLOCKS_ENDPOINT, blockFaceOrder, TILE_BASE_URL, gpuHooks,
    BlockType, type CustomBlock, type FaceTileInfo, type BlockFaceKey
  } from './core'

  // Block Grid State
  let blockGridEl: HTMLDivElement

  // Face Viewer State
  const faceLabels = ['Top (+Y)', 'Bottom (-Y)', 'Front (+Z)', 'Back (-Z)', 'Right (+X)', 'Left (-X)']
  let faceCanvases: (HTMLCanvasElement | null)[] = []

  // Texture Generator State
  let isGenerating = false
  let buttonText = 'Generate Face Tile'
  $: canGenerate = $texturePrompt.trim().length > 0 && $selectedFace !== null && !isGenerating

  // Block selection handlers
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
    if (!confirm(`Delete "${block.name}"? This will also remove the texture files.`)) return

    try {
      if (block.remoteId) {
        const res = await fetch(`${BLOCKS_ENDPOINT}/${block.remoteId}`, { method: 'DELETE' })
        if (res.ok) console.log(`Deleted block ${block.remoteId} from server`)
      }

      if (block.faceBitmaps) {
        blockFaceOrder.forEach(face => block.faceBitmaps?.[face]?.close?.())
      }

      customBlocks.update(blocks => blocks.filter(b => b.id !== block.id))

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
    if (!blockName?.trim()) return

    const nextId = $customBlocks.reduce((max, block) => Math.max(max, block.id), 999) + 1
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

  function renderBlockPreview(canvas: HTMLCanvasElement, block: { type?: BlockType; custom?: CustomBlock }) {
    if (block.custom) {
      const hasBitmaps = Object.values(block.custom.faceBitmaps || {}).some(Boolean)
      drawIsometricBlock(canvas, block.custom.colors, hasBitmaps ? block.custom.faceBitmaps : undefined)
    } else if (block.type !== undefined) {
      const palette = blockPalette[block.type]
      if (palette) drawIsometricBlock(canvas, palette)
    }
  }

  // Face selection
  function handleFaceClick(face: BlockFaceKey) {
    selectedFace.set(face)
  }

  function updateFacePreviews() {
    if ($selectedCustomBlock) {
      faceCanvases.forEach((canvas, index) => {
        if (!canvas) return
        const ctx = canvas.getContext('2d')
        if (!ctx) return
        const face = blockFaceOrder[index]
        if (!face) return
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        const bitmap = $selectedCustomBlock!.faceBitmaps?.[face]
        if (bitmap) {
          ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height)
        } else {
          const colorKey: 'top' | 'bottom' | 'side' = face === 'top' ? 'top' : face === 'bottom' ? 'bottom' : 'side'
          const color = $selectedCustomBlock!.colors[colorKey]
          if (color) {
            ctx.fillStyle = `rgb(${Math.floor(color[0] * 255)}, ${Math.floor(color[1] * 255)}, ${Math.floor(color[2] * 255)})`
            ctx.fillRect(0, 0, canvas.width, canvas.height)
          }
        }
      })
    } else {
      const palette = blockPalette[$selectedBlockType]
      if (palette) {
        const faceColors = [palette.top, palette.bottom, palette.side, palette.side, palette.side, palette.side]
        faceCanvases.forEach((canvas, index) => {
          if (!canvas) return
          const ctx = canvas.getContext('2d')
          const color = faceColors[index]
          if (ctx && color) {
            ctx.fillStyle = `rgb(${Math.floor(color[0] * 255)}, ${Math.floor(color[1] * 255)}, ${Math.floor(color[2] * 255)})`
            ctx.fillRect(0, 0, canvas.width, canvas.height)
          }
        })
      }
    }
  }

  // Texture generation
  async function handleGenerate() {
    const prompt = $texturePrompt.trim()
    if (!prompt || !$selectedFace) {
      alert('Select a block face before generating a tile.')
      return
    }

    isGenerating = true
    buttonText = 'Generating...'

    try {
      const isExisting = Boolean($selectedCustomBlock)
      const defaultName = (prompt.split(',')[0] ?? '').trim() || 'Custom Block'
      const targetName = isExisting ? $selectedCustomBlock!.name : defaultName

      const res = await fetch(GENERATE_TILE_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          face: $selectedFace,
          block_id: $selectedCustomBlock?.remoteId ?? null,
          block_name: targetName
        })
      })

      if (!res.ok) {
        const err = await res.json().catch(async () => ({ detail: await res.text() }))
        throw new Error(err?.detail || `HTTP ${res.status}`)
      }

      const data = await res.json()
      const tile = data.tile as FaceTileInfo & { face: BlockFaceKey }
      const blockData = data.block as { id: number; name?: string }
      const imageBase64 = data.image_base64 as string

      const binary = Uint8Array.from(atob(imageBase64), char => char.charCodeAt(0))
      const blob = new Blob([binary], { type: 'image/png' })
      const bitmap = await createImageBitmap(blob, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' })

      let targetBlock = $selectedCustomBlock
      if (!targetBlock) {
        const nextId = $customBlocks.reduce((max, b) => Math.max(max, b.id), 999) + 1
        const newBlock: CustomBlock = {
          id: nextId,
          name: blockData?.name || targetName,
          colors: { top: [0.5, 0.5, 0.5], bottom: [0.4, 0.4, 0.4], side: [0.45, 0.45, 0.45] },
          faceBitmaps: {},
          faceTiles: {},
          remoteId: blockData?.id
        }
        customBlocks.update(blocks => [...blocks, newBlock])
        selectedCustomBlock.set(newBlock)
        targetBlock = newBlock
      } else if (blockData?.id) {
        customBlocks.update(blocks => {
          const block = blocks.find(b => b.id === targetBlock!.id)
          if (block) {
            block.remoteId = blockData.id
            if (blockData.name) block.name = blockData.name
          }
          return blocks
        })
      }

      const faceKey = tile.face
      targetBlock.faceBitmaps?.[faceKey]?.close?.()

      customBlocks.update(blocks => {
        const block = blocks.find(b => b.id === targetBlock?.id)
        if (block) {
          if (!block.faceBitmaps) block.faceBitmaps = {}
          if (!block.faceTiles) block.faceTiles = {}
          block.faceBitmaps[faceKey] = bitmap
          block.faceTiles[faceKey] = { sequence: tile.sequence, path: tile.path, url: tile.url, prompt: tile.prompt }
        }
        return blocks
      })

      if ($gpuHooks.requestFaceBitmaps && $gpuHooks.uploadFaceBitmapsToGPU) {
        const allTiles = targetBlock.faceTiles || {}
        const bitmaps = await $gpuHooks.requestFaceBitmaps(allTiles)
        $gpuHooks.uploadFaceBitmapsToGPU(bitmaps, targetBlock)
      }

      buttonText = 'Generate Face Tile'
    } catch (err) {
      console.error('Generation failed:', err)
      alert(`Failed to generate: ${err instanceof Error ? err.message : String(err)}`)
      buttonText = 'Generate Face Tile'
    } finally {
      isGenerating = false
    }
  }

  // Mount logic
  onMount(async () => {
    // Render all block previews
    const canvases = blockGridEl.querySelectorAll('canvas')
    canvases.forEach((canvas, idx) => {
      if (idx < availableBlocks.length) {
        const block = availableBlocks[idx]
        if (block) renderBlockPreview(canvas as HTMLCanvasElement, { type: block.type })
      }
    })

    // Custom block rendering happens reactively
    const unsubscribe = customBlocks.subscribe(() => {
      setTimeout(() => {
        canvases.forEach((canvas, idx) => {
          const customIdx = idx - availableBlocks.length
          if (customIdx >= 0 && customIdx < $customBlocks.length) {
            const block = $customBlocks[customIdx]
            if (block) renderBlockPreview(canvas as HTMLCanvasElement, { custom: block })
          }
        })
      }, 0)
    })

    // Load existing textures from server
    try {
      const res = await fetch(TEXTURES_ENDPOINT)
      if (!res.ok) return

      const data = await res.json()
      const blocks = Array.isArray(data.blocks) ? data.blocks : []
      const MAX_CUSTOM_BLOCKS = 8
      let loadedCount = 0

      for (const blockData of blocks) {
        if (loadedCount >= MAX_CUSTOM_BLOCKS) break
        if (!blockData || typeof blockData !== 'object') continue

        const remoteId = typeof blockData.id === 'number' ? blockData.id : undefined
        const blockName = typeof blockData.name === 'string' && blockData.name.trim()
          ? blockData.name.trim()
          : `Block ${remoteId ?? loadedCount + 1000}`

        const customBlock: CustomBlock = {
          id: 1000 + loadedCount,
          name: blockName,
          colors: { top: [0.5, 0.5, 0.5], bottom: [0.4, 0.4, 0.4], side: [0.45, 0.45, 0.45] },
          faceBitmaps: {},
          faceTiles: {},
          remoteId
        }

        const faces = blockData.faces ?? {}
        let hasAnyTile = false

        for (const [faceKey, tileInfo] of Object.entries(faces) as [BlockFaceKey, any][]) {
          if (!blockFaceOrder.includes(faceKey) || !tileInfo?.path) continue
          const url = `${TILE_BASE_URL}/${tileInfo.path.replace(/^\/+/, '')}`

          try {
            const bitmap = await fetchTileBitmap(url)
            customBlock.faceBitmaps![faceKey] = bitmap
            customBlock.faceTiles![faceKey] = {
              sequence: tileInfo.sequence ?? 0,
              path: tileInfo.path,
              url,
              prompt: tileInfo.prompt ?? ''
            }
            hasAnyTile = true
          } catch (err) {
            console.warn(`Failed to load tile ${faceKey} for block ${blockName}:`, err)
          }
        }

        if (hasAnyTile && $gpuHooks.uploadFaceBitmapsToGPU) {
          const gpuBitmaps: Record<BlockFaceKey, ImageBitmap> = {} as Record<BlockFaceKey, ImageBitmap>
          blockFaceOrder.forEach(face => {
            const bmp = customBlock.faceBitmaps?.[face]
            if (bmp) gpuBitmaps[face] = bmp
          })
          customBlock.textureLayer = loadedCount
          $gpuHooks.uploadFaceBitmapsToGPU(gpuBitmaps, customBlock)
        }

        customBlocks.update(blocks => [...blocks, customBlock])
        loadedCount++
      }
    } catch (err) {
      console.error('Error loading existing textures:', err)
    }

    return unsubscribe
  })

  $: if (faceCanvases.length > 0) updateFacePreviews()
  $: $selectedBlockType, $selectedCustomBlock, updateFacePreviews()
</script>

<div class="sidebar">
  <!-- Block Grid -->
  <div class="block-grid" bind:this={blockGridEl}>
    <div class="grid-header">
      <h3>Block Palette</h3>
      <button class="new-block-btn" on:click={createNewBlock}>+</button>
    </div>
    <div class="grid-content">
      {#each availableBlocks as block}
        <button
          class="block-item"
          class:selected={$selectedBlockType === block.type && !$selectedCustomBlock}
          on:click={() => selectDefaultBlock(block.type)}
        >
          <canvas width="64" height="64"></canvas>
          <span>{block.name}</span>
        </button>
      {/each}
      {#each $customBlocks as block}
        <button class="block-item custom" class:selected={$selectedCustomBlock?.id === block.id} on:click={() => selectCustomBlock(block)}>
          <canvas width="64" height="64"></canvas>
          <span>{block.name}</span>
          <button class="delete-btn" on:click={(e) => deleteBlock(block, e)}>×</button>
        </button>
      {/each}
    </div>
  </div>

  <!-- Face Viewer -->
  <div class="face-viewer">
    <h3>Block Faces</h3>
    <div class="face-grid">
      {#each faceLabels as label, index}
        <div
          class="face-box"
          class:selected={$selectedFace === blockFaceOrder[index]}
          on:click={() => { const face = blockFaceOrder[index]; if (face) handleFaceClick(face); }}
          on:mousedown={(e) => e.stopPropagation()}
        >
          <label>{label}</label>
          <canvas class="face-preview" width="48" height="48" bind:this={faceCanvases[index]}></canvas>
        </div>
      {/each}
    </div>
  </div>

  <!-- Texture Generator -->
  <div class="texture-generator">
    <h3>Generate Texture</h3>
    <textarea
      placeholder="Describe the texture (e.g., weathered stone, mossy brick)..."
      bind:value={$texturePrompt}
      rows="3"
    ></textarea>
    <button class="generate-btn" disabled={!canGenerate} on:click={handleGenerate}>
      {buttonText}
    </button>
  </div>
</div>

<style>
  .sidebar {
    width: 380px;
    height: calc(100vh - 32px);
    margin: 16px 0 16px 16px;
    background: rgba(20, 30, 45, 0.95);
    border: 1px solid rgba(210, 223, 244, 0.2);
    border-radius: 16px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
    color: #d2dff4;
  }

  /* Block Grid */
  .block-grid {
    max-height: 120px; /* Drastically reduced vertical space */
    display: flex;
    flex-direction: column;
    border-bottom: 1px solid rgba(210, 223, 244, 0.15);
  }

  .grid-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid rgba(210, 223, 244, 0.1);
  }

  .new-block-btn {
    width: 28px;
    height: 28px;
    border-radius: 6px;
    background: rgba(48, 66, 88, 0.6);
    border: 1px solid rgba(210, 223, 244, 0.3);
    color: #d2dff4;
    font-size: 18px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .new-block-btn:hover {
    background: rgba(68, 86, 108, 0.7);
  }

  .grid-content {
    flex: 1;
    overflow-y: auto;
    padding: 10px 12px 6px;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
    gap: 4px;
  }

  .block-item {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1px;
    padding: 6px 6px 1px;
    background: rgba(34, 50, 68, 0.5);
    border: 2px solid rgba(190, 210, 230, 0.2);
    border-radius: 9px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .block-item:hover {
    background: rgba(48, 66, 88, 0.6);
    border-color: rgba(190, 210, 230, 0.4);
  }

  .block-item.selected {
    background: rgba(68, 96, 128, 0.7);
    border-color: rgba(120, 180, 240, 0.8);
  }

  .block-item canvas {
    width: 52px;
    height: 52px;
  }

  .block-item span {
    font-size: 11px;
    color: #c8d5e8;
    text-align: center;
    line-height: 1.1;
    margin: 0;
  }

  .delete-btn {
    position: absolute;
    top: 4px;
    right: 4px;
    width: 20px;
    height: 20px;
    border-radius: 4px;
    background: rgba(220, 60, 60, 0.8);
    border: none;
    color: white;
    font-size: 16px;
    line-height: 1;
    cursor: pointer;
    transition: background 0.2s;
  }

  .delete-btn:hover {
    background: rgba(255, 80, 80, 0.9);
  }

  /* Face Viewer */
  .face-viewer {
    padding: 16px 20px;
    border-bottom: 1px solid rgba(210, 223, 244, 0.15);
  }

  .face-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-top: 12px;
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

  .face-box label {
    font-size: 10px;
    color: #8fa0b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .face-preview {
    width: 48px;
    height: 48px;
    border-radius: 4px;
  }

  /* Texture Generator */
  .texture-generator {
    padding: 16px 20px;
  }

  .texture-generator textarea {
    width: 100%;
    margin-top: 12px;
    padding: 10px;
    background: rgba(34, 50, 68, 0.6);
    border: 1px solid rgba(190, 210, 230, 0.25);
    border-radius: 8px;
    color: #e3ebf7;
    font-size: 13px;
    font-family: inherit;
    resize: vertical;
  }

  .generate-btn {
    width: 100%;
    margin-top: 10px;
    padding: 12px;
    background: rgba(68, 96, 128, 0.7);
    border: 1px solid rgba(120, 180, 240, 0.5);
    border-radius: 8px;
    color: #d2dff4;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
  }

  .generate-btn:hover:not(:disabled) {
    background: rgba(88, 116, 148, 0.8);
    border-color: rgba(140, 200, 255, 0.7);
  }

  .generate-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
