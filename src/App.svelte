<script lang="ts">
  import Sidebar from './lib/Sidebar.svelte'
  import WebGPUCanvas from './lib/WebGPUCanvas.svelte'
  import { onMount } from 'svelte'
  import { customBlocks, gpuHooks, nextTextureLayer } from './stores'
  import { TEXTURES_ENDPOINT, blockFaceOrder, TILE_BASE_URL, fetchTileBitmap } from './blockUtils'
  import type { CustomBlock, FaceTileInfo } from './stores'
  import type { BlockFaceKey } from './chunks'

  onMount(async () => {
    // Load existing textures from server
    try {
      const response = await fetch(TEXTURES_ENDPOINT)
      if (!response.ok) {
        console.warn('Failed to load existing blocks from server')
        return
      }

      const data = await response.json()
      const blocks = Array.isArray(data.blocks) ? data.blocks : []
      console.log(`Loading ${blocks.length} existing custom blocks from server`)

      const MAX_CUSTOM_BLOCKS = 8
      let loadedCount = 0

      for (const blockData of blocks) {
        if (loadedCount >= MAX_CUSTOM_BLOCKS) {
          console.warn('Maximum custom blocks reached. Skipping additional blocks from server.')
          break
        }

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
        const entries = Object.entries(faces) as [BlockFaceKey, any][]
        let hasAnyTile = false

        for (const [faceKey, tileInfo] of entries) {
          if (!blockFaceOrder.includes(faceKey)) continue
          if (!tileInfo || typeof tileInfo !== 'object') continue
          const path = typeof tileInfo.path === 'string' ? tileInfo.path : ''
          if (!path) continue
          const sequence = typeof tileInfo.sequence === 'number' ? tileInfo.sequence : undefined
          const prompt = typeof tileInfo.prompt === 'string' ? tileInfo.prompt : ''
          const url = `${TILE_BASE_URL}/${path.replace(/^\/+/, '')}`

          try {
            const bitmap = await fetchTileBitmap(url)
            customBlock.faceBitmaps![faceKey] = bitmap
            customBlock.faceTiles![faceKey] = {
              sequence: sequence ?? 0,
              path,
              url,
              prompt
            }
            hasAnyTile = true
          } catch (tileErr) {
            console.warn(`Failed to load tile ${faceKey} for block ${blockName}:`, tileErr)
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
          console.log(`Loaded face tiles for existing block: ${blockName}`)
        }

        customBlocks.update(blocks => [...blocks, customBlock])
        loadedCount++
      }

      nextTextureLayer.set(loadedCount)
    } catch (err) {
      console.error('Error loading existing textures:', err)
    }
  })
</script>

<div class="app-container">
  <Sidebar />
  <WebGPUCanvas />
</div>

<style>
  :global(*, *::before, *::after) {
    box-sizing: border-box;
  }

  :global(html, body) {
    margin: 0;
    height: 100%;
    overflow: hidden;
  }

  :global(body) {
    min-height: 100vh;
    display: flex;
    background: #223244;
    color: #e3ebf7;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }

  .app-container {
    width: 100%;
    height: 100vh;
    display: flex;
  }
</style>
