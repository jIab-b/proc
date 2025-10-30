<script lang="ts">
  import { onMount } from 'svelte'
  import UI from './UI.svelte'
  import Canvas from './Canvas.svelte'
  import ApiKeyModal from './ApiKeyModal.svelte'
  import ResizableWindow from './lib/ResizableWindow.svelte'
  import { openaiApiKey, backendConfig } from './stores'

  let showApiKeyModal = true

  // Sidebar window
  let sidebarX = 12
  let sidebarY = 12
  let sidebarWidth = 350
  let sidebarHeight = window.innerHeight - 40

  // Canvas window
  let canvasX = 380
  let canvasY = 12
  let canvasWidth = window.innerWidth - 410
  let canvasHeight = window.innerHeight - 40

  onMount(async () => {
    // Fetch backend config to determine if API key modal is needed
    try {
      const response = await fetch('http://localhost:8000/api/config')
      if (response.ok) {
        const config = await response.json()
        backendConfig.set({
          mode: config.mode,
          requiresApiKey: config.requiresApiKey
        })
        console.log('[App] Backend mode:', config.mode, 'Requires API key:', config.requiresApiKey)

        // In dev mode, try to auto-load API key from backend
        if (config.mode === 'dev') {
          try {
            const keyResponse = await fetch('http://localhost:8000/api/dev/openai-key')
            if (keyResponse.ok) {
              const keyData = await keyResponse.json()
              if (keyData.apiKey) {
                openaiApiKey.set(keyData.apiKey)
                console.log('[App] Auto-loaded API key from backend in dev mode')
                showApiKeyModal = false
              } else {
                console.warn('[App] Dev mode but no API key in response')
                showApiKeyModal = true
              }
            } else {
              console.warn('[App] Failed to fetch API key in dev mode:', keyResponse.status)
              showApiKeyModal = true
            }
          } catch (keyError) {
            console.warn('[App] Could not auto-load API key in dev mode:', keyError)
            showApiKeyModal = true
          }
        } else {
          // Prod mode: always require API key from user
          showApiKeyModal = config.requiresApiKey
        }
      }
    } catch (error) {
      console.error('[App] Failed to fetch backend config:', error)
      // Default to showing modal on error
      showApiKeyModal = true
    }
  })
</script>

{#if showApiKeyModal}
  <ApiKeyModal />
{/if}

<div class="app-container">
  <ResizableWindow
    title="Block Controls"
    bind:x={sidebarX}
    bind:y={sidebarY}
    bind:width={sidebarWidth}
    bind:height={sidebarHeight}
    minWidth={200}
    maxWidth={600}
    minHeight={300}
    maxHeight={window.innerHeight - 20}
  >
    <UI />
  </ResizableWindow>

  <ResizableWindow
    title="Graphics Display"
    bind:x={canvasX}
    bind:y={canvasY}
    bind:width={canvasWidth}
    bind:height={canvasHeight}
    minWidth={400}
    maxWidth={window.innerWidth - 20}
    minHeight={300}
    maxHeight={window.innerHeight - 20}
  >
    <Canvas />
  </ResizableWindow>
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
    position: relative;
    width: 100%;
    height: 100vh;
  }
</style>
