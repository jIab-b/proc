<script>
  import { onMount } from 'svelte'
  import UI from './UI.svelte'
  import Canvas from './Canvas.svelte'
  import ApiKeyModal from './ApiKeyModal.svelte'
  import { openaiApiKey, backendConfig } from './stores'

  let showApiKeyModal = true

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
        showApiKeyModal = config.requiresApiKey
        console.log('[App] Backend mode:', config.mode, 'Requires API key:', config.requiresApiKey)
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
  <UI />
  <Canvas />
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
