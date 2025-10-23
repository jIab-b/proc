<script>
  import UI from './UI.svelte'
  import Canvas from './Canvas.svelte'
</script>

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
