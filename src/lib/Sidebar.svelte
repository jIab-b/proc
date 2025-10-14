<script lang="ts">
  import BlockGrid from './BlockGrid.svelte'
  import FaceViewer from './FaceViewer.svelte'
  import TextureGenerator from './TextureGenerator.svelte'
</script>

<div class="sidebar">
  <BlockGrid />
  <div class="face-viewer-container">
    <FaceViewer />
    <TextureGenerator />
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

  .face-viewer-container {
    display: flex;
    flex-direction: column;
  }
</style>
