<script lang="ts">
  import { onMount } from 'svelte'

  export let x: number = 0
  export let y: number = 0
  export let width: number = 400
  export let height: number = 600
  export let minWidth: number = 200
  export let maxWidth: number = 800
  export let minHeight: number = 300
  export let maxHeight: number = 1200

  let isDragging = false
  let dragType: 'n' | 's' | 'e' | 'w' | 'ne' | 'nw' | 'se' | 'sw' | null = null
  let startX = 0
  let startY = 0
  let startWidth = 0
  let startHeight = 0
  let startLeft = 0
  let startTop = 0

  const HANDLE_SIZE = 6

  function handleMouseDown(type: 'n' | 's' | 'e' | 'w' | 'ne' | 'nw' | 'se' | 'sw') {
    return (e: MouseEvent) => {
      isDragging = true
      dragType = type
      startX = e.clientX
      startY = e.clientY
      startWidth = width
      startHeight = height
      startLeft = x
      startTop = y
      document.body.style.userSelect = 'none'
    }
  }

  function handleMouseUp() {
    isDragging = false
    dragType = null
    document.body.style.userSelect = 'auto'
  }

  function handleMouseMove(e: MouseEvent) {
    if (!isDragging || !dragType) return

    const dx = e.clientX - startX
    const dy = e.clientY - startY

    if (dragType.includes('e')) {
      width = Math.max(minWidth, Math.min(maxWidth, startWidth + dx))
    }
    if (dragType.includes('w')) {
      const newWidth = startWidth - dx
      if (newWidth >= minWidth && newWidth <= maxWidth) {
        width = newWidth
        x = startLeft + dx
      }
    }
    if (dragType.includes('s')) {
      height = Math.max(minHeight, Math.min(maxHeight, startHeight + dy))
    }
    if (dragType.includes('n')) {
      const newHeight = startHeight - dy
      if (newHeight >= minHeight && newHeight <= maxHeight) {
        height = newHeight
        y = startTop + dy
      }
    }
  }

  onMount(() => {
    document.addEventListener('mouseup', handleMouseUp)
    document.addEventListener('mousemove', handleMouseMove)

    return () => {
      document.removeEventListener('mouseup', handleMouseUp)
      document.removeEventListener('mousemove', handleMouseMove)
    }
  })

  function getCursor(type: string): string {
    const cursorMap: Record<string, string> = {
      n: 'ns-resize',
      s: 'ns-resize',
      e: 'ew-resize',
      w: 'ew-resize',
      ne: 'nesw-resize',
      nw: 'nwse-resize',
      se: 'nwse-resize',
      sw: 'nesw-resize'
    }
    return cursorMap[type] || 'auto'
  }
</script>

<div
  class="window"
  style="left: {x}px; top: {y}px; width: {width}px; height: {height}px;"
>
  <!-- Resize Handles -->
  <!-- Top -->
  <div
    class="handle handle-n"
    on:mousedown={handleMouseDown('n')}
    style="cursor: {getCursor('n')}"
  />
  <!-- Bottom -->
  <div
    class="handle handle-s"
    on:mousedown={handleMouseDown('s')}
    style="cursor: {getCursor('s')}"
  />
  <!-- Left -->
  <div
    class="handle handle-w"
    on:mousedown={handleMouseDown('w')}
    style="cursor: {getCursor('w')}"
  />
  <!-- Right -->
  <div
    class="handle handle-e"
    on:mousedown={handleMouseDown('e')}
    style="cursor: {getCursor('e')}"
  />
  <!-- Corners -->
  <div
    class="handle handle-nw"
    on:mousedown={handleMouseDown('nw')}
    style="cursor: {getCursor('nw')}"
  />
  <div
    class="handle handle-ne"
    on:mousedown={handleMouseDown('ne')}
    style="cursor: {getCursor('ne')}"
  />
  <div
    class="handle handle-sw"
    on:mousedown={handleMouseDown('sw')}
    style="cursor: {getCursor('sw')}"
  />
  <div
    class="handle handle-se"
    on:mousedown={handleMouseDown('se')}
    style="cursor: {getCursor('se')}"
  />

  <!-- Content -->
  <div class="window-content">
    <slot />
  </div>
</div>

<style>
  .window {
    position: absolute;
    background: rgba(20, 30, 45, 0.95);
    border: 1px solid rgba(210, 223, 244, 0.2);
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .window-content {
    flex: 1;
    overflow: auto;
    min-width: 0;
    min-height: 0;
  }

  /* Resize Handles */
  .handle {
    position: absolute;
    background: transparent;
    z-index: 10;
  }

  /* Edges */
  .handle-n {
    top: 0;
    left: 0;
    right: 0;
    height: 6px;
  }

  .handle-s {
    bottom: 0;
    left: 0;
    right: 0;
    height: 6px;
  }

  .handle-e {
    right: 0;
    top: 0;
    bottom: 0;
    width: 6px;
  }

  .handle-w {
    left: 0;
    top: 0;
    bottom: 0;
    width: 6px;
  }

  /* Corners */
  .handle-nw {
    top: 0;
    left: 0;
    width: 12px;
    height: 12px;
  }

  .handle-ne {
    top: 0;
    right: 0;
    width: 12px;
    height: 12px;
  }

  .handle-sw {
    bottom: 0;
    left: 0;
    width: 12px;
    height: 12px;
  }

  .handle-se {
    bottom: 0;
    right: 0;
    width: 12px;
    height: 12px;
  }

  .handle:hover {
    background: rgba(120, 180, 240, 0.2);
  }

  .handle:active {
    background: rgba(120, 180, 240, 0.4);
  }
</style>
