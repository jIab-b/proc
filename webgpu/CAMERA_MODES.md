# Camera Modes - Player & Overview

Added a dual-camera system with **Player mode** (first-person) and **Overview mode** (orbital/Blender-style).

## Quick Start

### Toggle Camera Modes

**Tab Key** - Quick toggle between player and overview modes

**UI Toggle** - Use the camera mode buttons at the top of the sidebar

## Player Mode (First-Person)

Classic FPS-style camera for building and exploration.

### Controls

- **Click canvas** - Lock mouse pointer for looking around
- **Mouse** - Look around (when pointer locked)
- **WASD** - Move horizontally
- **Space / E** - Move up
- **Ctrl / Q** - Move down
- **Shift** - Move faster (2x speed)
- **Esc** - Unlock pointer

### Features

- First-person perspective
- Pointer lock for smooth mouse look
- Yaw and pitch rotation
- Collision-free movement

## Overview Mode (Orbital)

Blender-style orbital camera for wide-area viewing and fast navigation.

### Controls

- **Drag (Left/Right mouse)** - Orbit around target point
- **Scroll wheel** - Zoom in/out
- **Middle mouse drag** - Pan view
- **WASD** - Pan horizontally (fast)
- **Space / E** - Pan up
- **Ctrl / Q** - Pan down
- **Shift** - Pan faster (2x speed)

### Features

- Orbital rotation around a target point
- Smooth zooming (5 to 500 units)
- Fast panning with keyboard
- No pointer lock - free mouse movement
- Perfect for viewing terrain from all angles

## Camera Mode Details

### Player Mode

```typescript
Mode: 'player'
Pointer Lock: Yes (click to lock)
Movement Speed: 10 units/sec (20 with Shift)
Rotation: Yaw/Pitch with mouse
Controls: WASD + Mouse look
```

### Overview Mode

```typescript
Mode: 'overview'
Pointer Lock: No
Movement Speed: 20 units/sec (40 with Shift)
Rotation: Orbit around target
Controls: Drag to orbit, Scroll to zoom, WASD to pan
Zoom Range: 5 to 500 units
```

## Technical Implementation

### Camera State

```typescript
// Player mode
cameraPos: Vec3         // Camera position
yaw: number            // Horizontal rotation
pitch: number          // Vertical rotation

// Overview mode
orbitTarget: Vec3      // Point to orbit around
orbitDistance: number  // Distance from target
orbitYaw: number       // Orbital horizontal angle
orbitPitch: number     // Orbital vertical angle
```

### Mode Switching

When switching from Player to Overview:
- Orbit target is initialized to current player position
- Orbit angles match current player yaw/pitch
- Distance is set to 50 units
- Pointer is unlocked
- Camera becomes unpaused

When switching from Overview to Player:
- Player position remains unchanged
- Camera becomes paused (click to lock pointer)
- Player yaw/pitch can be initialized from orbit angles

### Store Integration

Camera mode is managed via Svelte store:

```typescript
import { cameraMode } from './core'

// Set mode
cameraMode.set('overview')

// Get mode
const mode = get(cameraMode)

// Subscribe to changes
cameraMode.subscribe(mode => {
  console.log(`Camera mode: ${mode}`)
})
```

### Keyboard Shortcut

Tab key is globally bound to toggle camera modes:

```typescript
window.addEventListener('keydown', (ev) => {
  if (ev.code === 'Tab') {
    ev.preventDefault()
    const newMode = cameraMode === 'player' ? 'overview' : 'player'
    cameraModeStore.set(newMode)
  }
})
```

## Use Cases

### Player Mode - Best For

- **Building** - Place and remove blocks precisely
- **Exploration** - First-person immersive view
- **Testing** - Walk through your creations
- **Close-up work** - Detailed block placement

### Overview Mode - Best For

- **Planning** - View large terrain areas
- **Terrain generation** - See the full generated landscape
- **Camera angles** - Find interesting viewpoints
- **Fast navigation** - Quickly move across large distances
- **Inspection** - View structures from all angles

## Tips & Tricks

### Player Mode

1. **Hold Shift** for faster movement when exploring large areas
2. **Use Q/E** for vertical movement when building tall structures
3. **Esc to unlock** pointer when you need to use UI

### Overview Mode

1. **Drag anywhere** to rotate - no need to lock pointer
2. **Scroll in/out** for quick distance adjustments
3. **WASD for precise panning** - much faster than middle mouse
4. **Shift + WASD** for very fast panning across large terrains
5. **Target follows WASD** - you're moving the point you're looking at

### Workflow

1. Start in **Player mode** to position yourself
2. Switch to **Overview** (Tab) to see the full area
3. Use **WASD + Shift** in overview to quickly pan to new areas
4. **Drag to rotate** and find the best angle
5. Switch back to **Player** (Tab) to do detailed work

## Camera Mode UI Component

The camera mode toggle is at the top of the sidebar:

```svelte
<CameraModeToggle />
```

Features:
- Two-button toggle (Player / Overview)
- Active state highlighting
- Tooltips with control hints
- Dynamic hint text based on mode
- Syncs with Tab key toggle

## Files Modified

1. **src/core.ts**
   - Added `cameraMode` store

2. **src/renderer.ts**
   - Added dual camera system
   - Implemented orbital camera controls
   - Added Tab key toggle
   - Subscribe to cameraMode store

3. **src/lib/CameraModeToggle.svelte**
   - New UI component
   - Mode toggle buttons
   - Control hints

4. **src/UI.svelte**
   - Import and render CameraModeToggle

## Future Enhancements

Possible improvements:

- [ ] Save preferred camera mode in localStorage
- [ ] Smooth transitions between modes
- [ ] Customizable keybindings
- [ ] Camera presets/bookmarks
- [ ] Animation paths
- [ ] Focus on selected blocks
- [ ] Snap to grid in overview mode
- [ ] Minimap in overview mode
- [ ] Third-person player mode option
