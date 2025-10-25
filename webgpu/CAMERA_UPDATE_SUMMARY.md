# Camera Modes Update - Summary

Successfully implemented a dual-camera system with **Player** (FPS) and **Overview** (orbital/Blender-style) modes.

## What Was Added

### 1. Dual Camera System in Renderer

**File:** `src/renderer.ts`

Added two camera modes:

#### Player Mode (FPS)
- First-person camera with pointer lock
- Mouse look with yaw/pitch
- WASD movement
- Click canvas to lock pointer

#### Overview Mode (Orbital)
- Orbital camera around a target point
- Drag to orbit (left/right mouse)
- Scroll to zoom (5-500 units)
- Middle mouse to pan
- WASD for fast panning
- No pointer lock - free mouse movement

### 2. Camera Mode Store

**File:** `src/core.ts`

Added Svelte store for camera mode:

```typescript
export const cameraMode: Writable<'player' | 'overview'> = writable('player')
```

This allows the UI and renderer to stay in sync.

### 3. Camera Mode Toggle UI

**File:** `src/lib/CameraModeToggle.svelte`

New component with:
- Two-button toggle (Player / Overview)
- Active state highlighting
- Dynamic control hints
- Syncs with Tab key

### 4. Integration

**File:** `src/UI.svelte`

Added `<CameraModeToggle />` to the top of the sidebar.

## How to Use

### Quick Toggle

Press **Tab** to instantly switch between camera modes.

### UI Toggle

Click the **🎮 Player** or **🌐 Overview** buttons in the sidebar.

### Player Mode Controls

```
Click canvas → Lock pointer
Mouse → Look around
WASD → Move
Space/E → Move up
Ctrl/Q → Move down
Shift → Move faster
Esc → Unlock pointer
Tab → Switch to overview
```

### Overview Mode Controls

```
Drag → Orbit around target
Scroll → Zoom in/out
Middle drag → Pan view
WASD → Pan horizontally (fast!)
Space/E → Pan up
Ctrl/Q → Pan down
Shift → Pan faster (2x speed)
Tab → Switch to player
```

## Technical Details

### Camera State Variables

```typescript
// Player mode
cameraPos: Vec3 = [0, y, z]
yaw: number = 0
pitch: number = 0

// Overview mode
orbitTarget: Vec3 = [0, y, 0]
orbitDistance: number = 50
orbitYaw: number = 0
orbitPitch: number = -0.5
```

### Mode Switching Logic

When switching **TO Overview mode**:
1. Exit pointer lock
2. Set paused = false
3. Initialize orbit target to current player position
4. Copy player yaw/pitch to orbit angles
5. Set distance to 50 units

When switching **TO Player mode**:
1. Set paused = true (requires click to lock)
2. Player position unchanged
3. Player can resume from where they were

### Mouse Event Handling

**Player mode:**
- Pointer lock required
- `mousemove` updates yaw/pitch
- `mousedown` places/removes blocks

**Overview mode:**
- No pointer lock
- `mousedown` starts drag
- `mousemove` orbits or pans
- `mouseup` ends drag
- `wheel` zooms in/out

## Files Modified

1. **src/core.ts** - Added cameraMode store
2. **src/renderer.ts** - Implemented dual camera system (200+ lines)
3. **src/lib/CameraModeToggle.svelte** - New UI component
4. **src/UI.svelte** - Added camera toggle to sidebar

## Files Created

1. **CAMERA_MODES.md** - Complete documentation
2. **CAMERA_UPDATE_SUMMARY.md** - This file

## Testing

Run the dev server:

```bash
cd /home/beed/splats/proc/webgpu
npm run dev
```

Then:

1. Open the app in browser
2. Look for camera mode toggle in sidebar
3. Click **🌐 Overview** or press **Tab**
4. Try dragging, scrolling, and WASD
5. Switch back to **🎮 Player** with Tab
6. Click canvas to lock pointer
7. Use mouse to look around

## Use Cases

### Player Mode → Best For:
- Building and placing blocks
- First-person exploration
- Close-up detail work
- Immersive experience

### Overview Mode → Best For:
- Viewing generated terrain
- Planning large structures
- Finding camera angles
- Fast navigation across terrain
- Getting a bird's eye view

## Overview Mode Speed Comparison

| Action | Speed | Notes |
|--------|-------|-------|
| WASD pan | 20 units/sec | Very fast horizontal movement |
| WASD + Shift | 40 units/sec | 2x speed for large terrains |
| Drag orbit | Variable | Smooth rotation |
| Scroll zoom | Variable | Distance-based zoom speed |

This makes it perfect for exploring the new grid-based terrain system (thousands of blocks wide)!

## Integration with Terrain Generation

The overview mode works perfectly with the random terrain generation:

1. Click **🎲 Generate Random Terrain**
2. Switch to **Overview mode** (Tab)
3. Use **WASD + Shift** to quickly pan across the terrain
4. **Scroll to zoom** for different perspectives
5. **Drag to orbit** to see the terrain from all angles

## Keyboard Shortcuts Summary

| Key | Action |
|-----|--------|
| Tab | Toggle camera mode |
| WASD | Move/pan |
| Space/E | Move up |
| Ctrl/Q | Move down |
| Shift | 2x movement speed |
| Esc | Unlock pointer (player mode) |
| Click | Lock pointer (player mode) |
| Drag | Orbit (overview mode) |
| Scroll | Zoom (overview mode) |
| Middle drag | Pan (overview mode) |

## Future Enhancements

Possible improvements:

- [ ] Save preferred camera mode to localStorage
- [ ] Smooth camera transitions between modes
- [ ] Customizable keybindings
- [ ] Camera presets (save positions)
- [ ] Focus on selected blocks
- [ ] Third-person player mode
- [ ] Camera animation paths
- [ ] Minimap in overview mode
- [ ] Grid snapping in overview mode

## Performance

No performance impact:
- Only one camera mode active at a time
- Store subscription is lightweight
- Mode switching is instant
- No additional rendering overhead

## Browser Compatibility

Works in all browsers that support:
- WebGPU
- Pointer Lock API
- Wheel events
- Mouse events

Tested in:
- Chrome/Edge (Chromium)
- Firefox (with WebGPU enabled)

## Tips for Users

### Getting Started

1. **Try Tab first** - Quickest way to see both modes
2. **In overview, use WASD** - Much faster than dragging
3. **Hold Shift** - Doubles movement speed in both modes
4. **Scroll generously** - Zoom range is 5 to 500 units

### Common Workflows

**Terrain Viewing:**
1. Generate random terrain
2. Tab to overview
3. Shift + WASD to explore
4. Drag to find best angle

**Precision Building:**
1. Tab to player mode
2. Click to lock pointer
3. Use WASD + mouse for precise positioning
4. Place blocks with left/right click

**Quick Inspection:**
1. Tab to overview
2. Scroll out to see full structure
3. Drag to rotate
4. Tab back to player

The camera system is now complete and ready to use! 🎮🌐
