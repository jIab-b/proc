### WebGPU redesign plan (scoped to `webgpu/` only)

#### Objectives
- Abstract GPU/compute into a swappable render backend.
- Route high‑level actions via a language‑agnostic DSL.
- Keep current voxel editor behavior intact while reducing files/LOC.
- Prepare for non‑voxel scenes without coupling UI/business logic to WebGPU.

#### Scope
- Only files in `webgpu/`. No backend or Python changes here.

---

### Current state (short)
- Rendering and input logic are mixed in `src/renderer.ts`.
- Domain state lives in `src/world/*` and `src/core.ts`.
- Legacy/duplicate UI and examples exist under `src/lib/*`, `examples/`, and grid helpers.
- Two parallel shapes for types/state (`core.ts` vs `stores.ts`/`chunks.ts`).

---

### Target architecture (inside `webgpu/`)
- DSL layer: single JSON/TypeScript definition for world actions.
- Engine: applies DSL to `WorldState`, publishes `WorldChange`.
- Render backend interface: WebGPU adapter implements it; future adapters can be added without touching UI.
- Input controller: emits DSL, no direct world mutations.

#### New modules to add
- `src/dsl/commands.ts`: TS types for DSL actions (mirrors JSON schema).
- `src/engine/worldEngine.ts`: receives DSL, calls `world.apply`, forwards changes to backend.
- `src/render/renderBackend.ts`: `RenderBackend` interface.
- `src/render/webgpuBackend.ts`: wraps `createRenderer` from `src/renderer.ts` to implement `RenderBackend`.
- `src/input/inputController.ts`: mouse/keyboard → DSL commands.
- `dsl.schema.json` (optional reference, colocated at repo root of `webgpu/`): JSON schema for the DSL.

Notes:
- No shader changes required (`src/pipelines/render/*.wgsl`).
- `src/renderer.ts` remains the low‑level WebGPU implementation; it should stop mutating world directly once `inputController` is in place.

---

### DSL (initial surface)
- Required now: block placement/removal, highlight selection, terrain region generation.
- Versioned payloads for forward compatibility.

Example actions (JSON):
```json
{"v":1,"type":"set_block","edit":{"position":[x,y,z],"blockType":4},"source":"player"}
```
```json
{"v":1,"type":"set_block","edit":{"position":[x,y,z],"blockType":0},"source":"player"}
```
```json
{"v":1,"type":"highlight_set","selection":{"center":[x,y,z],"shape":"plane","planeSizeX":8,"planeSizeZ":8}}
```
```json
{"v":1,"type":"terrain_region","params":{"action":"generate","region":{"min":[...],"max":[...]},"profile":"rolling_hills","selectionType":"plane","params":{"seed":1337,"amplitude":10,"roughness":2.4,"elevation":0.35}}}
```

---

### File changes (webgpu/ only)

#### Keep
- `src/renderer.ts`
- `src/world/*`
- `src/procedural/terrainGenerator.ts`
- `src/core.ts`
- `src/engine.ts` (until `worldEngine.ts` fully replaces DSL bits)
- `src/Canvas.svelte`, `src/UI.svelte`
- `src/lib/ProceduralTerrainPanel.svelte`, `src/lib/CameraModeToggle.svelte`
- `src/pipelines/render/*.wgsl`
- `src/types.d.ts`, `src/webgpu.d.ts`, `src/vite-env.d.ts`
- `vite.config.ts`, `package.json`, `index.html`

#### Add (new)
- `dsl.schema.json` (optional reference schema)
- `src/dsl/commands.ts`
- `src/engine/worldEngine.ts`
- `src/render/renderBackend.ts`
- `src/render/webgpuBackend.ts`
- `src/input/inputController.ts`

#### Remove (immediate, safe)
- `src/webgpuEngine.ts.backup`
- `src/App.svelte.backup`
- `src/lib/WebGPUCanvas.svelte`
- `src/lib/Sidebar.svelte`
- `src/lib/BlockGrid.svelte`
- `src/lib/FaceViewer.svelte`
- `src/lib/TextureGenerator.svelte`
- `examples/grid-terrain-example.ts`
- `src/gridMeshBuilder.ts`
- `src/chunkGrid.ts`
- `src/procedural/gridTerrainGenerator.ts`
- `src/blockUtils.ts` (unused by live app once the lib/* removals land)

#### Remove (later, after a tiny refactor)
- `src/chunks.ts` and `src/stores.ts` after:
  - move `openaiApiKey` to `src/core.ts` and switch imports.
  - update `stores.ts` consumers to use `core.ts` types, then delete `stores.ts` and `chunks.ts` together.

---

### Migration steps (incremental)
1) De‑bloat: delete “Remove (immediate, safe)” files. Build and run.
2) Add `src/render/renderBackend.ts` and `src/render/webgpuBackend.ts`. Wrap existing `createRenderer` with a thin adapter. No behavior change.
3) Add `src/engine/worldEngine.ts` that wires `WorldState` → backend and exposes `apply(command)`.
4) Add `src/input/inputController.ts`. Move mouse handlers from `src/renderer.ts` to emit DSL via the engine. Keep keybindings identical.
5) Add `src/dsl/commands.ts` (and optional `dsl.schema.json`). Update `src/world/commands.ts` to re‑export or conform to this shape.
6) Refactor `renderer.ts` to stop calling `world.apply` directly. It should use engine hooks or callbacks.
7) Unify types: move `openaiApiKey` to `core.ts`, make consumers import from there; then remove `stores.ts` and `chunks.ts`.

Build checks after steps 1, 4, and 7.

---

### Non‑voxel extensibility (inside `webgpu/`)
- Extend DSL with scene ops later: `spawn`, `destroy`, `set_transform`, `upload_mesh`, `set_material`, `set_camera`.
- Add `src/scene/*` for a minimal scene graph (nodes, transforms, materials, meshes). Keep voxel as one subsystem.
- Implement scene drawing inside `webgpuBackend` without touching UI.

---

### Testing checklist
- Block placement/removal via mouse still works.
- Highlight select (cube/sphere/ellipsoid/plane) still sets and renders overlays.
- Terrain generation via UI and selection still applies to chunk region.
- Custom block textures still upload and render.
- No references to removed files remain.

---

### Rollback
- All deletions target files unused by the live app; revert by restoring from VCS if needed.

---

### Notes
- Keep WGSL shaders untouched.
- Changes are confined to `webgpu/` per scope.


