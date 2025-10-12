# Comprehensive World Generation Plan

## Goals
- Provide a reproducible, Minecraft-inspired world simulation with deterministic seeds.
- Emit rich function-level traces so any world build can be replayed or inspected.
- Offer an LLM-friendly DSL that exposes high-level knobs without leaking low-level engine details.
- Keep the runtime compatible with the WebGPU renderer and existing chunk mesh pipeline.

## Architecture Overview
- **DSL Layer (`WorldSpec`)**
  - Human-readable enums and discrete bins for all tunable knobs (biomes, terrain profiles, feature budgets, shader sets).
  - Explicit stage seeds (climate, terrain, structures, surface) derived from the base seed for predictable perturbations.
  - JSON schema compatible so grammar-constrained decoding is trivial for LoRA-tuned LLMs.

- **Tracing Layer (`TracingContext`)**
  - Hierarchical stage scopes with begin/end semantics.
  - Each scope records the stage id, seed, input params, summary statistics, RNG usage, and timing.
  - `TracingContext.serialize()` exposes clean JSON for downstream dataset capture and debugging.

- **Deterministic RNG (`StageRandom`)**
  - Seeded Mulberry32 variant with stat aggregation (samples, min/max/mean).
  - Integrated with tracing to prove reproducibility and to give the LLM supervised examples of seed influence.

- **World Builder Pipeline (`generateChunk`)**
  1. `sampleClimate` – FBM-based temperature & moisture lattices driven by climate seed.
  2. `computeHeightfield` – terrain noise with ridge frequency, elevation variance, and domain warp bins.
  3. `applySurfaceLayers` – biome-aware block stacks, sea-level handling, and palette selection.
  4. `placeFeatures` – lightweight structure hooks (e.g., tree columns) governed by feature budgets and random traces.
  - Every stage logs summaries (value ranges, distribution histograms, counts) to the trace.

- **Runtime Integration**
  - New `worldgen` module produces `WorldSpec`, tracing utilities, and `generateChunk`.
  - `src/main.ts` instantiates a default spec, generates the chunk via the new pipeline, and exposes the trace via `window.worldGenTrace`.
  - Existing mesh building/render loop remains untouched aside from palette growth.

## Implementation Steps
1. Add DSL definitions and helper factories (`src/worldgen/dsl.ts`).
2. Implement tracing + deterministic RNG utilities (`src/worldgen/trace.ts`, `src/worldgen/random.ts`).
3. Introduce noise helpers and world builder pipeline (`src/worldgen/noise.ts`, `src/worldgen/generator.ts`, `src/worldgen/index.ts`).
4. Expand block palette to cover new biome surfaces (`src/chunks.ts`).
5. Replace `generateDefaultTerrain` usage in `src/main.ts` with the new pipeline and publish traces for inspection.
6. Light-touch validation via console logging and ensuring render integrity.

## Verification Plan
- Type-check the new modules (Vite build already runs `tsc --noEmit`).
- Launch `npm run dev` to confirm terrain renders and trace data appears on `window.worldGenTrace`.
- Spot-check trace JSON to confirm seeds, stage stats, and RNG logs exist.
