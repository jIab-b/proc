# World System Usage Guide

This guide demonstrates how to use the new predictable procedural generation world system.

## Quick Start

### Basic World Generation

```typescript
import { createDefaultWorldSpec, executeGeneration, ChunkManager } from './worldgen'

// Create a world specification
const spec = createDefaultWorldSpec(12345)

// Generate the world
const chunk = new ChunkManager(spec.dimensions)
await executeGeneration(chunk, spec)
```

## Core Features

### 1. World Registry

Track and manage multiple worlds:

```typescript
import { WorldRegistry, inferTags, hashObject } from './worldgen'

const registry = new WorldRegistry()

// Register a world
const metadata = {
  id: hashObject(spec),
  spec: worldSpec,
  created: Date.now(),
  tags: inferTags(spec),
  generationVersion: '1.0.0',
  traceDigest: fingerprint.traceHash
}
registry.register(metadata)

// Query worlds
const temperateWorlds = registry.getByBiome('temperate')
const worldsBySeed = registry.getBySeed(12345)
const taggedWorlds = registry.getByTag('has_rivers')
```

### 2. Seed Derivation Hierarchy

Generate infinite worlds with deterministic chunk-based seeding:

```typescript
import { deriveChunkSeed, deriveRegionSeed } from './worldgen'

// Derive seed for a specific chunk
const chunkSeed = deriveChunkSeed(worldSeed, stageSeed, { cx: 10, cz: 5 })

// Derive seed for a region (16×16 chunks)
const regionSeed = deriveRegionSeed(worldSeed, regionX, regionZ)
```

### 3. Pipeline System with Dependencies

The generation pipeline automatically handles stage dependencies:

```typescript
import { executeGeneration, validateDependencies, GENERATION_STAGES } from './worldgen'

// Validate pipeline before running
if (!validateDependencies()) {
  throw new Error('Pipeline dependency validation failed')
}

// Execute with full tracing
const trace = createTracingContext('my_world')
await executeGeneration(chunk, spec, trace)

// Inspect stages
console.log('Stages:', GENERATION_STAGES.map(s => s.stage))
```

### 4. Incremental Generation

Generate worlds progressively with checkpoints:

```typescript
import { IncrementalGenerator, generateWithProgress } from './worldgen'

// With progress callback
await generateWithProgress(chunk, spec, trace, (checkpoint) => {
  console.log(`Progress: ${(checkpoint.progress * 100).toFixed(1)}%`)
  console.log(`Completed stages: ${checkpoint.completedStages.join(', ')}`)
})

// Manual incremental control
const generator = new IncrementalGenerator(chunk, spec, trace)

for await (const checkpoint of generator.generateIncremental()) {
  console.log(`Stage ${checkpoint.completedStages.length}/${GENERATION_STAGES.length}`)

  // Can pause and resume later
  if (checkpoint.progress >= 0.5) {
    const saved = serializeCheckpoint(checkpoint)
    // Save checkpoint, resume later
    break
  }
}
```

### 5. Variations & Mutations

Create controlled variants of worlds:

```typescript
import { applyMutation, generateVariantSet, createDiversitySet, createParametricSweep } from './worldgen'

// Single mutation
const variant = applyMutation(baseSpec, {
  type: 'biome_swap',
  params: { targetBiome: 'tundra' }
})

// Multiple variants with same seed, different parameters
const variants = generateVariantSet(baseSpec, [
  { type: 'elevation_shift', params: { delta: 1 } },
  { type: 'feature_density', params: { treeDensity: 2 } }
], 5)

// Diversity set with different biomes and terrains
const diverse = createDiversitySet(baseSpec, {
  biomes: ['temperate', 'tundra', 'desert'],
  terrainForms: ['plains', 'ridged', 'valley'],
  elevationRange: [1, 4],
  preserveSeed: false
}, 20)

// Parametric sweep for one parameter
const elevationSweep = createParametricSweep(baseSpec, 'elevation')
// Generates 5 worlds with very_low, low, medium, high, very_high elevation
```

### 6. Verification & Determinism

Ensure worlds generate identically across runs:

```typescript
import {
  generateAndVerify,
  verifyReproducibility,
  VerificationDatabase,
  compareFingerprints
} from './worldgen'

// Generate and capture fingerprint
const { chunk, trace, fingerprint } = generateAndVerify(spec)
console.log('Block checksum:', fingerprint.blockChecksum)

// Verify later
const isReproducible = verifyReproducibility(spec, fingerprint)

// Use database for tracking
const verifyDB = new VerificationDatabase()
verifyDB.store('world_12345', fingerprint)

// Later verification
const report = verifyDB.verify('world_12345', spec)
if (report && !report.passed) {
  console.error('Verification failed:', report)
}
```

### 7. Batch Generation

Generate datasets for ML training:

```typescript
import { generateBatch, generateBalancedBatch, generateCurriculumBatch } from './worldgen'

// Random sampling with distribution
const result = await generateBatch({
  count: 1000,
  samplingStrategy: 'uniform',
  biomeDistribution: {
    temperate: 0.4,
    tundra: 0.2,
    desert: 0.2,
    islands: 0.2
  },
  generationVersion: '1.0.0'
})

console.log(`Generated ${result.worlds.length} worlds in ${result.totalTime}ms`)
console.log(`Average: ${result.averageTimePerWorld.toFixed(2)}ms per world`)

// Balanced sampling (equal representation)
const balanced = generateBalancedBatch(100)

// Curriculum learning (progressive difficulty)
const curriculum = generateCurriculumBatch([
  {
    count: 50,
    biomes: ['temperate'],
    terrainForms: ['plains'],
    elevationRange: [1, 2] // Easy: low variation
  },
  {
    count: 100,
    biomes: ['temperate', 'tundra'],
    terrainForms: ['plains', 'ridged'],
    elevationRange: [1, 3] // Medium
  },
  {
    count: 200,
    biomes: ['temperate', 'tundra', 'desert', 'islands'],
    terrainForms: ['plains', 'ridged', 'valley', 'mesa'],
    elevationRange: [0, 4] // Hard: full variation
  }
])
```

## Browser Console API

When running the app, the following are exposed on `window`:

```javascript
// In browser console:

// Access trace data
window.worldGenTrace
// { label: "world", stages: [...] }

// Access registry
window.worldRegistry.getAll()
window.worldRegistry.getByBiome('temperate')

// Access fingerprint
window.worldFingerprint
// { specHash: "...", blockChecksum: "...", traceHash: "..." }

// Access metadata
window.worldMetadata
// { id: "...", spec: {...}, tags: [...], ... }
```

## ML Training Workflow

Complete pipeline for dataset generation:

```typescript
import {
  generateBatch,
  WorldRegistry,
  VerificationDatabase,
  generateAndVerify
} from './worldgen'

// 1. Generate training set
const batchResult = await generateBatch({
  count: 50000,
  samplingStrategy: 'uniform',
  generationVersion: '1.0.0'
})

// 2. Store registry
const registryJSON = batchResult.registry.serialize()
// Save to file: training_registry.json

// 3. Verify determinism on subset
const verifyDB = new VerificationDatabase()
for (const world of batchResult.worlds.slice(0, 100)) {
  const { fingerprint } = generateAndVerify(world.spec)
  verifyDB.store(world.id, fingerprint)
}

// 4. Later: verify reproducibility
const dbExport = verifyDB.exportDatabase()
// Save to file: verification_db.json

// 5. Regenerate and verify
for (const world of batchResult.worlds.slice(0, 100)) {
  const report = verifyDB.verify(world.id, world.spec)
  if (!report?.passed) {
    console.error(`World ${world.id} failed verification`)
  }
}
```

## Advanced: Multi-Scale Context (Future)

The system supports multi-scale generation context:

```typescript
import { MultiScaleContext, executeGeneration } from './worldgen'

// Currently a placeholder for future multi-chunk worlds
const multiScale: MultiScaleContext = {
  scale: 'local',
  continental: undefined, // Future: broad biome distribution
  regional: undefined,    // Future: river networks, mountain ranges
  local: undefined        // Future: current chunk heightfield
}

await executeGeneration(chunk, spec, trace, multiScale)
```

## Performance Tips

1. **Batch processing**: Use `generateBatch` instead of individual `generateChunk` calls
2. **Incremental for UI**: Use `IncrementalGenerator` for progress indicators
3. **Verification sampling**: Only verify a subset (5-10%) for large batches
4. **Checkpoint saving**: Serialize checkpoints to pause/resume long batches

## Next Steps

- **Camera Rigs**: Add camera positioning system for multi-view rendering
- **Export System**: Enhance `exporter.ts` to save blocks/traces to disk
- **Distributed Generation**: Split batch across workers/machines using deterministic seeds
- **Parameter-Space Diffusion**: Train small model to sample diverse WorldSpecs
