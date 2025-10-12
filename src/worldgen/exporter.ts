import { ChunkDimensions, ChunkManager } from '../chunks'
import { createDefaultWorldSpec, WorldSpec } from './dsl'
import { assertWorldSpec } from './schema'
import { createTracingContext } from './trace'
import type { TracingContext } from './trace'
import { generateChunk } from './generator'

type TracePayload = ReturnType<TracingContext['serialize']>

export interface WorldSnapshot {
  spec: WorldSpec
  trace: TracePayload
  chunkSize: ChunkDimensions
  blocks?: Uint8Array
}

export interface SnapshotOptions {
  spec?: WorldSpec
  chunkSize?: ChunkDimensions
  includeBlocks?: boolean
}

export function buildWorldSnapshot(options: SnapshotOptions = {}): WorldSnapshot {
  let spec = options.spec ?? createDefaultWorldSpec()
  assertWorldSpec(spec)
  const chunkSize = options.chunkSize ?? spec.dimensions
  if (
    chunkSize.x !== spec.dimensions.x ||
    chunkSize.y !== spec.dimensions.y ||
    chunkSize.z !== spec.dimensions.z
  ) {
    spec = { ...spec, dimensions: { ...chunkSize } }
  }
  const chunk = new ChunkManager(chunkSize)
  const trace = createTracingContext(spec.label ?? 'world')
  generateChunk({ chunk, spec, trace })
  return {
    spec,
    trace: trace.serialize(),
    chunkSize,
    blocks: options.includeBlocks === false ? undefined : chunk.snapshotBlocks()
  }
}
