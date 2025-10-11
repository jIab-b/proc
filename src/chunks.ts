export type ChunkHandle = { id: number }

export class ChunkManager {
  private nextId = 1
  constructor() {}
  createChunk(): ChunkHandle {
    const h = { id: this.nextId++ }
    return h
  }
}

