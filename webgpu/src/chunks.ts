export type ChunkDimensions = { x: number; y: number; z: number }

export enum BlockType {
  Air = 0,
  Grass = 1,
  Dirt = 2,
  Stone = 3,
  Plank = 4,
  Snow = 5,
  Sand = 6,
  Water = 7,
  AlpineRock = 8,
  AlpineGrass = 9,
  GlacierIce = 10,
  Gravel = 11
}

export type ChunkMesh = { vertexData: Float32Array; vertexCount: number }

export type BlockFaceKey = 'top' | 'bottom' | 'north' | 'south' | 'east' | 'west'

export type BlockTextureFaceIndices = Record<BlockFaceKey, number>

const blockTextureLayers: Partial<Record<BlockType, BlockTextureFaceIndices>> = {}

const enum FaceIndex {
  PX,
  NX,
  PY,
  NY,
  PZ,
  NZ
}

type FaceDefinition = {
  normal: [number, number, number];
  offset: [number, number, number];
  corners: [number, number, number][];
  colorSlot: 'top' | 'bottom' | 'side';
}

const faceIndexToKey: Record<FaceIndex, BlockFaceKey> = {
  [FaceIndex.PX]: 'east',
  [FaceIndex.NX]: 'west',
  [FaceIndex.PY]: 'top',
  [FaceIndex.NY]: 'bottom',
  [FaceIndex.PZ]: 'south',
  [FaceIndex.NZ]: 'north'
}

const faceUVs: Record<FaceIndex, [number, number][]> = {
  [FaceIndex.PX]: [
    [0, 1],
    [0, 0],
    [1, 0],
    [1, 1]
  ],
  [FaceIndex.NX]: [
    [1, 1],
    [0, 1],
    [0, 0],
    [1, 0]
  ],
  [FaceIndex.PY]: [
    [0, 1],
    [0, 0],
    [1, 0],
    [1, 1]
  ],
  [FaceIndex.NY]: [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
  ],
  [FaceIndex.PZ]: [
    [0, 1],
    [1, 1],
    [1, 0],
    [0, 0]
  ],
  [FaceIndex.NZ]: [
    [1, 1],
    [1, 0],
    [0, 0],
    [0, 1]
  ]
}

type BlockPaletteEntry = {
  top: [number, number, number];
  bottom: [number, number, number];
  side: [number, number, number];
}

const faceDefs: Record<FaceIndex, FaceDefinition> = {
  [FaceIndex.PX]: {
    normal: [1, 0, 0],
    offset: [1, 0, 0],
    corners: [
      [1, 0, 0],
      [1, 1, 0],
      [1, 1, 1],
      [1, 0, 1]
    ],
    colorSlot: 'side'
  },
  [FaceIndex.NX]: {
    normal: [-1, 0, 0],
    offset: [-1, 0, 0],
    corners: [
      [0, 0, 0],
      [0, 0, 1],
      [0, 1, 1],
      [0, 1, 0]
    ],
    colorSlot: 'side'
  },
  [FaceIndex.PY]: {
    normal: [0, 1, 0],
    offset: [0, 1, 0],
    corners: [
      [0, 1, 0],
      [0, 1, 1],
      [1, 1, 1],
      [1, 1, 0]
    ],
    colorSlot: 'top'
  },
  [FaceIndex.NY]: {
    normal: [0, -1, 0],
    offset: [0, -1, 0],
    corners: [
      [0, 0, 0],
      [1, 0, 0],
      [1, 0, 1],
      [0, 0, 1]
    ],
    colorSlot: 'bottom'
  },
  [FaceIndex.PZ]: {
    normal: [0, 0, 1],
    offset: [0, 0, 1],
    corners: [
      [0, 0, 1],
      [1, 0, 1],
      [1, 1, 1],
      [0, 1, 1]
    ],
    colorSlot: 'side'
  },
  [FaceIndex.NZ]: {
    normal: [0, 0, -1],
    offset: [0, 0, -1],
    corners: [
      [0, 0, 0],
      [0, 1, 0],
      [1, 1, 0],
      [1, 0, 0]
    ],
    colorSlot: 'side'
  }
}

const faceIndices = [0, 1, 2, 0, 2, 3]

export const blockPalette: Record<BlockType, BlockPaletteEntry | undefined> = {
  [BlockType.Air]: undefined,
  [BlockType.Grass]: {
    top: [0.34, 0.68, 0.36],
    bottom: [0.40, 0.30, 0.16],
    side: [0.45, 0.58, 0.30]
  },
  [BlockType.Dirt]: {
    top: [0.42, 0.32, 0.20],
    bottom: [0.38, 0.26, 0.16],
    side: [0.40, 0.30, 0.18]
  },
  [BlockType.Stone]: {
    top: [0.58, 0.60, 0.64],
    bottom: [0.55, 0.57, 0.60],
    side: [0.56, 0.58, 0.62]
  },
  [BlockType.Plank]: {
    top: [0.78, 0.68, 0.50],
    bottom: [0.72, 0.60, 0.42],
    side: [0.74, 0.63, 0.45]
  },
  [BlockType.Snow]: {
    top: [0.92, 0.94, 0.96],
    bottom: [0.90, 0.92, 0.94],
    side: [0.88, 0.90, 0.93]
  },
  [BlockType.Sand]: {
    top: [0.88, 0.82, 0.60],
    bottom: [0.86, 0.78, 0.56],
    side: [0.87, 0.80, 0.58]
  },
  [BlockType.Water]: {
    top: [0.22, 0.40, 0.66],
    bottom: [0.20, 0.34, 0.60],
    side: [0.20, 0.38, 0.64]
  },
  [BlockType.AlpineRock]: {
    top: [0.45, 0.48, 0.52],
    bottom: [0.43, 0.46, 0.50],
    side: [0.44, 0.47, 0.51]
  },
  [BlockType.AlpineGrass]: {
    top: [0.26, 0.58, 0.32],
    bottom: [0.22, 0.44, 0.28],
    side: [0.24, 0.50, 0.30]
  },
  [BlockType.GlacierIce]: {
    top: [0.78, 0.88, 0.96],
    bottom: [0.72, 0.82, 0.90],
    side: [0.74, 0.84, 0.92]
  },
  [BlockType.Gravel]: {
    top: [0.52, 0.52, 0.50],
    bottom: [0.48, 0.48, 0.46],
    side: [0.50, 0.50, 0.48]
  }
}

export class ChunkManager {
  readonly size: ChunkDimensions
  private blocks: Uint8Array

  constructor(size: ChunkDimensions = { x: 32, y: 32, z: 32 }) {
    this.size = size
    this.blocks = new Uint8Array(size.x * size.y * size.z)
  }

  getBlock(x: number, y: number, z: number): BlockType {
    if (!this.inBounds(x, y, z)) return BlockType.Air
    return this.blocks[this.index(x, y, z)] as BlockType
  }

  setBlock(x: number, y: number, z: number, type: BlockType) {
    if (!this.inBounds(x, y, z)) return
    this.blocks[this.index(x, y, z)] = type
  }

  snapshotBlocks() {
    return new Uint8Array(this.blocks)
  }

  private inBounds(x: number, y: number, z: number) {
    return x >= 0 && y >= 0 && z >= 0 && x < this.size.x && y < this.size.y && z < this.size.z
  }

  private index(x: number, y: number, z: number) {
    return x + this.size.x * (z + this.size.z * y)
  }
}

export function buildChunkMesh(chunk: ChunkManager, worldScale = 1): ChunkMesh {
  const vertices: number[] = []
  const { x: sx, y: sy, z: sz } = chunk.size
  const offsetX = -sx / 2
  const offsetZ = -sz / 2

  for (let y = 0; y < sy; y++) {
    for (let z = 0; z < sz; z++) {
      for (let x = 0; x < sx; x++) {
        const block = chunk.getBlock(x, y, z)
        if (block === BlockType.Air) continue
        const palette = blockPalette[block]!
        const textureConfig = blockTextureLayers[block]

        for (let f = 0; f < 6; f++) {
          const face = faceDefs[f as FaceIndex]
          const nx = x + face.offset[0]
          const ny = y + face.offset[1]
          const nz = z + face.offset[2]
          if (chunk.getBlock(nx, ny, nz) !== BlockType.Air) continue

          const color = palette[face.colorSlot]
          const faceKey = faceIndexToKey[f as FaceIndex]
          const textureLayer = textureConfig ? textureConfig[faceKey] ?? -1 : -1
          const uvs = faceUVs[f as FaceIndex]
          const baseX = x + offsetX
          const baseY = y
          const baseZ = z + offsetZ
          for (let i = 0; i < faceIndices.length; i++) {
            const idx = faceIndices[i]!
            const corner = face.corners[idx]!
            const uv = uvs[idx]!
            vertices.push(
              (baseX + corner[0]) * worldScale,
              (baseY + corner[1]) * worldScale,
              (baseZ + corner[2]) * worldScale,
              face.normal[0],
              face.normal[1],
              face.normal[2],
              color[0],
              color[1],
              color[2],
              uv[0],
              uv[1],
              textureLayer
            )
          }
        }
      }
    }
  }

  const vertexData = new Float32Array(vertices)
  const vertexCount = vertexData.length / 12
  return { vertexData, vertexCount }
}

export function setBlockTextureIndices(block: BlockType, config: BlockTextureFaceIndices | null) {
  if (!config) {
    delete blockTextureLayers[block]
    return
  }
  blockTextureLayers[block] = { ...config }
}
