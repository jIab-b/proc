import { ChunkManager, BlockType } from '../chunks'
import { fbmNoise2D } from './noise'

export interface SimpleWorldConfig {
  seed: number
  dimensions: { x: number; y: number; z: number }
}

export function generateRollingHills(chunk: ChunkManager, config: SimpleWorldConfig) {
  const { x: sx, y: sy, z: sz } = chunk.size

  // Validate dimensions match
  if (sx !== config.dimensions.x || sy !== config.dimensions.y || sz !== config.dimensions.z) {
    throw new Error('Chunk dimensions must match config dimensions')
  }

  const seed = config.seed
  const baseFreq = 1 / 32  // Controls rolling hill frequency

  for (let x = 0; x < sx; x++) {
    for (let z = 0; z < sz; z++) {
      // Generate rolling hills using multi-octave noise
      const elevation = fbmNoise2D(x * baseFreq, z * baseFreq, seed, 4, 2.0, 0.5)

      // Map elevation to height (0-1 normalized to height range)
      // Keep terrain in the middle range for rolling hills
      const normalizedHeight = (elevation + 1) / 2  // Convert from [-1,1] to [0,1]
      const minHeight = Math.floor(sy * 0.2)
      const maxHeight = Math.floor(sy * 0.6)
      const heightRange = maxHeight - minHeight
      const columnHeight = Math.floor(minHeight + normalizedHeight * heightRange)

      // Build column from bottom to top
      for (let y = 0; y < sy; y++) {
        if (y > columnHeight) {
          chunk.setBlock(x, y, z, BlockType.Air)
        } else if (y === columnHeight) {
          // Top layer is grass
          chunk.setBlock(x, y, z, BlockType.Grass)
        } else if (y >= columnHeight - 3) {
          // 3 layers of dirt below grass
          chunk.setBlock(x, y, z, BlockType.Dirt)
        } else {
          // Everything else is stone
          chunk.setBlock(x, y, z, BlockType.Stone)
        }
      }
    }
  }
}

export function createSimpleWorldConfig(seed: number = Date.now()): SimpleWorldConfig {
  return {
    seed,
    dimensions: { x: 128, y: 64, z: 128 }
  }
}
