import type { ChunkManager } from '../core'
import type { CellularAutomataParams, BoundingBox, VoxelEdit } from '../dsl/commands'

function seededRandom(seed: number) {
  let state = seed
  return () => {
    state = (state * 1664525 + 1013904223) % 4294967296
    return state / 4294967296
  }
}

export function generateCellularAutomata(
  chunk: ChunkManager,
  params: CellularAutomataParams,
  region: BoundingBox,
  seed: number
): VoxelEdit[] {
  const { min, max } = region
  const sizeX = max[0] - min[0] + 1
  const sizeY = max[1] - min[1] + 1
  const sizeZ = max[2] - min[2] + 1

  // Initialize with random noise
  const rng = seededRandom(seed)
  let grid = new Uint8Array(sizeX * sizeY * sizeZ)

  for (let i = 0; i < grid.length; i++) {
    grid[i] = rng() < params.fillProbability ? 1 : 0
  }

  // Helper to get neighbor count
  const getNeighbors = (x: number, y: number, z: number): number => {
    let count = 0
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dz = -1; dz <= 1; dz++) {
          if (dx === 0 && dy === 0 && dz === 0) continue

          const nx = x + dx
          const ny = y + dy
          const nz = z + dz

          if (nx < 0 || nx >= sizeX || ny < 0 || ny >= sizeY || nz < 0 || nz >= sizeZ) {
            count++ // Treat out of bounds as filled
            continue
          }

          const idx = nx + ny * sizeX + nz * sizeX * sizeY
          if (grid[idx] === 1) count++
        }
      }
    }
    return count
  }

  // Iterate cellular automata
  for (let iter = 0; iter < params.iterations; iter++) {
    const newGrid = new Uint8Array(grid.length)

    for (let z = 0; z < sizeZ; z++) {
      for (let y = 0; y < sizeY; y++) {
        for (let x = 0; x < sizeX; x++) {
          const idx = x + y * sizeX + z * sizeX * sizeY
          const neighbors = getNeighbors(x, y, z)

          if (grid[idx] === 1) {
            // Cell is alive
            newGrid[idx] = neighbors >= params.deathLimit ? 1 : 0
          } else {
            // Cell is dead
            newGrid[idx] = neighbors >= params.birthLimit ? 1 : 0
          }
        }
      }
    }

    grid = newGrid
  }

  // Convert to voxel edits
  const edits: VoxelEdit[] = []
  for (let z = 0; z < sizeZ; z++) {
    for (let y = 0; y < sizeY; y++) {
      for (let x = 0; x < sizeX; x++) {
        const idx = x + y * sizeX + z * sizeX * sizeY
        const worldPos: [number, number, number] = [min[0] + x, min[1] + y, min[2] + z]

        const blockType = grid[idx] === 1 ? params.fillBlockType : params.emptyBlockType

        edits.push({ position: worldPos, blockType })
      }
    }
  }

  return edits
}
