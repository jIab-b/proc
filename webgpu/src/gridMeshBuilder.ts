// Grid Mesh Builder - Builds combined meshes for multiple chunks
// Optimized for rendering large terrain grids efficiently

import { ChunkGrid } from './chunkGrid'
import { buildChunkMesh, BlockType, type ChunkManager } from './core'

export interface GridMesh {
  vertexData: Float32Array
  vertexCount: number
  chunkCount: number
}

/**
 * Build a combined mesh for all visible chunks in the grid
 * This merges multiple chunk meshes into a single vertex buffer
 */
export function buildGridMesh(grid: ChunkGrid, worldScale = 1): GridMesh {
  const visibleChunks = grid.getVisibleChunks()

  if (visibleChunks.length === 0) {
    return {
      vertexData: new Float32Array(0),
      vertexCount: 0,
      chunkCount: 0
    }
  }

  // Build mesh for each chunk and collect vertices
  const allVertices: number[] = []
  let totalVertexCount = 0

  for (const chunkData of visibleChunks) {
    const { cx, cz } = chunkData.coord
    const chunk = chunkData.chunk

    // Build mesh for this chunk
    const mesh = buildChunkMesh(chunk, worldScale)

    if (mesh.vertexCount === 0) continue

    // Calculate world offset for this chunk
    const chunkSize = chunk.size.x
    const offsetX = cx * chunkSize * worldScale
    const offsetZ = cz * chunkSize * worldScale

    // Add vertices with offset applied
    // Each vertex has 12 floats: pos(3) + normal(3) + color(3) + uv(2) + texLayer(1)
    for (let i = 0; i < mesh.vertexData.length; i += 12) {
      // Position (offset by chunk position)
      allVertices.push(mesh.vertexData[i]! + offsetX)     // x
      allVertices.push(mesh.vertexData[i + 1]!)           // y (no offset)
      allVertices.push(mesh.vertexData[i + 2]! + offsetZ) // z

      // Normal, color, uv, texLayer (copy as-is)
      for (let j = 3; j < 12; j++) {
        allVertices.push(mesh.vertexData[i + j]!)
      }
    }

    totalVertexCount += mesh.vertexCount
  }

  return {
    vertexData: new Float32Array(allVertices),
    vertexCount: totalVertexCount,
    chunkCount: visibleChunks.length
  }
}

/**
 * Build meshes for only dirty chunks and return them separately
 * Useful for incremental updates
 */
export function buildDirtyChunks(grid: ChunkGrid, worldScale = 1): Map<string, GridMesh> {
  const dirtyMeshes = new Map<string, GridMesh>()
  const allChunks = grid.getLoadedChunks()

  for (const chunkData of allChunks) {
    if (!chunkData.dirty) continue

    const { cx, cz } = chunkData.coord
    const chunk = chunkData.chunk

    // Build mesh for this chunk
    const mesh = buildChunkMesh(chunk, worldScale)

    if (mesh.vertexCount === 0) continue

    // Calculate world offset for this chunk
    const chunkSize = chunk.size.x
    const offsetX = cx * chunkSize * worldScale
    const offsetZ = cz * chunkSize * worldScale

    // Add vertices with offset applied
    const vertices: number[] = []
    for (let i = 0; i < mesh.vertexData.length; i += 12) {
      // Position (offset by chunk position)
      vertices.push(mesh.vertexData[i]! + offsetX)     // x
      vertices.push(mesh.vertexData[i + 1]!)           // y
      vertices.push(mesh.vertexData[i + 2]! + offsetZ) // z

      // Normal, color, uv, texLayer
      for (let j = 3; j < 12; j++) {
        vertices.push(mesh.vertexData[i + j]!)
      }
    }

    const key = `${cx}_${cz}`
    dirtyMeshes.set(key, {
      vertexData: new Float32Array(vertices),
      vertexCount: mesh.vertexCount,
      chunkCount: 1
    })
  }

  return dirtyMeshes
}

/**
 * Get bounds of all visible chunks (for camera culling)
 */
export function getVisibleBounds(grid: ChunkGrid, chunkSize = 32, worldScale = 1) {
  const visibleChunks = grid.getVisibleChunks()

  if (visibleChunks.length === 0) {
    return {
      min: [0, 0, 0] as [number, number, number],
      max: [0, 0, 0] as [number, number, number]
    }
  }

  let minX = Infinity
  let maxX = -Infinity
  let minZ = Infinity
  let maxZ = -Infinity

  for (const chunkData of visibleChunks) {
    const { cx, cz } = chunkData.coord
    const x1 = cx * chunkSize * worldScale
    const x2 = (cx + 1) * chunkSize * worldScale
    const z1 = cz * chunkSize * worldScale
    const z2 = (cz + 1) * chunkSize * worldScale

    minX = Math.min(minX, x1)
    maxX = Math.max(maxX, x2)
    minZ = Math.min(minZ, z1)
    maxZ = Math.max(maxZ, z2)
  }

  return {
    min: [minX, 0, minZ] as [number, number, number],
    max: [maxX, chunkSize * worldScale, maxZ] as [number, number, number]
  }
}
