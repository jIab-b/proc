import type { Vec3, ChunkManager } from '../core'
import type { LSystemParams, VoxelEdit } from '../dsl/commands'

interface LSystemState {
  position: Vec3
  direction: Vec3
  thickness: number
}

interface LSystemStack {
  state: LSystemState
}

function distance(a: Vec3, b: Vec3): number {
  const dx = b[0] - a[0]
  const dy = b[1] - a[1]
  const dz = b[2] - a[2]
  return Math.sqrt(dx * dx + dy * dy + dz * dz)
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

function rotateAroundAxis(v: Vec3, axis: Vec3, angle: number): Vec3 {
  // Rodrigues' rotation formula
  const cosA = Math.cos(angle)
  const sinA = Math.sin(angle)
  const dot = v[0] * axis[0] + v[1] * axis[1] + v[2] * axis[2]

  return [
    v[0] * cosA + axis[0] * dot * (1 - cosA) + (axis[1] * v[2] - axis[2] * v[1]) * sinA,
    v[1] * cosA + axis[1] * dot * (1 - cosA) + (axis[2] * v[0] - axis[0] * v[2]) * sinA,
    v[2] * cosA + axis[2] * dot * (1 - cosA) + (axis[0] * v[1] - axis[1] * v[0]) * sinA
  ]
}

export function generateLSystem(
  chunk: ChunkManager,
  params: LSystemParams,
  startPos: Vec3
): VoxelEdit[] {
  const edits: VoxelEdit[] = []

  // Expand axiom using rules
  let current = params.axiom
  for (let i = 0; i < params.iterations; i++) {
    let next = ''
    for (const char of current) {
      next += params.rules[char] ?? char
    }
    current = next

    // Safety: limit string length
    if (current.length > 100000) {
      console.warn('[L-System] String too long, truncating')
      break
    }
  }

  // Interpret symbols
  const stack: LSystemStack[] = []
  let state: LSystemState = {
    position: [...startPos],
    direction: [0, 1, 0], // up
    thickness: params.thickness
  }

  const angleRad = (params.angle * Math.PI) / 180

  for (const symbol of current) {
    switch (symbol) {
      case 'F': // Move forward and draw
        {
          const newPos: Vec3 = [
            state.position[0] + state.direction[0],
            state.position[1] + state.direction[1],
            state.position[2] + state.direction[2]
          ]

          // Draw line segment (voxelize)
          const steps = Math.ceil(distance(state.position, newPos))
          for (let i = 0; i <= steps; i++) {
            const t = i / steps
            const pos: Vec3 = [
              Math.floor(lerp(state.position[0], newPos[0], t)),
              Math.floor(lerp(state.position[1], newPos[1], t)),
              Math.floor(lerp(state.position[2], newPos[2], t))
            ]

            // Add thickness
            const r = Math.max(1, Math.floor(state.thickness))
            for (let dx = -r; dx <= r; dx++) {
              for (let dy = -r; dy <= r; dy++) {
                for (let dz = -r; dz <= r; dz++) {
                  if (dx * dx + dy * dy + dz * dz <= r * r) {
                    edits.push({
                      position: [pos[0] + dx, pos[1] + dy, pos[2] + dz],
                      blockType: params.blockType
                    })
                  }
                }
              }
            }

            // Maybe add leaf
            if (params.leafBlockType && Math.random() < (params.leafProbability ?? 0)) {
              edits.push({
                position: [pos[0], pos[1], pos[2]],
                blockType: params.leafBlockType
              })
            }
          }

          state.position = newPos
          state.thickness *= params.taper
        }
        break

      case '+': // Rotate right
        state.direction = rotateAroundAxis(state.direction, [0, 1, 0], angleRad)
        break

      case '-': // Rotate left
        state.direction = rotateAroundAxis(state.direction, [0, 1, 0], -angleRad)
        break

      case '[': // Push state
        stack.push({
          state: {
            ...state,
            position: [...state.position],
            direction: [...state.direction]
          }
        })
        break

      case ']': // Pop state
        const popped = stack.pop()
        if (popped) state = popped.state
        break
    }
  }

  return edits
}
