/**
 * Demo scene showcasing PBR materials, lighting, and procedural generation
 */

import type { DSLCommand } from '../dsl/commands'
import { withVersion } from '../dsl/commands'
import { BlockType } from '../core'

export function generateDemoScene(): DSLCommand[] {
  const commands: DSLCommand[] = []

  // === 1. Setup Lighting: Sunset scene ===
  commands.push(withVersion({
    type: 'set_lighting',
    params: {
      sun: {
        direction: [-0.3, -0.7, -0.6], // Low angle for sunset
        color: [1.0, 0.7, 0.5],        // Warm orange sunset color
        intensity: 1.2
      },
      sky: {
        zenithColor: [0.3, 0.4, 0.7],    // Deep blue sky
        horizonColor: [1.0, 0.6, 0.4],   // Orange horizon
        groundColor: [0.2, 0.15, 0.1],   // Dark ground reflection
        intensity: 0.6
      },
      ambient: {
        color: [0.9, 0.8, 0.7],          // Warm ambient
        intensity: 0.25
      }
    },
    source: 'demoScene'
  }))

  // === 2. Create a base platform with stone material ===
  const platformY = 30
  const platformSize = 20
  for (let x = -platformSize; x <= platformSize; x++) {
    for (let z = -platformSize; z <= platformSize; z++) {
      commands.push(withVersion({
        type: 'set_block',
        edit: {
          position: [x, platformY, z],
          blockType: BlockType.Stone
        },
        source: 'demoScene'
      }))
    }
  }

  // === 3. Set PBR materials for different block types ===

  // Stone: Rough, non-metallic
  commands.push(withVersion({
    type: 'set_material',
    blockType: BlockType.Stone,
    material: {
      albedo: [0.6, 0.6, 0.6],
      roughness: 0.9,
      metallic: 0.0,
      ao: 0.8
    },
    source: 'demoScene'
  }))

  // Grass: Natural, slightly rough
  commands.push(withVersion({
    type: 'set_material',
    blockType: BlockType.Grass,
    material: {
      albedo: [0.3, 0.6, 0.2],
      roughness: 0.85,
      metallic: 0.0,
      ao: 1.0
    },
    source: 'demoScene'
  }))

  // Plank: Wood material
  commands.push(withVersion({
    type: 'set_material',
    blockType: BlockType.Plank,
    material: {
      albedo: [0.6, 0.4, 0.2],
      roughness: 0.7,
      metallic: 0.0,
      ao: 0.9
    },
    source: 'demoScene'
  }))

  // Emissive glowing block
  commands.push(withVersion({
    type: 'set_material',
    blockType: BlockType.Air + 1, // Use an unused block type
    material: {
      albedo: [0.1, 0.1, 0.1],
      roughness: 0.3,
      metallic: 0.0,
      emissive: [1.0, 0.8, 0.3], // Warm glow
      emissiveStrength: 5.0,
      ao: 1.0
    },
    source: 'demoScene'
  }))

  // Metallic block
  commands.push(withVersion({
    type: 'set_material',
    blockType: BlockType.Air + 2,
    material: {
      albedo: [0.8, 0.8, 0.9],
      roughness: 0.2,
      metallic: 1.0,
      ao: 1.0
    },
    source: 'demoScene'
  }))

  // === 4. Generate L-system trees ===
  const treePositions = [
    [-10, platformY + 1, -10],
    [10, platformY + 1, -10],
    [-10, platformY + 1, 10],
    [10, platformY + 1, 10]
  ]

  treePositions.forEach((pos) => {
    const x = pos[0]
    const y = pos[1]
    const z = pos[2]
    commands.push(withVersion({
      type: 'generate_structure',
      generator: {
        type: 'l-system',
        region: {
          min: [x - 5, y, z - 5],
          max: [x + 5, y + 15, z + 5]
        },
        seed: Math.floor(Math.random() * 10000),
        lSystem: {
          axiom: 'F',
          rules: {
            'F': 'F[+F]F[-F][F]' // Branching tree pattern
          },
          iterations: 4,
          angle: 25,
          thickness: 2.0,
          taper: 0.85,
          blockType: BlockType.Plank,  // Trunk
          leafBlockType: BlockType.Grass, // Leaves
          leafProbability: 0.3
        }
      },
      source: 'demoScene'
    }))
  })

  // === 5. Generate cave using cellular automata ===
  commands.push(withVersion({
    type: 'generate_structure',
    generator: {
      type: 'cellular_automata',
      region: {
        min: [-15, platformY - 10, -15],
        max: [-5, platformY - 1, -5]
      },
      seed: 12345,
      cellularAutomata: {
        fillProbability: 0.45,
        birthLimit: 4,
        deathLimit: 3,
        iterations: 5,
        fillBlockType: BlockType.Stone,
        emptyBlockType: BlockType.Air
      }
    },
    source: 'demoScene'
  }))

  // === 6. Add point lights ===

  // Central torch light
  commands.push(withVersion({
    type: 'add_point_light',
    id: 'center_torch',
    light: {
      position: [0, platformY + 5, 0],
      color: [1.0, 0.7, 0.3],
      intensity: 10.0,
      radius: 20.0
    },
    source: 'demoScene'
  }))

  // Corner accent lights
  const cornerLights = [
    { id: 'corner1', pos: [-15, platformY + 3, -15], color: [0.3, 0.7, 1.0] },
    { id: 'corner2', pos: [15, platformY + 3, -15], color: [1.0, 0.3, 0.7] },
    { id: 'corner3', pos: [-15, platformY + 3, 15], color: [0.7, 1.0, 0.3] },
    { id: 'corner4', pos: [15, platformY + 3, 15], color: [1.0, 0.9, 0.3] }
  ]

  cornerLights.forEach((item) => {
    commands.push(withVersion({
      type: 'add_point_light',
      id: item.id,
      light: {
        position: [item.pos[0], item.pos[1], item.pos[2]],
        color: [item.color[0], item.color[1], item.color[2]],
        intensity: 5.0,
        radius: 15.0
      },
      source: 'demoScene'
    }))
  })

  // === 7. Add some decorative emissive blocks ===
  const emissivePositions = [
    [0, platformY + 1, 0],
    [5, platformY + 1, 0],
    [-5, platformY + 1, 0],
    [0, platformY + 1, 5],
    [0, platformY + 1, -5]
  ]

  emissivePositions.forEach((pos) => {
    const x = pos[0]
    const y = pos[1]
    const z = pos[2]
    commands.push(withVersion({
      type: 'set_block',
      edit: {
        position: [x, y, z],
        blockType: BlockType.Air + 1 // Emissive block
      },
      source: 'demoScene'
    }))
  })

  // === 8. Add metallic spheres using simple voxelization ===
  const sphereCenterX = 0
  const sphereCenterY = platformY + 8
  const sphereCenterZ = 0
  const sphereRadius = 3

  for (let dx = -sphereRadius; dx <= sphereRadius; dx++) {
    for (let dy = -sphereRadius; dy <= sphereRadius; dy++) {
      for (let dz = -sphereRadius; dz <= sphereRadius; dz++) {
        const distSq = dx * dx + dy * dy + dz * dz
        if (distSq <= sphereRadius * sphereRadius) {
          commands.push(withVersion({
            type: 'set_block',
            edit: {
              position: [sphereCenterX + dx, sphereCenterY + dy, sphereCenterZ + dz],
              blockType: BlockType.Air + 2 // Metallic block
            },
            source: 'demoScene'
          }))
        }
      }
    }
  }

  return commands
}
