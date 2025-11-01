#!/usr/bin/env node
/**
 * Terrain DSL - LLM-accessible CLI for large-scale terrain generation
 *
 * This tool provides a simple command-line interface for generating terrain
 * in grid blocks. It's designed to be easily called by LLMs for automated
 * terrain generation workflows.
 *
 * Usage:
 *   node terrain-dsl.ts <command> [args...]
 *
 * Commands:
 *   generate <cx1> <cz1> <cx2> <cz2> <amplitude> <roughness> <elevation> [seed]
 *   list
 *   stats
 *   clear
 *   help
 *
 * Example:
 *   # Generate a 10x10 chunk region (320x320 blocks)
 *   node terrain-dsl.ts generate 0 0 9 9 10 2.4 0.35 1337
 */

import { ChunkGrid } from './src/chunkGrid'
import {
  generateGridRegion,
  getRegionDimensions,
  getRegionBlockCount,
  type GridRegion,
  type GridTerrainParams
} from './src/procedural/gridTerrainGenerator'
import * as fs from 'fs'
import * as path from 'path'
import * as dotenv from 'dotenv'

// Load .env configuration
const envPath = path.join(__dirname, '.env')
if (fs.existsSync(envPath)) {
  dotenv.config({ path: envPath })
  console.error('[DSL] Loaded configuration from .env')
}

// Global chunk grid instance
const grid = new ChunkGrid({
  chunkSize: { x: 32, y: 32, z: 32 },
  maxLoadedChunks: parseInt(process.env.MAX_LOADED_CHUNKS || '256'),
  viewDistance: parseInt(process.env.VIEW_DISTANCE || '16')
})

interface CommandResult {
  success: boolean
  command: string
  data?: any
  error?: string
  stats?: any
}

/**
 * Generate terrain in a grid region
 */
function cmdGenerate(args: string[]): CommandResult {
  if (args.length < 7) {
    return {
      success: false,
      command: 'generate',
      error: 'Usage: generate <cx1> <cz1> <cx2> <cz2> <amplitude> <roughness> <elevation> [seed]'
    }
  }

  const cx1 = parseInt(args[0]!)
  const cz1 = parseInt(args[1]!)
  const cx2 = parseInt(args[2]!)
  const cz2 = parseInt(args[3]!)
  const amplitude = parseFloat(args[4]!)
  const roughness = parseFloat(args[5]!)
  const elevation = parseFloat(args[6]!)
  const seed = args[7] ? parseInt(args[7]) : undefined

  if (amplitude < 8 || amplitude > 18) {
    return {
      success: false,
      command: 'generate',
      error: `amplitude ${amplitude} out of range [8, 18]`
    }
  }
  if (roughness < 2.2 || roughness > 2.8) {
    return {
      success: false,
      command: 'generate',
      error: `roughness ${roughness} out of range [2.2, 2.8]`
    }
  }
  if (elevation < 0.35 || elevation > 0.50) {
    return {
      success: false,
      command: 'generate',
      error: `elevation ${elevation} out of range [0.35, 0.50]`
    }
  }

  const region: GridRegion = {
    minChunk: { cx: cx1, cz: cz1 },
    maxChunk: { cx: cx2, cz: cz2 }
  }

  const params: GridTerrainParams = {
    amplitude,
    roughness,
    elevation,
    seed
  }

  const startTime = Date.now()
  const dims = getRegionDimensions(region)

  try {
    generateGridRegion(grid, region, params)
    const elapsed = Date.now() - startTime

    return {
      success: true,
      command: 'generate',
      data: {
        region: {
          chunks: { min: region.minChunk, max: region.maxChunk },
          dimensions: dims
        },
        params: {
          seed: params.seed,
          amplitude: params.amplitude,
          roughness: params.roughness,
          elevation: params.elevation
        },
        elapsed_ms: elapsed
      },
      stats: grid.getStats()
    }
  } catch (error) {
    return {
      success: false,
      command: 'generate',
      error: String(error)
    }
  }
}

/**
 * List all loaded chunks
 */
function cmdList(): CommandResult {
  const chunks = grid.getLoadedChunks()

  return {
    success: true,
    command: 'list',
    data: {
      chunks: chunks.map(c => ({
        coord: c.coord,
        dirty: c.dirty,
        lastAccessed: new Date(c.lastAccessed).toISOString()
      })),
      count: chunks.length
    },
    stats: grid.getStats()
  }
}

/**
 * Get grid statistics
 */
function cmdStats(): CommandResult {
  const stats = grid.getStats()

  return {
    success: true,
    command: 'stats',
    data: stats
  }
}

/**
 * Clear all chunks
 */
function cmdClear(): CommandResult {
  grid.clear()

  return {
    success: true,
    command: 'clear',
    data: { message: 'All chunks cleared' },
    stats: grid.getStats()
  }
}

/**
 * Show help
 */
function cmdHelp(): CommandResult {
  const help = `
Terrain DSL - LLM-accessible CLI for large-scale terrain generation

COMMANDS:

  generate <cx1> <cz1> <cx2> <cz2> <amplitude> <roughness> <elevation> [seed]
    Generate terrain in a grid region from chunk (cx1,cz1) to (cx2,cz2)

    Arguments:
      cx1, cz1     - Starting chunk coordinates
      cx2, cz2     - Ending chunk coordinates
      amplitude    - Height amplitude (float, 8-18)
      roughness    - Terrain roughness/detail (float, 2.2-2.8)
      elevation    - Base elevation (float, 0.35-0.50)
      seed         - Optional: Random seed (integer)

    Example: Generate a 10x10 chunk region (320x320 blocks)
      generate 0 0 9 9 10 2.4 0.35 1337

  list
    List all currently loaded chunks

  stats
    Show grid statistics (loaded chunks, view distance, etc.)

  clear
    Clear all loaded chunks from memory

  help
    Show this help message

ENVIRONMENT VARIABLES (.env):

  MAX_LOADED_CHUNKS   - Maximum chunks to keep in memory (default: 256)
  VIEW_DISTANCE       - View distance in chunks (default: 16)

OUTPUT:

  All commands output JSON to stdout for easy LLM consumption.
  Status messages go to stderr.

EXAMPLES:

  # Generate a small region (5x5 chunks = 160x160 blocks) with gentle terrain
  node terrain-dsl.ts generate 0 0 4 4 9.0 2.2 0.36

  # Generate a large region (50x50 chunks = 1600x1600 blocks) with dramatic terrain
  node terrain-dsl.ts generate 0 0 49 49 17.0 2.7 0.48 7331

  # Generate with custom seed (same params, different layout)
  node terrain-dsl.ts generate -10 -10 10 10 12.0 2.4 0.42 42

  # List all chunks
  node terrain-dsl.ts list

  # Get statistics
  node terrain-dsl.ts stats
`

  return {
    success: true,
    command: 'help',
    data: { help }
  }
}

/**
 * Main command router
 */
function main() {
  const args = process.argv.slice(2)

  if (args.length === 0) {
    const result = cmdHelp()
    console.log(JSON.stringify(result, null, 2))
    process.exit(0)
  }

  const command = args[0]!
  const commandArgs = args.slice(1)

  let result: CommandResult

  switch (command) {
    case 'generate':
      result = cmdGenerate(commandArgs)
      break
    case 'list':
      result = cmdList()
      break
    case 'stats':
      result = cmdStats()
      break
    case 'clear':
      result = cmdClear()
      break
    case 'help':
      result = cmdHelp()
      break
    default:
      result = {
        success: false,
        command,
        error: `Unknown command: ${command}. Run 'help' for usage.`
      }
  }

  // Output JSON to stdout
  console.log(JSON.stringify(result, null, 2))

  // Exit with appropriate code
  process.exit(result.success ? 0 : 1)
}

// Run if executed directly
if (require.main === module) {
  main()
}

export { grid, cmdGenerate, cmdList, cmdStats, cmdClear, cmdHelp }
