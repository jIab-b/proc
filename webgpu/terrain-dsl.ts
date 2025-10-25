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
 *   generate <cx1> <cz1> <cx2> <cz2> <profile> [seed] [amplitude] [roughness] [elevation]
 *   list
 *   stats
 *   clear
 *   help
 *
 * Example:
 *   # Generate a 10x10 chunk region (320x320 blocks)
 *   node terrain-dsl.ts generate 0 0 9 9 rolling_hills 1337 10 2.4 0.35
 */

import { ChunkGrid } from './src/chunkGrid'
import {
  generateGridRegion,
  getRegionDimensions,
  getRegionBlockCount,
  type GridRegion,
  type GridTerrainParams
} from './src/procedural/gridTerrainGenerator'
import type { TerrainProfile } from './src/procedural/terrainGenerator'
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
  if (args.length < 5) {
    return {
      success: false,
      command: 'generate',
      error: 'Usage: generate <cx1> <cz1> <cx2> <cz2> <profile> [seed] [amplitude] [roughness] [elevation]'
    }
  }

  const cx1 = parseInt(args[0]!)
  const cz1 = parseInt(args[1]!)
  const cx2 = parseInt(args[2]!)
  const cz2 = parseInt(args[3]!)
  const profile = args[4] as TerrainProfile

  if (!['rolling_hills', 'mountain', 'hybrid'].includes(profile)) {
    return {
      success: false,
      command: 'generate',
      error: `Invalid profile: ${profile}. Must be one of: rolling_hills, mountain, hybrid`
    }
  }

  const region: GridRegion = {
    minChunk: { cx: cx1, cz: cz1 },
    maxChunk: { cx: cx2, cz: cz2 }
  }

  const params: GridTerrainParams = {
    profile,
    seed: args[5] ? parseInt(args[5]) : undefined,
    amplitude: args[6] ? parseFloat(args[6]) : undefined,
    roughness: args[7] ? parseFloat(args[7]) : undefined,
    elevation: args[8] ? parseFloat(args[8]) : undefined
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
          profile,
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

  generate <cx1> <cz1> <cx2> <cz2> <profile> [seed] [amplitude] [roughness] [elevation]
    Generate terrain in a grid region from chunk (cx1,cz1) to (cx2,cz2)

    Arguments:
      cx1, cz1     - Starting chunk coordinates
      cx2, cz2     - Ending chunk coordinates
      profile      - Terrain profile: rolling_hills, mountain, or hybrid
      seed         - Optional: Random seed (integer)
      amplitude    - Optional: Height amplitude (float, 8-18)
      roughness    - Optional: Terrain roughness (float, 2.2-2.8)
      elevation    - Optional: Base elevation (float, 0.35-0.5)

    Example: Generate a 10x10 chunk region (320x320 blocks)
      generate 0 0 9 9 rolling_hills 1337 10 2.4 0.35

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

  # Generate a small region (5x5 chunks = 160x160 blocks)
  node terrain-dsl.ts generate 0 0 4 4 rolling_hills

  # Generate a large region (50x50 chunks = 1600x1600 blocks)
  node terrain-dsl.ts generate 0 0 49 49 mountain 7331 18 2.8 0.5

  # Generate with custom seed
  node terrain-dsl.ts generate -10 -10 10 10 hybrid 42

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
