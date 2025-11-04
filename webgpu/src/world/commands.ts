import type { TerrainGenerateParams, Vec3 } from '../core'
import { BlockType } from '../core'
import type { DSLCommand, VoxelEdit, StructureGenerator } from '../dsl/commands'
import { createCommandEmitter } from '../dsl/commandEmitter'

export type { VoxelEdit }

export type WorldCommand =
  | { type: 'set_block'; edit: VoxelEdit; source?: string }
  | { type: 'set_blocks'; edits: VoxelEdit[]; source?: string }
  | { type: 'load_snapshot'; blocks: VoxelEdit[]; clear?: boolean; worldScale?: number; source?: string }
  | { type: 'clear_all'; source?: string }
  | { type: 'terrain_region'; params: TerrainGenerateParams; source?: string }
  | { type: 'generate_structure'; generator: StructureGenerator; source?: string }

export type ParseResult = { commands: WorldCommand[]; errors: string[] }

const TOKEN_SPLIT = /\s+/

function parseTokenAsInt(token: string, name: string, errors: string[]) {
  const value = Number.parseInt(token, 10)
  if (Number.isNaN(value)) {
    errors.push(`Expected integer for ${name}, got "${token}"`)
  }
  return value
}

function parseTokenAsFloat(token: string, name: string, errors: string[]) {
  const value = Number.parseFloat(token)
  if (Number.isNaN(value)) {
    errors.push(`Expected number for ${name}, got "${token}"`)
  }
  return value
}

export function parseWorldCommands(input: string): ParseResult {
  const commands: WorldCommand[] = []
  const errors: string[] = []
  const text = input.trim()
  if (!text) return { commands, errors }

  const tryJson = () => {
    try {
      const parsed = JSON.parse(text)
      const array = Array.isArray(parsed) ? parsed : [parsed]
      array.forEach((cmd) => {
        const normalized = normalizeCommand(cmd)
        if (normalized) commands.push(normalized)
      })
    } catch (err) {
      errors.push(`Failed to parse JSON DSL: ${err instanceof Error ? err.message : String(err)}`)
    }
  }

  const lines = text.split('\n').map(line => line.trim()).filter(Boolean)
  const looksJson = text.startsWith('{') || text.startsWith('[')

  if (looksJson) {
    tryJson()
  } else {
    for (const line of lines) {
      const [head, ...rest] = line.split(TOKEN_SPLIT).filter(Boolean)
      if (!head) continue
      switch (head.toLowerCase()) {
        case 'set': {
          if (rest.length < 4) {
            errors.push(`set command requires 4 arguments, got "${line}"`)
            break
          }
          const [xToken, yToken, zToken, typeToken] = rest
          const position: Vec3 = [
            parseTokenAsInt(xToken!, 'x', errors),
            parseTokenAsInt(yToken!, 'y', errors),
            parseTokenAsInt(zToken!, 'z', errors)
          ]
          const block = parseBlock(typeToken!, errors)
          commands.push({
            type: 'set_block',
            edit: { position, blockType: block }
          })
          break
        }
        case 'remove': {
          if (rest.length < 3) {
            errors.push(`remove command requires 3 arguments, got "${line}"`)
            break
          }
          const [xToken, yToken, zToken] = rest
          const position: Vec3 = [
            parseTokenAsInt(xToken!, 'x', errors),
            parseTokenAsInt(yToken!, 'y', errors),
            parseTokenAsInt(zToken!, 'z', errors)
          ]
          commands.push({
            type: 'set_block',
            edit: { position, blockType: BlockType.Air }
          })
          break
        }
        case 'terrain': {
          if (rest.length < 10) {
            errors.push(`terrain command requires 10+ arguments, got "${line}"`)
            break
          }
          const [
            actionToken,
            minX, minY, minZ,
            maxX, maxY, maxZ,
            amplitudeToken,
            roughnessToken,
            elevationToken,
            seedToken,
            selectionTypeToken,
            ...ellipsoidTokens
          ] = rest

          const params: TerrainGenerateParams = {
            action: actionToken as TerrainGenerateParams['action'],
            region: {
              min: [
                parseTokenAsFloat(minX!, 'minX', errors),
                parseTokenAsFloat(minY!, 'minY', errors),
                parseTokenAsFloat(minZ!, 'minZ', errors)
              ],
              max: [
                parseTokenAsFloat(maxX!, 'maxX', errors),
                parseTokenAsFloat(maxY!, 'maxY', errors),
                parseTokenAsFloat(maxZ!, 'maxZ', errors)
              ]
            },
            selectionType: (selectionTypeToken as TerrainGenerateParams['selectionType']) ?? 'default',
            params: {
              amplitude: parseTokenAsFloat(amplitudeToken!, 'amplitude', errors),
              roughness: parseTokenAsFloat(roughnessToken!, 'roughness', errors),
              elevation: parseTokenAsFloat(elevationToken!, 'elevation', errors),
              seed: seedToken ? parseTokenAsInt(seedToken, 'seed', errors) : undefined
            }
          }

          if (ellipsoidTokens[0]?.toLowerCase() === 'ellipsoid' && ellipsoidTokens.length >= 6) {
            params.ellipsoidMask = {
              center: [
                parseTokenAsFloat(ellipsoidTokens[1]!, 'ellipsoidCenterX', errors),
                parseTokenAsFloat(ellipsoidTokens[2]!, 'ellipsoidCenterY', errors),
                parseTokenAsFloat(ellipsoidTokens[3]!, 'ellipsoidCenterZ', errors)
              ],
              radiusX: parseTokenAsFloat(ellipsoidTokens[4]!, 'radiusX', errors),
              radiusY: parseTokenAsFloat(ellipsoidTokens[5]!, 'radiusY', errors),
              radiusZ: parseTokenAsFloat(ellipsoidTokens[6]!, 'radiusZ', errors)
            }
          }

          commands.push({
            type: 'terrain_region',
            params
          })
          break
        }
        default:
          errors.push(`Unknown command "${head}"`)
      }
    }
  }

  return { commands, errors }
}

export function formatWorldCommand(command: WorldCommand): string {
  switch (command.type) {
    case 'set_block': {
      const { position, blockType } = command.edit
      return `set ${position.join(' ')} ${blockType}`
    }
    case 'set_blocks':
      return JSON.stringify(command)
    case 'load_snapshot':
      return JSON.stringify(command)
    case 'clear_all':
      return 'clear_all'
    case 'terrain_region': {
      const { params } = command
      const base = [
        'terrain',
        params.action,
        ...params.region.min,
        ...params.region.max,
        params.params.amplitude,
        params.params.roughness,
        params.params.elevation
      ]
      if (params.params.seed !== undefined) {
        base.push(params.params.seed)
      }
      if (params.selectionType !== 'default') {
        base.push(params.selectionType)
      }
      if (params.ellipsoidMask) {
        base.push(
          'ellipsoid',
          ...params.ellipsoidMask.center,
          params.ellipsoidMask.radiusX,
          params.ellipsoidMask.radiusY,
          params.ellipsoidMask.radiusZ
        )
      }
      return base.join(' ')
    }
    default:
      return JSON.stringify(command)
  }
}

function parseBlock(token: string, errors: string[]) {
  if (!Number.isNaN(Number(token))) {
    return Number(token)
  }
  const key = token.toUpperCase() as keyof typeof BlockType
  if (BlockType[key] !== undefined) {
    return BlockType[key]
  }
  errors.push(`Unknown block type "${token}"`)
  return BlockType.Air
}

function normalizeCommand(input: any): WorldCommand | null {
  if (!input || typeof input !== 'object') {
    return null
  }

  if (typeof input.v === 'number') {
    const cmd = input as DSLCommand
    switch (cmd.type) {
      case 'set_block':
        return { type: 'set_block', edit: cmd.edit, source: cmd.source }
      case 'set_blocks':
        return { type: 'set_blocks', edits: cmd.edits, source: cmd.source }
      case 'clear_all':
        return { type: 'clear_all', source: cmd.source }
      case 'load_snapshot':
        return {
          type: 'load_snapshot',
          blocks: cmd.blocks,
          worldScale: cmd.worldScale,
          clear: cmd.clear,
          source: cmd.source
        }
      case 'terrain_region':
        return { type: 'terrain_region', params: cmd.params, source: cmd.source }
      default:
        return null
    }
  }

  if (input.type === 'set_block') {
    return {
      type: 'set_block',
      edit: {
        position: [...input.edit.position] as Vec3,
        blockType: Number(input.edit.blockType)
      },
      source: input.source
    }
  }
  if (input.type === 'set_blocks') {
    return {
      type: 'set_blocks',
      edits: input.edits.map((e: any) => ({
        position: [...e.position] as Vec3,
        blockType: Number(e.blockType)
      })),
      source: input.source
    }
  }
  if (input.type === 'clear_all') {
    return { type: 'clear_all', source: input.source }
  }
  if (input.type === 'load_snapshot') {
    return {
      type: 'load_snapshot',
      clear: Boolean(input.clear),
      worldScale: typeof input.worldScale === 'number' ? input.worldScale : undefined,
      blocks: Array.isArray(input.blocks)
        ? input.blocks.map((b: any) => ({
            position: [...b.position] as Vec3,
            blockType: Number(b.blockType)
          }))
        : [],
      source: input.source
    }
  }
  if (input.type === 'terrain_region') {
    return {
      type: 'terrain_region',
      params: input.params,
      source: input.source
    }
  }
  return null
}
