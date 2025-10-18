// Shared DSL (TypeScript reference)
// Canonical action format used by both WebGPU client and Python side.

export type BlockTypeName =
  | 'Air'
  | 'Grass'
  | 'Dirt'
  | 'Stone'
  | 'Plank'
  | 'Snow'
  | 'Sand'
  | 'Water'

export type Position = [number, number, number]

export type PlaceBlock = {
  type: 'place_block'
  params: { position: Position; blockType: BlockTypeName; customBlockId?: number }
}

export type RemoveBlock = {
  type: 'remove_block'
  params: { position: Position }
}

export type DSLAction = PlaceBlock | RemoveBlock

// Parse DSL from free-form text into canonical actions.
// Supports two forms:
//   place_block({ position: [x,y,z], blockType: 'Stone' })
//   remove_block({ position: [x,y,z] })
// and a minimal token form:
//   place_block x y z BlockType
//   remove_block x y z
export function parseDSL(text: string): DSLAction[] {
  const actions: DSLAction[] = []
  const src = text || ''

  // JSON-ish object form
  const placeObj = /place_block\s*\(\s*\{([^}]+)\}\s*\)/gi
  const removeObj = /remove_block\s*\(\s*\{([^}]+)\}\s*\)/gi

  let m: RegExpExecArray | null
  while ((m = placeObj.exec(src)) !== null) {
    const body = m[1]!
    const pos = body.match(/position\s*:\s*\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]/)
    const type = body.match(/blockType\s*:\s*["'](\w+)["']/)
    const custom = body.match(/customBlockId\s*:\s*(\d+)/)
    if (!pos || !type) continue
    const position: Position = [parseInt(pos[1]!, 10), parseInt(pos[2]!, 10), parseInt(pos[3]!, 10)]
    const blockType = type[1]! as BlockTypeName
    const customBlockId = custom ? parseInt(custom[1]!, 10) : undefined
    actions.push({ type: 'place_block', params: { position, blockType, customBlockId } })
  }

  while ((m = removeObj.exec(src)) !== null) {
    const body = m[1]!
    const pos = body.match(/position\s*:\s*\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]/)
    if (!pos) continue
    const position: Position = [parseInt(pos[1]!, 10), parseInt(pos[2]!, 10), parseInt(pos[3]!, 10)]
    actions.push({ type: 'remove_block', params: { position } })
  }

  // Minimal token form
  for (const rawLine of src.split(/\r?\n/)) {
    const line = rawLine.trim()
    if (!line) continue
    if (line.startsWith('place_block ') && !line.includes('{')) {
      const parts = line.split(/\s+/)
      if (parts.length >= 5) {
        const x = parseInt(parts[1]!, 10)
        const y = parseInt(parts[2]!, 10)
        const z = parseInt(parts[3]!, 10)
        const blockType = parts[4] as BlockTypeName
        if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
          actions.push({ type: 'place_block', params: { position: [x, y, z], blockType } })
        }
      }
      continue
    }
    if (line.startsWith('remove_block ') && !line.includes('{')) {
      const parts = line.split(/\s+/)
      if (parts.length >= 4) {
        const x = parseInt(parts[1]!, 10)
        const y = parseInt(parts[2]!, 10)
        const z = parseInt(parts[3]!, 10)
        if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
          actions.push({ type: 'remove_block', params: { position: [x, y, z] } })
        }
      }
      continue
    }
  }

  return actions
}

