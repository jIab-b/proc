"""
Shared DSL (Python reference)

Defines canonical action types and minimal helpers for applying actions to a
numpy-backed voxel grid or an abstract world interface.

This mirrors dsl/ts/index.ts BlockTypeName and action shapes.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Tuple, Optional, Protocol, Dict, Any

BlockTypeName = Literal['Air','Grass','Dirt','Stone','Plank','Snow','Sand','Water']

Position = Tuple[int, int, int]

@dataclass
class PlaceBlock:
    type: Literal['place_block']
    params: Dict[str, Any]

@dataclass
class RemoveBlock:
    type: Literal['remove_block']
    params: Dict[str, Any]

DSLAction = PlaceBlock | RemoveBlock


class WorldLike(Protocol):
    def set_block(self, x: int, y: int, z: int, block_type: BlockTypeName) -> None: ...
    def clear_block(self, x: int, y: int, z: int) -> None: ...
    def in_bounds(self, x: int, y: int, z: int) -> bool: ...


def apply_actions(world: WorldLike, actions: List[DSLAction]) -> None:
    for a in actions:
        if a.type == 'place_block':
            pos = a.params.get('position')
            bt: BlockTypeName = a.params.get('blockType')
            if not isinstance(pos, (list, tuple)) or len(pos) != 3:
                continue
            x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
            if hasattr(world, 'in_bounds') and not world.in_bounds(x, y, z):
                continue
            world.set_block(x, y, z, bt)
        elif a.type == 'remove_block':
            pos = a.params.get('position')
            if not isinstance(pos, (list, tuple)) or len(pos) != 3:
                continue
            x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
            if hasattr(world, 'in_bounds') and not world.in_bounds(x, y, z):
                continue
            world.clear_block(x, y, z)


def parse_dsl(text: str) -> List[DSLAction]:
    import re
    actions: List[DSLAction] = []
    src = text or ''

    place_obj = re.compile(r"place_block\s*\(\s*\{([^}]+)\}\s*\)", re.IGNORECASE)
    remove_obj = re.compile(r"remove_block\s*\(\s*\{([^}]+)\}\s*\)", re.IGNORECASE)

    for m in place_obj.finditer(src):
        body = m.group(1)
        pos = re.search(r"position\s*:\s*\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]", body)
        typ = re.search(r"blockType\s*:\s*['\"](\w+)['\"]", body)
        custom = re.search(r"customBlockId\s*:\s*(\d+)", body)
        if not pos or not typ:
            continue
        x, y, z = int(pos.group(1)), int(pos.group(2)), int(pos.group(3))
        bt = typ.group(1)
        action: DSLAction = PlaceBlock(type='place_block', params={'position': (x, y, z), 'blockType': bt})
        if custom:
            action.params['customBlockId'] = int(custom.group(1))
        actions.append(action)

    for m in remove_obj.finditer(src):
        body = m.group(1)
        pos = re.search(r"position\s*:\s*\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]", body)
        if not pos:
            continue
        x, y, z = int(pos.group(1)), int(pos.group(2)), int(pos.group(3))
        actions.append(RemoveBlock(type='remove_block', params={'position': (x, y, z)}))

    # Minimal token form
    for raw in src.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith('place_block ') and '{' not in line:
            parts = line.split()
            if len(parts) >= 5:
                x, y, z = int(parts[1]), int(parts[2]), int(parts[3])
                bt = parts[4]
                actions.append(PlaceBlock(type='place_block', params={'position': (x, y, z), 'blockType': bt}))
        if line.startswith('remove_block ') and '{' not in line:
            parts = line.split()
            if len(parts) >= 4:
                x, y, z = int(parts[1]), int(parts[2]), int(parts[3])
                actions.append(RemoveBlock(type='remove_block', params={'position': (x, y, z)}))

    return actions

