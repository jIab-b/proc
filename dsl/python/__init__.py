"""
Shared DSL (Python reference, canonical)

Defines canonical action types (Pydantic) and minimal helpers for applying
actions to a world-like adapter. Also exposes schema export and a parser that
normalizes free-form text to canonical JSON actions.
"""
from __future__ import annotations
from typing import List, Literal, Tuple, Optional, Protocol, Dict, Any, Union
from pydantic import BaseModel, Field, validator

BlockTypeName = Literal['Air','Grass','Dirt','Stone','Plank','Snow','Sand','Water']

Position = Tuple[int, int, int]


class PlaceBlock(BaseModel):
    type: Literal['place_block'] = 'place_block'
    params: Dict[str, Any]


class RemoveBlock(BaseModel):
    type: Literal['remove_block'] = 'remove_block'
    params: Dict[str, Any]


DSLAction = Union[PlaceBlock, RemoveBlock]


class DSLWorldDimensions(BaseModel):
    x: int
    y: int
    z: int


class DSLWorldConfig(BaseModel):
    dimensions: DSLWorldDimensions
    seed: Optional[int] = None


class DSLHeader(BaseModel):
    dslVersion: str = Field('1.0', description='Canonical DSL version')
    worldConfig: Optional[DSLWorldConfig] = None
    worldScale: Optional[float] = Field(None, description='Scale between voxel units and world units')


class WorldLike(Protocol):
    def set_block(self, x: int, y: int, z: int, block_type: BlockTypeName) -> None: ...
    def clear_block(self, x: int, y: int, z: int) -> None: ...
    def in_bounds(self, x: int, y: int, z: int) -> bool: ...


def apply_actions(world: WorldLike, actions: List[Dict[str, Any] | DSLAction]) -> None:
    for a in actions:
        if isinstance(a, BaseModel):
            a = a.model_dump()
        if a.get('type') == 'place_block':
            pos = a.get('params', {}).get('position')
            bt: BlockTypeName = a.get('params', {}).get('blockType')
            if not isinstance(pos, (list, tuple)) or len(pos) != 3:
                continue
            x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
            if hasattr(world, 'in_bounds') and not world.in_bounds(x, y, z):
                continue
            world.set_block(x, y, z, bt)
        elif a.get('type') == 'remove_block':
            pos = a.get('params', {}).get('position')
            if not isinstance(pos, (list, tuple)) or len(pos) != 3:
                continue
            x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
            if hasattr(world, 'in_bounds') and not world.in_bounds(x, y, z):
                continue
            world.clear_block(x, y, z)


def parse_dsl(text: str) -> List[Dict[str, Any]]:
    import re
    actions: List[Dict[str, Any]] = []
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
        action: Dict[str, Any] = {'type': 'place_block', 'params': {'position': [x, y, z], 'blockType': bt}}
        if custom:
            action['params']['customBlockId'] = int(custom.group(1))
        actions.append(action)

    for m in remove_obj.finditer(src):
        body = m.group(1)
        pos = re.search(r"position\s*:\s*\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]", body)
        if not pos:
            continue
        x, y, z = int(pos.group(1)), int(pos.group(2)), int(pos.group(3))
        actions.append({'type': 'remove_block', 'params': {'position': [x, y, z]}})

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
                actions.append({'type': 'place_block', 'params': {'position': [x, y, z], 'blockType': bt}})
        if line.startswith('remove_block ') and '{' not in line:
            parts = line.split()
            if len(parts) >= 4:
                x, y, z = int(parts[1]), int(parts[2]), int(parts[3])
                actions.append({'type': 'remove_block', 'params': {'position': [x, y, z]}})

    return actions


def to_canonical(actions: List[Dict[str, Any]]) -> List[DSLAction]:
    """Validate and normalize a list of actions to canonical Pydantic models."""
    canon: List[DSLAction] = []
    for a in actions:
        t = a.get('type')
        if t == 'place_block':
            canon.append(PlaceBlock(**a))
        elif t == 'remove_block':
            canon.append(RemoveBlock(**a))
    return canon


def schema() -> Dict[str, Any]:
    """Return a JSON Schema for the canonical DSL envelope."""
    # Schema of a list of DSLAction, with optional header serialized separately by callers.
    # Pydantic v2: model_json_schema
    # We provide a simple envelope for convenience.
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://example.com/dsl.schema.json",
        "title": "Voxel DSL",
        "type": "object",
        "properties": {
            "dslVersion": {"type": "string", "default": "1.0"},
            "worldConfig": {
                "type": ["object", "null"],
                "properties": {
                    "dimensions": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "integer", "minimum": 1},
                            "y": {"type": "integer", "minimum": 1},
                            "z": {"type": "integer", "minimum": 1}
                        },
                        "required": ["x", "y", "z"]
                    },
                    "seed": {"type": ["integer", "null"]}
                },
                "required": ["dimensions"]
            },
            "worldScale": {"type": ["number", "null"]},
            "actions": {
                "type": "array",
                "items": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "type": {"const": "place_block"},
                                "params": {
                                    "type": "object",
                                    "properties": {
                                        "position": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3},
                                        "blockType": {"type": "string"},
                                        "customBlockId": {"type": ["integer", "null"]}
                                    },
                                    "required": ["position", "blockType"]
                                }
                            },
                            "required": ["type", "params"]
                        },
                        {
                            "type": "object",
                            "properties": {
                                "type": {"const": "remove_block"},
                                "params": {
                                    "type": "object",
                                    "properties": {
                                        "position": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3}
                                    },
                                    "required": ["position"]
                                }
                            },
                            "required": ["type", "params"]
                        }
                    ]
                }
            }
        },
        "required": ["actions"]
    }
