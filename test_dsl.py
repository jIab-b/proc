#!/usr/bin/env python3
"""Integration test that drives the DSL end-to-end via the OpenAI Responses API.

The script:
1. Loads prompts that exercise the voxel DSL.
2. Calls OpenAI's Responses API to obtain the DSL program.
3. Executes the program using a lightweight voxel world that mirrors the
   browser-side TypeScript implementation (bounds + block palette).
4. Persists the resulting world via the FastAPI server's `save_map` helper so the
   output lands in `maps/` just like saves triggered from the client.
5. Logs rich metadata for each run through the FastAPI logging helper and a
   developer-friendly text file (`logtest.txt`).
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx

# ---------------------------------------------------------------------------
# Environment initialisation
# ---------------------------------------------------------------------------

ENV_PATH = Path(".env")
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value

# The FastAPI app initialises directory layout + logging helpers on import.
import main as server  # noqa: E402  (import after env load)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

LOG_FILE = Path("logtest.txt")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
WORLD_BOUNDS = (64, 48, 64)  # Matches the WebGPU client's chunk size
WORLD_SCALE = 2.0
DEFAULT_SEED = 0

# Block palette lifted from the TypeScript BlockType enum
BLOCK_NAME_TO_ID: Dict[str, int] = {
    "air": 0,
    "stone": 1,
    "dirt": 2,
    "grass": 3,
    "sand": 4,
    "water": 5,
    "wood": 6,
    "leaves": 7,
    "plank": 8,
    "planks": 8,
    "oak_plank": 8,
    "oak_planks": 8,
    "coalore": 9,
    "coal_ore": 9,
    "ironore": 10,
    "iron_ore": 10,
    "diamondore": 11,
    "diamond_ore": 11,
    "whitewool": 12,
    "white_wool": 12,
}
BLOCK_ID_TO_NAME: Dict[int, str] = {
    0: "Air",
    1: "Stone",
    2: "Dirt",
    3: "Grass",
    4: "Sand",
    5: "Water",
    6: "Wood",
    7: "Leaves",
    8: "Plank",
    9: "CoalOre",
    10: "IronOre",
    11: "DiamondOre",
    12: "WhiteWool",
}

# ---------------------------------------------------------------------------
# DSL execution helpers
# ---------------------------------------------------------------------------


@dataclass
class ActionResult:
    success: bool
    detail: Dict[str, Any]


class VoxelWorld:
    """Minimal voxel world mirroring the bounds logic used by the client DSL."""

    def __init__(self, size: Tuple[int, int, int] = WORLD_BOUNDS) -> None:
        self.size_x, self.size_y, self.size_z = size
        self.blocks: Dict[Tuple[int, int, int], int] = {}
        self.execution_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Core world operations
    # ------------------------------------------------------------------

    def _in_bounds(self, position: Sequence[int]) -> bool:
        x, y, z = position
        return (
            0 <= x < self.size_x
            and 0 <= y < self.size_y
            and 0 <= z < self.size_z
        )

    def _normalise_block(self, value: Any) -> int:
        if isinstance(value, int):
            return value if value in BLOCK_ID_TO_NAME else 1
        if isinstance(value, str):
            key = value.strip().lower().replace(" ", "_")
            return BLOCK_NAME_TO_ID.get(key, 1)
        return 1

    def place_block(
        self,
        position: Sequence[int],
        block_type: Any,
        *,
        custom_block_id: Optional[int] = None,
        source: str = "llm",
    ) -> ActionResult:
        entry: Dict[str, Any] = {
            "action": "place_block",
            "params": {
                "position": list(position),
                "blockType": block_type,
                "customBlockId": custom_block_id,
                "source": source,
            },
        }

        if not self._in_bounds(position):
            result = {"success": False, "reason": "out_of_bounds"}
        else:
            block_id = self._normalise_block(block_type)
            self.blocks[tuple(position)] = block_id
            result = {
                "success": True,
                "blockType": block_id,
                "blockTypeName": BLOCK_ID_TO_NAME.get(block_id, str(block_id)),
            }
            if custom_block_id is not None:
                result["customBlockId"] = custom_block_id

        entry["result"] = result
        self.execution_log.append(entry)
        return ActionResult(result["success"], result)

    def remove_block(
        self,
        position: Sequence[int],
        *,
        source: str = "llm",
    ) -> ActionResult:
        entry: Dict[str, Any] = {
            "action": "remove_block",
            "params": {
                "position": list(position),
                "source": source,
            },
        }

        if not self._in_bounds(position):
            result = {"success": False, "reason": "out_of_bounds"}
        else:
            key = tuple(position)
            previous = self.blocks.get(key)
            if previous is None:
                result = {"success": False, "reason": "already_empty"}
            else:
                self.blocks.pop(key)
                result = {
                    "success": True,
                    "previousBlock": previous,
                    "previousBlockName": BLOCK_ID_TO_NAME.get(previous, str(previous)),
                }

        entry["result"] = result
        self.execution_log.append(entry)
        return ActionResult(result["success"], result)

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        successes = sum(1 for entry in self.execution_log if entry["result"].get("success"))
        return {
            "total_blocks": len(self.blocks),
            "execution_count": len(self.execution_log),
            "successful_actions": successes,
            "failed_actions": len(self.execution_log) - successes,
        }

    def to_map_blocks(self) -> List[Dict[str, Any]]:
        records = []
        for (x, y, z), block_id in sorted(self.blocks.items()):
            if block_id == 0:
                continue
            records.append(
                {
                    "position": [x, y, z],
                    "blockType": BLOCK_ID_TO_NAME.get(block_id, str(block_id)),
                }
            )
        return records


# ---------------------------------------------------------------------------
# DSL parsing
# ---------------------------------------------------------------------------


def parse_dsl(text: str) -> List[Dict[str, Any]]:
    """Extract DSL actions from free-form model output."""
    actions: List[Dict[str, Any]] = []
    lines = text.splitlines()

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        # Plain tokenised form: place_block x y z blockType
        if line.startswith("place_block ") and "{" not in line:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    x, y, z = map(int, parts[1:4])
                except ValueError:
                    continue
                actions.append(
                    {
                        "type": "place_block",
                        "params": {"position": [x, y, z], "blockType": parts[4]},
                    }
                )
            continue

        if line.startswith("remove_block ") and "{" not in line:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    x, y, z = map(int, parts[1:4])
                except ValueError:
                    continue
                actions.append(
                    {"type": "remove_block", "params": {"position": [x, y, z]}}
                )
            continue

        # Curly-brace payload form used by the client prompt
        if "place_block" in line and "{" in line:
            payload = _extract_payload(line, "place_block")
            if payload:
                actions.append({"type": "place_block", "params": payload})
            continue

        if "remove_block" in line and "{" in line:
            payload = _extract_payload(line, "remove_block")
            if payload:
                actions.append({"type": "remove_block", "params": payload})
            continue

    return actions


def _extract_payload(line: str, keyword: str) -> Optional[Dict[str, Any]]:
    """Pull the argument payload between braces for the given keyword."""
    start = line.lower().find(f"{keyword}(")
    if start == -1:
        return None

    brace_start = line.find("{", start)
    if brace_start == -1:
        return None

    depth = 0
    brace_end = None
    for idx in range(brace_start, len(line)):
        char = line[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                brace_end = idx
                break
    if brace_end is None:
        return None

    body = line[brace_start + 1 : brace_end]

    import re

    pos_match = re.search(r"position\s*:\s*\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]", body)
    if not pos_match:
        return None
    position = [int(pos_match.group(i)) for i in range(1, 4)]

    if keyword == "place_block":
        block_match = re.search(r"blockType\s*:\s*[\"']?(\w+)[\"']?", body)
        if not block_match:
            return None
        payload: Dict[str, Any] = {
            "position": position,
            "blockType": block_match.group(1),
        }
        custom_match = re.search(r"customBlockId\s*:\s*(\d+)", body)
        if custom_match:
            payload["customBlockId"] = int(custom_match.group(1))
        return payload

    return {"position": position}


# ---------------------------------------------------------------------------
# OpenAI Responses API helper
# ---------------------------------------------------------------------------


async def call_openai(prompt: str, system_prompt: str) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        ],
        "max_output_tokens": 4096,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            "https://api.openai.com/v1/responses", json=payload, headers=headers
        )

    if response.status_code >= 400:
        raise RuntimeError(f"OpenAI API error: {response.status_code} - {response.text}")

    data = response.json()
    output_text = data.get("output_text") or ""
    if not output_text:
        for item in data.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") in {"output_text", "text"}:
                        segment = content.get("text", "")
                        if segment:
                            output_text = f"{output_text}\n{segment}" if output_text else segment

    return {
        "model": data.get("model", OPENAI_MODEL),
        "output": output_text,
        "usage": data.get("usage"),
    }


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = (
    "You are an expert procedural voxel world designer. "
    "Generate detailed world specifications using DSL commands for a Minecraft-inspired editor. "
    "Use only the commands:\n"
    "- place_block({position: [x, y, z], blockType: \"block_name\"})\n"
    "- remove_block({position: [x, y, z]})\n"
    "Coordinates must stay within x: 0-63, y: 0-47, z: 0-63. "
    "Allowed block types: Stone, Dirt, Grass, Sand, Water, Wood, Leaves, CoalOre, IronOre, DiamondOre, WhiteWool, Plank.\n"
    "Return a clean list of commands."
)

PROMPTS = [
    ("Simple House", "Generate a 5x5 wooden house with stone foundation, dirt roof, door opening, and one window."),
    ("Mountain Peak", "Create a stone mountain up to height 10 with coal ore veins mid-way and diamond ore near the peak. Add white wool snow on top."),
    ("Desert Oasis", "Produce a desert oasis with sand terrain, a 2x2 water pool, and palm trees (wood trunk, leaves canopy)."),
    ("Underground Cave", "Carve an underground cave by removing stone, include iron and coal ore deposits, and a small subterranean water pool."),
    ("Forest Clearing", "Lay out a grass clearing surrounded by varying-height trees, plus decorative bushes or flowers using dirt/grass."),
    ("Simple Castle", "Build a rectangular castle: stone walls 4 tall, corner towers, battlements, and a gatehouse opening."),
    ("Farming Field", "Design a tilled dirt farming field bordered by wood fence, central water channel, and rows of crops using white wool."),
    ("Volcano", "Assemble a volcanic mountain of stone with a crater, simulate lava using coal ore, and scatter ash deposits."),
]


@dataclass
class TestResult:
    name: str
    status: str
    error: Optional[str] = None
    map_sequence: Optional[int] = None
    block_count: int = 0
    command_count: int = 0


class DSLTester:
    def __init__(self) -> None:
        self.log_file = LOG_FILE
        self._ensure_log_file()

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _ensure_log_file(self) -> None:
        if not self.log_file.exists():
            self.log_file.write_text("", encoding="utf-8")

    def log_response(
        self,
        test_name: str,
        prompt: str,
        response: Dict[str, Any],
        world: VoxelWorld,
        map_sequence: Optional[int],
        *,
        error: Optional[str] = None,
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        summary = world.summary()

        entry = {
            "event": "dsl_test",
            "timestamp": timestamp,
            "test": test_name,
            "prompt": prompt,
            "model": response.get("model"),
            "output": response.get("output"),
            "usage": response.get("usage"),
            "summary": summary,
            "mapSequence": map_sequence,
            "error": error,
        }
        server.write_log_entry(entry)

        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(f"TIMESTAMP: {timestamp}\n")
            fh.write(f"TEST: {test_name}\n")
            fh.write(f"MODEL: {response.get('model')}\n")
            fh.write(f"COMMANDS EXECUTED: {summary['execution_count']}\n")
            fh.write(f"BLOCKS PLACED: {summary['total_blocks']}\n")
            fh.write(f"MAP SEQUENCE: {map_sequence}\n")
            if error:
                fh.write(f"ERROR: {error}\n")
            fh.write("PROMPT:\n")
            fh.write(prompt + "\n")
            fh.write("OUTPUT:\n")
            fh.write(response.get("output", "(empty)") + "\n")
            fh.write("-" * 60 + "\n\n")

    # ------------------------------------------------------------------
    # Map persistence
    # ------------------------------------------------------------------

    async def save_map(self, test_name: str, world: VoxelWorld) -> Dict[str, Any]:
        capture_id = (
            "dsl-"
            + test_name.lower().replace(" ", "-")
            + "-"
            + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        )
        blocks = world.to_map_blocks()

        payload = server.MapSavePayload(
            sequence=None,
            captureId=capture_id,
            worldScale=WORLD_SCALE,
            worldConfig={
                "seed": DEFAULT_SEED,
                "dimensions": {
                    "x": world.size_x,
                    "y": world.size_y,
                    "z": world.size_z,
                },
            },
            blocks=blocks,
            customBlocks=[],
        )

        return await server.save_map(payload)

    # ------------------------------------------------------------------
    # Test execution
    # ------------------------------------------------------------------

    async def run_prompt(self, name: str, prompt: str) -> TestResult:
        world = VoxelWorld()
        response: Dict[str, Any]
        try:
            response = await call_openai(prompt, SYSTEM_PROMPT)
        except Exception as exc:  # noqa: BLE001
            self.log_response(
                name,
                prompt,
                {"model": OPENAI_MODEL, "output": "", "usage": None},
                world,
                map_sequence=None,
                error=str(exc),
            )
            raise

        try:
            actions = parse_dsl(response.get("output", ""))
        except Exception as exc:  # noqa: BLE001
            self.log_response(name, prompt, response, world, None, error=f"parse_error: {exc}")
            raise

        for action in actions:
            params = action.get("params", {})
            position = params.get("position", [0, 0, 0])
            if action["type"] == "place_block":
                world.place_block(
                    position,
                    params.get("blockType", "Stone"),
                    custom_block_id=params.get("customBlockId"),
                )
            elif action["type"] == "remove_block":
                world.remove_block(position)

        try:
            map_result = await self.save_map(name, world)
        except Exception as exc:  # noqa: BLE001
            self.log_response(name, prompt, response, world, None, error=f"save_map_error: {exc}")
            raise

        self.log_response(name, prompt, response, world, map_result.get("sequence"))

        summary = world.summary()
        return TestResult(
            name=name,
            status="success",
            map_sequence=map_result.get("sequence"),
            block_count=summary["total_blocks"],
            command_count=summary["execution_count"],
        )

    async def run(self) -> List[TestResult]:
        print("🧪 Starting DSL Interaction Test")
        print(f"📝 Logging responses to: {self.log_file}")
        print(f"🤖 Using OpenAI model: {OPENAI_MODEL}")
        print()

        results: List[TestResult] = []

        for idx, (name, prompt) in enumerate(PROMPTS, start=1):
            print(f"📍 Test {idx}/{len(PROMPTS)}: {name}")
            print(f"   Prompt: {prompt[:80]}...")
            try:
                result = await self.run_prompt(name, prompt)
                print(
                    f"   ✅ Success - {result.block_count} blocks across {result.command_count} commands (map #{result.map_sequence})"
                )
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                error_msg = str(exc)
                print(f"   ❌ Failed: {error_msg}")
                server.write_log_entry(
                    {
                        "event": "dsl_test_error",
                        "test": name,
                        "prompt": prompt,
                        "error": error_msg,
                    }
                )
                results.append(TestResult(name=name, status="failed", error=error_msg))
            print()
            await asyncio.sleep(1)

        success_count = sum(1 for r in results if r.status == "success")
        block_total = sum(r.block_count for r in results if r.status == "success")
        command_total = sum(r.command_count for r in results if r.status == "success")

        print("📊 Test Summary")
        print(f"   ✅ Successful: {success_count}/{len(results)}")
        print(f"   🔧 Total Commands: {command_total}")
        print(f"   🧱 Blocks Placed: {block_total}")

        return results


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not configured. Set it in .env and retry.")
        return

    tester = DSLTester()
    asyncio.run(tester.run())


if __name__ == "__main__":
    main()
