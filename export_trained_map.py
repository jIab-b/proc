#!/usr/bin/env python3
"""Export a trained voxel map into the project's maps registry.

Steps performed:
1. Normalise ``out_local/logs`` sub-directories to simple numeric IDs (1, 2, ...).
2. Pick a run (latest by default) and load its ``map.json`` artefact.
3. Allocate a new map sequence using ``maps/registry.json`` and copy the map in.
4. Update the registry so the backend can serve the new map.
"""
from __future__ import annotations

import argparse
import json
from json import JSONDecodeError
import shutil
from pathlib import Path
from typing import List, Tuple
from uuid import uuid4

LOGS_DIR = Path("out_local/logs")
MAPS_DIR = Path("maps")
REGISTRY_PATH = MAPS_DIR / "registry.json"


def normalise_training_runs() -> List[Path]:
    """Rename training run directories to sequential numeric IDs."""
    if not LOGS_DIR.exists():
        raise FileNotFoundError(f"Logs directory not found: {LOGS_DIR}")

    run_dirs = sorted(p for p in LOGS_DIR.iterdir() if p.is_dir())
    temp_moves: List[Tuple[Path, Path]] = []

    for idx, run_dir in enumerate(run_dirs, start=1):
        desired = LOGS_DIR / str(idx)
        if run_dir == desired:
            continue
        tmp = LOGS_DIR / f"._ren_{idx}_{uuid4().hex}"
        run_dir.rename(tmp)
        temp_moves.append((tmp, desired))

    # Second pass to land at final destinations.
    for tmp, dest in temp_moves:
        if dest.exists():
            shutil.rmtree(dest)
        tmp.rename(dest)

    return sorted(p for p in LOGS_DIR.iterdir() if p.is_dir())


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REGISTRY_PATH.open("w", encoding="utf-8") as fh:
            json.dump({"next_sequence": 1}, fh, indent=2)

    with REGISTRY_PATH.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if "next_sequence" not in data or not isinstance(data["next_sequence"], int):
        raise ValueError("registry.json missing a valid 'next_sequence' integer")
    return data


def save_registry(registry: dict) -> None:
    with REGISTRY_PATH.open("w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2)


def export_map(run_id: int | None) -> Path:
    runs = normalise_training_runs()
    if not runs:
        raise RuntimeError("No training runs found under out_local/logs")

    if run_id is None:
        run_path = runs[-1]
    else:
        run_path = LOGS_DIR / str(run_id)
        if not run_path.exists():
            raise FileNotFoundError(f"Training run {run_id} not found in {LOGS_DIR}")

    map_path = run_path / "map.json"
    if not map_path.exists():
        raise FileNotFoundError(f"Map artefact not found at {map_path}")

    with map_path.open("r", encoding="utf-8") as fh:
        try:
            map_data = json.load(fh)
        except JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse map artefact at {map_path}. Ensure the training run finished "
                "and re-sync outputs from Modal."
            ) from exc

    registry = load_registry()
    sequence = registry["next_sequence"]

    # Update map sequence metadata before writing.
    map_data["sequence"] = sequence
    meta = map_data.setdefault("metadata", {})
    if not isinstance(meta, dict):
        raise ValueError("Expected 'metadata' field to be an object if present in map.json")
    meta["training_run"] = int(run_path.name)

    target_dir = MAPS_DIR / str(sequence)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_map_path = target_dir / "map.json"
    with target_map_path.open("w", encoding="utf-8") as fh:
        json.dump(map_data, fh, indent=2)

    registry["next_sequence"] = sequence + 1
    save_registry(registry)

    return target_map_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained voxel map into maps/ directory")
    parser.add_argument("--run-id", type=int, default=None, help="Numeric training run ID (defaults to latest)")
    args = parser.parse_args()

    target_map_path = export_map(args.run_id)
    print(f"Exported map to {target_map_path}")
    print(f"Updated registry: {REGISTRY_PATH}")


if __name__ == "__main__":
    main()
