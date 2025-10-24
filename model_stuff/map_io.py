"""Minimal map I/O for material-only voxel optimisation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F

from .nv_diff_render.materials import MATERIALS, material_name_to_index


def load_map_to_grid(
    map_path: str | Path,
    device: torch.device | str = "cuda",
) -> Tuple[torch.Tensor, Tuple[int, int, int], float]:
    """Load `map.json` into material logits aligned with the voxel grid."""
    map_path = Path(map_path)
    with map_path.open("r") as f:
        data = json.load(f)

    dims = data["worldConfig"]["dimensions"]
    grid_size = (int(dims["x"]), int(dims["y"]), int(dims["z"]))
    world_scale = float(data["worldConfig"].get("worldScale", 2.0))
    device = torch.device(device)

    num_materials = len(MATERIALS)
    air_idx = MATERIALS.index("Air")

    material_logits = torch.zeros((*grid_size, num_materials), dtype=torch.float32, device=device)
    material_logits[..., air_idx] = 5.0  # favour sky for empty voxels

    blocks_loaded = 0
    for block in data.get("blocks", []):
        x, y, z = (int(v) for v in block["position"])
        try:
            mat_idx = material_name_to_index(block["blockType"])
        except ValueError:
            continue
        material_logits[x, y, z, :] = 0.0
        material_logits[x, y, z, mat_idx] = 5.0
        blocks_loaded += 1

    print(f"Loading map from {map_path}")
    print(f"  Grid size: {grid_size[0]}×{grid_size[1]}×{grid_size[2]}")
    print(f"  World scale: {world_scale}")
    print(f"  Blocks in file: {blocks_loaded}")

    return material_logits, grid_size, world_scale


def save_grid_to_map(
    material_logits: torch.Tensor,
    output_path: str | Path,
    world_scale: float = 2.0,
    threshold: float = 0.5,
    metadata: dict | None = None,
) -> int:
    """Write material logits back to `map.json`, dropping voxels ≈ sky."""
    output_path = Path(output_path)
    grid_size = material_logits.shape[:3]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    probs = F.softmax(material_logits, dim=-1)
    air_idx = MATERIALS.index("Air")
    occ = 1.0 - probs[..., air_idx]

    print(f"Exporting grid to {output_path}")
    print(f"  Grid size: {grid_size[0]}×{grid_size[1]}×{grid_size[2]}")
    print(f"  Occupancy threshold: {threshold}")

    count = 0
    with output_path.open("w") as f:
        f.write("{")
        f.write(
            '"sequence": 1,'
            '"worldConfig": {"dimensions": {"x": %d, "y": %d, "z": %d}, "worldScale": %s},'
            % (*grid_size, json.dumps(world_scale))
        )
        if metadata:
            f.write('"metadata": %s,' % json.dumps(metadata))
        f.write('"blocks": [')
        first = True
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                for z in range(grid_size[2]):
                    if occ[x, y, z].item() <= threshold:
                        continue
                    mat_idx = int(torch.argmax(probs[x, y, z]).item())
                    mat_name = MATERIALS[mat_idx]
                    if mat_name == "Air":
                        continue
                    if not first:
                        f.write(',')
                    json.dump({"position": [x, y, z], "blockType": mat_name}, f)
                    first = False
                    count += 1
        f.write(']}')

    print(f"  Exported {count} blocks")
    return count
