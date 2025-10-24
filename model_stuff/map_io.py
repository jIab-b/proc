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

    blocks = [block for block in data.get("blocks", []) if block.get("blockType") is not None]
    if blocks:
        positions = torch.tensor([block["position"] for block in blocks], device=device, dtype=torch.long)
        mat_names = [block["blockType"] for block in blocks]
        mat_indices = []
        for name in mat_names:
            try:
                mat_indices.append(material_name_to_index(name))
            except ValueError:
                mat_indices.append(air_idx)
        mat_idx_tensor = torch.tensor(mat_indices, device=device, dtype=torch.long)

        valid_mask = mat_idx_tensor != air_idx
        if valid_mask.any():
            pos_valid = positions[valid_mask]
            mat_valid = mat_idx_tensor[valid_mask]
            material_logits[pos_valid[:, 0], pos_valid[:, 1], pos_valid[:, 2]] = 0.0
            material_logits[pos_valid[:, 0], pos_valid[:, 1], pos_valid[:, 2]] = (
                F.one_hot(mat_valid, num_materials).to(material_logits.dtype) * 5.0
            )
        blocks_loaded = int(valid_mask.sum().item())
    else:
        blocks_loaded = 0

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

    mat_idx = torch.argmax(probs, dim=-1)
    solid_mask = (occ > threshold) & (mat_idx != air_idx)
    positions = torch.nonzero(solid_mask, as_tuple=False)
    if positions.numel() > 0:
        positions = positions.cpu()
        mat_indices = mat_idx[solid_mask].cpu().tolist()
        block_entries = [
            {"position": [int(px), int(py), int(pz)], "blockType": MATERIALS[int(mi)]}
            for (px, py, pz), mi in zip(positions.tolist(), mat_indices)
        ]
    else:
        block_entries = []

    payload = {
        "sequence": 1,
        "worldConfig": {
            "dimensions": {"x": grid_size[0], "y": grid_size[1], "z": grid_size[2]},
            "worldScale": world_scale,
        },
        "blocks": block_entries,
    }
    if metadata:
        payload["metadata"] = metadata

    with output_path.open("w") as f:
        json.dump(payload, f)

    count = len(block_entries)
    print(f"  Exported {count} blocks")
    return count
