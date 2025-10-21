"""
Map I/O utilities for converting between map.json and voxel grids.

Handles:
- Loading map.json → occupancy_logits + material_logits
- Saving grid → map.json
- Material name ↔ index mapping
"""

import json
import torch
from pathlib import Path
from typing import Tuple

from .nv_diff_render.materials import MATERIALS, material_name_to_index


def load_map_to_grid(map_path: str | Path, device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int], float]:
    """
    Load map.json into occupancy + material logits.

    Args:
        map_path: Path to map.json file
        device: Device to place tensors on

    Returns:
        occupancy_logits: (X, Y, Z) tensor - sigmoid → [0,1] occupancy probs
        material_logits: (X, Y, Z, M) tensor - softmax → material probs
        grid_size: (X, Y, Z) tuple
        world_scale: float (default 2.0)
    """
    map_path = Path(map_path)

    with open(map_path) as f:
        data = json.load(f)

    # Get grid dimensions
    dims = data['worldConfig']['dimensions']
    X, Y, Z = dims['x'], dims['y'], dims['z']
    world_scale = data['worldConfig'].get('worldScale', 2.0)

    M = len(MATERIALS)

    print(f"Loading map from {map_path}")
    print(f"  Grid size: {X}×{Y}×{Z}")
    print(f"  World scale: {world_scale}")
    print(f"  Blocks in file: {len(data['blocks'])}")

    # Initialize logits
    # Default: low occupancy (air)
    occupancy_logits = torch.full((X, Y, Z), -5.0, dtype=torch.float32, device=device)

    # Default: uniform material distribution
    material_logits = torch.zeros((X, Y, Z, M), dtype=torch.float32, device=device)

    # Fill from blocks
    blocks_loaded = 0
    for block in data['blocks']:
        x, y, z = block['position']
        mat_name = block['blockType']

        try:
            mat_idx = material_name_to_index(mat_name)
        except ValueError:
            # Unknown material, skip
            continue

        # High occupancy (sigmoid(5.0) ≈ 0.993)
        occupancy_logits[x, y, z] = 5.0

        # Strong material preference
        material_logits[x, y, z, :] = 0.0
        material_logits[x, y, z, mat_idx] = 10.0

        blocks_loaded += 1

    print(f"  Loaded {blocks_loaded} blocks into grid")

    return occupancy_logits, material_logits, (X, Y, Z), world_scale


def save_grid_to_map(
    occupancy_logits: torch.Tensor,
    material_logits: torch.Tensor,
    output_path: str | Path,
    world_scale: float = 2.0,
    threshold: float = 0.5,
    metadata: dict = None
) -> int:
    """
    Export grid to map.json format.

    Args:
        occupancy_logits: (X, Y, Z) tensor
        material_logits: (X, Y, Z, M) tensor
        output_path: Where to save map.json
        world_scale: World scale parameter
        threshold: Occupancy threshold for solid blocks (default 0.5)
        metadata: Optional metadata dict to include

    Returns:
        Number of blocks exported
    """
    output_path = Path(output_path)

    X, Y, Z = occupancy_logits.shape

    print(f"Exporting grid to {output_path}")
    print(f"  Grid size: {X}×{Y}×{Z}")
    print(f"  Occupancy threshold: {threshold}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, 'w') as f:
        f.write('{')
        f.write('"sequence": 1,')
        f.write('"worldConfig": {"dimensions": {"x": %d, "y": %d, "z": %d}, "worldScale": %s},' % (X, Y, Z, json.dumps(world_scale)))
        if metadata:
            f.write('"metadata": %s,' % json.dumps(metadata))
        f.write('"blocks": [')
        first = True
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    occ = torch.sigmoid(occupancy_logits[x, y, z]).item()
                    if occ > threshold:
                        mat_idx = int(torch.argmax(material_logits[x, y, z]).item())
                        mat_name = MATERIALS[mat_idx]
                        if mat_name != "Air":
                            if not first:
                                f.write(',')
                            json.dump({"position": [x, y, z], "blockType": mat_name}, f)
                            first = False
                            count += 1
        f.write(']}')

    print(f"  Exported {count} blocks")

    return count


def init_grid_from_primitive(
    grid_size: Tuple[int, int, int],
    primitive_type: str = 'ground_plane',
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize occupancy + material grids from architectural primitive.

    Args:
        grid_size: (X, Y, Z) dimensions
        primitive_type: 'ground_plane', 'cube', or 'empty'
        device: Device to place tensors on

    Returns:
        occupancy_logits: (X, Y, Z) tensor
        material_logits: (X, Y, Z, M) tensor
    """
    X, Y, Z = grid_size
    M = len(MATERIALS)

    # Initialize with air
    occupancy_logits = torch.full((X, Y, Z), -5.0, dtype=torch.float32, device=device)
    material_logits = torch.zeros((X, Y, Z, M), dtype=torch.float32, device=device)

    # Set Air as default material preference
    AIR_IDX = material_name_to_index("Air")
    material_logits[:, :, :, AIR_IDX] = 5.0

    if primitive_type == 'ground_plane':
        # Solid ground layer at y=0
        occupancy_logits[:, 0, :] = 3.0  # sigmoid(3) ≈ 0.95

        # Grass on top
        GRASS_IDX = material_name_to_index("Grass")
        material_logits[:, 0, :, :] = 0.0
        material_logits[:, 0, :, GRASS_IDX] = 5.0

        # Add some noise to upper layers (for exploration)
        occupancy_logits[:, 1:, :] = -3.0 + torch.randn(X, Y-1, Z, device=device) * 0.3

        print(f"Initialized ground_plane: {X}×{Y}×{Z}")

    elif primitive_type == 'cube':
        # Central cube structure
        cx, cy, cz = X // 2, 2, Z // 2
        size = min(4, X // 4, Z // 4)

        # Ground plane
        occupancy_logits[:, 0, :] = 3.0
        GRASS_IDX = material_name_to_index("Grass")
        material_logits[:, 0, :, :] = 0.0
        material_logits[:, 0, :, GRASS_IDX] = 5.0

        # Stone cube
        occupancy_logits[cx-size:cx+size, cy:cy+size*2, cz-size:cz+size] = 3.0
        STONE_IDX = material_name_to_index("Stone")
        material_logits[cx-size:cx+size, cy:cy+size*2, cz-size:cz+size, :] = 0.0
        material_logits[cx-size:cx+size, cy:cy+size*2, cz-size:cz+size, STONE_IDX] = 5.0

        print(f"Initialized cube: {X}×{Y}×{Z}, center=({cx},{cy},{cz}), size={size}")

    elif primitive_type == 'empty':
        # Already initialized to air
        print(f"Initialized empty grid: {X}×{Y}×{Z}")

    else:
        raise ValueError(f"Unknown primitive type: {primitive_type}")

    return occupancy_logits, material_logits


def get_grid_stats(occupancy_logits: torch.Tensor, material_logits: torch.Tensor) -> dict:
    """
    Get statistics about current grid state.

    Args:
        occupancy_logits: (X, Y, Z) tensor
        material_logits: (X, Y, Z, M) tensor

    Returns:
        Dict with grid statistics
    """
    occupancy_probs = torch.sigmoid(occupancy_logits)
    material_probs = torch.softmax(material_logits, dim=-1)

    # Count active voxels at different thresholds
    active_50 = (occupancy_probs > 0.5).sum().item()
    active_90 = (occupancy_probs > 0.9).sum().item()

    # Material distribution (for voxels with occupancy > 0.5)
    active_mask = occupancy_probs > 0.5
    material_dist = {}

    if active_mask.any():
        active_materials = material_probs[active_mask].sum(dim=0)
        total = active_materials.sum().item()

        for i, mat_name in enumerate(MATERIALS):
            count = active_materials[i].item()
            material_dist[mat_name] = {
                "count": int(count),
                "percentage": float(count / total * 100) if total > 0 else 0.0
            }

    return {
        "total_voxels": occupancy_logits.numel(),
        "active_voxels_50": active_50,
        "active_voxels_90": active_90,
        "occupancy_mean": float(occupancy_probs.mean().item()),
        "occupancy_std": float(occupancy_probs.std().item()),
        "material_distribution": material_dist
    }
