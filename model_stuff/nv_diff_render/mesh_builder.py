"""
Mesh builder for block-based geometry matching WebGPU exactly.

References:
- src/chunks.ts:234-289 (buildChunkMesh)
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Callable
from .materials import FACE_DEFS, FACE_INDICES, get_material_palette
from .utils import get_face_vertex_world, is_in_bounds


def build_block_mesh(
    positions: List[Tuple[int, int, int]],
    material_logits: torch.Tensor,
    grid_size: Tuple[int, int, int],
    world_scale: float = 2.0,
    neighbor_check: Optional[Callable[[Tuple[int, int, int]], bool]] = None,
    temperature: float = 1.0,
    hard_assignment: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Build triangle mesh from block placements with differentiable materials.

    Matches chunks.ts:234-289 buildChunkMesh() exactly.

    Args:
        positions: List of (x, y, z) block grid coordinates
        material_logits: (N, M) material logits per block
        grid_size: (sx, sy, sz) world dimensions
        world_scale: Scale multiplier (default: 2.0)
        neighbor_check: Optional function(pos) -> is_solid for culling
        temperature: Softmax temperature for material mixing
        hard_assignment: Use hard one-hot if True

    Returns:
        vertices: (V, 3) world-space positions
        faces: (F, 3) triangle indices
        attributes: {
            'normals': (V, 3) face normals,
            'colors': (V, 3) base colors (material-weighted),
            'uvs': (V, 2) texture coordinates,
            'material_weights': (V, M) soft material probabilities
        }
    """
    device = material_logits.device
    N = len(positions)
    M = material_logits.shape[1]

    # Convert material logits to probabilities
    if hard_assignment:
        material_probs = F.gumbel_softmax(material_logits, tau=temperature, hard=True, dim=-1)
    else:
        material_probs = F.softmax(material_logits / temperature, dim=-1)

    # Get material palette (M, 3, 3) -> [material, face_type, RGB]
    palette = get_material_palette().to(device)

    # Build position lookup for neighbor checking
    if neighbor_check is None:
        # Default: check if position is in our block list
        pos_set = set(positions)
        neighbor_check = lambda p: p in pos_set

    # Collect all weighted colors (this is what needs gradients)
    all_weighted_colors = []
    all_mat_probs = []
    all_vertex_info = []  # (block_idx, face_idx, corner_idx) for building geometry

    # For each block
    for block_idx, (bx, by, bz) in enumerate(positions):
        block_mat_probs = material_probs[block_idx]  # (M,)

        # For each face direction
        for face_idx in range(6):
            face_def = FACE_DEFS[face_idx]

            # Check if neighbor is solid (cull if so)
            offset = face_def['offset']
            neighbor_pos = (bx + offset[0], by + offset[1], bz + offset[2])

            # Render face if neighbor is Air or out of bounds
            if neighbor_check(neighbor_pos):
                continue  # Neighbor is solid, cull this face

            # Get palette slot for this face (0=top, 1=bottom, 2=side)
            palette_slot = face_def['palette_slot']

            # Compute weighted color for this face
            # palette shape: (M, 3, 3)
            # palette[:, palette_slot, :] -> (M, 3) colors for this face type
            face_colors = palette[:, palette_slot, :]  # (M, 3)
            weighted_color = (block_mat_probs.unsqueeze(-1) * face_colors).sum(dim=0)  # (3,)

            # Store for each vertex in this face (6 vertices = 2 triangles)
            for tri_idx in FACE_INDICES:
                corner_idx = tri_idx
                all_weighted_colors.append(weighted_color)
                all_mat_probs.append(block_mat_probs)
                all_vertex_info.append((bx, by, bz, face_idx, corner_idx))

    # Handle empty mesh
    if len(all_vertex_info) == 0:
        return (
            torch.zeros((0, 3), dtype=torch.float32, device=device),
            torch.zeros((0, 3), dtype=torch.int32, device=device),
            {
                'normals': torch.zeros((0, 3), dtype=torch.float32, device=device),
                'colors': torch.zeros((0, 3), dtype=torch.float32, device=device),
                'uvs': torch.zeros((0, 2), dtype=torch.float32, device=device),
                'material_weights': torch.zeros((0, M), dtype=torch.float32, device=device)
            }
        )

    # Stack weighted colors (CRITICAL: this must maintain gradients)
    colors = torch.stack(all_weighted_colors, dim=0)  # (V, 3)
    mat_weights = torch.stack(all_mat_probs, dim=0)  # (V, M)

    # Build geometry (positions, normals, UVs) without requiring gradients
    num_vertices = len(all_vertex_info)
    vertices = torch.zeros((num_vertices, 3), dtype=torch.float32, device=device)
    normals = torch.zeros((num_vertices, 3), dtype=torch.float32, device=device)
    uvs = torch.zeros((num_vertices, 2), dtype=torch.float32, device=device)

    for v_idx, (bx, by, bz, face_idx, corner_idx) in enumerate(all_vertex_info):
        face_def = FACE_DEFS[face_idx]

        # World position
        wx, wy, wz = get_face_vertex_world(
            bx, by, bz, face_idx, corner_idx, grid_size, world_scale
        )
        vertices[v_idx, 0] = wx
        vertices[v_idx, 1] = wy
        vertices[v_idx, 2] = wz

        # Normal
        normals[v_idx, 0] = face_def['normal'][0]
        normals[v_idx, 1] = face_def['normal'][1]
        normals[v_idx, 2] = face_def['normal'][2]

        # UV
        uv = face_def['uvs'][corner_idx]
        uvs[v_idx, 0] = uv[0]
        uvs[v_idx, 1] = uv[1]

    # Build face indices (every 3 vertices is a triangle)
    faces = torch.arange(num_vertices, dtype=torch.int32, device=device).reshape(-1, 3)

    attributes = {
        'normals': normals,
        'colors': colors,
        'uvs': uvs,
        'material_weights': mat_weights
    }

    return vertices, faces, attributes


def build_mesh_from_grid(
    block_grid: torch.Tensor,
    material_logits: torch.Tensor,
    world_scale: float = 2.0,
    temperature: float = 1.0,
    hard_assignment: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Build mesh from dense voxel grid.

    Args:
        block_grid: (X, Y, Z) bool tensor (True = block exists)
        material_logits: (X, Y, Z, M) material logits per voxel
        world_scale: Scale multiplier
        temperature: Softmax temperature
        hard_assignment: Use hard one-hot

    Returns:
        Same as build_block_mesh()
    """
    device = block_grid.device
    X, Y, Z = block_grid.shape
    grid_size = (X, Y, Z)

    # Find occupied voxels
    occupied = torch.nonzero(block_grid, as_tuple=False)  # (N, 3)
    positions = [(int(x), int(y), int(z)) for x, y, z in occupied]

    if len(positions) == 0:
        M = material_logits.shape[-1]
        return (
            torch.zeros((0, 3), dtype=torch.float32, device=device),
            torch.zeros((0, 3), dtype=torch.int32, device=device),
            {
                'normals': torch.zeros((0, 3), dtype=torch.float32, device=device),
                'colors': torch.zeros((0, 3), dtype=torch.float32, device=device),
                'uvs': torch.zeros((0, 2), dtype=torch.float32, device=device),
                'material_weights': torch.zeros((0, M), dtype=torch.float32, device=device)
            }
        )

    # Extract material logits for occupied voxels
    logits = material_logits[occupied[:, 0], occupied[:, 1], occupied[:, 2]]  # (N, M)

    # Neighbor check using grid
    def neighbor_check(pos):
        x, y, z = pos
        if not is_in_bounds(x, y, z, grid_size):
            return False
        return block_grid[x, y, z].item()

    return build_block_mesh(
        positions,
        logits,
        grid_size,
        world_scale,
        neighbor_check,
        temperature,
        hard_assignment
    )
