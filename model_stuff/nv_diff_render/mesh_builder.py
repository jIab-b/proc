"""
Mesh builder for block-based geometry matching WebGPU exactly.

References:
- src/chunks.ts:234-289 (buildChunkMesh)
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Callable
from .materials import FACE_DEFS, FACE_INDICES, get_material_palette
from .utils import get_face_vertex_world, is_in_bounds, world_to_clip


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

    if neighbor_check is None:
        pos_set = set(positions)
        neighbor_check = lambda p: p in pos_set

    # First pass: count visible faces to pre-allocate tensors
    num_visible_vertices = 0
    for block_idx, (bx, by, bz) in enumerate(positions):
        for face_idx in range(6):
            face_def = FACE_DEFS[face_idx]
            offset = face_def['offset']
            neighbor_pos = (bx + offset[0], by + offset[1], bz + offset[2])

            if not neighbor_check(neighbor_pos):
                num_visible_vertices += 6  # 6 vertices per face

    # Handle empty mesh
    if num_visible_vertices == 0:
        empty_vec3 = torch.zeros((0, 3), dtype=torch.float32, device=device)
        empty_vec2 = torch.zeros((0, 2), dtype=torch.float32, device=device)
        empty_occ = torch.zeros((0, 1), dtype=torch.float32, device=device)
        return (
            empty_vec3,
            torch.zeros((0, 3), dtype=torch.int32, device=device),
            {
                'normals': empty_vec3,
                'colors': empty_vec3,
                'uvs': empty_vec2,
                'material_weights': torch.zeros((0, M), dtype=torch.float32, device=device),
                'occupancy': empty_occ
            }
        )

    colors = torch.zeros((num_visible_vertices, 3), dtype=torch.float32, device=device)
    vertices = torch.zeros((num_visible_vertices, 3), dtype=torch.float32, device=device)
    normals = torch.zeros((num_visible_vertices, 3), dtype=torch.float32, device=device)
    uvs = torch.zeros((num_visible_vertices, 2), dtype=torch.float32, device=device)

    v_idx = 0
    for block_idx, (bx, by, bz) in enumerate(positions):
        block_mat_probs = material_probs[block_idx]  # (M,)

        for face_idx in range(6):
            face_def = FACE_DEFS[face_idx]

            offset = face_def['offset']
            neighbor_pos = (bx + offset[0], by + offset[1], bz + offset[2])

            if neighbor_check(neighbor_pos):
                continue  # Neighbor is solid, cull this face

            palette_slot = face_def['palette_slot']

            face_colors = palette[:, palette_slot, :]  # (M, 3)
            weighted_color = (block_mat_probs.unsqueeze(-1) * face_colors).sum(dim=0)  # (3,)

            for tri_idx in FACE_INDICES:
                corner_idx = tri_idx
                colors[v_idx] = weighted_color
                normals[v_idx, 0] = face_def['normal'][0]
                normals[v_idx, 1] = face_def['normal'][1]
                normals[v_idx, 2] = face_def['normal'][2]
                uv = face_def['uvs'][corner_idx]
                uvs[v_idx, 0] = uv[0]
                uvs[v_idx, 1] = uv[1]
                wx, wy, wz = get_face_vertex_world(bx, by, bz, face_idx, corner_idx, grid_size, world_scale)
                vertices[v_idx, 0] = wx
                vertices[v_idx, 1] = wy
                vertices[v_idx, 2] = wz
                v_idx += 1

    # Build face indices (every 3 vertices is a triangle)
    faces = torch.arange(v_idx, dtype=torch.int32, device=device).reshape(-1, 3)

    attributes = {
        'normals': normals,
        'colors': colors,
        'uvs': uvs
    }

    return vertices, faces, attributes


def build_mesh_from_grid(
    block_grid: torch.Tensor,
    material_logits: torch.Tensor,
    camera_view: torch.Tensor,
    camera_proj: torch.Tensor,
    world_scale: float = 2.0,
    temperature: float = 1.0,
    hard_assignment: bool = False,
    occupancy_probs: Optional[torch.Tensor] = None
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
        Same as build_block_mesh(), plus an 'occupancy' attribute carrying per-vertex
        occupancy weights for differentiable alpha blending.
    """
    device = block_grid.device
    X, Y, Z = block_grid.shape
    sx, sy, sz = X, Y, Z
    grid_size = (X, Y, Z)

    if not torch.any(block_grid):
        M = material_logits.shape[-1]
        empty_vec3 = torch.zeros((0, 3), dtype=torch.float32, device=device)
        empty_vec2 = torch.zeros((0, 2), dtype=torch.float32, device=device)
        empty_occ = torch.zeros((0, 1), dtype=torch.float32, device=device)
        return (
            empty_vec3,
            torch.zeros((0, 3), dtype=torch.int32, device=device),
            {
                'normals': empty_vec3,
                'colors': empty_vec3,
                'uvs': empty_vec2,
                'material_weights': torch.zeros((0, M), dtype=torch.float32, device=device),
                'occupancy': empty_occ
            }
        )

    if hard_assignment:
        mat_probs_full = F.gumbel_softmax(material_logits, tau=temperature, hard=True, dim=-1)
    else:
        mat_probs_full = F.softmax(material_logits / temperature, dim=-1)
    palette = get_material_palette().to(device)

    if occupancy_probs is not None:
        occupancy_probs = occupancy_probs.to(device=device, dtype=torch.float32)
    else:
        occupancy_probs = torch.ones_like(block_grid, dtype=torch.float32, device=device)

    # Block AABB frustum culling (conservative)
    occ_idx = torch.nonzero(block_grid, as_tuple=False)
    if occ_idx.shape[0] > 0:
        offset_x = -sx / 2.0
        offset_y = 0.0
        offset_z = -sz / 2.0
        bx = occ_idx[:, 0].float()
        by = occ_idx[:, 1].float()
        bz = occ_idx[:, 2].float()
        base = torch.stack([
            (bx + offset_x) * world_scale,
            (by + offset_y) * world_scale,
            (bz + offset_z) * world_scale
        ], dim=1)
        corner_offsets = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ], dtype=torch.float32, device=device)
        corners_world = base[:, None, :] + corner_offsets[None, :, :] * world_scale
        corners_flat = corners_world.reshape(-1, 3)
        clip = world_to_clip(corners_flat, camera_view.to(device), camera_proj.to(device))
        w = clip[:, 3].clamp(min=1e-6)
        ndc = (clip[:, :3] / w.unsqueeze(1)).reshape(-1, 8, 3)
        ndc_min = ndc.min(dim=1).values
        ndc_max = ndc.max(dim=1).values
        culled = (ndc_max[:, 0] < -1.0) | (ndc_min[:, 0] > 1.0) | \
                 (ndc_max[:, 1] < -1.0) | (ndc_min[:, 1] > 1.0) | \
                 (ndc_max[:, 2] < -1.0) | (ndc_min[:, 2] > 1.0)
        inside_mask = ~culled
        if inside_mask.any():
            pruned_grid = torch.zeros_like(block_grid)
            kept = occ_idx[inside_mask]
            pruned_grid[kept[:, 0], kept[:, 1], kept[:, 2]] = True
            block_grid = pruned_grid
        else:
            empty_vec3 = torch.zeros((0, 3), dtype=torch.float32, device=device)
            empty_vec2 = torch.zeros((0, 2), dtype=torch.float32, device=device)
            empty_occ = torch.zeros((0, 1), dtype=torch.float32, device=device)
            return (
                empty_vec3,
                torch.zeros((0, 3), dtype=torch.int32, device=device),
                {
                    'normals': empty_vec3,
                    'colors': empty_vec3,
                    'uvs': empty_vec2,
                    'occupancy': empty_occ
                }
            )

    mask_px = torch.zeros_like(block_grid, dtype=torch.bool)
    mask_px[:-1, :, :] = block_grid[:-1, :, :] & ~block_grid[1:, :, :]
    mask_px[-1, :, :] = block_grid[-1, :, :]

    mask_nx = torch.zeros_like(block_grid, dtype=torch.bool)
    mask_nx[1:, :, :] = block_grid[1:, :, :] & ~block_grid[:-1, :, :]
    mask_nx[0, :, :] = block_grid[0, :, :]

    mask_py = torch.zeros_like(block_grid, dtype=torch.bool)
    mask_py[:, :-1, :] = block_grid[:, :-1, :] & ~block_grid[:, 1:, :]
    mask_py[:, -1, :] = block_grid[:, -1, :]

    mask_ny = torch.zeros_like(block_grid, dtype=torch.bool)
    mask_ny[:, 1:, :] = block_grid[:, 1:, :] & ~block_grid[:, :-1, :]
    mask_ny[:, 0, :] = block_grid[:, 0, :]

    mask_pz = torch.zeros_like(block_grid, dtype=torch.bool)
    mask_pz[:, :, :-1] = block_grid[:, :, :-1] & ~block_grid[:, :, 1:]
    mask_pz[:, :, -1] = block_grid[:, :, -1]

    mask_nz = torch.zeros_like(block_grid, dtype=torch.bool)
    mask_nz[:, :, 1:] = block_grid[:, :, 1:] & ~block_grid[:, :, :-1]
    mask_nz[:, :, 0] = block_grid[:, :, 0]

    faces_out_vertices = []
    faces_out_colors = []
    faces_out_normals = []
    faces_out_uvs = []
    faces_out_occupancy = []

    def emit_for_mask(mask: torch.Tensor, face_index: int, palette_slot: int):
        idx = torch.nonzero(mask, as_tuple=False)
        if idx.shape[0] == 0:
            return
        bx = idx[:, 0].float()
        by = idx[:, 1].float()
        bz = idx[:, 2].float()
        offset = torch.tensor([-(sx / 2.0), 0.0, -(sz / 2.0)], dtype=torch.float32, device=device)
        base = torch.stack([(bx + offset[0]) * world_scale, (by + offset[1]) * world_scale, (bz + offset[2]) * world_scale], dim=1)
        corners_list = FACE_DEFS[face_index]['corners']
        order = torch.tensor(FACE_INDICES, dtype=torch.long, device=device)
        corners = torch.tensor(corners_list, dtype=torch.float32, device=device)[order]
        verts = base[:, None, :] + corners[None, :, :] * world_scale
        verts = verts.reshape(-1, 3)
        mat_probs = mat_probs_full[idx[:, 0].long(), idx[:, 1].long(), idx[:, 2].long(), :]
        face_colors = palette[:, palette_slot, :]
        weighted = mat_probs @ face_colors
        weighted = weighted[:, None, :].expand(-1, 6, -1).reshape(-1, 3)
        normal_vals = torch.tensor(FACE_DEFS[face_index]['normal'], dtype=torch.float32, device=device)
        normals = normal_vals.view(1, 3).expand(idx.shape[0] * 6, 3)
        uvs_list = FACE_DEFS[face_index]['uvs']
        uv_ordered = torch.tensor([uvs_list[i] for i in FACE_INDICES], dtype=torch.float32, device=device)
        uvs_expanded = uv_ordered.view(1, 6, 2).expand(idx.shape[0], 6, 2).reshape(-1, 2)
        occ_vals = occupancy_probs[idx[:, 0], idx[:, 1], idx[:, 2]].reshape(-1, 1)
        occ_expanded = occ_vals.repeat_interleave(6, dim=0)
        faces_out_vertices.append(verts)
        faces_out_colors.append(weighted)
        faces_out_normals.append(normals)
        faces_out_uvs.append(uvs_expanded)
        faces_out_occupancy.append(occ_expanded)

    emit_for_mask(mask_px, 0, 2)
    emit_for_mask(mask_nx, 1, 2)
    emit_for_mask(mask_py, 2, 0)
    emit_for_mask(mask_ny, 3, 1)
    emit_for_mask(mask_pz, 4, 2)
    emit_for_mask(mask_nz, 5, 2)

    if not faces_out_vertices:
        empty_vec3 = torch.zeros((0, 3), dtype=torch.float32, device=device)
        empty_vec2 = torch.zeros((0, 2), dtype=torch.float32, device=device)
        empty_occ = torch.zeros((0, 1), dtype=torch.float32, device=device)
        return (
            empty_vec3,
            torch.zeros((0, 3), dtype=torch.int32, device=device),
            {
                'normals': empty_vec3,
                'colors': empty_vec3,
                'uvs': empty_vec2,
                'occupancy': empty_occ
            }
        )

    vertices = torch.cat(faces_out_vertices, dim=0)
    colors = torch.cat(faces_out_colors, dim=0)
    normals = torch.cat(faces_out_normals, dim=0)
    uvs = torch.cat(faces_out_uvs, dim=0)
    occupancy = torch.cat(faces_out_occupancy, dim=0)

    faces = torch.arange(vertices.shape[0], dtype=torch.int32, device=device).reshape(-1, 3)
    attributes = {
        'normals': normals,
        'colors': colors,
        'uvs': uvs,
        'occupancy': occupancy
    }
    return vertices, faces, attributes
