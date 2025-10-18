"""
Utility functions for coordinate transforms and camera handling.

References:
- src/chunks.ts:237-238 (chunk origin offset)
- src/camera.ts:4-54 (camera matrices)
- src/webgpuEngine.ts:101-117 (projection)
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any


def block_to_world(
    x: int,
    y: int,
    z: int,
    grid_size: Tuple[int, int, int],
    world_scale: float = 2.0
) -> Tuple[float, float, float]:
    """
    Convert block grid coordinates to world coordinates.

    Matches chunks.ts:237-238, 267-269, 318.

    Args:
        x, y, z: Block grid coordinates
        grid_size: (sx, sy, sz) dimensions
        world_scale: Scale multiplier (default: 2.0)

    Returns:
        (wx, wy, wz) world coordinates
    """
    sx, sy, sz = grid_size

    # Chunk origin offset (chunks.ts:318)
    offset_x = -sx / 2.0
    offset_z = -sz / 2.0
    offset_y = 0.0

    # World position (chunks.ts:267-269)
    wx = (x + offset_x) * world_scale
    wy = (y + offset_y) * world_scale
    wz = (z + offset_z) * world_scale

    return (wx, wy, wz)


def get_face_vertex_world(
    block_x: int,
    block_y: int,
    block_z: int,
    face_index: int,
    corner_index: int,
    grid_size: Tuple[int, int, int],
    world_scale: float = 2.0
) -> Tuple[float, float, float]:
    """
    Get world-space vertex position for a block face corner.

    Args:
        block_x, block_y, block_z: Block grid position
        face_index: 0-5 (PX, NX, PY, NY, PZ, NZ)
        corner_index: 0-3 (quad corner)
        grid_size: Grid dimensions
        world_scale: Scale multiplier

    Returns:
        (x, y, z) world position
    """
    from .materials import FACE_DEFS

    # Get base world position
    base_x, base_y, base_z = block_to_world(block_x, block_y, block_z, grid_size, world_scale)

    # Get corner offset from face definition
    corner = FACE_DEFS[face_index]['corners'][corner_index]

    # Add corner offset scaled by world_scale
    return (
        base_x + corner[0] * world_scale,
        base_y + corner[1] * world_scale,
        base_z + corner[2] * world_scale
    )


def load_camera_matrices_from_metadata(
    metadata: Dict[str, Any],
    view_index: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load camera matrices from dataset metadata.

    WebGPU stores matrices in column-major order.
    PyTorch expects row-major, so we transpose.

    Args:
        metadata: Dict from metadata.json
        view_index: Which view to load

    Returns:
        view: (4, 4) view matrix (row-major)
        proj: (4, 4) projection matrix (row-major)
    """
    view_data = metadata['views'][view_index]

    # Load from column-major (WebGPU) and transpose to row-major (PyTorch)
    view_col = np.array(view_data['viewMatrix']).reshape(4, 4)
    proj_col = np.array(view_data['projectionMatrix']).reshape(4, 4)

    view = torch.from_numpy(view_col.T).float()
    proj = torch.from_numpy(proj_col.T).float()

    return view, proj


def create_perspective_matrix(
    fov_y_rad: float,
    aspect: float,
    near: float,
    far: float
) -> torch.Tensor:
    """
    Create perspective projection matrix matching camera.ts:4-14.

    OpenGL convention: Z in [-1, 1], column-major output.

    Args:
        fov_y_rad: Vertical field of view in radians
        aspect: Width / height
        near: Near clipping plane
        far: Far clipping plane

    Returns:
        (4, 4) projection matrix (row-major for PyTorch)
    """
    f = 1.0 / np.tan(fov_y_rad / 2.0)
    nf = 1.0 / (near - far)

    # Build in column-major order (matching WebGPU)
    mat_col = np.zeros((4, 4), dtype=np.float32)
    mat_col[0, 0] = f / aspect  # [0,0]
    mat_col[1, 1] = f           # [1,1]
    mat_col[2, 2] = (far + near) * nf      # [2,2]
    mat_col[2, 3] = -1.0                   # [2,3]
    mat_col[3, 2] = (2.0 * far * near) * nf  # [3,2]

    # Transpose to row-major for PyTorch
    return torch.from_numpy(mat_col.T)


def create_look_at_matrix(
    eye: Tuple[float, float, float],
    center: Tuple[float, float, float],
    up: Tuple[float, float, float]
) -> torch.Tensor:
    """
    Create view matrix matching camera.ts:16-39.

    Args:
        eye: Camera position
        center: Look-at target
        up: Up vector

    Returns:
        (4, 4) view matrix (row-major for PyTorch)
    """
    eye = np.array(eye, dtype=np.float32)
    center = np.array(center, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    # Forward vector (camera points in -Z)
    forward = eye - center
    forward_len = np.linalg.norm(forward)
    if forward_len < 1e-6:
        forward = np.array([0.0, 0.0, 1.0])
    else:
        forward = forward / forward_len

    # Right vector
    right = np.cross(up, forward)
    right_len = np.linalg.norm(right)
    if right_len < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / right_len

    # Recompute up
    up_vec = np.cross(forward, right)

    # Build view matrix (column-major)
    mat_col = np.zeros((4, 4), dtype=np.float32)
    mat_col[0, 0:3] = right
    mat_col[1, 0:3] = up_vec
    mat_col[2, 0:3] = forward
    mat_col[0, 3] = -np.dot(right, eye)
    mat_col[1, 3] = -np.dot(up_vec, eye)
    mat_col[2, 3] = -np.dot(forward, eye)
    mat_col[3, 3] = 1.0

    # Transpose to row-major
    return torch.from_numpy(mat_col.T)


def world_to_clip(
    positions: torch.Tensor,
    view_matrix: torch.Tensor,
    proj_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Transform world-space positions to clip space.

    Args:
        positions: (N, 3) world positions
        view_matrix: (4, 4) view matrix
        proj_matrix: (4, 4) projection matrix

    Returns:
        (N, 4) homogeneous clip-space coordinates
    """
    # Add homogeneous coordinate
    ones = torch.ones((positions.shape[0], 1), dtype=positions.dtype, device=positions.device)
    pos_h = torch.cat([positions, ones], dim=1)  # (N, 4)

    # Transform: clip = proj @ view @ world
    view_pos = pos_h @ view_matrix.T  # (N, 4)
    clip_pos = view_pos @ proj_matrix.T  # (N, 4)

    return clip_pos


def clip_to_ndc(clip_pos: torch.Tensor) -> torch.Tensor:
    """
    Convert clip-space to NDC by perspective divide.

    Args:
        clip_pos: (N, 4) homogeneous clip coordinates

    Returns:
        (N, 3) NDC coordinates in [-1, 1]
    """
    w = clip_pos[:, 3:4].clamp(min=1e-6)
    ndc = clip_pos[:, :3] / w
    return ndc


def ndc_to_screen(
    ndc: torch.Tensor,
    img_h: int,
    img_w: int
) -> torch.Tensor:
    """
    Convert NDC coordinates to screen space.

    Args:
        ndc: (N, 3) NDC coordinates
        img_h: Image height
        img_w: Image width

    Returns:
        (N, 2) screen coordinates
    """
    screen_x = (ndc[:, 0] * 0.5 + 0.5) * img_w
    screen_y = (-ndc[:, 1] * 0.5 + 0.5) * img_h  # Flip Y
    return torch.stack([screen_x, screen_y], dim=1)


def is_in_bounds(
    x: int,
    y: int,
    z: int,
    grid_size: Tuple[int, int, int]
) -> bool:
    """Check if grid position is within bounds."""
    sx, sy, sz = grid_size
    return 0 <= x < sx and 0 <= y < sy and 0 <= z < sz
