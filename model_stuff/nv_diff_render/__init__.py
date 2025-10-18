"""
nvdiffrast-based differentiable block renderer.

High-accuracy renderer matching WebGPU output with gradient flow through
material parameters.
"""

from .materials import (
    MATERIALS,
    FACE_DEFS,
    FACE_INDICES,
    FaceIndex,
    get_material_palette,
    get_material_densities,
    material_name_to_index,
    material_index_to_name
)

from .utils import (
    block_to_world,
    get_face_vertex_world,
    load_camera_matrices_from_metadata,
    create_perspective_matrix,
    create_look_at_matrix,
    world_to_clip,
    clip_to_ndc,
    ndc_to_screen,
    is_in_bounds
)

from .mesh_builder import (
    build_block_mesh,
    build_mesh_from_grid
)

from .shading import (
    TerrainShader,
    composite_over_sky
)

from .renderer import DifferentiableBlockRenderer

from .diff_world import DifferentiableBlockWorld

__all__ = [
    # Materials
    'MATERIALS',
    'FACE_DEFS',
    'FACE_INDICES',
    'FaceIndex',
    'get_material_palette',
    'get_material_densities',
    'material_name_to_index',
    'material_index_to_name',

    # Utils
    'block_to_world',
    'get_face_vertex_world',
    'load_camera_matrices_from_metadata',
    'create_perspective_matrix',
    'create_look_at_matrix',
    'world_to_clip',
    'clip_to_ndc',
    'ndc_to_screen',
    'is_in_bounds',

    # Mesh builder
    'build_block_mesh',
    'build_mesh_from_grid',

    # Shading
    'TerrainShader',
    'composite_over_sky',

    # Renderer
    'DifferentiableBlockRenderer',

    # World
    'DifferentiableBlockWorld',
]
