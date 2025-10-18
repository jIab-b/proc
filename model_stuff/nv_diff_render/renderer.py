"""
Main nvdiffrast renderer for differentiable block rendering.

This renderer produces output matching WebGPU exactly while maintaining
differentiability through material parameters.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional, Callable

try:
    import nvdiffrast.torch as dr
    NVDIFFRAST_AVAILABLE = True
except ImportError:
    NVDIFFRAST_AVAILABLE = False
    print("Warning: nvdiffrast not available. Install with: pip install nvdiffrast")

from .mesh_builder import build_block_mesh
from .shading import TerrainShader, composite_over_sky
from .utils import world_to_clip


class DifferentiableBlockRenderer(nn.Module):
    """
    High-accuracy nvdiffrast renderer matching WebGPU output.

    This renderer:
    - Generates triangle meshes from block placements
    - Supports differentiable material selection via logits
    - Matches WebGPU lighting and rendering exactly
    """

    def __init__(
        self,
        grid_size: Tuple[int, int, int] = (64, 48, 64),
        world_scale: float = 2.0,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize renderer.

        Args:
            grid_size: (sx, sy, sz) world dimensions
            world_scale: Scale multiplier (default: 2.0)
            device: Torch device
        """
        super().__init__()

        if not NVDIFFRAST_AVAILABLE:
            raise ImportError("nvdiffrast is required. Install with: pip install nvdiffrast")

        self.grid_size = grid_size
        self.world_scale = world_scale
        self.device = device

        # Initialize shader
        self.shader = TerrainShader(device=device)

        # nvdiffrast context (will be created on first render)
        self.glctx = None

    def to(self, device: torch.device) -> 'DifferentiableBlockRenderer':
        """Move renderer to device."""
        super().to(device)
        self.device = device
        self.shader = self.shader.to(device)
        return self

    def _ensure_context(self):
        """Ensure nvdiffrast context exists. Use CUDA for WSL2 compatibility."""
        if self.glctx is None:
            # Use CUDA rasterizer instead of GL (WSL2/headless compatible)
            self.glctx = dr.RasterizeCudaContext()

    def render(
        self,
        positions: List[Tuple[int, int, int]],
        material_logits: torch.Tensor,
        camera_view: torch.Tensor,
        camera_proj: torch.Tensor,
        img_h: int,
        img_w: int,
        neighbor_check: Optional[Callable[[Tuple[int, int, int]], bool]] = None,
        temperature: float = 1.0,
        hard_materials: bool = False,
        return_depth: bool = False,
        return_normals: bool = False
    ) -> torch.Tensor:
        """
        Render blocks from camera view.

        Args:
            positions: List of (x, y, z) block positions
            material_logits: (N, M) material logits per block
            camera_view: (4, 4) view matrix
            camera_proj: (4, 4) projection matrix
            img_h: Output image height
            img_w: Output image width
            neighbor_check: Optional function for face culling
            temperature: Material softmax temperature
            hard_materials: Use hard one-hot assignment
            return_depth: Also return depth map
            return_normals: Also return normal map

        Returns:
            (1, 4, H, W) RGBA image composited over sky
            If return_depth: also returns (1, 1, H, W) depth
            If return_normals: also returns (1, 3, H, W) normals
        """
        self._ensure_context()

        device = material_logits.device
        camera_view = camera_view.to(device)
        camera_proj = camera_proj.to(device)

        # Handle empty scene
        if len(positions) == 0:
            rgb = self.shader.sky_color.view(1, 3, 1, 1).expand(1, 3, img_h, img_w)
            alpha = torch.zeros(1, 1, img_h, img_w, device=device)
            result = torch.cat([rgb, alpha], dim=1)  # (1, 4, H, W)

            if return_depth or return_normals:
                extras = []
                if return_depth:
                    extras.append(torch.zeros(1, 1, img_h, img_w, device=device))
                if return_normals:
                    extras.append(torch.zeros(1, 3, img_h, img_w, device=device))
                return (result, *extras)
            return result

        # Build mesh
        vertices, faces, attributes = build_block_mesh(
            positions,
            material_logits,
            self.grid_size,
            self.world_scale,
            neighbor_check,
            temperature,
            hard_materials
        )

        # Handle empty mesh (all faces culled)
        if vertices.shape[0] == 0:
            rgb = self.shader.sky_color.view(1, 3, 1, 1).expand(1, 3, img_h, img_w)
            alpha = torch.zeros(1, 1, img_h, img_w, device=device)
            result = torch.cat([rgb, alpha], dim=1)

            if return_depth or return_normals:
                extras = []
                if return_depth:
                    extras.append(torch.zeros(1, 1, img_h, img_w, device=device))
                if return_normals:
                    extras.append(torch.zeros(1, 3, img_h, img_w, device=device))
                return (result, *extras)
            return result

        # Transform vertices to clip space
        clip_pos = world_to_clip(vertices, camera_view, camera_proj)

        # Add batch dimension for nvdiffrast
        clip_pos_batch = clip_pos.unsqueeze(0)  # (1, V, 4)
        # CUDA rasterizer requires int32 (GL uses int64)
        faces_int32 = faces.int()

        # Rasterize
        rast, rast_db = dr.rasterize(
            self.glctx,
            clip_pos_batch,
            faces_int32,
            resolution=[img_h, img_w]
        )

        # Interpolate attributes
        normals, _ = dr.interpolate(
            attributes['normals'].unsqueeze(0),  # (1, V, 3)
            rast,
            faces_int32
        )
        normals = normals[0]  # (H, W, 3)

        colors, _ = dr.interpolate(
            attributes['colors'].unsqueeze(0),  # (1, V, 3)
            rast,
            faces_int32
        )
        colors = colors[0]  # (H, W, 3)

        # Mask for valid pixels
        mask = (rast[0, :, :, 3:4] > 0).float()  # (H, W, 1)

        # Apply lighting shader
        lit_colors = self.shader.shade(normals, colors, mask)

        # Composite over sky
        rgb = composite_over_sky(lit_colors, mask, self.shader.sky_color)

        # Flip vertically (nvdiffrast uses bottom-left origin, we want top-left)
        rgb = torch.flip(rgb, dims=[0])
        mask = torch.flip(mask, dims=[0])

        # Format output as (1, 4, H, W)
        rgb_out = rgb.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        alpha_out = mask.permute(2, 0, 1).unsqueeze(0)  # (1, 1, H, W)
        result = torch.cat([rgb_out, alpha_out], dim=1)  # (1, 4, H, W)

        # Optional depth and normals
        extras = []
        if return_depth:
            # Extract depth from rasterized Z
            depth = rast[0, :, :, 2:3]  # (H, W, 1)
            depth = torch.flip(depth, dims=[0])  # Flip vertically
            depth_out = depth.permute(2, 0, 1).unsqueeze(0)  # (1, 1, H, W)
            extras.append(depth_out)

        if return_normals:
            # Return world-space normals
            normals_flipped = torch.flip(normals, dims=[0])  # Flip vertically
            normals_out = normals_flipped.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            extras.append(normals_out)

        if extras:
            return (result, *extras)

        return result

    def render_from_grid(
        self,
        block_grid: torch.Tensor,
        material_logits: torch.Tensor,
        camera_view: torch.Tensor,
        camera_proj: torch.Tensor,
        img_h: int,
        img_w: int,
        **kwargs
    ) -> torch.Tensor:
        """
        Render from dense voxel grid.

        Args:
            block_grid: (X, Y, Z) bool tensor
            material_logits: (X, Y, Z, M) material logits
            camera_view, camera_proj: Camera matrices
            img_h, img_w: Output size
            **kwargs: Additional args for render()

        Returns:
            Same as render()
        """
        from .utils import is_in_bounds

        # Find occupied positions
        occupied = torch.nonzero(block_grid, as_tuple=False)
        positions = [(int(x), int(y), int(z)) for x, y, z in occupied]

        if len(positions) == 0:
            device = material_logits.device
            rgb = self.shader.sky_color.view(1, 3, 1, 1).expand(1, 3, img_h, img_w).to(device)
            alpha = torch.zeros(1, 1, img_h, img_w, device=device)
            return torch.cat([rgb, alpha], dim=1)

        # Extract material logits for occupied voxels
        logits = material_logits[occupied[:, 0], occupied[:, 1], occupied[:, 2]]

        # Neighbor check using grid
        def neighbor_check(pos):
            x, y, z = pos
            if not is_in_bounds(x, y, z, self.grid_size):
                return False
            return block_grid[x, y, z].item()

        return self.render(
            positions,
            logits,
            camera_view,
            camera_proj,
            img_h,
            img_w,
            neighbor_check=neighbor_check,
            **kwargs
        )
