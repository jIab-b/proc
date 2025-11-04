"""
Main nvdiffrast renderer for differentiable block rendering.

This renderer produces output matching WebGPU exactly while maintaining
differentiability through material parameters.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional, Callable
from pathlib import Path
import math  # for sqrt if needed, but not

try:
    import nvdiffrast.torch as dr
    NVDIFFRAST_AVAILABLE = True
except ImportError:
    NVDIFFRAST_AVAILABLE = False
    print("Warning: nvdiffrast not available. Install with: pip install nvdiffrast")

from .mesh_builder import build_block_mesh
from .shading import TerrainShader, composite_over_sky
from .utils import world_to_clip


def log_tensor_stats(name, tensor, entry):
    if tensor is None:
        entry[f"{name}"] = "None"
        return

    # Handle boolean tensors by converting to float for statistics
    if tensor.dtype == torch.bool:
        tensor_for_stats = tensor.float()
        has_nan = False
        has_inf = False
        is_finite = True
    else:
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        is_finite = torch.isfinite(tensor).all()
        tensor_for_stats = tensor

    if not is_finite:
        stats = f"non-finite (min={tensor_for_stats.min():.6f}, max={tensor_for_stats.max():.6f}, mean={tensor_for_stats.mean():.6f}, norm={torch.linalg.norm(tensor_for_stats):.6f}, has_nan={has_nan}, has_inf={has_inf})"
    else:
        stats = f"finite (min={tensor_for_stats.min():.6f}, max={tensor_for_stats.max():.6f}, mean={tensor_for_stats.mean():.6f}, norm={torch.linalg.norm(tensor_for_stats):.6f})"
    entry[f"{name}"] = stats
    if hasattr(tensor, 'shape'):
        entry[f"{name}_shape"] = str(tensor.shape)
    if name == "block_grid":
        entry[f"{name}_sum"] = int(torch.sum(tensor).item()) if torch.is_tensor(tensor) else "not tensor"
    if name == "faces":
        entry[f"{name}_num_faces"] = tensor.shape[0] if hasattr(tensor, 'shape') else 0


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
        self.last_debug: Dict[str, float] = {}

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
            self.last_debug = {
                'num_vertices': 0.0,
                'num_faces': 0.0,
                'mask_mean': 0.0,
                'colors_req_grad': 0.0,
            }

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
            self.last_debug = {
                'num_vertices': 0.0,
                'num_faces': 0.0,
                'mask_mean': 0.0,
                'colors_req_grad': 0.0,
            }

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
        rast, _ = dr.rasterize(
            self.glctx,
            clip_pos_batch,
            faces_int32,
            resolution=[img_h, img_w],
            grad_db=True
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
        lit_colors = torch.nan_to_num(lit_colors.clamp(0.0, 1.0))

        # Composite over sky
        rgb = composite_over_sky(lit_colors, mask, self.shader.sky_color)
        rgb = torch.nan_to_num(rgb.clamp(0.0, 1.0))

        # Flip vertically (nvdiffrast uses bottom-left origin, we want top-left)
        rgb = torch.flip(rgb, dims=[0])
        mask = torch.flip(mask, dims=[0])

        # Format output as (1, 4, H, W)
        rgb_out = rgb.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        alpha_out = mask.permute(2, 0, 1).unsqueeze(0)  # (1, 1, H, W)
        result = torch.cat([rgb_out, alpha_out], dim=1)  # (1, 4, H, W)
        result = torch.nan_to_num(result.clamp(0.0, 1.0))

        # Cache debug info
        self.last_debug = {
            'num_vertices': float(vertices.shape[0]),
            'num_faces': float(faces_int32.shape[0]),
            'mask_mean': float(mask.mean().detach().cpu()),
            'colors_req_grad': 1.0 if attributes['colors'].requires_grad else 0.0,
        }

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
        occupancy_probs: torch.Tensor | None = None,
        palette: torch.Tensor | None = None,
        material_probs: torch.Tensor | None = None,
        debug_path: Optional[str] = None,
        step: Optional[int] = None,
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

        # Ensure rasterizer context
        self._ensure_context()

        debug_entry = {}
        if debug_path and step is not None:
            log_tensor_stats("block_grid", block_grid, debug_entry)

        # Fast empty check
        if not torch.any(block_grid):
            device = material_logits.device
            rgb = self.shader.sky_color.view(1, 3, 1, 1).expand(1, 3, img_h, img_w).to(device)
            alpha = torch.zeros(1, 1, img_h, img_w, device=device)
            debug_entry["empty_scene"] = True
            log_tensor_stats("result_empty_rgba", torch.cat([rgb, alpha], dim=1), debug_entry)
            if debug_path and step is not None:
                with open(debug_path, "a", encoding="utf-8") as f:
                    f.write(f"Renderer Step {step} - Empty scene:\n")
                    for k, v in debug_entry.items():
                        f.write(f"  {k}: {v}\n")
                    f.write("\n")
                    f.flush()
            return torch.cat([rgb, alpha], dim=1)

        from .mesh_builder import build_mesh_from_grid

        vertices, faces, attributes = build_mesh_from_grid(
            block_grid,
            material_logits,
            camera_view,
            camera_proj,
            world_scale=self.world_scale,
            temperature=kwargs.get('temperature', 1.0),
            hard_assignment=kwargs.get('hard_materials', False),
            occupancy_probs=occupancy_probs,
            palette=palette,
            material_probs=material_probs,
        )

        if vertices.shape[0] == 0:
            device = material_logits.device
            rgb = self.shader.sky_color.view(1, 3, 1, 1).expand(1, 3, img_h, img_w).to(device)
            alpha = torch.zeros(1, 1, img_h, img_w, device=device)
<<<<<<< HEAD
            self.last_debug = {
                'num_vertices': 0.0,
                'num_faces': 0.0,
                'mask_mean': 0.0,
                'colors_req_grad': 0.0,
                'occ_total': float(attributes.get('debug_total_occ', torch.tensor(0)).item()) if isinstance(attributes.get('debug_total_occ', None), torch.Tensor) else 0.0,
                'occ_kept': float(attributes.get('debug_kept_occ', torch.tensor(0)).item()) if isinstance(attributes.get('debug_kept_occ', None), torch.Tensor) else 0.0,
            }
=======
            debug_entry["mesh_empty"] = True
            # Write empty log
            with open(debug_path, "a", encoding="utf-8") as f:
                f.write(f"Renderer Step {step} - Empty mesh:\n")
                for k, v in debug_entry.items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")
                f.flush()
>>>>>>> d6210d93b310fd5862d24541de732cbd6bc68227
            return torch.cat([rgb, alpha], dim=1)

        clip_pos = world_to_clip(vertices, camera_view.to(vertices.device), camera_proj.to(vertices.device))
        clip_pos_batch = clip_pos.unsqueeze(0)
        faces_int32 = faces.int()

<<<<<<< HEAD
        rast, _ = dr.rasterize(self.glctx, clip_pos_batch, faces_int32, resolution=[img_h, img_w], grad_db=True)
=======
        rast, _ = dr.rasterize(self.glctx, clip_pos_batch, faces_int32, resolution=[img_h, img_w], grad_db=False)
        # Log rast stats (barycentrics in [0,1,2], z, w)
        rast_flat = rast[0].view(-1, 4)  # H*W x 4
        log_tensor_stats("rast_bary_z_w", rast_flat, debug_entry)
>>>>>>> d6210d93b310fd5862d24541de732cbd6bc68227

        normals, _ = dr.interpolate(attributes['normals'].unsqueeze(0), rast, faces_int32)
        normals = normals[0]
        log_tensor_stats("normals_interp", normals, debug_entry)

        colors, _ = dr.interpolate(attributes['colors'].unsqueeze(0), rast, faces_int32)
        colors = colors[0]
        log_tensor_stats("colors_interp", colors, debug_entry)

        occupancy = None
        if 'occupancy' in attributes:
            occupancy_attr = attributes['occupancy'].unsqueeze(0)
            occupancy, _ = dr.interpolate(occupancy_attr, rast, faces_int32)
            occupancy = occupancy[0]
            log_tensor_stats("occupancy_interp", occupancy, debug_entry)

        mask = (rast[0, :, :, 3:4] > 0).float()
        if occupancy is not None:
            occupancy = torch.nan_to_num(occupancy.clamp(0.0, 1.0))
            if torch.isnan(occupancy).any() or torch.isinf(occupancy).any():
                debug_entry["occupancy_corrected"] = "NaN/inf fixed"
            mask = mask * occupancy

        log_tensor_stats("mask", mask, debug_entry)

        lit_colors = self.shader.shade(normals, colors, mask)
        lit_colors = torch.nan_to_num(lit_colors.clamp(0.0, 1.0))
        if torch.isnan(lit_colors).any() or torch.isinf(lit_colors).any():
            debug_entry["lit_colors_corrected"] = "NaN/inf fixed"
        log_tensor_stats("lit_colors", lit_colors, debug_entry)

        rgb = composite_over_sky(lit_colors, mask, self.shader.sky_color)
        rgb = torch.nan_to_num(rgb.clamp(0.0, 1.0))
        if torch.isnan(rgb).any() or torch.isinf(rgb).any():
            debug_entry["rgb_corrected"] = "NaN/inf fixed"
        log_tensor_stats("rgb", rgb, debug_entry)

        rgb = torch.flip(rgb, dims=[0])
        mask = torch.flip(mask, dims=[0])
        rgb_out = rgb.permute(2, 0, 1).unsqueeze(0)
        alpha_out = mask.permute(2, 0, 1).unsqueeze(0)
        result = torch.cat([rgb_out, alpha_out], dim=1)
        result = torch.nan_to_num(result.clamp(0.0, 1.0))
<<<<<<< HEAD
        self.last_debug = {
            'num_vertices': float(vertices.shape[0]),
            'num_faces': float(faces_int32.shape[0]),
            'mask_mean': float(mask.mean().detach().cpu()),
            'colors_req_grad': 1.0 if attributes['colors'].requires_grad else 0.0,
            'occ_total': float(attributes.get('debug_total_occ', torch.tensor(0)).item()) if isinstance(attributes.get('debug_total_occ', None), torch.Tensor) else 0.0,
            'occ_kept': float(attributes.get('debug_kept_occ', torch.tensor(0)).item()) if isinstance(attributes.get('debug_kept_occ', None), torch.Tensor) else 0.0,
        }
=======
        if torch.isnan(result).any() or torch.isinf(result).any():
            debug_entry["result_corrected"] = "NaN/inf fixed"
        log_tensor_stats("result_rgba", result, debug_entry)

        # Write to file
        if debug_path and step is not None:
            with open(debug_path, "a", encoding="utf-8") as f:
                f.write(f"Renderer Step {step}:\n")
                for k, v in debug_entry.items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")
                f.flush()

>>>>>>> d6210d93b310fd5862d24541de732cbd6bc68227
        return result
