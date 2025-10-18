from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class RendererConfig:
    # Canonical naming (match WebGPU):
    width: Optional[int] = None
    height: Optional[int] = None
    # Back-compat aliases (deprecated):
    image_height: Optional[int] = None
    image_width: Optional[int] = None
    steps: Optional[int] = None
    temperature: float = 1.0
    step_size: float = 0.25
    srgb: bool = True
    # Shading/geometry parity knobs (Phase 1)
    normal_sharpness: float = 0.0  # 0 disables face-aligned approximation; >0 enables
    occupancy_hardness: float = 0.0  # 0 disables sigmoid hardening; >0 enables
    occupancy_threshold: float = 0.35
    # Perspective camera (optional). If provided, render_perspective is used.
    camera_view: Optional[torch.Tensor] = None  # 4x4, world <- camera
    camera_proj: Optional[torch.Tensor] = None  # 4x4, clip <- camera
    world_scale: float = 2.0  # must match WebGPU engine (default 2)
    # Textures (Phase 2)
    use_textures: bool = False
    texture_scale: float = 1.0  # UV scale inside a voxel (1.0 = one tile per voxel)
    # WebGPU color/parity helpers
    # If True, adjust defaults to better match the WebGPU dataset captures
    # (no sRGB gamma at the end, harder occupancy, face-aligned normals, and
    # optional per-face palette colors when textures are unavailable).
    webgpu_parity: bool = False
    # Use per-face palette colors (top/bottom/side) instead of a single base color
    # per material when textures are not used. Defaults to False; enabled automatically
    # if webgpu_parity is True.
    use_palette_faces: bool = False
    # Backend: None/'vol' for volumetric (default), 'nv' for nvdiffrast triangle path
    backend: Optional[str] = None

    def resolve_size(self) -> tuple[int, int]:
        h = self.height if self.height is not None else self.image_height
        w = self.width if self.width is not None else self.image_width
        if h is None or w is None:
            raise ValueError("RendererConfig requires width/height (or image_width/image_height)")
        return h, w


class DifferentiableRenderer:
    def __init__(self, sigma_m: torch.Tensor, c_m: torch.Tensor, texture_atlas: Optional[torch.Tensor] = None):
        self.sigma_m = sigma_m
        self.c_m = c_m
        # texture_atlas shape: (M, 6, 4, Ht, Wt) in RGBA, values in [0,1]
        self.texture_atlas = texture_atlas

    def set_textures(self, texture_atlas: Optional[torch.Tensor]) -> None:
        self.texture_atlas = texture_atlas

    def render(self, W_logits: torch.Tensor, config: RendererConfig) -> torch.Tensor:
        from .core import render_ortho, render_perspective
        # Resolve size and derive parity-driven overrides without mutating the input dataclass
        H, W = config.resolve_size()
        use_palette_faces = bool(config.use_palette_faces or config.webgpu_parity)
        normal_sharpness = config.normal_sharpness
        occupancy_hardness = config.occupancy_hardness
        step_size = config.step_size
        srgb_out = config.srgb
        if config.webgpu_parity:
            # Emulate WebGPU capture look by default:
            # - no gamma encode at end (canvas capture path already ends in sRGB)
            # - encourage face-aligned normals/hard occupancy if not explicitly set
            srgb_out = False if config.srgb is True else config.srgb
            if not normal_sharpness or normal_sharpness <= 0:
                normal_sharpness = 10.0
            if not occupancy_hardness or occupancy_hardness <= 0:
                occupancy_hardness = 12.0
            # Step size used across codebase for dataset rendering parity
            if step_size is None or step_size == 0.25:
                step_size = 0.2

        # Optional nvdiffrast backend for WebGPU-parity triangle rendering
        backend = (config.backend or '').lower()
        if backend in ('nv', 'nvdiffrast', 'tri'):
            if config.camera_view is None or config.camera_proj is None:
                raise ValueError("nvdiffrast backend requires perspective camera_view and camera_proj")
            # Lazy import to avoid hard dep when unused
            from ..nv_diff_render import NvDiffRenderer, NvRenderConfig  # type: ignore
            nv = NvDiffRenderer(self.sigma_m, self.c_m, texture_atlas=self.texture_atlas)
            img = nv.render(
                W_logits,
                NvRenderConfig(
                    height=H,
                    width=W,
                    temperature=config.temperature,
                    world_scale=config.world_scale,
                    camera_view=config.camera_view,
                    camera_proj=config.camera_proj,
                    face_steepness=float(occupancy_hardness) if (occupancy_hardness and occupancy_hardness > 0) else 12.0,
                    face_threshold=0.01,
                    use_palette_faces=use_palette_faces,
                    srgb=srgb_out,
                ),
            )
            return img

        if config.camera_view is not None and config.camera_proj is not None:
            I = render_perspective(
                W_logits,
                self.sigma_m,
                self.c_m,
                H,
                W,
                config.temperature,
                camera_view=config.camera_view,
                camera_proj=config.camera_proj,
                steps=config.steps,
                step_size=step_size,
                world_scale=config.world_scale,
                normal_sharpness=normal_sharpness,
                occupancy_hardness=occupancy_hardness,
                occupancy_threshold=config.occupancy_threshold,
                use_textures=config.use_textures,
                use_palette_faces=use_palette_faces,
                texture_scale=config.texture_scale,
                texture_atlas=self.texture_atlas,
            )
        else:
            I = render_ortho(
                W_logits,
                self.sigma_m,
                self.c_m,
                H,
                W,
                config.temperature,
                steps=config.steps,
                step_size=step_size,
                normal_sharpness=normal_sharpness,
                occupancy_hardness=occupancy_hardness,
                occupancy_threshold=config.occupancy_threshold,
                use_textures=config.use_textures,
                use_palette_faces=use_palette_faces,
                texture_scale=config.texture_scale,
                texture_atlas=self.texture_atlas,
            )
        if srgb_out:
            # Apply simple sRGB EOTF to match typical canvas export look
            rgb = I[..., :3, :, :].clamp(0, 1)
            rgb = torch.pow(rgb + 1e-8, 1.0 / 2.2)
            I = torch.cat([rgb, I[..., 3:4, :, :]], dim=-3)
        return I
