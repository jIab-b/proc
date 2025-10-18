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
    # Perspective camera (optional). If provided, render_perspective is used.
    camera_view: Optional[torch.Tensor] = None  # 4x4, world <- camera
    camera_proj: Optional[torch.Tensor] = None  # 4x4, clip <- camera
    world_scale: float = 2.0  # must match WebGPU engine (default 2)

    def resolve_size(self) -> tuple[int, int]:
        h = self.height if self.height is not None else self.image_height
        w = self.width if self.width is not None else self.image_width
        if h is None or w is None:
            raise ValueError("RendererConfig requires width/height (or image_width/image_height)")
        return h, w


class DifferentiableRenderer:
    def __init__(self, sigma_m: torch.Tensor, c_m: torch.Tensor):
        self.sigma_m = sigma_m
        self.c_m = c_m

    def render(self, W_logits: torch.Tensor, config: RendererConfig) -> torch.Tensor:
        from .core import render_ortho, render_perspective
        H, W = config.resolve_size()
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
                step_size=config.step_size,
                world_scale=config.world_scale,
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
                step_size=config.step_size,
            )
        if config.srgb:
            # Apply simple sRGB EOTF to match typical canvas export look
            rgb = I[..., :3, :, :].clamp(0, 1)
            rgb = torch.pow(rgb + 1e-8, 1.0 / 2.2)
            I = torch.cat([rgb, I[..., 3:4, :, :]], dim=-3)
        return I
