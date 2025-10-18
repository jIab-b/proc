from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class RendererConfig:
    image_height: int
    image_width: int
    steps: Optional[int] = None
    temperature: float = 1.0


class DifferentiableRenderer:
    def __init__(self, sigma_m: torch.Tensor, c_m: torch.Tensor):
        self.sigma_m = sigma_m
        self.c_m = c_m

    def render(self, W_logits: torch.Tensor, config: RendererConfig) -> torch.Tensor:
        from .core import render_ortho
        return render_ortho(
            W_logits,
            self.sigma_m,
            self.c_m,
            config.image_height,
            config.image_width,
            config.temperature,
            steps=config.steps,
        )


