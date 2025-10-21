"""
Fragment shader matching terrain.wgsl exactly.

References:
- src/pipelines/render/terrain.wgsl:34-54 (fs_main)
"""

import torch
from typing import Optional


class TerrainShader:
    """
    Lighting shader matching WebGPU terrain.wgsl fragment shader.

    Constants from terrain.wgsl:35-40:
    - sunDir: normalize(vec3(-0.4, -0.85, -0.5))
    - sunColor: vec3(1.0, 0.97, 0.9)
    - skyColor: vec3(0.53, 0.81, 0.92)
    - horizonColor: vec3(0.18, 0.16, 0.12)
    - ambient scale: 0.55
    """

    def __init__(self, device: torch.device = torch.device('cpu')):
        """
        Initialize shader with lighting constants.

        Args:
            device: Torch device for constants
        """
        # Lighting constants (terrain.wgsl:35-40)
        self.sun_dir = torch.tensor([-0.4, -0.85, -0.5], device=device, dtype=torch.float32)
        self.sun_dir = self.sun_dir / torch.linalg.norm(self.sun_dir)  # normalize

        self.sun_color = torch.tensor([1.0, 0.97, 0.9], device=device, dtype=torch.float32)
        self.sky_color = torch.tensor([0.53, 0.81, 0.92], device=device, dtype=torch.float32)
        self.horizon_color = torch.tensor([0.18, 0.16, 0.12], device=device, dtype=torch.float32)
        self.ambient_scale = 0.55

        self.device = device

    def to(self, device: torch.device) -> 'TerrainShader':
        """Move shader constants to device."""
        self.sun_dir = self.sun_dir.to(device)
        self.sun_color = self.sun_color.to(device)
        self.sky_color = self.sky_color.to(device)
        self.horizon_color = self.horizon_color.to(device)
        self.device = device
        return self

    def shade(
        self,
        normals: torch.Tensor,
        colors: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply lighting to surface colors.

        Matches terrain.wgsl:41-53 exactly.

        Args:
            normals: (H, W, 3) normalized surface normals
            colors: (H, W, 3) surface colors (base or textured)
            mask: Optional (H, W, 1) validity mask

        Returns:
            (H, W, 3) lit RGB values
        """
        # Ensure normals are normalized (terrain.wgsl:41)
        n = normals / (torch.linalg.norm(normals, dim=-1, keepdim=True) + 1e-6)
        n = torch.nan_to_num(n)

        # Sun diffuse lighting (terrain.wgsl:42)
        # sunDiffuse = max(dot(n, -sunDir), 0.0)
        sun_dot = torch.sum(n * (-self.sun_dir.view(1, 1, 3)), dim=-1, keepdim=True)
        sun_diffuse = torch.clamp(sun_dot, min=0.0)

        # Sky factor based on normal Y (terrain.wgsl:43)
        # skyFactor = clamp(n.y * 0.5 + 0.5, 0.0, 1.0)
        sky_factor = torch.clamp(n[..., 1:2] * 0.5 + 0.5, min=0.0, max=1.0)

        # Ambient lighting (terrain.wgsl:44)
        # ambient = (horizonColor * (1.0 - skyFactor) + skyColor * skyFactor) * 0.55
        ambient = (
            self.horizon_color.view(1, 1, 3) * (1.0 - sky_factor) +
            self.sky_color.view(1, 1, 3) * sky_factor
        ) * self.ambient_scale

        # Diffuse lighting (terrain.wgsl:45)
        # diffuse = sunDiffuse * sunColor
        diffuse = sun_diffuse * self.sun_color.view(1, 1, 3)

        # Final shading (terrain.wgsl:53)
        # litColor = surfaceColor * (ambient + diffuse)
        lit_color = colors * (ambient + diffuse)
        lit_color = torch.nan_to_num(lit_color)

        # Apply mask if provided
        if mask is not None:
            lit_color = lit_color * mask
            lit_color = torch.nan_to_num(lit_color)

        return lit_color


def composite_over_sky(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    sky_color: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Composite RGB over sky background.

    Args:
        rgb: (H, W, 3) foreground color
        alpha: (H, W, 1) foreground alpha
        sky_color: (3,) or (1,1,3) sky color (default: WebGPU clear color)

    Returns:
        (H, W, 3) composited RGB
    """
    if sky_color is None:
        # Default to WebGPU clear color (terrain.wgsl clear value)
        sky_color = torch.tensor([0.53, 0.81, 0.92], device=rgb.device, dtype=rgb.dtype)

    if sky_color.dim() == 1:
        sky_color = sky_color.view(1, 1, 3)

    # Standard over operator: C = C_fg * alpha + C_bg * (1 - alpha)
    composited = rgb * alpha + sky_color * (1.0 - alpha)

    return composited
