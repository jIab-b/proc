import torch
import torch.nn.functional as F


def soft_fields(W: torch.Tensor, sigma_m: torch.Tensor, c_m: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    # Encourage AIR as the default when evidence is absent to avoid "uniform fog".
    # Identify AIR channels as those with near-zero density in sigma_m.
    device = W.device
    sigma_m_d = sigma_m.to(device)
    c_m_d = c_m.to(device)
    air_mask = (sigma_m_d <= 1e-6)
    # A modest positive bias makes AIR dominate when logits are ~0, while
    # still allowing strong non-air logits (e.g., ~6) to win decisively.
    air_bias = torch.where(air_mask, torch.tensor(5.0, device=device, dtype=W.dtype), torch.tensor(0.0, device=device, dtype=W.dtype))
    P = torch.softmax((W + air_bias) / temperature, dim=-1)
    sigma = (P * sigma_m_d.view(1, 1, 1, -1)).sum(-1)
    rgb = P @ c_m_d
    return sigma, rgb


def _build_grid_ortho(img_h: int, img_w: int, X: int, Y: int, Z: int, steps: int, device: torch.device) -> torch.Tensor:
    xs = torch.linspace(0, X - 1, img_w, device=device)
    ys = torch.linspace(0, Y - 1, img_h, device=device)
    zs = torch.linspace(0, Z - 1, steps, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    # Normalize to [-1, 1] for grid_sample with align_corners=True
    xg = 2.0 * (xx / max(X - 1, 1)) - 1.0
    yg = 2.0 * (yy / max(Y - 1, 1)) - 1.0
    zg = 2.0 * (zs / max(Z - 1, 1)) - 1.0
    xg = xg.unsqueeze(0).unsqueeze(0).expand(1, steps, img_h, img_w)
    yg = yg.unsqueeze(0).unsqueeze(0).expand(1, steps, img_h, img_w)
    zg = zg.view(1, steps, 1, 1).expand(1, steps, img_h, img_w)
    grid = torch.stack([xg, yg, zg], dim=-1)
    return grid


def render_ortho(
    W: torch.Tensor,
    sigma_m: torch.Tensor,
    c_m: torch.Tensor,
    img_h: int,
    img_w: int,
    temperature: float,
    steps: int | None = None,
    step_size: float = 0.25,
) -> torch.Tensor:
    device = W.device
    X, Y, Z, _ = W.shape
    sigma, rgb = soft_fields(W, sigma_m.to(device), c_m.to(device), temperature)

    # Build tri-linear sample volumes in (N=1,C,D=Z,H=Y,W=X)
    sigma_vol = sigma.permute(2, 1, 0).contiguous().unsqueeze(0).unsqueeze(0)
    rgb_vol = rgb.permute(2, 1, 0, 3).contiguous().permute(3, 0, 1, 2).unsqueeze(0)

    # Spatial gradients for normals (central differences)
    gx = torch.zeros_like(sigma)
    gy = torch.zeros_like(sigma)
    gz = torch.zeros_like(sigma)
    if X > 2:
        gx[1:-1, :, :] = (sigma[2:, :, :] - sigma[:-2, :, :]) * 0.5
    if Y > 2:
        gy[:, 1:-1, :] = (sigma[:, 2:, :] - sigma[:, :-2, :]) * 0.5
    if Z > 2:
        gz[:, :, 1:-1] = (sigma[:, :, 2:] - sigma[:, :, :-2]) * 0.5
    gx_vol = gx.permute(2, 1, 0).contiguous().unsqueeze(0).unsqueeze(0)
    gy_vol = gy.permute(2, 1, 0).contiguous().unsqueeze(0).unsqueeze(0)
    gz_vol = gz.permute(2, 1, 0).contiguous().unsqueeze(0).unsqueeze(0)

    S = steps if steps is not None else Z
    grid = _build_grid_ortho(img_h, img_w, X, Y, Z, S, device)

    # grid_sample returns (N,C,S,H,W)
    sigma_s = F.grid_sample(sigma_vol, grid, align_corners=True, mode='bilinear', padding_mode='zeros').squeeze(0).squeeze(0)
    rgb_s = F.grid_sample(rgb_vol, grid, align_corners=True, mode='bilinear', padding_mode='zeros').squeeze(0)
    # rgb_s: (3,S,H,W) -> (H,W,S,3)
    rgb_s = rgb_s.permute(2, 3, 1, 0)

    gx_s = F.grid_sample(gx_vol, grid, align_corners=True, mode='bilinear', padding_mode='zeros').squeeze(0).squeeze(0)
    gy_s = F.grid_sample(gy_vol, grid, align_corners=True, mode='bilinear', padding_mode='zeros').squeeze(0).squeeze(0)
    gz_s = F.grid_sample(gz_vol, grid, align_corners=True, mode='bilinear', padding_mode='zeros').squeeze(0).squeeze(0)
    # (S,H,W) to (H,W,S)
    sigma_line = sigma_s.permute(1, 2, 0).clamp_min(0)
    gx_line = gx_s.permute(1, 2, 0)
    gy_line = gy_s.permute(1, 2, 0)
    gz_line = gz_s.permute(1, 2, 0)
    n_norm = torch.sqrt(gx_line * gx_line + gy_line * gy_line + gz_line * gz_line + 1e-6)
    nx = gx_line / n_norm
    ny = gy_line / n_norm
    nz = gz_line / n_norm
    # Lighting constants: match WebGPU terrain.wgsl
    sun_dir = torch.tensor([-0.4, -0.85, -0.5], device=device, dtype=rgb.dtype)
    sun_dir = sun_dir / torch.linalg.norm(sun_dir)
    sun_color = torch.tensor([1.0, 0.97, 0.9], device=device, dtype=rgb.dtype)
    sky_color = torch.tensor([0.53, 0.81, 0.92], device=device, dtype=rgb.dtype)
    horizon_color = torch.tensor([0.18, 0.16, 0.12], device=device, dtype=rgb.dtype)
    sun_dot = (nx * (-sun_dir[0]) + ny * (-sun_dir[1]) + nz * (-sun_dir[2])).clamp_min(0.0)
    sky_factor = (ny * 0.5 + 0.5).clamp(0.0, 1.0)
    # Shapes to (H,W,S,3)
    ambient = ((1.0 - sky_factor)[..., None] * horizon_color.view(1, 1, 1, 3) + sky_factor[..., None] * sky_color.view(1, 1, 1, 3)) * 0.55
    diffuse = sun_dot[..., None] * sun_color.view(1, 1, 1, 3)
    shading = ambient + diffuse
    # rgb_s is (H,W,S,3). Apply per-sample shading then composite.
    rgb_line = (rgb_s * shading).clamp(0.0, 1.0)
    # Step distance in voxel units, scaled by step_size for controllable opacity.
    base_delta = (Z - 1) / max(S - 1, 1)
    delta = float(step_size) * float(base_delta)
    alpha = 1.0 - torch.exp(-sigma_line * delta)
    one_m_alpha = 1.0 - alpha + 1e-6
    Tcum = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), one_m_alpha], dim=-1), dim=-1)[..., :-1]
    w = alpha * Tcum
    C = (w.unsqueeze(-1) * rgb_line).sum(dim=-2)
    A = w.sum(dim=-1, keepdim=True).clamp(0, 1)
    # Composite over WebGPU sky clear color to match dataset look.
    sky_bg = sky_color.view(1, 1, -1)
    C = C + (1.0 - A) * sky_bg
    I = torch.cat([C, A], dim=-1).permute(2, 0, 1)
    return I.unsqueeze(0)
