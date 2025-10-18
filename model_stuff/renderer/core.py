import torch


def soft_fields(W: torch.Tensor, sigma_m: torch.Tensor, c_m: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    P = torch.softmax(W / temperature, dim=-1)
    sigma = (P * sigma_m.view(1, 1, 1, -1).to(P.device)).sum(-1)
    rgb = P @ c_m.to(P.device)
    return sigma, rgb


def render_ortho(W: torch.Tensor, sigma_m: torch.Tensor, c_m: torch.Tensor, img_h: int, img_w: int, temperature: float, steps: int | None = None) -> torch.Tensor:
    device = W.device
    X, Y, Z, _ = W.shape
    sigma, rgb = soft_fields(W, sigma_m.to(device), c_m.to(device), temperature)
    xs = torch.linspace(0, X - 1, img_w, device=device).round().long()
    ys = torch.linspace(0, Y - 1, img_h, device=device).round().long()
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    sigma_line = sigma[xx, yy, :].clamp_min(0)
    rgb_line = rgb[xx, yy, :, :]
    if steps is not None and steps < Z:
        idx = torch.linspace(0, Z - 1, steps, device=device).round().long()
        sigma_line = sigma_line.index_select(-1, idx)
        rgb_line = rgb_line.index_select(-2, idx)
    gx = torch.zeros_like(sigma)
    gy = torch.zeros_like(sigma)
    gz = torch.zeros_like(sigma)
    if X > 2:
        gx[1:-1, :, :] = (sigma[2:, :, :] - sigma[:-2, :, :]) * 0.5
    if Y > 2:
        gy[:, 1:-1, :] = (sigma[:, 2:, :] - sigma[:, :-2, :]) * 0.5
    if Z > 2:
        gz[:, :, 1:-1] = (sigma[:, :, 2:] - sigma[:, :, :-2]) * 0.5
    gx_line = gx[xx, yy, :]
    gy_line = gy[xx, yy, :]
    gz_line = gz[xx, yy, :]
    n_norm = torch.sqrt(gx_line * gx_line + gy_line * gy_line + gz_line * gz_line + 1e-6)
    nx = gx_line / n_norm
    ny = gy_line / n_norm
    nz = gz_line / n_norm
    sun_dir = torch.tensor([-0.4, -0.85, -0.5], device=device, dtype=rgb_line.dtype)
    sun_dir = sun_dir / torch.linalg.norm(sun_dir)
    sun_color = torch.tensor([1.0, 0.97, 0.9], device=device, dtype=rgb_line.dtype)
    sky_color = torch.tensor([0.53, 0.81, 0.92], device=device, dtype=rgb_line.dtype)
    horizon_color = torch.tensor([0.18, 0.16, 0.12], device=device, dtype=rgb_line.dtype)
    sun_dot = (nx * (-sun_dir[0]) + ny * (-sun_dir[1]) + nz * (-sun_dir[2])).clamp_min(0.0)
    sky_factor = (ny * 0.5 + 0.5).clamp(0.0, 1.0)
    ambient = (horizon_color.view(1, 1, -1) * (1.0 - sky_factor).unsqueeze(-1) + sky_color.view(1, 1, -1) * sky_factor.unsqueeze(-1)) * 0.55
    diffuse = sun_dot.unsqueeze(-1) * sun_color.view(1, 1, -1)
    shading = ambient + diffuse
    rgb_line = (rgb_line * shading).clamp(0.0, 1.0)
    delta = 1.0
    alpha = 1.0 - torch.exp(-sigma_line * delta)
    one_m_alpha = 1.0 - alpha + 1e-6
    Tcum = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), one_m_alpha], dim=-1), dim=-1)[..., :-1]
    w = alpha * Tcum
    C = (w.unsqueeze(-1) * rgb_line).sum(dim=-2)
    A = w.sum(dim=-1, keepdim=True).clamp(0, 1)
    I = torch.cat([C, A], dim=-1).permute(2, 0, 1)
    return I.unsqueeze(0)


