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


def render_perspective(
    W: torch.Tensor,
    sigma_m: torch.Tensor,
    c_m: torch.Tensor,
    img_h: int,
    img_w: int,
    temperature: float,
    *,
    camera_view: torch.Tensor,
    camera_proj: torch.Tensor,
    steps: int | None = None,
    step_size: float = 0.25,
    world_scale: float = 2.0,
) -> torch.Tensor:
    device = W.device
    X, Y, Z, _ = W.shape
    sigma, rgb = soft_fields(W, sigma_m.to(device), c_m.to(device), temperature)

    # Build volumes (N=1,C,D=Z,H=Y,W=X)
    sigma_vol = sigma.permute(2, 1, 0).contiguous().unsqueeze(0).unsqueeze(0)
    rgb_vol = rgb.permute(2, 1, 0, 3).contiguous().permute(3, 0, 1, 2).unsqueeze(0)

    # Gradients for normals
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

    S = steps if steps is not None else max(X, Y, Z)

    # Camera matrices
    # camera_view: world <- camera (view matrix)
    # camera_proj: clip <- camera (OpenGL-style, z in [-1,1])
    inv_view = torch.linalg.inv(camera_view.to(device))
    inv_proj = torch.linalg.inv(camera_proj.to(device))

    # Build per-pixel ray directions in world space
    xs = torch.linspace(0.5, img_w - 0.5, img_w, device=device)
    ys = torch.linspace(0.5, img_h - 0.5, img_h, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    x_ndc = (xx / img_w) * 2.0 - 1.0
    y_ndc = 1.0 - (yy / img_h) * 2.0
    ones = torch.ones_like(x_ndc)
    # Clip space point at near plane (z = -1 in GL NDC)
    p_clip = torch.stack([x_ndc, y_ndc, -ones, ones], dim=-1)  # (H,W,4)
    # Camera space
    p_cam = p_clip @ inv_proj.T  # (H,W,4)
    # Homogenize
    p_cam = p_cam[..., :3] / p_cam[..., 3:4].clamp_min(1e-6)
    # Camera origin in world
    cam_o = (inv_view @ torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)).view(4)
    origin = cam_o[:3]
    # Direction in world
    p_world_h = torch.cat([p_cam, torch.ones_like(p_cam[..., :1])], dim=-1) @ inv_view.T
    p_world = p_world_h[..., :3] / p_world_h[..., 3:4].clamp_min(1e-6)
    dir_ws = p_world - origin.view(1, 1, 3)
    dir_ws = dir_ws / torch.linalg.norm(dir_ws, dim=-1, keepdim=True).clamp_min(1e-6)

    # Voxel AABB in world space
    s = float(world_scale)
    aabb_min = torch.tensor([-(X * s) * 0.5, 0.0, -(Z * s) * 0.5], device=device, dtype=dir_ws.dtype)
    aabb_max = torch.tensor([(X * s) * 0.5, Y * s, (Z * s) * 0.5], device=device, dtype=dir_ws.dtype)

    # Ray-AABB intersection (vectorized slab method)
    o = origin.view(1, 1, 3)
    d = dir_ws
    inv_d = 1.0 / d.clamp(min=-1e10, max=1e10)
    t0s = (aabb_min.view(1, 1, 3) - o) * inv_d
    t1s = (aabb_max.view(1, 1, 3) - o) * inv_d
    tmin = torch.minimum(t0s, t1s).amax(dim=-1)
    tmax = torch.maximum(t0s, t1s).amin(dim=-1)
    # Valid intersection and in front of camera
    hit = (tmax > torch.maximum(tmin, torch.zeros_like(tmin)))
    # Clamp t range to be positive
    t_start = torch.where(hit, torch.maximum(tmin, torch.zeros_like(tmin)), torch.zeros_like(tmin))
    t_end = torch.where(hit, tmax, torch.zeros_like(tmax))

    # Sample along ray segment
    # Create per-pixel per-step param in [0,1]
    t_lin = torch.linspace(0.0, 1.0, S, device=device).view(1, 1, S)
    seg_len = (t_end - t_start).unsqueeze(-1)  # (H,W,1)
    t_vals = t_start.unsqueeze(-1) + t_lin * seg_len  # (H,W,S)
    # World positions
    pos_ws = o.unsqueeze(-2) + t_vals.unsqueeze(-1) * d.unsqueeze(-2)  # (H,W,S,3)

    # Map to grid index coordinates
    # chunk origin offset [-X*s/2, 0, -Z*s/2]
    offset = torch.tensor([-(X * s) * 0.5, 0.0, -(Z * s) * 0.5], device=device, dtype=dir_ws.dtype)
    pos_grid = (pos_ws - offset.view(1, 1, 1, 3)) / s  # in voxel units

    # Normalize to [-1,1] for grid_sample with align_corners=True
    xg = 2.0 * (pos_grid[..., 0] / max(X - 1, 1)) - 1.0
    yg = 2.0 * (pos_grid[..., 1] / max(Y - 1, 1)) - 1.0
    zg = 2.0 * (pos_grid[..., 2] / max(Z - 1, 1)) - 1.0
    grid = torch.stack([xg, yg, zg], dim=-1)  # (H,W,S,3)
    # Reorder to (N=1,S,H,W,3)
    grid = grid.permute(2, 0, 1, 3).unsqueeze(0)

    # Sample sigma/rgb/normals
    sigma_s = F.grid_sample(sigma_vol, grid, align_corners=True, mode='bilinear', padding_mode='zeros').squeeze(0).squeeze(0)
    rgb_s = F.grid_sample(rgb_vol, grid, align_corners=True, mode='bilinear', padding_mode='zeros').squeeze(0)
    rgb_s = rgb_s.permute(2, 3, 1, 0)  # (H,W,S,3)
    gx_s = F.grid_sample(gx_vol, grid, align_corners=True, mode='bilinear', padding_mode='zeros').squeeze(0).squeeze(0)
    gy_s = F.grid_sample(gy_vol, grid, align_corners=True, mode='bilinear', padding_mode='zeros').squeeze(0).squeeze(0)
    gz_s = F.grid_sample(gz_vol, grid, align_corners=True, mode='bilinear', padding_mode='zeros').squeeze(0).squeeze(0)
    sigma_line = sigma_s.permute(1, 2, 0).clamp_min(0)
    gx_line = gx_s.permute(1, 2, 0)
    gy_line = gy_s.permute(1, 2, 0)
    gz_line = gz_s.permute(1, 2, 0)
    n_norm = torch.sqrt(gx_line * gx_line + gy_line * gy_line + gz_line * gz_line + 1e-6)
    nx = gx_line / n_norm
    ny = gy_line / n_norm
    nz = gz_line / n_norm

    # Lighting (match WebGPU terrain.wgsl)
    sun_dir = torch.tensor([-0.4, -0.85, -0.5], device=device, dtype=rgb.dtype)
    sun_dir = sun_dir / torch.linalg.norm(sun_dir)
    sun_color = torch.tensor([1.0, 0.97, 0.9], device=device, dtype=rgb.dtype)
    sky_color = torch.tensor([0.53, 0.81, 0.92], device=device, dtype=rgb.dtype)
    horizon_color = torch.tensor([0.18, 0.16, 0.12], device=device, dtype=rgb.dtype)
    sun_dot = (nx * (-sun_dir[0]) + ny * (-sun_dir[1]) + nz * (-sun_dir[2])).clamp_min(0.0)
    sky_factor = (ny * 0.5 + 0.5).clamp(0.0, 1.0)
    ambient = ((1.0 - sky_factor)[..., None] * horizon_color.view(1, 1, 1, 3) + sky_factor[..., None] * sky_color.view(1, 1, 1, 3)) * 0.55
    diffuse = sun_dot[..., None] * sun_color.view(1, 1, 1, 3)
    shading = ambient + diffuse
    rgb_line = (rgb_s * shading).clamp(0.0, 1.0)

    # Alpha compositing along ray with per-pixel step length
    # Convert world segment length per step to voxel units
    dt_world = (seg_len / S).clamp_min(1e-8)  # (H,W,1)
    dt_grid = dt_world / s
    delta = float(step_size) * dt_grid  # broadcast
    alpha = 1.0 - torch.exp(-sigma_line * delta)
    one_m_alpha = 1.0 - alpha + 1e-6
    Tcum = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), one_m_alpha], dim=-1), dim=-1)[..., :-1]
    w = alpha * Tcum
    C = (w.unsqueeze(-1) * rgb_line).sum(dim=-2)
    A = w.sum(dim=-1, keepdim=True).clamp(0, 1)
    # Composite over sky
    sky_bg = sky_color.view(1, 1, -1)
    C = C + (1.0 - A) * sky_bg
    I = torch.cat([C, A], dim=-1).permute(2, 0, 1)
    return I.unsqueeze(0)
