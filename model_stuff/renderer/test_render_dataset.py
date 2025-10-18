from pathlib import Path
from typing import Tuple
import torch
import numpy as np
from PIL import Image
from ..config import DATA_IMAGES, DEVICE, GRID_XYZ, IMG_HW, TEST_IMGS_DIR
from ..materials import c_m, sigma_m, AIR
from .api import RendererConfig, DifferentiableRenderer


def resize_image_to_hw(img: Image.Image, hw: Tuple[int, int]) -> Image.Image:
    h, w = hw
    return img.resize((w, h), Image.BILINEAR)


def image_to_material_indices(img: Image.Image, palette: torch.Tensor, dark_air_threshold: float = 0.08) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    h, w, _ = arr.shape
    lum = (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2])
    pal = palette[1:].cpu().numpy()
    arr_flat = arr.reshape(-1, 3)[:, None, :]
    pal_flat = pal[None, :, :]
    d2 = ((arr_flat - pal_flat) ** 2).sum(-1)
    idx_non_air = d2.argmin(axis=1).astype(np.int64) + 1
    idx = idx_non_air.reshape(h, w)
    idx[lum < dark_air_threshold] = AIR
    return idx


def build_w_logits_from_indices(indices: np.ndarray, grid_xyz: Tuple[int, int, int], num_mats: int, device: str) -> torch.Tensor:
    X, Y, Z = grid_xyz
    h, w = indices.shape
    xs = np.linspace(0, X - 1, w).round().astype(np.int64)
    ys = np.linspace(0, Y - 1, h).round().astype(np.int64)
    W = torch.zeros((X, Y, Z, num_mats), device=device, dtype=torch.float32)
    z_mid = Z // 2
    for j, yy in enumerate(ys):
        for i, xx in enumerate(xs):
            m = int(indices[j, i])
            if m == AIR:
                continue
            z0 = max(0, z_mid - 1)
            z1 = min(Z - 1, z_mid + 1)
            W[xx, yy, z0:z1 + 1, m] = 6.0
    return W


def main() -> None:
    out_dir = TEST_IMGS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(DATA_IMAGES)
    files = sorted([p for p in images_dir.glob("*.png")])
    if not files:
        return
    h, w = IMG_HW
    X, Y, Z = GRID_XYZ
    num_mats = int(c_m.size(0))
    renderer = DifferentiableRenderer(sigma_m.to(DEVICE), c_m.to(DEVICE))
    for idx, fp in enumerate(files[:12]):
        img = Image.open(fp)
        img_r = resize_image_to_hw(img, (h, w))
        mat_idx = image_to_material_indices(img_r, c_m, dark_air_threshold=0.08)
        W_logits = build_w_logits_from_indices(mat_idx, (X, Y, Z), num_mats, DEVICE)
        I = renderer.render(W_logits, RendererConfig(image_height=h, image_width=w, temperature=0.7))
        x = I.detach().cpu()[0, :3].permute(1, 2, 0).numpy()
        img_out = (x * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_out).save(out_dir / f"{idx:04d}.png")


if __name__ == "__main__":
    main()


