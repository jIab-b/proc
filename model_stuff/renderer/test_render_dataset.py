from pathlib import Path
from typing import Tuple
import torch
import numpy as np
from PIL import Image
from ..config import DATA_IMAGES, DEVICE, GRID_XYZ, IMG_HW, TEST_IMGS_DIR, MAPS_DIR
from ..materials import c_m, sigma_m, AIR
from .api import RendererConfig, DifferentiableRenderer
from ..dsl import actions_to_logits


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

    # Prefer rendering via DSL map in the same dataset sequence (maps/<seq>/map.json)
    dataset_seq = images_dir.parent.name  # e.g., '1'
    map_path = MAPS_DIR / dataset_seq / 'map.json'

    # Load dataset metadata for camera & resolution
    meta_file = images_dir.parent / 'metadata.json'
    meta = None
    if meta_file.exists():
        try:
            import json
            with meta_file.open('r', encoding='utf-8') as fh:
                meta = json.load(fh)
        except Exception:
            meta = None

    if meta is not None:
        w = int(meta.get('imageSize', {}).get('width', IMG_HW[1]))
        h = int(meta.get('imageSize', {}).get('height', IMG_HW[0]))
    else:
        h, w = IMG_HW

    # Build logits from DSL map if available; otherwise fall back to the image-colorization heuristic
    if map_path.exists():
        import json
        map_data = json.loads(map_path.read_text())
        blocks = map_data.get('blocks', [])
        # Convert blocks to actions for shared path
        actions = [{ 'type': 'place_block', 'params': { 'position': b.get('position'), 'blockType': b.get('blockType') } } for b in blocks]
        dims = map_data.get('worldConfig', {}).get('dimensions', {})
        X = int(dims.get('x', GRID_XYZ[0]))
        Y = int(dims.get('y', GRID_XYZ[1]))
        Z = int(dims.get('z', GRID_XYZ[2]))
        world_scale = float(map_data.get('worldScale', 2.0))
        num_mats = int(c_m.size(0))
        W_logits = actions_to_logits(actions, (X, Y, Z), num_mats, DEVICE)
        renderer = DifferentiableRenderer(sigma_m.to(DEVICE), c_m.to(DEVICE))
        views = meta.get('views', []) if meta else []
        for idx, view in enumerate(views[:len(files)]):
            vm = torch.tensor(view['viewMatrix'], dtype=torch.float32, device=DEVICE).view(4, 4)
            pm = torch.tensor(view['projectionMatrix'], dtype=torch.float32, device=DEVICE).view(4, 4)
            I = renderer.render(
                W_logits,
                RendererConfig(height=h, width=w, temperature=0.7, step_size=0.2, srgb=True, camera_view=vm, camera_proj=pm, world_scale=world_scale)
            )
            x = I.detach().cpu()[0, :3].permute(1, 2, 0).numpy()
            img_out = (x * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(img_out).save(out_dir / f"{idx:04d}.png")
        return

    # Fallback: colorize and render (orthographic or perspective if meta exists)
    X, Y, Z = GRID_XYZ
    num_mats = int(c_m.size(0))
    renderer = DifferentiableRenderer(sigma_m.to(DEVICE), c_m.to(DEVICE))
    for idx, fp in enumerate(files[:12]):
        img = Image.open(fp)
        img_r = resize_image_to_hw(img, (h, w))
        mat_idx = image_to_material_indices(img_r, c_m, dark_air_threshold=0.08)
        W_logits = build_w_logits_from_indices(mat_idx, (X, Y, Z), num_mats, DEVICE)
        if meta is not None and idx < len(meta.get('views', [])):
            view = meta['views'][idx]
            vm = torch.tensor(view['viewMatrix'], dtype=torch.float32, device=DEVICE).view(4, 4)
            pm = torch.tensor(view['projectionMatrix'], dtype=torch.float32, device=DEVICE).view(4, 4)
            I = renderer.render(
                W_logits,
                RendererConfig(height=h, width=w, temperature=0.7, step_size=0.2, srgb=True, camera_view=vm, camera_proj=pm, world_scale=2.0)
            )
        else:
            I = renderer.render(W_logits, RendererConfig(height=h, width=w, temperature=0.7, step_size=0.2, srgb=True))
        x = I.detach().cpu()[0, :3].permute(1, 2, 0).numpy()
        img_out = (x * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_out).save(out_dir / f"{idx:04d}.png")


if __name__ == "__main__":
    main()
