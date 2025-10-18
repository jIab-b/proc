from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch

from dsl.python import parse_dsl, apply_actions, DSLAction, BlockTypeName
from ..materials import MATERIALS, c_m, sigma_m, AIR
from ..config import DEVICE, GRID_XYZ, IMG_HW, TEST_IMGS_DIR
from .api import RendererConfig, DifferentiableRenderer
from .textures import load_face_textures


class WorldGrid:
    def __init__(self, size: Tuple[int, int, int]):
        self.X, self.Y, self.Z = size
        # store block type indices matching MATERIALS names
        self.grid = torch.full((self.X, self.Y, self.Z), fill_value=AIR, dtype=torch.int64)

    def in_bounds(self, x: int, y: int, z: int) -> bool:
        return 0 <= x < self.X and 0 <= y < self.Y and 0 <= z < self.Z

    def set_block(self, x: int, y: int, z: int, block_type: BlockTypeName) -> None:
        if not self.in_bounds(x, y, z):
            return
        try:
            idx = MATERIALS.index(block_type)
        except ValueError:
            return
        self.grid[x, y, z] = int(idx)

    def clear_block(self, x: int, y: int, z: int) -> None:
        if self.in_bounds(x, y, z):
            self.grid[x, y, z] = AIR

    def to_logits(self, num_mats: int, device: str, logit_solid: float = 6.0) -> torch.Tensor:
        X, Y, Z = self.X, self.Y, self.Z
        W = torch.zeros((X, Y, Z, num_mats), device=device, dtype=torch.float32)
        # Write a strong logit on the selected material channel per voxel
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    m = int(self.grid[x, y, z].item())
                    if m == AIR:
                        continue
                    W[x, y, z, m] = logit_solid
        return W


def load_actions_from_path(path: Path) -> List[DSLAction]:
    text = path.read_text(encoding="utf-8")
    text_stripped = text.lstrip()
    # Accept JSON array of actions or free-form text to parse
    if text_stripped.startswith("["):
        data = json.loads(text)
        # Trust shape; minimal validation
        return data  # type: ignore[return-value]
    return parse_dsl(text)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a DSL scene with the differentiable renderer")
    ap.add_argument("dsl_file", type=str, help="Path to DSL text or JSON file")
    ap.add_argument("--out", type=str, default=str(TEST_IMGS_DIR / "dsl_render.png"))
    ap.add_argument("--size", type=str, default=None, help="HxW (e.g. 512x512). Defaults to dataset IMG_HW or metadata if used elsewhere.")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--step-size", dest="step_size", type=float, default=0.2)
    ap.add_argument("--use-textures", action="store_true", help="Use per-material face textures if present")
    ap.add_argument("--tile-size", type=int, default=64, help="Face texture tile size for loading")
    ap.add_argument("--normal-sharpness", type=float, default=10.0, help="Soft face-normal sharpness (0 disables)")
    ap.add_argument("--occupancy-hardness", type=float, default=12.0, help="Sigmoid hardening factor for occupancy (0 disables)")
    ap.add_argument("--occupancy-threshold", type=float, default=0.35, help="Sigmoid threshold for occupancy hardening")
    ap.add_argument("--texture-scale", type=float, default=1.0, help="UV scale inside a voxel for textures")
    ap.add_argument("--backend", type=str, default="vol", choices=["vol","nv"], help="Rendering backend: vol (default) or nv (nvdiffrast)")
    ap.add_argument("--parity", action="store_true", help="Enable WebGPU parity knobs (palette faces, no sRGB gamma, harder faces)")
    args = ap.parse_args()

    actions = load_actions_from_path(Path(args.dsl_file))
    X, Y, Z = GRID_XYZ
    world = WorldGrid((X, Y, Z))
    apply_actions(world, actions)

    num_mats = int(c_m.size(0))
    W_logits = world.to_logits(num_mats=num_mats, device=DEVICE)

    if args.size:
        try:
            h_str, w_str = args.size.lower().split("x", 1)
            H, W = int(h_str), int(w_str)
        except Exception:
            raise SystemExit("--size must be HxW, e.g. 512x512")
    else:
        H, W = IMG_HW

    texture_atlas = None
    if args.use_textures:
        try:
            texture_atlas = load_face_textures(MATERIALS, base_dir="textures", tile_size=args.tile_size, device=DEVICE)
        except Exception as e:
            print(f"[warn] Failed to load textures: {e}")
            texture_atlas = None
    renderer = DifferentiableRenderer(sigma_m.to(DEVICE), c_m.to(DEVICE), texture_atlas=texture_atlas)
    I = renderer.render(
        W_logits,
        RendererConfig(
            height=H,
            width=W,
            temperature=args.temperature,
            step_size=args.step_size,
            srgb=not args.parity,
            use_textures=bool(texture_atlas is not None and args.use_textures),
            normal_sharpness=args.normal_sharpness,
            occupancy_hardness=args.occupancy_hardness,
            occupancy_threshold=args.occupancy_threshold,
            texture_scale=args.texture_scale,
            backend=("nv" if args.backend=="nv" else None),
            webgpu_parity=bool(args.parity),
            use_palette_faces=bool(args.parity),
        ),
    )
    img = (I[0, :3].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype("uint8")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image  # type: ignore
        Image.fromarray(img).save(out_path)
    except Exception as e:
        # Fallback to numpy .npy if PIL not available
        npy_path = out_path.with_suffix(".npy")
        import numpy as np
        np.save(npy_path, img)
        print(f"PIL not available; wrote numpy array to {npy_path}")
        return
    print(str(out_path))


if __name__ == "__main__":
    main()
