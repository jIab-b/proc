from __future__ import annotations
from pathlib import Path
from typing import Iterable, Sequence

import torch

FACE_ORDER: Sequence[str] = ("top", "bottom", "north", "south", "east", "west")


def _try_face_paths(base: Path, material: str, face: str) -> list[Path]:
    cand: list[Path] = []
    # Common layouts
    cand.append(base / material / f"{face}.png")
    cand.append(base / material / f"{face}.jpg")
    cand.append(base / f"{material}_{face}.png")
    cand.append(base / f"{material}_{face}.jpg")
    # Lowercase variants
    m = material.lower()
    f = face.lower()
    cand.append(base / m / f"{f}.png")
    cand.append(base / m / f"{f}.jpg")
    cand.append(base / f"{m}_{f}.png")
    cand.append(base / f"{m}_{f}.jpg")
    # Generic tiles directory
    cand.append(base / "tiles" / m / f"{f}.png")
    cand.append(base / "tiles" / m / f"{f}.jpg")
    return cand


def load_face_textures(
    materials: Iterable[str],
    base_dir: str | Path = "textures",
    tile_size: int = 64,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Load per-material, per-face textures as an atlas tensor.

    Returns a tensor of shape (M, 6, 4, H, W) in RGBA with values in [0,1].
    Missing faces default to alpha=0 (falls back to base material color).
    """
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PIL is required to load textures") from exc

    base = Path(base_dir)
    mats = list(materials)
    M = len(mats)
    H = W = int(tile_size)
    out = torch.zeros((M, 6, 4, H, W), dtype=torch.float32)
    for mi, mat in enumerate(mats):
        for fi, face in enumerate(FACE_ORDER):
            img_path = None
            for p in _try_face_paths(base, mat, face):
                if p.exists():
                    img_path = p
                    break
            if img_path is None:
                # leave zeros (alpha=0)
                continue
            img = Image.open(img_path).convert("RGBA").resize((W, H), Image.NEAREST)
            arr = torch.from_numpy(torch.ByteTensor(bytearray(img.tobytes()))).view(H, W, 4).float() / 255.0
            # to (4,H,W)
            arr = arr.permute(2, 0, 1).contiguous()
            out[mi, fi] = arr
    if device is not None:
        out = out.to(device)
    return out

