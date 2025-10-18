from __future__ import annotations
from pathlib import Path
import json
import math
import torch

from ..config import DATA_IMAGES, GRID_XYZ


def _load_meta() -> dict | None:
    meta_file = Path(DATA_IMAGES).parent / 'metadata.json'
    if not meta_file.exists():
        return None
    try:
        return json.loads(meta_file.read_text())
    except Exception:
        return None


def _camera_origin_from_view(view: torch.Tensor) -> torch.Tensor:
    inv_view = torch.linalg.inv(view)
    o = inv_view @ torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=view.dtype, device=view.device)
    return o[:3]


def _project(point_ws: torch.Tensor, view: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    p = torch.cat([point_ws, torch.ones_like(point_ws[:1])])  # (4,)
    clip = proj @ (view @ p)
    ndc = clip[:3] / clip[3].clamp_min(1e-6)
    return ndc  # x,y in [-1,1]


def main() -> None:
    meta = _load_meta()
    if meta is None:
        print('[test_cameras] metadata.json not found; skipping camera checks.')
        return
    views = meta.get('views') or []
    if not views:
        print('[test_cameras] No views in metadata; skipping camera checks.')
        return
    v = views[0]
    vm = torch.tensor(v['viewMatrix'], dtype=torch.float32).view(4, 4)
    pm = torch.tensor(v['projectionMatrix'], dtype=torch.float32).view(4, 4)
    expected_pos = torch.tensor(v['position'], dtype=torch.float32)
    origin_row = _camera_origin_from_view(vm)
    err_row = torch.linalg.norm(origin_row - expected_pos).item()
    origin_col = _camera_origin_from_view(vm.T)
    err_col = torch.linalg.norm(origin_col - expected_pos).item()
    err = min(err_row, err_col)
    which = 'row' if err_row <= err_col else 'col(T)'
    print(f"[test_cameras] camera origin error: {err:.6f} using {which} orientation")
    assert err < 1e-3, (
        f"Camera origin mismatch. row_err={err_row:.6f} origin_row={origin_row.tolist()} "
        f"col_err={err_col:.6f} origin_col={origin_col.tolist()} expected={expected_pos.tolist()}"
    )

    # Project a center point and verify it lands inside the image bounds
    X, Y, Z = GRID_XYZ
    world_scale = float(meta.get('worldScale', 2.0)) if isinstance(meta.get('worldScale', None), (int, float)) else 2.0
    center_ws = torch.tensor([0.0, Y * world_scale * 0.5, 0.0], dtype=torch.float32)
    # Use the orientation that matched origin
    vm_use = vm if which == 'row' else vm.T
    pm_use = pm if which == 'row' else pm.T
    ndc = _project(center_ws, vm_use, pm_use)
    assert (ndc[:2].abs() <= 1.0 + 1e-4).all(), f"Projected center outside frustum: {ndc}"
    print('[test_cameras] OK')


if __name__ == '__main__':
    main()
