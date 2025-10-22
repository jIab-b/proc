"""Dataset utilities for posed Minecraft-style renders.

Loads posed RGB images and associated camera intrinsics/extrinsics from
`datasets/<id>/` produced by the capture tooling. Each dataset entry
contains an `images/` directory of PNGs and a `metadata.json` describing
camera matrices in WebGPU column-major order. We convert the matrices to
PyTorch row-major form and expose helpers for sampling views during SDS
training.

The loader keeps metadata and camera tensors on the requested device but
lazily loads images from disk to avoid duplicating GPU memory. Images are
returned as float tensors in `[0, 1]` with shape `(3, H, W)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .nv_diff_render.utils import load_camera_matrices_from_metadata


@dataclass(frozen=True)
class ViewRecord:
    """Metadata bundle for a single posed capture."""

    index: int
    rgb_path: Path
    view_matrix: torch.Tensor
    proj_matrix: torch.Tensor
    width: int
    height: int
    intrinsics: Dict


class MultiViewDataset:
    """Lightweight dataset wrapper for `datasets/<id>` directories."""

    def __init__(
        self,
        dataset_id: int = 1,
        root: str | Path = "datasets",
        device: torch.device | str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        self.root = Path(root)
        self.dataset_id = dataset_id
        self.dataset_dir = self.root / str(dataset_id)
        metadata_path = self.dataset_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")

        with metadata_path.open("r") as f:
            self._metadata = json_load(f)

        self.image_dir = self.dataset_dir / "images"
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Dataset images directory missing: {self.image_dir}")

        self.device = torch.device(device)

        width = int(self._metadata["imageSize"]["width"])
        height = int(self._metadata["imageSize"]["height"])
        self._image_size = (height, width)

        records: List[ViewRecord] = []
        for view_meta in self._metadata["views"]:
            idx = int(view_meta["index"])
            rgb_rel = Path(view_meta["rgbPath"])
            rgb_path = self.image_dir / rgb_rel.name
            if not rgb_path.exists():
                raise FileNotFoundError(f"RGB image missing: {rgb_path}")
            view_mat, proj_mat = load_camera_matrices_from_metadata(self._metadata, idx)
            intrinsics = dict(view_meta.get("intrinsics", {}))
            records.append(
                ViewRecord(
                    index=idx,
                    rgb_path=rgb_path,
                    view_matrix=view_mat.to(self.device),
                    proj_matrix=proj_mat.to(self.device),
                    width=width,
                    height=height,
                    intrinsics=intrinsics,
                )
            )

        self._records = sorted(records, key=lambda r: r.index)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._records)

    @property
    def image_size(self) -> Tuple[int, int]:
        """Return `(height, width)` for source images."""
        return self._image_size

    @property
    def metadata(self) -> Dict:
        """Expose raw metadata for advanced uses (read-only)."""
        return self._metadata

    def get_view(self, index: int) -> ViewRecord:
        """Return the cached `ViewRecord` for `index` (0 ≤ idx < len)."""
        return self._records[index % len(self._records)]

    def sample_indices(self, count: int = 1) -> List[int]:
        idx = torch.randint(low=0, high=len(self._records), size=(count,))
        return idx.tolist()

    def load_image(self, record: ViewRecord) -> torch.Tensor:
        """Load an RGB image as `float32` tensor in `[0, 1]` with shape `(3,H,W)`."""
        with Image.open(record.rgb_path) as img:
            rgb = torch.from_numpy(np.array(img.convert("RGB"), dtype=np.uint8))
        rgb = rgb.permute(2, 0, 1).float() / 255.0
        return rgb

    def iter_views(self) -> Iterable[Tuple[ViewRecord, torch.Tensor]]:
        for rec in self._records:
            yield rec, self.load_image(rec)


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def json_load(fp) -> Dict:
    import json

    return json.load(fp)
