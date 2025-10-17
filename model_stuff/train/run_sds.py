#!/usr/bin/env python3
"""End-to-end optimisation loop for the probabilistic voxel DSL.

At a high level the script performs the following steps:

1. Load the exported dataset (RGB images + camera metadata) from
   ``datasets/<sequence>/``.
2. Load the existing ``map.json`` that we intend to edit and initialise a
   probabilistic voxel grid whose logits mirror the current block layout.
3. Iterate through random camera views, render differentiable rays with the
   volume renderer, and optimise the logits so the rendered colours match the
   target pixels.
4. Write checkpoints, logs, and an updated ``map.json`` into ``out_local`` so
   they can be synced back to the local workstation.

The optimisation loss defaults to simple L2 reconstruction which is a safe
starting point on commodity GPUs.  The scaffolding for score distillation is in
place via the ``--mode sds`` flag, but it raises ``NotImplementedError`` until a
project-specific diffusion prior is integrated.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image

from model_stuff.dsl.probabilistic_world import BlockPalette, ProbabilisticVoxelGrid
from model_stuff.render.volume_renderer import CameraView, VolumeRenderer
from model_stuff.utils.logging_utils import configure_logging, create_run_directory


@dataclass
class ViewRecord:
    camera: CameraView
    image: torch.Tensor  # (H, W, 3) in [0, 1]
    name: str


class DatasetBatcher:
    def __init__(self, dataset_dir: Path, device: torch.device) -> None:
        self.dataset_dir = dataset_dir
        self.device = device
        self.metadata = self._load_metadata(dataset_dir)
        self.views = self._load_views()
        logging.info("Loaded dataset with %d views from %s", len(self.views), dataset_dir)

    def _load_metadata(self, dataset_dir: Path) -> Dict[str, object]:
        meta_path = dataset_dir / "metadata.json"
        with meta_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data

    def _load_views(self) -> List[ViewRecord]:
        views: List[ViewRecord] = []
        image_size = self.metadata.get("imageSize", {})
        for view in self.metadata.get("views", []):
            image_rel = view.get("rgbPath")
            if not image_rel:
                continue
            img_path = self.dataset_dir / image_rel
            if not img_path.exists():
                logging.warning("Image %s referenced by metadata but not found", img_path)
                continue

            pil_img = Image.open(img_path).convert("RGB")
            img_np = np.array(pil_img, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(img_np).to(self.device)

            camera = CameraView.from_metadata(view, image_size=image_size, device=self.device)
            views.append(ViewRecord(camera=camera, image=image_tensor, name=view.get("id", "view")))
        if not views:
            raise RuntimeError("No valid views were loaded from the dataset")
        return views

    def sample_batch(self, rays_per_batch: int) -> Tuple[CameraView, torch.Tensor, torch.Tensor, str]:
        view = random.choice(self.views)
        h, w, _ = view.image.shape
        ys = torch.randint(0, h, (rays_per_batch,), device=self.device)
        xs = torch.randint(0, w, (rays_per_batch,), device=self.device)
        pixel_indices = torch.stack([ys, xs], dim=1)
        colours = view.image[ys, xs]
        return view.camera, pixel_indices, colours, view.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimise voxel DSL with differentiable rendering")
    parser.add_argument("--dataset-sequence", type=int, help="Dataset sequence ID under datasets/", required=False)
    parser.add_argument("--dataset-dir", type=str, help="Path to dataset directory (overrides --dataset-sequence)")
    parser.add_argument("--map-sequence", type=int, help="Map sequence ID under maps/", required=False)
    parser.add_argument("--map-path", type=str, help="Path to map.json (overrides --map-sequence)")
    parser.add_argument("--output-dir", type=str, default="out_local", help="Directory for run artefacts")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--rays-per-batch", type=int, default=2048)
    parser.add_argument("--num-samples", type=int, default=64, help="Samples per ray for volume rendering")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--min-probability", type=float, default=0.6, help="Threshold when collapsing logits to blocks")
    parser.add_argument("--step-size", type=float, default=None, help="Override ray step size (world units)")
    parser.add_argument("--mode", choices=["l2", "sds"], default="l2", help="Optimisation loss")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    dataset_dir: Optional[Path] = None
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
    elif args.dataset_sequence is not None:
        dataset_dir = Path("datasets") / str(args.dataset_sequence)
    if not dataset_dir or not dataset_dir.exists():
        raise FileNotFoundError("Dataset directory not found. Provide --dataset-dir or --dataset-sequence")

    map_path: Optional[Path] = None
    if args.map_path:
        map_path = Path(args.map_path)
    elif args.map_sequence is not None:
        map_path = Path("maps") / str(args.map_sequence) / "map.json"
    if not map_path or not map_path.exists():
        raise FileNotFoundError("Map file not found. Provide --map-path or --map-sequence")

    return dataset_dir.resolve(), map_path.resolve()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    dataset_dir, map_path = resolve_paths(args)
    output_root = Path(args.output_dir)
    run_dir = create_run_directory(output_root, run_name=args.run_name)
    configure_logging(run_dir)
    logging.info("Starting training run: dataset=%s map=%s", dataset_dir, map_path)

    device = torch.device(args.device)
    palette = BlockPalette.default(device=device)
    grid = ProbabilisticVoxelGrid.from_map(map_path, palette=palette, device=device)
    grid.logits = nn.Parameter(grid.logits.requires_grad_(True))

    renderer = VolumeRenderer(grid, device=device, step_size=args.step_size)
    dataset = DatasetBatcher(dataset_dir=dataset_dir, device=device)

    optimiser = torch.optim.Adam([grid.logits], lr=args.lr)

    losses: List[float] = []
    for step in range(1, args.steps + 1):
        camera, pixel_indices, target_colours, view_name = dataset.sample_batch(args.rays_per_batch)

        optimiser.zero_grad(set_to_none=True)
        renderer.refresh_volumes()
        pred_rgb, _, _ = renderer.render_pixels(
            camera=camera,
            pixel_indices=pixel_indices,
            num_samples=args.num_samples,
        )

        if args.mode == "l2":
            loss = F.mse_loss(pred_rgb, target_colours)
        else:
            raise NotImplementedError("Score distillation is not yet integrated for this project")

        loss.backward()
        optimiser.step()

        losses.append(loss.item())

        if step % args.log_every == 0:
            with torch.no_grad():
                probs = grid.material_probabilities().mean(dim=(0, 1, 2))
                entropy = (-probs * (probs + 1e-9).log()).sum().item()
                logging.info(
                    "step=%04d loss=%.6f entropy=%.4f target_view=%s",
                    step,
                    loss.item(),
                    entropy,
                    view_name,
                )

        if step % args.eval_every == 0:
            checkpoint_path = run_dir / f"logits_step_{step:04d}.pt"
            torch.save(grid.logits.detach().cpu(), checkpoint_path)
            logging.info("Saved checkpoint to %s", checkpoint_path)

    # Final outputs -----------------------------------------------------------------
    final_checkpoint = run_dir / "logits_final.pt"
    torch.save(grid.logits.detach().cpu(), final_checkpoint)
    logging.info("Stored final logits checkpoint at %s", final_checkpoint)

    updated_map_path = run_dir / "map.json"
    grid.update_map_json(
        input_map_path=map_path,
        output_map_path=updated_map_path,
        min_probability=args.min_probability,
    )
    logging.info("Wrote updated map JSON to %s", updated_map_path)

    metrics = {
        "steps": args.steps,
        "loss_history": losses,
        "final_loss": losses[-1] if losses else None,
        "dataset_dir": str(dataset_dir),
        "map_path": str(map_path),
    }
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logging.info("Dumped metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
