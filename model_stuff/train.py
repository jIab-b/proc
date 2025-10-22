"""End-to-end SDS training loop for voxel scenes."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .cameras import CameraSampler
from .dataset import MultiViewDataset
from .losses import photometric_loss, regularisation_losses
from .model import VoxelScene
from .renderer_nvd import VoxelRenderer
from .sds import score_distillation_loss
from .sdxl_lightning import SDXLLightning


# ----------------------------------------------------------------------
# Configuration container
# ----------------------------------------------------------------------


@dataclass
class TrainConfig:
    prompt: str
    dataset_id: int = 1
    map_path: Optional[str] = None
    steps: int = 200
    learning_rate: float = 0.01
    cfg_scale: float = 7.5
    temperature_start: float = 2.0
    temperature_end: float = 0.5
    sds_weight: float = 1.0
    photo_weight: float = 1.0
    lambda_sparsity: float = 1e-3
    lambda_entropy: float = 1e-4
    lambda_tv: float = 0.0
    novel_view_prob: float = 0.2
    train_height: int = 192
    train_width: int = 192
    max_blocks: Optional[int] = 50000
    output_dir: str = "out_local/voxel_sds"
    log_interval: int = 10
    image_interval: int = 50
    map_interval: int = 100
    seed: int = 42


# ----------------------------------------------------------------------


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_image(rgb: torch.Tensor, path: Path) -> None:
    arr = rgb.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    arr = (arr * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


# ----------------------------------------------------------------------


def train(config: TrainConfig) -> Path:
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MultiViewDataset(dataset_id=config.dataset_id, device=device)
    map_path = Path(config.map_path) if config.map_path else Path(f"maps/{config.dataset_id}/map.json")
    if not map_path.exists():
        raise FileNotFoundError(f"Map not found: {map_path}")

    scene = VoxelScene.from_map(map_path, device=device)
    renderer = VoxelRenderer(scene.grid)

    sampler = CameraSampler(
        dataset=dataset,
        grid_size=scene.grid.grid_size,
        world_scale=scene.world_scale,
        novel_view_prob=config.novel_view_prob,
        device=device,
    )

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    sdxl = SDXLLightning(
        model_id=model_id,
        device=device,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        height=config.train_height,
        width=config.train_width,
    )
    embeddings = sdxl.encode_prompt(config.prompt)

    optimizer = torch.optim.Adam(scene.parameters(), lr=config.learning_rate)

    run_dir = ensure_dir(Path(config.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S"))
    images_dir = ensure_dir(run_dir / "images")
    maps_dir = ensure_dir(run_dir / "maps")
    log_path = run_dir / "train.jsonl"

    with (run_dir / "config.json").open("w") as f:
        json.dump(asdict(config), f, indent=2)

    # Preview initial state
    jpg_view = dataset.get_view(0)
    preview_rgba = renderer.render(
        jpg_view.view_matrix,
        jpg_view.proj_matrix,
        config.train_height,
        config.train_width,
        temperature=config.temperature_start,
        max_blocks=config.max_blocks,
    )
    save_image(preview_rgba[0, :3], images_dir / "preview_init.png")

    for step in range(1, config.steps + 1):
        t = (step - 1) / max(config.steps - 1, 1)
        temperature = lerp(config.temperature_start, config.temperature_end, t)

        sample = sampler.sample()
        rgba = renderer.render(
            sample.view,
            sample.proj,
            config.train_height,
            config.train_width,
            temperature=temperature,
            max_blocks=config.max_blocks,
        )

        rgb_pred = rgba[:, :3]

        total_loss = torch.zeros((), device=device)
        losses: Dict[str, float] = {}

        # Photometric supervision (dataset cameras only)
        if sample.from_dataset and config.photo_weight > 0.0:
            gt = sample.rgb.unsqueeze(0).to(device)
            gt_resized = F.interpolate(
                gt,
                size=(config.train_height, config.train_width),
                mode="bilinear",
                align_corners=False,
            )
            l_photo = photometric_loss(rgb_pred, gt_resized) * config.photo_weight
            total_loss = total_loss + l_photo
            losses["photometric"] = float(l_photo.item())

        if config.sds_weight > 0.0:
            l_sds = score_distillation_loss(rgba, sdxl, embeddings, cfg_scale=config.cfg_scale) * config.sds_weight
            total_loss = total_loss + l_sds
            losses["sds"] = float(l_sds.item())

        occ = scene.occupancy_probs()
        mats = scene.material_probs(temperature=max(temperature, 1e-4))
        reg = regularisation_losses(
            occ_probs=occ,
            mat_probs=mats,
            lambda_sparsity=config.lambda_sparsity,
            lambda_entropy=config.lambda_entropy,
            lambda_tv=config.lambda_tv,
        )

        for key, value in reg.items():
            total_loss = total_loss + value
            losses[key] = float(value.item())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % config.log_interval == 0 or step == 1:
            stats = scene.stats()
            log_entry = {
                "step": step,
                "temperature": temperature,
                "loss_total": float(total_loss.item()),
                "losses": losses,
                "stats": stats,
                "sample": {
                    "from_dataset": sample.from_dataset,
                    "index": sample.index,
                },
            }
            with log_path.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")
            print(
                f"Step {step}/{config.steps} "
                f"loss={total_loss.item():.4f} "
                f"active={stats['num_active_voxels']}/{stats['total_voxels']}"
            )

        if step % config.image_interval == 0:
            save_image(rgb_pred[0], images_dir / f"step_{step:04d}.png")

        if step % config.map_interval == 0 or step == config.steps:
            scene.save_map(
                maps_dir / f"step_{step:04d}.json",
                metadata={"prompt": config.prompt, "step": step},
            )

    final_map = run_dir / "final_map.json"
    scene.save_map(final_map, metadata={"prompt": config.prompt, "steps": config.steps})
    save_image(rgb_pred[0], images_dir / "final.png")

    print(f"Training finished. Outputs saved in {run_dir}")
    return run_dir


# ----------------------------------------------------------------------


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Voxel SDS training")
    parser.add_argument("--prompt", required=True, help="Text prompt for SDS guidance")
    parser.add_argument("--dataset_id", type=int, default=1)
    parser.add_argument("--map_path", type=str, default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--temperature_start", type=float, default=2.0)
    parser.add_argument("--temperature_end", type=float, default=0.5)
    parser.add_argument("--sds_weight", type=float, default=1.0)
    parser.add_argument("--photo_weight", type=float, default=1.0)
    parser.add_argument("--lambda_sparsity", type=float, default=1e-3)
    parser.add_argument("--lambda_entropy", type=float, default=1e-4)
    parser.add_argument("--lambda_tv", type=float, default=0.0)
    parser.add_argument("--novel_view_prob", type=float, default=0.2)
    parser.add_argument("--train_height", type=int, default=192)
    parser.add_argument("--train_width", type=int, default=192)
    parser.add_argument("--max_blocks", type=int, default=50000)
    parser.add_argument("--output_dir", type=str, default="out_local/voxel_sds")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--image_interval", type=int, default=50)
    parser.add_argument("--map_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(**vars(args))
    if args.max_blocks <= 0:
        config.max_blocks = None
    return config


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
