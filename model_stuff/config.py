"""
Training presets for SDS voxel optimization with SDXL guidance.

Small is a smoke test for ~8GB GPUs (e.g., RTX 2060 Super).
Medium and Large scale steps and resolution.
"""

from typing import Dict, Any


def preset_small(output_dir: str = "out_local/sds_training") -> Dict[str, Any]:
    return {
        # Core
        "steps": 60,
        "lr": 0.01,
        "cfg_scale": 6.0,
        "temp_start": 2.0,
        "temp_end": 0.5,
        # Reg
        "lambda_sparsity": 1e-3,
        "lambda_entropy": 1e-4,
        "lambda_smooth": 0.0,
        # Logging / I/O
        "log_every": 10,
        "image_every": 5,
        "save_map_every": 20,
        "output_dir": output_dir,
        # Render resolution (kept low for VRAM headroom)
        "train_h": 160,
        "train_w": 160,
    }


def preset_medium(output_dir: str = "out_local/sds_training") -> Dict[str, Any]:
    return {
        "steps": 300,
        "lr": 0.01,
        "cfg_scale": 7.5,
        "temp_start": 2.0,
        "temp_end": 0.5,
        "lambda_sparsity": 1e-3,
        "lambda_entropy": 1e-4,
        "lambda_smooth": 0.0,
        "log_every": 10,
        "image_every": 5,
        "save_map_every": 50,
        "output_dir": output_dir,
        "train_h": 256,
        "train_w": 256,
    }


def preset_large(output_dir: str = "out_local/sds_training") -> Dict[str, Any]:
    return {
        "steps": 1000,
        "lr": 0.01,
        "cfg_scale": 7.5,
        "temp_start": 2.0,
        "temp_end": 0.5,
        "lambda_sparsity": 1e-3,
        "lambda_entropy": 1e-4,
        "lambda_smooth": 0.0,
        "log_every": 10,
        "image_every": 5,
        "save_map_every": 50,
        "output_dir": output_dir,
        "train_h": 320,
        "train_w": 320,
    }


PRESETS = {
    "small": preset_small,
    "medium": preset_medium,
    "large": preset_large,
}


def get_preset(name: str, output_dir: str = "out_local/sds_training") -> Dict[str, Any]:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Valid: {list(PRESETS)}")
    return PRESETS[name](output_dir)

