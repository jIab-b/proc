"""
Training presets for SDS voxel optimization with SDXL guidance.

Small is a smoke test for ~8GB GPUs (e.g., RTX 2060 Super).
Medium and Large scale steps and resolution.
Tiny is ultra-reductionist for 4GB GPUs (~RTX 2050 / 3050).
"""

from typing import Dict, Any


def preset_tiny(output_dir: str = "out_local/sds_training") -> Dict[str, Any]:
    return {
        "steps": 20,
        "lr": 0.02,
        "cfg_scale": 5.0,
        "temp_start": 2.0,
        "temp_end": 0.5,
        "lambda_sparsity": 5e-3,
        "lambda_entropy": 1e-4,
        "lambda_smooth": 0.0,
        "log_every": 5,
        "image_every": 2,
        "save_map_every": 10,
        "output_dir": output_dir,
        "train_h": 96,
        "train_w": 96,
        "max_blocks": 1000,
    }


def preset_small(output_dir: str = "out_local/sds_training") -> Dict[str, Any]:
    return {
        # Core
        "steps": 60,
        "lr": 0.01,
        "cfg_scale": 1.0,
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
        # Renderer safety caps
        "max_blocks": 20000,
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
        "max_blocks": 60000,
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
        "max_blocks": 120000,
    }


PRESETS = {
    "tiny": preset_tiny,
    "small": preset_small,
    "medium": preset_medium,
    "large": preset_large,
}


def get_preset(name: str, output_dir: str = "out_local/sds_training") -> Dict[str, Any]:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Valid: {list(PRESETS)}")
    return PRESETS[name](output_dir)
