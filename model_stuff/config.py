from pathlib import Path
import torch

# Hardcoded environment: "local" or "modal"
ENV_MODE = "local"

if ENV_MODE == "modal":
    ROOT = Path("/workspace")
else:
    ROOT = Path.cwd()

DATASETS_DIR = ROOT / "datasets"
MAPS_DIR = ROOT / "maps"
OUT_LOCAL_DIR = ROOT / "out_local"

DATA_IMAGES = DATASETS_DIR / "1/images"
MAP_OUT = (ROOT / "model_stuff/map.json") if ENV_MODE == "modal" else (MAPS_DIR / "1/map.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRID_XYZ = (32, 32, 32)
IMG_HW = (192, 192)
CFG_SCALE = 5.0
STEPS = 50
LR = 1e-2
TEMP_START = 2.0
TEMP_END = 0.5
SEED = 42
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# Logging / outputs
LOGS_DIR = OUT_LOCAL_DIR / "logs"
LOGS_IMAGES_DIR = LOGS_DIR / "images"
TEST_IMGS_DIR = OUT_LOCAL_DIR / "test_imgs"
LOG_EVERY = 10
IMAGE_EVERY = 50


