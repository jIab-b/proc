import os
from pathlib import Path
from mlrunner import MLRunner

MODEL_REPOS = [
    ("stabilityai/stable-diffusion-xl-base-1.0", "sdxl-base"),
    ("stabilityai/stable-diffusion-xl-refiner-1.0", "sdxl-refiner"),
    ("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", "openclip-vit-b-32"),
]

runner = MLRunner(
    backend="modal",
    storage={"models": MODEL_REPOS},
)

def run_model():
    script = "-m model_stuff.run_smoke"
    runner.run(code=script, output_dir="./out_local")
    runner.sync_outputs(local_dir="./out_local")
if __name__ == "__main__":
    run_model()
