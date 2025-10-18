# modal_app.py
import os, signal, subprocess, json, shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import modal
from modal import Image, Volume, gpu
import argparse
import zipfile
import io
import hashlib

app = modal.App("voxels")


splats_wspace = Volume.from_name("workspace", create_if_missing=True)



image = (
    Image.from_registry("pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel")
    .env({"HF_HOME": "/workspace/hf"})
    .run_commands(
        "apt-get update && apt-get install -y curl ca-certificates gnupg",
        "curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i /tmp/cuda-keyring.deb && rm -f /tmp/cuda-keyring.deb",
        "apt-get update",
    )
    .apt_install(
        # Common tools
        "git", "wget", "curl", "build-essential", "ccache", "gdb",
        "cargo", "rustc", "pkg-config", "cmake", "ninja-build",
        "libnuma-dev",            # required by MSCCl++ and NUMA-aware components
        "rdma-core", "libibverbs-dev",  # optional RDMA/IB verbs support (non-fatal if unused)
        "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "libgomp1",
    )
    .uv_pip_install(
        "uv",
        "ninja",
        "setuptools",
        "wheel",
        "numpy",
        "pybase64",
        "huggingface_hub",
        "diffusers",
        "safetensors",
        "pillow",
        "torch",
        "torchvision",
        "transformers",
        "accelerate"
    )
)

MODEL_REPOS = [
    ("stabilityai/stable-diffusion-xl-base-1.0", "sdxl-base"),
    ("stabilityai/stable-diffusion-xl-refiner-1.0", "sdxl-refiner"),
]

MODEL_ROOT = Path("/workspace/models")
DEFAULT_SYNC_DIRS = ["model_stuff", "datasets", "maps"]

def compute_hashes(dir_path: str) -> dict:
    if not os.path.exists(dir_path):
        return {}
    hashes = {}
    for root, _, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, dir_path)
            with open(file_path, 'rb') as f:
                h = hashlib.sha256(f.read()).hexdigest()
            hashes[rel_path] = h
    return hashes


GPU_KIND = os.environ.get("MODAL_GPU", "L4").upper()
GPU      = {"L4": "L4", "L40S": "L40S", "A100": "A100-40GB", "H100": "H100"}.get(GPU_KIND, "A100")






@app.function(
    image=image,
    volumes={"/workspace": splats_wspace},
    gpu=gpu.A100(),
)
def run_sdxl_lightning_infer(
    prompt: str = "A girl smiling",
    num_inference_steps: int = 4,
    guidance_scale: float = 0,
    output_filename: str = "output.png"
) -> str:
    """Run SDXL Lightning inference with 4-step UNet and save output to /workspace/out_local."""
    import sys
    sys.path.insert(0, "/workspace/model_stuff")
    from infer import sdxl_lightning_infer

    return sdxl_lightning_infer(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_filename=output_filename,
        output_dir="/workspace/out_local"
    )


@app.local_entrypoint()
def run_infer(
    prompt: str = "A girl smiling",
    num_inference_steps: int = 4,
    guidance_scale: float = 0,
    output_filename: str = "output.png",
    local_dir: str = "./out_local"
):
    """Run SDXL Lightning inference on Modal and sync outputs to local directory."""
    print(f"Starting SDXL Lightning inference with prompt: '{prompt}'")

    # Upload model_stuff to Modal workspace
    print("Uploading model_stuff to workspace...")
    sync_workspace(["model_stuff"])

    # Run inference on Modal
    result = run_sdxl_lightning_infer.remote(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_filename=output_filename
    )
    print(f"Inference completed: {result}")

    # Sync outputs to local directory
    print(f"Syncing outputs to {local_dir}...")
    sync_outputs(local_dir=local_dir)
    print("Done!")


def _sync_workspace_dirs(dirs_to_sync):
    for src in dirs_to_sync:
        src = os.path.abspath(src)
        base = os.path.basename(src.rstrip(os.sep))
        if not os.path.exists(src):
            print(f"Skipping {src}; local path missing")
            continue

        remote_rel_root = base  # Path inside the Modal volume
        remote_abs_root = f"/workspace/{remote_rel_root}"

        print(f"Syncing {src} -> {remote_abs_root} ...")
        local_hashes = compute_hashes(src)
        remote_hashes = get_remote_hashes.remote(remote_abs_root)

        for rel in set(remote_hashes) - set(local_hashes):
            remote_file = os.path.join(remote_rel_root, rel).replace("\\", "/")
            subprocess.run(["modal", "volume", "rm", "workspace", remote_file], check=False)

        for rel, lh in local_hashes.items():
            remote_file = os.path.join(remote_rel_root, rel).replace("\\", "/")
            if rel not in remote_hashes or remote_hashes[rel] != lh:
                local_file = os.path.join(src, rel)
                subprocess.run(["modal", "volume", "rm", "workspace", remote_file], check=False)
                subprocess.run(["modal", "volume", "put", "workspace", local_file, remote_file], check=True)

        print(f"Done syncing {src}.")




@app.local_entrypoint()
def sync_workspace(dirs_to_sync=None):
    if dirs_to_sync is None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--dir",
            action="append",
            default=[],
            help=f"Additional directories to sync to workspace (defaults to {DEFAULT_SYNC_DIRS} if none provided)",
        )
        args, unknown = parser.parse_known_args()
        dirs_to_sync = args.dir
        if not dirs_to_sync:
            dirs_to_sync = DEFAULT_SYNC_DIRS

    _sync_workspace_dirs(dirs_to_sync)


@app.function(
    image=image,
    volumes={"/workspace": splats_wspace},
)
def zip_remote_dir(remote_dir: str = "/workspace/out_local") -> Optional[bytes]:
    if not os.path.exists(remote_dir):
        print(f"Remote path '{remote_dir}' not found")
        return None

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(remote_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, remote_dir)
                zipf.write(file_path, arcname)
    
    zip_buffer.seek(0)
    return zip_buffer.read()

@app.function(
    image=image,
    volumes={"/workspace": splats_wspace},
)
def get_remote_hashes(remote_dir: str) -> dict:
    return compute_hashes(remote_dir)


@app.local_entrypoint()
def sync_outputs(local_dir: str = "./out_local"):
    local_path = Path(local_dir).expanduser().resolve()
    remote_dir = "/workspace/out_local"

    print(f"Checking remote directory: {remote_dir}")
    remote_hashes = get_remote_hashes.remote(remote_dir)
    print(f"Remote files: {list(remote_hashes.keys())}")

    if not remote_hashes:
        print("No files found at /workspace/out_local")
        if local_path.exists():
            shutil.rmtree(local_path)
        local_path.mkdir(parents=True, exist_ok=True)
        print(f"Cleared {local_path}")
        return

    local_hashes = compute_hashes(str(local_path)) if local_path.exists() else {}

    if local_hashes == remote_hashes:
        print("Local already up to date")
        return

    print(f"Downloading {len(remote_hashes)} files from /workspace/out_local to {local_path}")
    zip_data = zip_remote_dir.remote()

    if local_path.exists():
        shutil.rmtree(local_path)
    local_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zipf:
        zipf.extractall(local_path)

    print(f"Downloaded to {local_path}")


@app.function(
    image=image,
    volumes={"/workspace": splats_wspace},
    gpu=gpu.A100(),
)
def run_probabilistic_training(config: dict) -> dict:
    import os
    import subprocess

    os.chdir("/workspace")

    args = ["python", "-m", "model_stuff.train_sds"]

    print(f"[modal] Executing training command: {' '.join(args)}")
    subprocess.run(args, check=True)
    return {"command": args, "output_dir": config.get("output_dir", "out_local")}


@app.local_entrypoint()
def train_voxel(
    dataset_sequence: int = 1,
    map_sequence: int = 1,
    steps: int = 500,
    rays_per_batch: int = 4096,
    num_samples: int = 96,
    lr: float = 5e-3,
    mode: str = "l2",
    prompt: str = "",
    negative_prompt: str = "",
    guidance_scale: float = 3.0,
    lambda_sds: float = 1.0,
    lambda_l2: float = 1.0,
    sds_image_size: int = 256,
    snapshot_every: int = 100,
    snapshot_view: Optional[str] = None,
    log_every: int = 10,
    eval_every: int = 100,
    min_probability: float = 0.6,
    step_size: Optional[float] = None,
    sdxl_root: str = "/workspace/models/sdxl-base",
    lightning_repo: str = "ByteDance/SDXL-Lightning",
    lightning_ckpt: str = "sdxl_lightning_4step_unet.safetensors",
    seed: int = 42,
    run_name: Optional[str] = None,
):
    """Sync assets, launch the differentiable training job on Modal, then download outputs."""

    dirs = DEFAULT_SYNC_DIRS.copy()
    if "out_local" not in dirs:
        dirs.append("out_local")
    _sync_workspace_dirs(dirs)

    config = {
        "dataset_sequence": dataset_sequence,
        "map_sequence": map_sequence,
        "steps": steps,
        "rays_per_batch": rays_per_batch,
        "num_samples": num_samples,
        "lr": lr,
        "mode": mode,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "guidance_scale": guidance_scale,
        "lambda_sds": lambda_sds,
        "lambda_l2": lambda_l2,
        "sds_image_size": sds_image_size,
        "snapshot_every": snapshot_every,
        "snapshot_view": snapshot_view,
        "log_every": log_every,
        "eval_every": eval_every,
        "min_probability": min_probability,
        "step_size": step_size,
        "sdxl_root": sdxl_root,
        "lightning_repo": lightning_repo,
        "lightning_ckpt": lightning_ckpt,
        "seed": seed,
        "output_dir": "/workspace/out_local",
        "run_name": run_name,
    }
    print("Dispatching training job with config:", json.dumps(config, indent=2))
    print("Launching training: python -m model_stuff.train_sds on Modal")

    print("Ensuring model assets are available on the worker...")
    download_ml_models.remote()
    print("Model assets check initiated. Starting training...")
    
    result = run_probabilistic_training.remote(config)
    print("Training finished. Result:", result)

    print("Syncing outputs back to local machine...")
    sync_outputs()
    print("All outputs available under ./out_local")


@app.function(
    image=image,
    volumes={"/workspace": splats_wspace},
)
def shell():
    import sys
    import subprocess
    subprocess.call(["/bin/bash"], cwd="/workspace", stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)


@app.function(
    image=image,
    volumes={"/workspace": splats_wspace},
    timeout=3600,
)
def download_ml_models():
    """Download pretrained assets into the shared Modal volume."""
    from huggingface_hub import snapshot_download

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")

    for repo_id, alias in MODEL_REPOS:
        target_dir = MODEL_ROOT / alias
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"[modal] Downloading {repo_id} -> {target_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            token=hf_token,
        )

    print("[modal] Model assets synced to /workspace/models")


@app.local_entrypoint()
def prepare_models():
    """Trigger remote model downloads so Flux/ControlNet assets are ready."""
    print("Dispatching download_ml_models job to Modal...")
    download_ml_models.remote()
    print("Download initiated. Monitor Modal logs for completion details.")
