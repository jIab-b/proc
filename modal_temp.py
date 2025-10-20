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
    .env({
        "HF_HOME": "/workspace/hf",
        "HUGGINGFACE_HUB_CACHE": "/workspace/hf",
    })
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
DEFAULT_SYNC_DIRS = ["model_stuff", "datasets", "maps", "third_party"]

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
    gpu=GPU,
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

        # Replicate top-level folder into /workspace
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

    # Also sync top-level requirements.txt if present (needed for installs)
    req = os.path.abspath("requirements.txt")
    if os.path.exists(req):
        print("Syncing requirements.txt -> /workspace/requirements.txt ...")
        subprocess.run(["modal", "volume", "rm", "workspace", "requirements.txt"], check=False)
        subprocess.run(["modal", "volume", "put", "workspace", req, "requirements.txt"], check=True)
        print("Done syncing requirements.txt.")




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


# --- SDS training on Modal (minimal) ---

@app.function(
    image=image,
    volumes={"/workspace": splats_wspace},
    gpu=GPU,
    timeout=86400,
)
def run_sds_train(train_args: List[str]) -> str:
    """
    Minimal SDS training runner:
    - Creates venv in /workspace/venv
    - Installs requirements.txt with uv
    - Installs local third_party/nvdiffrast
    - Runs `python -m model_stuff.train_sds_final ...`
    - Returns the timestamped run directory path
    """
    import os, subprocess, sys, shlex
    from pathlib import Path

    os.environ.setdefault("HF_HOME", "/workspace/hf")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/workspace/hf")
    Path("/workspace/hf").mkdir(parents=True, exist_ok=True)
    Path("/workspace/out_local").mkdir(parents=True, exist_ok=True)

    # Change to workspace and run training with venv activated
    os.chdir("/workspace")
    quoted_args = ' '.join(shlex.quote(arg) for arg in train_args)
    cmd = f"source /workspace/venv/bin/activate && python -m model_stuff.train_sds_final {quoted_args}"
    print("Running:", cmd)
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")

    # Identify latest timestamped run dir to return
    runs = sorted(Path("/workspace/out_local/sds_training").glob("*/"), key=lambda p: p.name)
    return str(runs[-1]) if runs else "/workspace/out_local/sds_training"


@app.local_entrypoint()
def run_train(
    prompt: str = "a stone tower",
    dataset_id: int = 1,
    init_mode: str = "ground_plane",
    preset: str = "small",
    steps: int = 60,
    image_every: int = 5,
    save_map_every: int = 20,
    train_h: int = 160,
    train_w: int = 160,
    max_blocks: int = 20000,
    local_dir: str = "./out_local",
):
    """Sync code, run SDS training on Modal, then sync out_local locally."""
    # Sync essentials, including third_party (nvdiffrast)
    sync_workspace(["model_stuff"])

    # Build train args list (ASCII hyphens only)
    args = [
        "--prompt", prompt,
        "--dataset_id", str(dataset_id),
        "--init_mode", init_mode,
        "--preset", preset,
        "--steps", str(steps),
        "--image_every", str(image_every),
        "--save_map_every", str(save_map_every),
        "--train_h", str(train_h),
        "--train_w", str(train_w),
        "--max_blocks", str(max_blocks),
        "--output_dir", "/workspace/out_local/sds_training",
    ]

    run_dir = run_sds_train.remote(args)
    print(f"Remote run dir: {run_dir}")

    # Sync out_local back to local filesystem
    sync_outputs(local_dir=local_dir)
    print("Training outputs synced.")
