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

app = modal.App("splats")


splats_wspace = Volume.from_name("workspace", create_if_missing=True)

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


image = (
    Image.from_registry("pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel")
    .env({"HF_HOME": "/workspace/hf"})
    .run_commands(
        # Prepare to add NVIDIA CUDA APT repo (for Nsight CLI tools)
        "apt-get update && apt-get install -y curl ca-certificates gnupg",
        "curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i /tmp/cuda-keyring.deb && rm -f /tmp/cuda-keyring.deb",
        "apt-get update",
    )
    .apt_install(
        # Common tools
        "git", "wget", "curl", "build-essential", "ccache", "gdb",
        # Rust toolchain + native build helpers for building Rust/Python extensions
        "cargo", "rustc", "pkg-config", "cmake", "ninja-build",
        # Kernel build deps
        "libnuma-dev",            # required by MSCCl++ and NUMA-aware components
        "rdma-core", "libibverbs-dev",  # optional RDMA/IB verbs support (non-fatal if unused)
        # Nsight CLI tools matching CUDA 12.8
        #"cuda-nsight-systems-12-8", "cuda-nsight-compute-12-8",
        # OpenGL libraries for headless rendering
        "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "libgomp1",
    )
    .uv_pip_install(
        # Ensure uv is present for runtime `python3 -m uv ...`
        "uv",
        # Build backends/tools for no-build-isolation flows
        "scikit-build-core",   # backend for sgl-kernel
        "setuptools-rust",     # backend for sgl-router
        "ninja",
        "setuptools",
        "wheel",
        "numpy",               # quiets PyTorch's numpy warning during configure
        # Runtime deps (kept)
        "pybase64",
        "huggingface_hub",
    )
)



GPU_KIND = os.environ.get("MODAL_GPU", "L4").upper()
GPU      = {"L4": "L4", "L40S": "L40S", "A100": "A100-40GB", "H100": "H100"}.get(GPU_KIND, "A100")






@app.local_entrypoint()
def sync_workspace():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", action="append", default=[], help="Additional directories to sync to workspace (defaults to ['dreamgaussian', 'diff-gaussian-rasterization'] if none provided)")
    args, unknown = parser.parse_known_args()
    dirs_to_sync = args.dir
    if not dirs_to_sync:
        dirs_to_sync = ['zero123plus', 'images']
        #dirs_to_sync = ['images']
    for src in dirs_to_sync:
        dest = f"/{os.path.basename(src)}"
        print(f"Syncing {src} -> {dest} ...")
        local_hashes = compute_hashes(src)
        remote_hashes = get_remote_hashes.remote(dest)
        for rel in set(remote_hashes) - set(local_hashes):
            remote_file = os.path.join(dest, rel).lstrip(os.sep)
            subprocess.run(["modal", "volume", "rm", "workspace", remote_file], check=False)
        for rel, lh in local_hashes.items():
            if rel not in remote_hashes or remote_hashes[rel] != lh:
                local_file = os.path.join(src, rel)
                remote_file = os.path.join(dest, rel).lstrip(os.sep)
                subprocess.run(["modal", "volume", "put", "workspace", local_file, remote_file], check=True)
        print(f"Done syncing {src}.")




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
    print(f"Downloading /workspace/out_local to {local_path}")
    remote_dir = "/workspace/out_local"
    remote_hashes = get_remote_hashes.remote(remote_dir)
    local_hashes = compute_hashes(str(local_path)) if local_path.exists() else {}
    if local_hashes == remote_hashes:
        print("Local already up to date")
        return
    zip_data = zip_remote_dir.remote()
    if not zip_data:
        print("No data found at /workspace/out_local")
        if local_path.exists():
            shutil.rmtree(local_path)
        local_path.mkdir(parents=True, exist_ok=True)
        print(f"Cleared {local_path}")
        return
    if local_path.exists():
        shutil.rmtree(local_path)
    local_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zipf:
        zipf.extractall(local_path)
    print(f"Downloaded to {local_path}")


@app.function(
    image=image,
    volumes={"/workspace": splats_wspace},
)
def shell():
    import sys
    import subprocess
    subprocess.call(["/bin/bash", "-c", "cd /workspace && /bin/bash"], stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)

