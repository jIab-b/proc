import os, subprocess, shutil
from pathlib import Path
from typing import Optional, List
import modal
from modal import Image, Volume
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
    ("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", "openclip-vit-b-32"),
]

MODEL_ROOT = Path("/workspace/models")
DEFAULT_SYNC_DIRS = ["model_stuff", "third_party"]

def compute_hashes(dir_path: str) -> dict:
    if not os.path.exists(dir_path):
        return {}
    hashes = {}
    for root, _, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, dir_path)
            # Normalize path separators to forward slashes for cross-platform compatibility
            rel_path = rel_path.replace("\\", "/")
            with open(file_path, 'rb') as f:
                h = hashlib.sha256(f.read()).hexdigest()
            hashes[rel_path] = h
    return hashes


GPU_KIND = os.environ.get("MODAL_GPU", "L4").upper()
GPU      = {"L4": "L4", "L40S": "L40S", "A100": "A100-40GB", "H100": "H100"}.get(GPU_KIND, "A100")





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

        files_to_upload = []
        for rel, lh in local_hashes.items():
            remote_file = os.path.join(remote_rel_root, rel).replace("\\", "/")
            if rel not in remote_hashes:
                reason = "new file"
                files_to_upload.append((rel, reason))
            elif remote_hashes[rel] != lh:
                reason = "hash changed"
                files_to_upload.append((rel, reason))

        if files_to_upload:
            print(f"Uploading {len(files_to_upload)} changed files:")
            for rel, reason in files_to_upload:
                print(f"  {rel} ({reason})")
                remote_file = os.path.join(remote_rel_root, rel).replace("\\", "/")
                local_file = os.path.join(src, rel)
                subprocess.run(["modal", "volume", "rm", "workspace", remote_file], check=False)
                subprocess.run(["modal", "volume", "put", "workspace", local_file, remote_file], check=True)
        else:
            print("No files need uploading - all hashes match")

        print(f"Done syncing {src}.")

    # Also sync top-level requirements.txt if present (needed for installs)
    # req = os.path.abspath("requirements.txt")
    # if os.path.exists(req):
    #     print("Syncing requirements.txt -> /workspace/requirements.txt ...")
    #     subprocess.run(["modal", "volume", "rm", "workspace", "requirements.txt"], check=False)
    #     subprocess.run(["modal", "volume", "put", "workspace", req, "requirements.txt"], check=True)
    #     print("Done syncing requirements.txt.")




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
    gpu=GPU,
    timeout=86400,
)
def run_smoke_remote(script: str = "run_smoke") -> str:
    import os, subprocess
    from pathlib import Path
    os.chdir("/workspace")
    Path("/workspace/out_local").mkdir(parents=True, exist_ok=True)
    cmd = f"source /workspace/venv/bin/activate && export HF_HUB_OFFLINE=1 && python -m model_stuff.{script}"
    print("Running:", cmd)
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")
    return "/workspace/out_local"


@app.function(
    image=image,
    volumes={"/workspace": splats_wspace},
    timeout=86400,
)
def prefetch_hf_models() -> str:
    import os
    from huggingface_hub import snapshot_download
    os.environ.setdefault("HF_HOME", "/workspace/hf")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/workspace/hf")
    Path("/workspace/hf").mkdir(parents=True, exist_ok=True)
    for repo, alias in MODEL_REPOS:
        target = f"/workspace/hf/{alias}"
        Path(target).mkdir(parents=True, exist_ok=True)
        print(f"Prefetching {repo} -> {target}")
        snapshot_download(repo_id=repo, local_dir=target, local_dir_use_symlinks=False)
    splats_wspace.commit()
    return "/workspace/hf"


@app.function(
    image=image,
    volumes={"/workspace": splats_wspace},
    timeout=86400,
)
def clear_out():
    import os, subprocess
    os.chdir("/workspace")
    if os.path.exists("/workspace/out_local"):
        shutil.rmtree("/workspace/out_local")
    os.makedirs("/workspace/out_local", exist_ok=True)
    

@app.local_entrypoint()
def run_model():
    

    script = "run_smoke"
    #script = "run_quick"
    _sync_workspace_dirs(["model_stuff"])
    path = prefetch_hf_models.remote()
    print(f"HF cached at {path}")
    run_smoke_remote.remote(script)
    sync_outputs(local_dir="./out_local")

