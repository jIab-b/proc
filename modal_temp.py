import os, shutil
from pathlib import Path
import modal
from modal import Image, Volume
from modal import gpu as mgpu
import argparse
import modal_utils as mu

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

GPU_KIND = os.environ.get("MODAL_GPU", "L4").upper()
GPU      = {"L4": "L4", "L40S": "L40S", "A100": "A100-40GB", "H100": "H100"}.get(GPU_KIND, "A100")
GPU_SPEC = {"L4": mgpu.L4(), "L40S": mgpu.L40S(), "A100": mgpu.A100(), "H100": mgpu.H100()}.get(GPU_KIND, mgpu.A100())

mu._image = image
mu._vol = splats_wspace
mu._gpu = GPU_SPEC
mu.app.bind(app)

MODEL_REPOS = [
    ("stabilityai/stable-diffusion-xl-base-1.0", "sdxl-base"),
    ("stabilityai/stable-diffusion-xl-refiner-1.0", "sdxl-refiner"),
    ("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", "openclip-vit-b-32"),
]

MODEL_ROOT = Path("/workspace/models")
DEFAULT_SYNC_DIRS = ["model_stuff", "third_party"]


 

 



@app.function(image=image, volumes={"/workspace": splats_wspace})
def shell():
    pass


@app.function(image=image, volumes={"/workspace": splats_wspace}, gpu=GPU_SPEC)
def gpu_shell():
    pass



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
    

    script = "-m model_stuff.run_smoke"
    mu.sync_workspace([
        "model_stuff",
    ], exclude=["**/*.pyc"], exclude_dirs_map={
        "model_stuff": ["**/__pycache__/**"],
    }) 
    mu.prefetch_hf_remote.remote(MODEL_REPOS, "/workspace/hf")
    try:
        splats_wspace.commit()
    except Exception:
        pass
    mu.run_scripts(script, venv="/workspace/venv", workdir="/workspace", gpu=GPU)
    mu.sync_outputs(local_dir="./out_local")

