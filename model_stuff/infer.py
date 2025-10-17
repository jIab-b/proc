"""SDXL Lightning inference module."""
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def sdxl_lightning_infer(
    prompt: str = "A girl smiling",
    num_inference_steps: int = 4,
    guidance_scale: float = 0,
    output_filename: str = "output.png",
    output_dir: str = "/workspace/out_local"
) -> str:
    """Run SDXL Lightning inference with 4-step UNet and save output."""
    # Ensure output directory exists
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_4step_unet.safetensors"

    print(f"Loading SDXL Lightning base model from {base}...")
    # Load model.
    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base,
        unet=unet,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")

    # Ensure sampler uses "trailing" timesteps.
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing"
    )

    print(f"Running inference with prompt: '{prompt}'")
    # Ensure using the same inference steps as the loaded model and CFG set to 0.
    image = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    ).images[0]

    output_path = out_dir / output_filename
    image.save(str(output_path))
    print(f"Saved output to {output_path}")

    return str(output_path)
