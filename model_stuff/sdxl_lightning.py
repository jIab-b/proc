import torch
from typing import Tuple
from diffusers import StableDiffusionXLPipeline, DDPMScheduler

LATENT_SCALING = 0.18215

class SDXLLightning:
    def __init__(self, model_id: str, device: str, dtype: torch.dtype = torch.float16, height: int = 256, width: int = 256):
        self.device = device
        self.dtype = dtype
        self.height = height
        self.width = width
        pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=True)
        pipe.to(device)
        self.pipe = pipe
        self.vae = pipe.vae
        self.unet = pipe.unet
        # Use a DDPM training schedule to define add_noise and timesteps
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pe, ue, pe_pooled, ue_pooled = self.pipe.encode_prompt(
            prompt=[prompt],
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
        if pe_pooled.ndim == 3 and pe_pooled.size(1) == 1:
            pe_pooled = pe_pooled.squeeze(1)
        if ue_pooled.ndim == 3 and ue_pooled.size(1) == 1:
            ue_pooled = ue_pooled.squeeze(1)
        proj_dim = getattr(getattr(self.pipe, "text_encoder_2", None), "config", None)
        proj_dim = getattr(proj_dim, "projection_dim", None)
        if proj_dim is None:
            proj_dim = 1280
        add_time_ids = self.pipe._get_add_time_ids(
            original_size=(self.height, self.width),
            crops_coords_top_left=(0, 0),
            target_size=(self.height, self.width),
            dtype=pe.dtype,
            text_encoder_projection_dim=proj_dim,
        ).to(self.device)
        if add_time_ids.ndim == 1:
            add_time_ids = add_time_ids.unsqueeze(0)
        if add_time_ids.ndim == 3 and add_time_ids.size(1) == 1:
            add_time_ids = add_time_ids.squeeze(1)
        return pe, pe_pooled, ue, ue_pooled, add_time_ids

    def vae_encode(self, images: torch.Tensor) -> torch.Tensor:
        latents = self.vae.encode(images).latent_dist.sample()
        return latents * LATENT_SCALING

    def add_noise(self, z0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.scheduler.add_noise(z0, noise, timesteps)

    def eps_pred_cfg(self, x_t: torch.Tensor, t: torch.Tensor, pe: torch.Tensor, pe_pooled: torch.Tensor, ue: torch.Tensor, ue_pooled: torch.Tensor, add_time_ids: torch.Tensor, cfg_scale: float) -> torch.Tensor:
        c = {
            "encoder_hidden_states": pe,
            "added_cond_kwargs": {"text_embeds": pe_pooled, "time_ids": add_time_ids},
        }
        u = {
            "encoder_hidden_states": ue,
            "added_cond_kwargs": {"text_embeds": ue_pooled, "time_ids": add_time_ids},
        }
        eps = self.unet(x_t, t, **c).sample
        eps_u = self.unet(x_t, t, **u).sample
        return eps_u + cfg_scale * (eps - eps_u)

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        num_train = self.scheduler.config.num_train_timesteps
        ts = torch.randint(low=0, high=num_train, size=(batch_size,), device=self.device, dtype=torch.long)
        return ts



