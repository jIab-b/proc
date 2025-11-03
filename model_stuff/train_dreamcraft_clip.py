import os
import json
import math
import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from diffusers import StableDiffusionXLPipeline, DDPMScheduler
    DIFFUSERS_AVAILABLE = True
except Exception:
    DIFFUSERS_AVAILABLE = False
try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except Exception:
    OPEN_CLIP_AVAILABLE = False

from .palette import (
    NAME_TO_BLOCK,
    AdjacencyPattern,
    get_block_count,
    get_palette_tensor,
    name_for_index,
)
from .nv_diff_render.renderer import DifferentiableBlockRenderer
from .nv_diff_render.utils import create_look_at_matrix, create_perspective_matrix


def positional_encoding(x, num_frequencies):
    freqs = []
    for i in range(num_frequencies):
        f = 2.0 ** i * math.pi
        freqs.append(torch.sin(f * x))
        freqs.append(torch.cos(f * x))
    return torch.cat([x] + freqs, dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128, depth=4):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DreamCraftImplicitModel(nn.Module):
    def __init__(self, grid_size, num_materials, pe_frequencies=6, hidden_dim=128, depth=4, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.grid_size = tuple(grid_size)
        self.num_materials = int(num_materials)
        self.pe_frequencies = int(pe_frequencies)
        in_dim = 3 + 2 * 3 * self.pe_frequencies
        self.air_mlp = MLP(in_dim, 1, hidden_dim=hidden_dim, depth=depth)
        self.solid_mlp = MLP(in_dim, self.num_materials, hidden_dim=hidden_dim, depth=depth)
        self.register_buffer("coord_grid", self._make_coord_grid(self.grid_size, device), persistent=False)
        with torch.no_grad():
            if isinstance(self.air_mlp.net[-1], nn.Linear):
                nn.init.constant_(self.air_mlp.net[-1].bias, -1.0)
            if isinstance(self.solid_mlp.net[-1], nn.Linear):
                nn.init.zeros_(self.solid_mlp.net[-1].bias)

        with torch.no_grad():
            # Bias air logits toward empty space initially.
            if isinstance(self.air_mlp.net[-1], nn.Linear):
                nn.init.constant_(self.air_mlp.net[-1].bias, -4.0)
            # Encourage solid logits to be near zero initially.
            if isinstance(self.solid_mlp.net[-1], nn.Linear):
                nn.init.constant_(self.solid_mlp.net[-1].bias, 0.0)

    @staticmethod
    def _make_coord_grid(grid_size, device):
        X, Y, Z = grid_size
        gx = torch.linspace(0.0, 1.0, X, device=device)
        gy = torch.linspace(0.0, 1.0, Y, device=device)
        gz = torch.linspace(0.0, 1.0, Z, device=device)
        zz, yy, xx = torch.meshgrid(gz, gy, gx, indexing="ij")
        coords = torch.stack([xx, yy, zz], dim=-1)  # (Z, Y, X, 3)
        coords = coords.permute(2, 1, 0, 3).contiguous()  # (X, Y, Z, 3)
        return coords

    def forward(self):
        X, Y, Z = self.grid_size
        coords = self.coord_grid.view(-1, 3)
        enc = positional_encoding(coords * 2.0 - 1.0, self.pe_frequencies)
        air_logits = self.air_mlp(enc).view(X, Y, Z)
        solids_logits = self.solid_mlp(enc).view(X, Y, Z, self.num_materials)
        if self.num_materials > 0:
            solids_logits[..., 0] = solids_logits[..., 0] - 1e4
        return air_logits, solids_logits


class SDXLSDS:
    def __init__(
        self,
        model_id,
        device,
        resolution=1024,
        guidance_scale=5.0,
        t_min=50,
        t_max=950,
        grad_scale=1.0,
        grad_clip=0.5,
        normalize_grad=True,
    ):
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers is required. Install with: pip install diffusers transformers accelerate scipy safetensors")
        dtype = torch.float16 if (device.type == "cuda") else torch.float32
        cache_dir_env = os.environ.get("HF_HOME")
        local_only = str(os.environ.get("HF_HUB_OFFLINE", "0")).lower() in ("1", "true", "yes")
        loader_kwargs = {
            "torch_dtype": dtype,
            "local_files_only": local_only,
        }
        if cache_dir_env:
            loader_kwargs["cache_dir"] = cache_dir_env
        load_id = model_id
        if local_only:
            base_dir = "/workspace/hf/sdxl-base"
            ref_dir = "/workspace/hf/sdxl-refiner"
            if ("stable-diffusion-xl-base" in model_id) and os.path.isdir(base_dir):
                load_id = base_dir
                loader_kwargs.pop("cache_dir", None)
            elif ("stable-diffusion-xl-refiner" in model_id) and os.path.isdir(ref_dir):
                load_id = ref_dir
                loader_kwargs.pop("cache_dir", None)
        self.pipe = StableDiffusionXLPipeline.from_pretrained(load_id, **loader_kwargs)
        self.pipe.to(device)
        self.pipe.vae.requires_grad_(False)
        self.pipe.unet.requires_grad_(False)
        if hasattr(self.pipe, "text_encoder"):
            self.pipe.text_encoder.requires_grad_(False)
        if hasattr(self.pipe, "text_encoder_2"):
            self.pipe.text_encoder_2.requires_grad_(False)
        self.device = device
        self.resolution = int(resolution)
        self.guidance_scale = float(guidance_scale)
        self.scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
        self.t_min = int(t_min)
        self.t_max = int(t_max)
        self.sdxl_scale = self.pipe.vae.config.scaling_factor
        self.model_dtype = self.pipe.unet.dtype
        self.grad_scale = float(grad_scale)
        self.grad_clip = float(grad_clip) if grad_clip is not None else None
        self.normalize_grad = bool(normalize_grad)

    def _encode_prompt(self, prompt, negative_prompt, batch_size):
        enc = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=batch_size,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )
        if len(enc) == 4:
            prompt_embeds, neg_prompt_embeds, pooled, neg_pooled = enc
        else:
            prompt_embeds, neg_prompt_embeds = enc
            pooled = torch.zeros((batch_size, 128), device=self.device, dtype=prompt_embeds.dtype)
            neg_pooled = torch.zeros_like(pooled)
        return prompt_embeds, neg_prompt_embeds, pooled, neg_pooled

    def _prep_image(self, rgb_bchw):
        img = F.interpolate(rgb_bchw, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
        img = img.clamp(0.0, 1.0)
        img = img * 2.0 - 1.0
        vae_dtype = self.pipe.vae.dtype
        return img.to(device=self.device, dtype=vae_dtype)

    def _prepare_time_ids(self, batch, dtype):
        orig = torch.tensor([self.resolution, self.resolution], device=self.device, dtype=dtype)
        crop = torch.tensor([0, 0], device=self.device, dtype=dtype)
        targ = torch.tensor([self.resolution, self.resolution], device=self.device, dtype=dtype)
        time_ids = torch.cat([orig, crop, targ], dim=0).view(1, -1).repeat(batch, 1)
        return time_ids

    def step_sds(self, rgb_bchw, prompt, negative_prompt=""):
        B = rgb_bchw.shape[0]
        with torch.no_grad():
            prompt_embeds, neg_prompt_embeds, pooled, neg_pooled = self._encode_prompt(prompt, negative_prompt, B)
        added_cond = {"text_embeds": pooled, "time_ids": self._prepare_time_ids(B, pooled.dtype)}
        added_cond_uncond = {"text_embeds": neg_pooled, "time_ids": self._prepare_time_ids(B, neg_pooled.dtype)}
        img = self._prep_image(rgb_bchw)
        latents = self.pipe.vae.encode(img).latent_dist.sample()
        latents = latents * self.sdxl_scale
        latents.requires_grad_(True)
        t = torch.randint(low=self.t_min, high=self.t_max + 1, size=(B,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(latents)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        a = alphas_cumprod.gather(0, t).view(B, 1, 1, 1).sqrt().to(dtype=latents.dtype)
        s = (1.0 - alphas_cumprod.gather(0, t)).view(B, 1, 1, 1).sqrt().to(dtype=latents.dtype)
        xt = a * latents + s * noise
        model_in = torch.cat([xt, xt], dim=0)
        t_in = torch.cat([t, t], dim=0)
        cond = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
        added = {
            "text_embeds": torch.cat([added_cond_uncond["text_embeds"], added_cond["text_embeds"]], dim=0),
            "time_ids": torch.cat([added_cond_uncond["time_ids"], added_cond["time_ids"]], dim=0),
        }
        noise_pred = self.pipe.unet(model_in, t_in, encoder_hidden_states=cond, added_cond_kwargs=added).sample
        noise_uncond, noise_text = noise_pred.chunk(2, dim=0)
        noise_guided = noise_uncond + self.guidance_scale * (noise_text - noise_uncond)
        noise_guided = noise_guided.detach()
        raw_grad = noise_guided - noise
        raw_grad = torch.nan_to_num(raw_grad, nan=0.0, posinf=0.0, neginf=0.0)
        raw_norm = torch.linalg.norm(raw_grad.float())
        stats = {
            "grad_raw_norm": float(raw_norm.cpu()) if torch.isfinite(raw_norm) else float("nan"),
            "grad_scale": self.grad_scale,
            "clipped": False,
            "normalized": bool(self.normalize_grad),
            "applied": False,
        }
        if not torch.isfinite(raw_norm) or raw_norm.item() < 1e-12:
            return stats
        grad = raw_grad
        if self.normalize_grad:
            grad = grad / raw_norm.clamp(min=1e-6)
        if self.grad_clip is not None and self.grad_clip > 0.0:
            grad = torch.clamp(grad, -self.grad_clip, self.grad_clip)
            stats["clipped"] = True
        grad = grad * self.grad_scale
        xt.backward(gradient=grad, retain_graph=False)
        stats["applied"] = True
        return stats


def get_clip_model(device, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", local_dir: str | None = None):
    if not OPEN_CLIP_AVAILABLE:
        raise ImportError("open_clip is required. Install with: pip install open_clip_torch")
    load_target = pretrained
    if local_dir:
        candidates = []
        if os.path.isabs(local_dir):
            candidates.append(Path(local_dir))
        else:
            candidates.append(Path(local_dir))
            hf_home = os.environ.get("HF_HOME")
            if hf_home:
                candidates.append(Path(hf_home) / local_dir)
            candidates.append(Path("/workspace/hf") / local_dir)
            candidates.append(Path("/workspace/models") / local_dir)
        weight_names = [
            "open_clip_pytorch_model.bin",
            "open_clip_pytorch_model.pt",
            "model.bin",
        ]
        for cand in candidates:
            for name in weight_names:
                weights_file = cand / name
                if weights_file.exists():
                    load_target = str(weights_file)
                    break
            if load_target != pretrained:
                break
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name=model_name, pretrained=load_target
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval().to(device)
    return model, preprocess_val, tokenizer


def mix_anneal(a_soft, a_hard, alpha):
    return alpha * a_hard + (1.0 - alpha) * a_soft


def compute_alpha(step: int, total_steps: int, start: float, end: float) -> float:
    frac = float(step) / float(max(1, total_steps))
    return start + (end - start) * frac


def sample_camera_orbit(step, total_steps, grid_size, img_w, img_h, radius_scale=2.2, fov_deg=50.0, seed=None):
    X, Y, Z = grid_size
    radius = radius_scale * max(X, Y, Z)
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(int(seed) + int(step))
        phi = torch.rand(1, generator=g).item() * 2.0 * math.pi
        theta = (0.35 + 0.5 * torch.rand(1, generator=g).item()) * math.pi  # avoid poles
    else:
        phi = (step / max(1, total_steps)) * 2.0 * math.pi
        theta = 0.6 * math.pi
    cx = (X - 1) / 2.0
    cy = (Y - 1) / 2.0
    cz = (Z - 1) / 2.0
    eye = (
        cx + radius * math.sin(theta) * math.cos(phi),
        cy + radius * math.cos(theta),
        cz + radius * math.sin(theta) * math.sin(phi),
    )
    center = (cx, cy, cz)
    up = (0.0, 1.0, 0.0)
    aspect = float(img_w) / float(img_h)
    view = create_look_at_matrix(eye, center, up).to(torch.float32)
    proj = create_perspective_matrix(math.radians(fov_deg), aspect, near=0.1, far=10000.0).to(torch.float32)
    return view, proj


def save_rgb(path, rgb_chw):
    try:
        from PIL import Image
        arr = (rgb_chw.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")
        Image.fromarray(arr).save(path)
    except Exception as e:
        torch.save(rgb_chw.cpu(), str(path) + ".pt")


def save_slice(path, volume, axis=1):
    try:
        from PIL import Image
        vol = volume.detach().cpu()
        axes = {0, 1, 2}
        if axis not in axes:
            axis = 1
        idx = vol.shape[axis] // 2
        if axis == 0:
            slice_ = vol[idx, :, :]
        elif axis == 1:
            slice_ = vol[:, idx, :]
        else:
            slice_ = vol[:, :, idx]
        slice_ = slice_.clamp(0.0, 1.0).numpy()
        slice_img = (slice_ * 255.0).astype("uint8")
        Image.fromarray(slice_img).save(path)
    except Exception:
        torch.save(volume.cpu(), str(path) + ".pt")


def load_adjacency_patterns(config_path: str, materials: int, device: torch.device) -> List[AdjacencyPattern]:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    patterns: List[AdjacencyPattern] = []
    entries = data.get("patterns", [])
    for entry in entries:
        size = entry.get("size", [1, 1, 1])
        if len(size) != 3:
            raise ValueError(f"Pattern size must be length-3 list, got {size}")
        sx, sy, sz = [int(max(1, s)) for s in size]
        kernel = torch.zeros((1, materials, sz, sy, sx), dtype=torch.float32, device=device)
        blocks = entry.get("blocks", [])
        if not blocks:
            continue
        count = 0
        for block in blocks:
            offset = block.get("offset")
            if offset is None or len(offset) != 3:
                raise ValueError(f"Pattern block missing offset: {block}")
            ox, oy, oz = [int(o) for o in offset]
            if not (0 <= ox < sx and 0 <= oy < sy and 0 <= oz < sz):
                raise ValueError(f"Pattern block offset {offset} outside size {size}")
            block_name = block.get("type")
            if block_name is None:
                raise ValueError(f"Pattern block missing type: {block}")
            if isinstance(block_name, int):
                block_idx = block_name
            else:
                normalized = block_name.replace(" ", "")
                if normalized not in NAME_TO_BLOCK:
                    raise ValueError(f"Unknown block name '{block_name}' in adjacency config.")
                block_idx = int(NAME_TO_BLOCK[normalized])
            if not (0 <= block_idx < materials):
                raise ValueError(f"Block index {block_idx} outside material range.")
            kernel[0, block_idx, oz, oy, ox] = 1.0
            count += 1
        if count == 0:
            continue
        weight = float(entry.get("weight", 1.0))
        mode = entry.get("mode", "penalize").lower()
        reward = mode in ("reward", "encourage")
        patterns.append(
            AdjacencyPattern(
                size=torch.Size([sx, sy, sz]),
                kernel=kernel,
                threshold=float(count) - 0.5,
                weight=weight,
                reward=reward,
            )
        )
    return patterns


def train(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.air_alpha_start is None:
        args.air_alpha_start = args.alpha_start
    if args.air_alpha_end is None:
        args.air_alpha_end = args.alpha_end
    if args.solid_alpha_start is None:
        args.solid_alpha_start = args.alpha_start
    if args.solid_alpha_end is None:
        args.solid_alpha_end = args.alpha_end

    if args.grid_xyz is not None and len(args.grid_xyz) == 3:
        grid = tuple(args.grid_xyz)
    else:
        grid = (args.grid, args.grid, args.grid)
    X, Y, Z = grid

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not DIFFUSERS_AVAILABLE:
        raise ImportError("Install diffusers: pip install diffusers transformers accelerate scipy safetensors")

    sds = SDXLSDS(
        model_id=args.sdxl_model,
        device=device,
        resolution=args.sdxl_res,
        guidance_scale=args.guidance_scale,
        t_min=args.t_min,
        t_max=args.t_max,
        grad_scale=args.sds_weight,
        grad_clip=args.sds_grad_clip,
        normalize_grad=args.sds_normalize_grad,
    )

    total_materials = int(args.materials)
    if total_materials <= 0:
        total_materials = get_block_count()

    palette_tensor = get_palette_tensor(device)
    adjacency_patterns: List[AdjacencyPattern] = []
    if args.adjacency_config:
        adjacency_patterns = load_adjacency_patterns(args.adjacency_config, total_materials, device)

    model = DreamCraftImplicitModel(
        grid_size=grid,
        num_materials=total_materials,
        pe_frequencies=args.pe_freqs,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        device=device,
    ).to(device)

    renderer = DifferentiableBlockRenderer(grid_size=grid, world_scale=args.world_scale, device=device).to(device)

    clip_model = None
    clip_tokenizer = None
    clip_text_feat = None
    clip_preprocess = None
    if args.clip_weight > 0:
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError("open_clip is required for clip_weight > 0. Install with: pip install open_clip_torch")
        clip_model, clip_preprocess, clip_tokenizer = get_clip_model(device, args.clip_model, args.clip_pretrained, args.clip_local_dir)
        with torch.no_grad():
            text_tokens = clip_tokenizer([args.prompt])
            clip_text_feat = clip_model.encode_text(text_tokens.to(device))
            clip_text_feat = clip_text_feat / clip_text_feat.norm(dim=-1, keepdim=True)

    opt = torch.optim.Adam([{"params": model.parameters(), "lr": args.lr}])

    preview_dir = out_dir / "previews"
    if args.preview_every > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "log.txt"
    with log_path.open("w", encoding="utf-8") as log_file:
        json.dump({
            "event": "train_start",
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "grid": list(grid),
            "materials": total_materials,
            "steps": args.steps,
            "lr": args.lr,
        }, log_file)
        log_file.write("\n")
        log_file.flush()

        for step in range(1, args.steps + 1):
            opt.zero_grad()
            air_logits, mat_logits = model()
            air_soft = torch.sigmoid(air_logits)

            air_two_class = torch.stack([-air_logits, air_logits], dim=-1)
            air_hard = F.gumbel_softmax(air_two_class, tau=max(1e-3, args.air_tau), hard=True, dim=-1)[..., 1]
            solid_hard = F.gumbel_softmax(mat_logits, tau=max(1e-3, args.solid_tau), hard=True, dim=-1)

            alpha_air = compute_alpha(step, args.steps, args.air_alpha_start, args.air_alpha_end)
            alpha_solid = compute_alpha(step, args.steps, args.solid_alpha_start, args.solid_alpha_end)

            a_mixed = mix_anneal(air_soft, air_hard, alpha_air)
            a_mixed = torch.nan_to_num(a_mixed, nan=0.0, posinf=1.0, neginf=0.0)

            solid_soft = torch.softmax(mat_logits / max(1e-6, args.tau), dim=-1)
            mat_probs = mix_anneal(solid_soft, solid_hard, alpha_solid)
            mat_probs = torch.nan_to_num(mat_probs, nan=0.0, posinf=1.0, neginf=0.0)
            mat_probs = mat_probs.clamp(min=1e-6)
            mat_probs = mat_probs / mat_probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            mat_logits_mixed = torch.log(mat_probs)

            block_grid = (a_mixed > args.occ_cull_thresh)
            view, proj = sample_camera_orbit(
                step,
                args.steps,
                grid,
                args.train_w,
                args.train_h,
                radius_scale=args.cam_radius_scale,
                fov_deg=args.fov_deg,
                seed=args.seed,
            )
            view = view.to(device)
            proj = proj.to(device)

            img_rgba = renderer.render_from_grid(
                block_grid=block_grid,
                material_logits=mat_logits_mixed,
                camera_view=view,
                camera_proj=proj,
                img_h=args.train_h,
                img_w=args.train_w,
                occupancy_probs=a_mixed,
                hard_materials=False,
                temperature=1.0,
                palette=palette_tensor,
                material_probs=mat_probs,
            )
            rgb = img_rgba[:, :3, :, :]

            sds_stats = sds.step_sds(rgb, prompt=args.prompt, negative_prompt=args.negative_prompt)
            clip_loss = torch.tensor(0.0, device=device)
            clip_sim = None
            if clip_model is not None:
                img_for_clip = F.interpolate(rgb, size=args.clip_image_res, mode="bilinear", align_corners=False)
                img_for_clip = img_for_clip.clamp(0.0, 1.0)
                img_for_clip = img_for_clip * 2.0 - 1.0
                image_feat = clip_model.encode_image(img_for_clip.to(device))
                image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
                clip_sim = torch.sum(image_feat * clip_text_feat)
                clip_loss = -args.clip_weight * clip_sim

            occ_reg = args.occ_reg * a_mixed.mean()
            tv = torch.tensor(0.0, device=device)
            if args.tv_reg > 0.0:
                dx = (a_mixed[1:, :, :] - a_mixed[:-1, :, :]).abs().mean()
                dy = (a_mixed[:, 1:, :] - a_mixed[:, :-1, :]).abs().mean()
                dz = (a_mixed[:, :, 1:] - a_mixed[:, :, :-1]).abs().mean()
                tv = args.tv_reg * (dx + dy + dz)

            dist_loss = torch.tensor(0.0, device=device)
            if args.dist_weight > 0.0 and args.dist_target is not None and len(args.dist_target) > 0:
                tgt = torch.tensor(args.dist_target, device=device, dtype=torch.float32)
                tgt = tgt.clamp(min=0)
                if tgt.sum() > 0:
                    tgt = tgt / tgt.sum()
                    probs_non_air = mat_probs[..., 1:]
                    weights = a_mixed[..., None]
                    counts = (weights * probs_non_air).sum(dim=(0, 1, 2))
                    if counts.sum() > 0:
                        counts = counts / counts.sum()
                        counts = torch.nan_to_num(counts, nan=0.0)
                        dist_loss = args.dist_weight * (counts - tgt).abs().sum()

            adjacency_loss = torch.tensor(0.0, device=device)
            if adjacency_patterns:
                hard_volume = solid_hard.permute(3, 2, 1, 0).unsqueeze(0)  # (1, M, Z, Y, X)
                for pattern in adjacency_patterns:
                    conv = F.conv3d(hard_volume, pattern.kernel, stride=1)
                    activation = torch.relu(conv - pattern.threshold)
                    mean_activation = activation.mean()
                    mean_activation = torch.nan_to_num(mean_activation, nan=0.0)
                    if pattern.reward:
                        adjacency_loss = adjacency_loss - pattern.weight * mean_activation
                    else:
                        adjacency_loss = adjacency_loss + pattern.weight * mean_activation

            loss = occ_reg + tv + dist_loss + adjacency_loss + clip_loss

            active_voxels = int((a_mixed > args.occ_cull_thresh).sum().item())
            step_record = {
                "event": "step",
                "step": step,
                "alpha_air": alpha_air,
                "alpha_solid": alpha_solid,
                "occ_mean": float(a_mixed.mean().detach().cpu()),
                "occ_max": float(a_mixed.max().detach().cpu()),
                "active_voxels": active_voxels,
                "sds": sds_stats,
            }

            if not torch.isfinite(loss).item():
                print(f"Warning: non-finite loss at step {step}, skipping update.")
                step_record["warning"] = "non_finite_loss"
                json.dump(step_record, log_file)
                log_file.write("\n")
                log_file.flush()
                opt.zero_grad(set_to_none=True)
                continue

            loss.backward()

            grads_corrected = False
            for param in model.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    grads_corrected = True

            if grads_corrected:
                print(f"Warning: non-finite gradients corrected at step {step}.")
                step_record.setdefault("warnings", []).append("non_finite_grad_corrected")

            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            step_record.update({
                "loss": float(loss.detach().cpu()),
                "occ_reg": float(occ_reg.detach().cpu()),
                "tv": float(tv.detach().cpu()),
                "dist_loss": float(dist_loss.detach().cpu()),
                "adj_loss": float(adjacency_loss.detach().cpu()),
                "clip_loss": float(clip_loss.detach().cpu()),
                "clip_sim": float(clip_sim.detach().cpu()) if clip_sim is not None else None,
            })

            with torch.no_grad():
                mat_counts = (a_mixed[..., None] * mat_probs).sum(dim=(0, 1, 2))
                step_record["material_counts"] = [float(x) for x in mat_counts.cpu()]

            json.dump(step_record, log_file)
            log_file.write("\n")
            log_file.flush()

            opt.step()

            if args.preview_every > 0 and (step % args.preview_every == 0 or step == 1):
                save_rgb(preview_dir / f"step_{step:05d}_rgb.png", rgb[0])
                save_slice(preview_dir / f"step_{step:05d}_occ.png", a_mixed)

            if step % args.log_every == 0 or step == 1:
                print(
                    f"step {step}/{args.steps} "
                    f"loss={step_record['loss']:.4f} "
                    f"occ_reg={step_record['occ_reg']:.4f} "
                    f"tv={step_record['tv']:.4f} "
                    f"dist={step_record['dist_loss']:.4f} "
                    f"adj={step_record['adj_loss']:.4f} "
                    f"clip={step_record['clip_loss']:.4f} "
                    f"alpha_air={alpha_air:.2f} "
                    f"alpha_solid={alpha_solid:.2f}"
                )

            if args.save_every > 0 and (step % args.save_every == 0 or step == args.steps):
                with torch.no_grad():
                    save_path = out_dir / f"step_{step:05d}.png"
                    save_rgb(save_path, rgb[0])
                    types = torch.argmax(mat_probs, dim=-1)
                    present = (a_mixed > args.occ_cull_thresh)
                    types_masked = torch.where(present, types, torch.zeros_like(types))
                    nz = torch.nonzero(types_masked != 0, as_tuple=False)
                    blocks: List[List[int]]
                    if nz.numel() > 0:
                        tvals = types_masked[nz[:, 0], nz[:, 1], nz[:, 2]].unsqueeze(1)
                        blocks = torch.cat([nz, tvals], dim=1).cpu().tolist()
                    else:
                        blocks = []
                        if args.export_force_nonempty:
                            flat = a_mixed.reshape(-1)
                            k = int(max(1, args.export_topk))
                            k = min(k, flat.numel())
                            vals, idxs = torch.topk(flat, k, largest=True)
                            xi = (idxs // (grid[1] * grid[2])).to(torch.long)
                            yi = ((idxs // grid[2]) % grid[1]).to(torch.long)
                            zi = (idxs % grid[2]).to(torch.long)
                            tsel = types[xi, yi, zi].unsqueeze(1)
                            keep = (tsel.squeeze(1) != 0)
                            xi, yi, zi, tsel = xi[keep], yi[keep], zi[keep], tsel[keep]
                            blocks = torch.stack([xi, yi, zi], dim=1)
                            if blocks.numel() > 0:
                                blocks = torch.cat([blocks, tsel], dim=1).cpu().tolist()
                            else:
                                blocks = []
                    block_records = []
                    for x, y, z, t in blocks:
                        t_int = int(t)
                        type_name = name_for_index(t_int)
                        block_records.append(
                            {
                                "position": [int(x), int(y), int(z)],
                                "typeIndex": t_int,
                                "blockType": type_name,
                                "typeName": type_name,
                            }
                        )
                    export_payload = {
                        "size": [int(grid[0]), int(grid[1]), int(grid[2])],
                        "blocks": block_records,
                        "prompt": args.prompt,
                        "worldScale": args.world_scale,
                        "alphaAir": alpha_air,
                        "alphaSolid": alpha_solid,
                        "materials": [name_for_index(i) for i in range(total_materials)],
                        "blockCount": len(block_records),
                    }
                    with open(out_dir / f"map_{step:05d}.json", "w", encoding="utf-8") as f:
                        json.dump(export_payload, f, indent=2)
                    json.dump({"event": "export", "step": step, "blockCount": len(block_records)}, log_file)
                    log_file.write("\n")
                    log_file.flush()

        json.dump({"event": "train_end"}, log_file)
        log_file.write("\n")
        log_file.flush()


def build_argparser():
    p = argparse.ArgumentParser("DreamCraft (paper-style) with SDXL SDS")
    p.add_argument("--prompt", type=str, default="a small stone tower")
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--grid", type=int, default=32, help="N for NxNxN grid")
    p.add_argument("--grid_xyz", type=int, nargs=3, default=None, help="Override grid with X Y Z")
    p.add_argument("--materials", type=int, default=12)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train_h", type=int, default=192)
    p.add_argument("--train_w", type=int, default=192)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--solid_tau", type=float, default=1.0)
    p.add_argument("--hard_after", type=int, default=100)
    p.add_argument("--air_tau", type=float, default=1.0)
    p.add_argument("--alpha_start", type=float, default=0.0)
    p.add_argument("--alpha_end", type=float, default=1.0)
    p.add_argument("--air_alpha_start", type=float, default=None)
    p.add_argument("--air_alpha_end", type=float, default=None)
    p.add_argument("--solid_alpha_start", type=float, default=None)
    p.add_argument("--solid_alpha_end", type=float, default=None)
    p.add_argument("--occ_cull_thresh", type=float, default=0.05)
    p.add_argument("--export_force_nonempty", action="store_true")
    p.add_argument("--export_topk", type=int, default=512)
    p.add_argument("--occ_reg", type=float, default=1e-3)
    p.add_argument("--tv_reg", type=float, default=2e-3)
    p.add_argument("--dist_weight", type=float, default=0.0)
    p.add_argument("--dist_target", type=float, nargs='*', default=None, help="Target distribution over non-air materials length M-1")
    p.add_argument("--adjacency_config", type=str, default=None, help="JSON file specifying adjacency reward/penalty patterns")
    p.add_argument("--grad_clip", type=float, default=1.0, help="Max-norm gradient clip; set <=0 to disable")
    p.add_argument("--cam_radius_scale", type=float, default=2.2)
    p.add_argument("--fov_deg", type=float, default=50.0)
    p.add_argument("--world_scale", type=float, default=2.0)
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--preview_every", type=int, default=10)
    p.add_argument("--out_dir", type=str, default="./out_local/dreamcraft_sdxl")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pe_freqs", type=int, default=6)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--sdxl_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--sdxl_res", type=int, default=1024)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--t_min", type=int, default=50)
    p.add_argument("--t_max", type=int, default=950)
    p.add_argument("--sds_weight", type=float, default=0.1, help="Scale applied to SDS gradient.")
    p.add_argument("--sds_grad_clip", type=float, default=0.5, help="Elementwise clamp applied to SDS gradient after optional normalization.")
    p.add_argument("--sds_normalize_grad", action="store_true", help="Normalize SDS gradient to unit norm before scaling.")
    p.set_defaults(sds_normalize_grad=True)
    p.add_argument("--clip_weight", type=float, default=0.0, help="Optional CLIP guidance weight.")
    p.add_argument("--clip_model", type=str, default="ViT-B-32")
    p.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k")
    p.add_argument("--clip_image_res", type=int, default=224)
    p.add_argument("--clip_local_dir", type=str, default="openclip-vit-b-32", help="Local directory containing OpenCLIP weights prefetched via modal")
    return p


def main():
    args = build_argparser().parse_args()
    if args.air_alpha_start is None:
        args.air_alpha_start = args.alpha_start
    if args.air_alpha_end is None:
        args.air_alpha_end = args.alpha_end
    if args.solid_alpha_start is None:
        args.solid_alpha_start = args.alpha_start
    if args.solid_alpha_end is None:
        args.solid_alpha_end = args.alpha_end
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    train(args)


if __name__ == "__main__":
    main()
