import os
import json
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from diffusers import StableDiffusionXLPipeline, DDPMScheduler
    DIFFUSERS_AVAILABLE = True
except Exception:
    DIFFUSERS_AVAILABLE = False

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
        air_out = self.air_mlp(enc).view(X, Y, Z)
        air_soft = torch.sigmoid(air_out)
        solids_logits = self.solid_mlp(enc).view(X, Y, Z, self.num_materials)
        if self.num_materials > 0:
            solids_logits[..., 0] = solids_logits[..., 0] - 1e4
        return air_soft, solids_logits


class SDXLSDS:
    def __init__(self, model_id, device, resolution=1024, guidance_scale=5.0, t_min=50, t_max=950):
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
        w = 1.0
        grad = w * (noise_guided - noise)
        xt.backward(gradient=grad, retain_graph=True)
        return latents


def get_clip_model(device):
    if not OPEN_CLIP_AVAILABLE:
        raise ImportError("open_clip is required. Install with: pip install open_clip_torch")
    model, _, _ = open_clip.create_model_and_transforms(
        model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval().to(device)
    return model, tokenizer


def mix_anneal(a_soft, a_hard, alpha):
    return alpha * a_hard + (1.0 - alpha) * a_soft


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


def train(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.grid_xyz is not None and len(args.grid_xyz) == 3:
        grid = tuple(args.grid_xyz)
    else:
        grid = (args.grid, args.grid, args.grid)
    X, Y, Z = grid

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not DIFFUSERS_AVAILABLE:
        raise ImportError("Install diffusers: pip install diffusers transformers accelerate scipy safetensors")

    sds = SDXLSDS(model_id=args.sdxl_model, device=device, resolution=args.sdxl_res, guidance_scale=args.guidance_scale, t_min=args.t_min, t_max=args.t_max)

    model = DreamCraftImplicitModel(grid_size=grid, num_materials=args.materials, pe_frequencies=args.pe_freqs, hidden_dim=args.hidden_dim, depth=args.depth, device=device).to(device)
    renderer = DifferentiableBlockRenderer(grid_size=grid, world_scale=args.world_scale, device=device).to(device)

    opt = torch.optim.Adam([{"params": model.parameters(), "lr": args.lr}])

    for step in range(1, args.steps + 1):
        opt.zero_grad()
        a_soft, mat_logits = model()
        with torch.no_grad():
            g = torch.Generator(device=device)
            g.manual_seed(int(args.seed) + step)
            logits_air = torch.stack([-(a_soft - 0.5), (a_soft - 0.5)], dim=-1)
            a_hard = F.gumbel_softmax(logits_air, tau=max(1e-3, args.air_tau), hard=True, dim=-1)[..., 1]
        alpha = float(step) / float(max(1, args.steps))
        alpha = args.alpha_start + (args.alpha_end - args.alpha_start) * alpha
        a_mixed = mix_anneal(a_soft, a_hard, alpha)
        block_grid = (a_mixed > args.occ_cull_thresh)
        view, proj = sample_camera_orbit(step, args.steps, grid, args.train_w, args.train_h, radius_scale=args.cam_radius_scale, fov_deg=args.fov_deg, seed=args.seed)
        view = view.to(device)
        proj = proj.to(device)

        img_rgba = renderer.render_from_grid(
            block_grid=block_grid,
            material_logits=mat_logits,
            camera_view=view,
            camera_proj=proj,
            img_h=args.train_h,
            img_w=args.train_w,
            occupancy_probs=a_mixed,
            hard_materials=(step >= args.hard_after),
            temperature=args.tau,
        )
        rgb = img_rgba[:, :3, :, :]

        sds.step_sds(rgb, prompt=args.prompt, negative_prompt=args.negative_prompt)

        occ_reg = args.occ_reg * a_soft.mean()
        tv = 0.0
        if args.tv_reg > 0.0:
            dx = (a_soft[1:, :, :] - a_soft[:-1, :, :]).abs().mean()
            dy = (a_soft[:, 1:, :] - a_soft[:, :-1, :]).abs().mean()
            dz = (a_soft[:, :, 1:] - a_soft[:, :, :-1]).abs().mean()
            tv = args.tv_reg * (dx + dy + dz)

        dist_loss = 0.0
        if args.dist_weight > 0.0 and args.dist_target is not None and len(args.dist_target) > 0:
            with torch.no_grad():
                tgt = torch.tensor(args.dist_target, device=device, dtype=torch.float32)
                tgt = tgt.clamp(min=0)
                if tgt.sum() > 0:
                    tgt = tgt / tgt.sum()
            probs = torch.softmax(mat_logits / max(1e-6, args.tau), dim=-1)
            probs = probs[..., 1:]
            weights = a_mixed[..., None]
            counts = (weights * probs).sum(dim=(0, 1, 2))
            if counts.sum() > 0:
                counts = counts / counts.sum()
                dist_loss = args.dist_weight * (counts - tgt).abs().sum()

        loss = occ_reg + tv + dist_loss
        loss.backward()
        opt.step()

        if step % args.log_every == 0 or step == 1:
            print(f"step {step}/{args.steps} loss={loss.item():.4f} occ_reg={occ_reg.item():.4f} tv={float(tv):.4f} dist={float(dist_loss):.4f} alpha={alpha:.2f} hard={(step>=args.hard_after)}")

        if args.save_every > 0 and (step % args.save_every == 0 or step == args.steps):
            with torch.no_grad():
                save_path = out_dir / f"step_{step:05d}.png"
                save_rgb(save_path, rgb[0])
                types = torch.argmax(mat_logits, dim=-1)
                present = (a_mixed > args.occ_cull_thresh)
                types_masked = torch.where(present, types, torch.zeros_like(types))
                nz = torch.nonzero(types_masked != 0, as_tuple=False)
                blocks: list
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
                j = {
                    "size": [int(grid[0]), int(grid[1]), int(grid[2])],
                    "blocks": blocks,
                    "prompt": args.prompt,
                }
                with open(out_dir / f"map_{step:05d}.json", "w") as f:
                    json.dump(j, f)


def build_argparser():
    p = argparse.ArgumentParser("DreamCraft (paper-style) with SDXL SDS")
    p.add_argument("--prompt", type=str, default="a small stone tower")
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--grid", type=int, default=32, help="N for NxNxN grid")
    p.add_argument("--grid_xyz", type=int, nargs=3, default=None, help="Override grid with X Y Z")
    p.add_argument("--materials", type=int, default=8)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--train_h", type=int, default=192)
    p.add_argument("--train_w", type=int, default=192)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--hard_after", type=int, default=100)
    p.add_argument("--air_tau", type=float, default=1.0)
    p.add_argument("--alpha_start", type=float, default=0.0)
    p.add_argument("--alpha_end", type=float, default=1.0)
    p.add_argument("--occ_cull_thresh", type=float, default=0.05)
    p.add_argument("--export_force_nonempty", action="store_true")
    p.add_argument("--export_topk", type=int, default=512)
    p.add_argument("--occ_reg", type=float, default=1e-3)
    p.add_argument("--tv_reg", type=float, default=2e-3)
    p.add_argument("--dist_weight", type=float, default=0.0)
    p.add_argument("--dist_target", type=float, nargs='*', default=None, help="Target distribution over non-air materials length M-1")
    p.add_argument("--cam_radius_scale", type=float, default=2.2)
    p.add_argument("--fov_deg", type=float, default=50.0)
    p.add_argument("--world_scale", type=float, default=2.0)
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--log_every", type=int, default=10)
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
    return p


def main():
    args = build_argparser().parse_args()
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    train(args)


if __name__ == "__main__":
    main()


