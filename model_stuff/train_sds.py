import torch
import os
import random
import json
from pathlib import Path
from PIL import Image
from .config import (
    DATA_IMAGES, MAP_OUT, DEVICE, GRID_XYZ, IMG_HW, CFG_SCALE, STEPS, LR,
    TEMP_START, TEMP_END, SEED, SDXL_MODEL_ID, LOGS_DIR, LOGS_IMAGES_DIR,
    LOG_EVERY, IMAGE_EVERY,
)
from .materials import sigma_m, c_m
from .renderer import RendererConfig, DifferentiableRenderer
from .dsl import grid_to_actions, write_map_json, actions_to_logits, parse_dsl_text
from .dataset import load_prompts
from .sdxl_lightning import SDXLLightning, LATENT_SCALING

def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)
    X, Y, Z = GRID_XYZ
    H, W = IMG_HW
    M = sigma_m.numel()
    # Optionally seed from DSL if provided via environment
    dsl_actions = None
    dsl_file = os.environ.get('DSL_FILE') or os.environ.get('DSL_PATH')
    dsl_text_env = os.environ.get('DSL_TEXT')
    if dsl_file:
        try:
            text = Path(dsl_file).read_text(encoding='utf-8')
            import json as _json
            if text.lstrip().startswith('{'):
                data = _json.loads(text)
                dsl_actions = data.get('actions') or data.get('dsl') or None
            elif text.lstrip().startswith('['):
                dsl_actions = _json.loads(text)
            else:
                dsl_actions = parse_dsl_text(text)
        except Exception:
            dsl_actions = None
    elif dsl_text_env:
        dsl_actions = parse_dsl_text(dsl_text_env)

    if dsl_actions:
        W0 = actions_to_logits(dsl_actions, (X, Y, Z), M, DEVICE)
        W_logits = W0.clone().detach().requires_grad_(True)
    else:
        W_logits = torch.randn(X, Y, Z, M, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([W_logits], lr=LR)
    prompts = load_prompts(DATA_IMAGES)
    prompt = prompts[0]
    logs_dir = LOGS_DIR
    images_dir = LOGS_IMAGES_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "train.jsonl"

    renderer = DifferentiableRenderer(sigma_m.to(DEVICE), c_m.to(DEVICE))
    sdxl = SDXLLightning(SDXL_MODEL_ID, device=DEVICE, dtype=torch.float16 if DEVICE == "cuda" else torch.float32, height=H, width=W)
    pe, pe_pooled, ue, ue_pooled, add_time_ids = sdxl.encode_prompt(prompt)
    for step in range(STEPS):
        t = TEMP_START + (TEMP_END - TEMP_START) * (step / max(STEPS - 1, 1))
        I = renderer.render(W_logits, RendererConfig(image_height=H, image_width=W, temperature=t)).to(sdxl.dtype if hasattr(sdxl, 'dtype') else I.dtype)
        I = I[..., :3, :, :].clamp(0, 1)
        z0 = sdxl.vae_encode(I)
        ts = sdxl.sample_timesteps(batch_size=z0.shape[0])
        noise = torch.randn_like(z0)
        x_t = sdxl.add_noise(z0, noise, ts)
        eps_cfg = sdxl.eps_pred_cfg(x_t, ts, pe, pe_pooled, ue, ue_pooled, add_time_ids, CFG_SCALE)
        loss_sds = (eps_cfg - noise).pow(2).mean()
        P = torch.softmax(W_logits / t, dim=-1)
        loss_entropy = (P * torch.log(P.clamp_min(1e-8))).sum() / (X * Y * Z)
        loss_sparse = (1 - P[..., 0]).mean()
        loss = loss_sds + 1e-3 * loss_entropy + 1e-3 * loss_sparse
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % LOG_EVERY == 0 or step == 0:
            record = {
                "step": step + 1,
                "temperature": float(t),
                "loss_total": float(loss.detach().cpu().item()),
                "loss_sds": float(loss_sds.detach().cpu().item()),
                "loss_entropy": float(loss_entropy.detach().cpu().item()),
                "loss_sparse": float(loss_sparse.detach().cpu().item()),
            }
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")

        if (step + 1) % IMAGE_EVERY == 0 or step in (0, STEPS - 1):
            img = (I[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype("uint8")
            Image.fromarray(img).save(images_dir / f"step_{step+1:04d}.png")
    actions = grid_to_actions(W_logits.detach().cpu())
    write_map_json(MAP_OUT, actions, {"prompt": prompt})
    try:
        # Also mirror to out_local for convenience
        mirror_path = Path("/workspace/out_local/map.json")
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        write_map_json(mirror_path, actions, {"prompt": prompt})
    except Exception:
        pass
    print(str(MAP_OUT))

if __name__ == "__main__":
    main()

