import os
from .palette import get_block_count
from .train_dreamcraft_clip import build_argparser, train


def main():
    cache_dir = "/workspace/hf" if os.path.isdir("/workspace/hf") else os.path.join(os.getcwd(), "hf_cache")
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache_dir)
    if os.path.isdir("/workspace/hf"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    parser = build_argparser()
    args = parser.parse_args([])
    args.prompt = "a small stone tower"
    args.negative_prompt = ""
    args.grid = 32
    args.grid_xyz = None
    args.materials = get_block_count()
    args.steps = 800
    args.lr = 8e-4
    args.train_h = 192
    args.train_w = 192
    args.tau = 1.0
    args.solid_tau = 1.0
    args.hard_after = 300
    args.air_tau = 1.0
    args.alpha_start = 0.0
    args.alpha_end = 1.0
    args.occ_cull_thresh = 0.12
    args.occ_reg = 5e-4
    args.tv_reg = 1.5e-3
    args.cam_radius_scale = 2.2
    args.fov_deg = 50.0
    args.world_scale = 2.0
    args.save_every = 200
    args.log_every = 20
    args.preview_every = 100
    args.out_dir = "./out_local/dreamcraft_quick"
    args.device = None
    args.seed = 42
    args.pe_freqs = 6
    args.hidden_dim = 128
    args.depth = 4
    args.sdxl_model = "stabilityai/stable-diffusion-xl-base-1.0"
    args.sdxl_res = 384
    args.guidance_scale = 5.0
    args.t_min = 50
    args.t_max = 950
    args.sds_weight = 0.08
    args.sds_grad_clip = 0.4
    args.sds_normalize_grad = True
    args.clip_weight = 0.25
    args.clip_local_dir = "openclip-vit-b-32"
    args.dist_weight = 0.0
    args.dist_target = None
    args.grad_clip = 1.0
    train(args)


if __name__ == "__main__":
    main()
