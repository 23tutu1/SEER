from __future__ import annotations

import argparse
import math

import torch

from src.clip_embedding import CLIPConfig, CLIPEmbedder
from src.data import RaysDataset, load_rays_payload
from src.nerf_model import NeRFModel
from src.seer_model import SEERModel
from src.rendering import render_rays
from src.metrics import psnr, ssim


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    payload = load_rays_payload(args.data)
    dataset = RaysDataset(payload)

    clip_embedder = CLIPEmbedder(
        CLIPConfig(
            backend=args.clip_backend,
            model_name=args.clip_model,
            pretrained=args.clip_pretrained,
            device=args.device,
            freeze=True,
        )
    )

    if payload.text_embeddings is None:
        if payload.texts is None:
            if payload.image_texts is not None and payload.image_indices is not None:
                image_embeddings = clip_embedder.encode_text(payload.image_texts).cpu()
                payload.text_embeddings = image_embeddings[payload.image_indices]
            else:
                payload.text_embeddings = torch.zeros(len(dataset), clip_embedder.text_dim)
        else:
            payload.text_embeddings = clip_embedder.encode_text(payload.texts).cpu()

    nerf_coarse = NeRFModel(
        pos_freqs=args.pos_freqs,
        dir_freqs=args.dir_freqs,
        hidden_dim=args.nerf_hidden_dim,
        num_layers=args.nerf_layers,
        skips=tuple(args.nerf_skips),
    ).to(device)
    nerf_fine = NeRFModel(
        pos_freqs=args.pos_freqs,
        dir_freqs=args.dir_freqs,
        hidden_dim=args.nerf_hidden_dim,
        num_layers=args.nerf_layers,
        skips=tuple(args.nerf_skips),
    ).to(device)
    model_coarse = SEERModel(
        nerf_coarse, clip_dim=clip_embedder.text_dim, hidden_dim=args.hidden_dim
    ).to(device)
    model_fine = SEERModel(
        nerf_fine, clip_dim=clip_embedder.text_dim, hidden_dim=args.hidden_dim
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_coarse" in checkpoint:
        model_coarse.load_state_dict(checkpoint["model_coarse"], strict=False)
    elif "model" in checkpoint:
        model_coarse.load_state_dict(checkpoint["model"], strict=False)
    if "model_fine" in checkpoint:
        model_fine.load_state_dict(checkpoint["model_fine"], strict=False)
    else:
        model_fine.load_state_dict(model_coarse.state_dict(), strict=False)
    model_coarse.eval()
    model_fine.eval()

    positions = payload.positions.to(device)
    directions = payload.directions.to(device)
    text_embeddings = payload.text_embeddings.to(device)
    rgbs = payload.rgbs.to(device)

    near = payload.meta.get("near", args.near)
    far = payload.meta.get("far", args.far)

    with torch.no_grad():
        result = render_rays(
            model_coarse,
            positions,
            directions,
            num_samples=args.num_samples,
            near=near,
            far=far,
            perturb=False,
            chunk_rays=args.chunk_rays,
            clip_embedding=text_embeddings,
            white_bkgd=True,
            num_importance=args.num_importance,
            model_fine=model_fine,
        )
        mse = torch.mean((result.rgb - rgbs) ** 2)
        psnr_value = psnr(mse).item()
        if args.compute_ssim:
            H = payload.meta.get("H")
            W = payload.meta.get("W")
            if H is not None and W is not None:
                pred = result.rgb.view(-1, H, W, 3).permute(0, 3, 1, 2)
                gt = rgbs.view(-1, H, W, 3).permute(0, 3, 1, 2)
                ssim_value = ssim(pred, gt).mean().item()
            else:
                ssim_value = float("nan")
        else:
            ssim_value = float("nan")

    print(f"mse={mse.item():.6f} psnr={psnr_value:.2f}dB ssim={ssim_value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--nerf-hidden-dim", type=int, default=256)
    parser.add_argument("--nerf-layers", type=int, default=8)
    parser.add_argument("--nerf-skips", type=int, nargs="*", default=[4])
    parser.add_argument("--pos-freqs", type=int, default=10)
    parser.add_argument("--dir-freqs", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--num-importance", type=int, default=64)
    parser.add_argument("--near", type=float, default=2.0)
    parser.add_argument("--far", type=float, default=6.0)
    parser.add_argument("--chunk-rays", type=int, default=8192)
    parser.add_argument("--compute-ssim", action="store_true")
    parser.add_argument("--clip-backend", type=str, default="open_clip")
    parser.add_argument("--clip-model", type=str, default="ViT-B-32")
    parser.add_argument("--clip-pretrained", type=str, default="openai")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
