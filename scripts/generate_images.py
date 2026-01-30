from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.clip_embedding import CLIPConfig, CLIPEmbedder
from src.data import load_rays_payload
from src.nerf_model import NeRFModel
from src.seer_model import SEERModel
from src.rendering import render_rays


def generate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    payload = load_rays_payload(args.rays)
    if "H" not in payload.meta or "W" not in payload.meta:
        raise ValueError("rays payload must include H and W for image reconstruction.")

    clip_embedder = CLIPEmbedder(
        CLIPConfig(
            backend=args.clip_backend,
            model_name=args.clip_model,
            pretrained=args.clip_pretrained,
            device=args.device,
            freeze=True,
        )
    )

    if args.prompt:
        clip_embedding = clip_embedder.encode_text(args.prompt)
    elif payload.image_texts is not None:
        clip_embedding = clip_embedder.encode_text(payload.image_texts[0])
    elif payload.text_embeddings is not None:
        clip_embedding = payload.text_embeddings[:1]
    elif payload.texts is not None:
        clip_embedding = clip_embedder.encode_text(payload.texts[0])
    else:
        clip_embedding = torch.zeros(1, clip_embedder.text_dim)

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
            clip_embedding=clip_embedding.to(device),
            white_bkgd=True,
            num_importance=args.num_importance,
            model_fine=model_fine,
        )

    H = payload.meta["H"]
    W = payload.meta["W"]
    image = result.rgb.view(H, W, 3).cpu().numpy() * 255.0
    image = image.astype("uint8")
    Image.fromarray(image).save(args.output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--rays", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--output", type=str, default="output.png")
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
    parser.add_argument("--clip-backend", type=str, default="open_clip")
    parser.add_argument("--clip-model", type=str, default="ViT-B-32")
    parser.add_argument("--clip-pretrained", type=str, default="openai")
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
