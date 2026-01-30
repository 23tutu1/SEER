from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from src.clip_embedding import CLIPConfig, CLIPEmbedder
from src.nerf_model import NeRFModel
from src.seer_model import SEERModel
from src.rendering import render_rays


def _load_transforms(dataset_dir: Path, split: str) -> dict:
    candidates = [
        dataset_dir / f"transforms_{split}.json",
        dataset_dir / "transforms.json",
    ]
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError("No transforms JSON found in dataset directory.")


def _build_rays(H: int, W: int, focal: float, c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    dirs = np.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -np.ones_like(i)], axis=-1)
    rays_d = (dirs[..., None, :] * c2w[:3, :3]).sum(axis=-1)
    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape)
    return rays_o, rays_d


def render_path(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    data = _load_transforms(Path(args.dataset_dir), args.split)

    camera_angle_x = data["camera_angle_x"]
    frames = data["frames"]
    if args.every > 1:
        frames = frames[:: args.every]

    clip_embedder = CLIPEmbedder(
        CLIPConfig(
            backend=args.clip_backend,
            model_name=args.clip_model,
            pretrained=args.clip_pretrained,
            device=args.device,
            freeze=True,
        )
    )
    clip_embedding = clip_embedder.encode_text(args.prompt)

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
    model_coarse = SEERModel(nerf_coarse, clip_dim=clip_embedder.text_dim, hidden_dim=args.hidden_dim).to(device)
    model_fine = SEERModel(nerf_fine, clip_dim=clip_embedder.text_dim, hidden_dim=args.hidden_dim).to(device)

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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, frame in enumerate(frames):
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        file_path = frame["file_path"]
        image_path = Path(args.dataset_dir) / file_path
        if image_path.exists():
            image = Image.open(image_path)
            W, H = image.size
        else:
            H = int(args.height)
            W = int(args.width)
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
        rays_o, rays_d = _build_rays(H, W, focal, c2w)
        rays_o = torch.from_numpy(rays_o.reshape(-1, 3)).to(device)
        rays_d = torch.from_numpy(rays_d.reshape(-1, 3)).to(device)

        with torch.no_grad():
            result = render_rays(
                model_coarse,
                rays_o,
                rays_d,
                num_samples=args.num_samples,
                near=args.near,
                far=args.far,
                perturb=False,
                chunk_rays=args.chunk_rays,
                clip_embedding=clip_embedding.to(device),
                white_bkgd=True,
                num_importance=args.num_importance,
                model_fine=model_fine,
            )

        image = result.rgb.view(H, W, 3).cpu().numpy() * 255.0
        Image.fromarray(image.astype("uint8")).save(output_dir / f"frame_{idx:04d}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="renders")
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
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--clip-backend", type=str, default="open_clip")
    parser.add_argument("--clip-model", type=str, default="ViT-B-32")
    parser.add_argument("--clip-pretrained", type=str, default="openai")
    args = parser.parse_args()

    render_path(args)


if __name__ == "__main__":
    main()
