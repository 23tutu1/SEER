from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.clip_embedding import CLIPConfig, CLIPEmbedder
from src.data import RaysDataset, RaysPayload, load_rays_payload
from src.nerf_model import NeRFModel
from src.seer_model import SEERModel
from src.rendering import render_rays, multiview_consistency_loss


def _build_text_embeddings(payload: RaysPayload, embedder: CLIPEmbedder) -> torch.Tensor:
    if payload.text_embeddings is not None:
        return payload.text_embeddings
    if payload.texts is None:
        if payload.image_texts is None or payload.image_indices is None:
            zeros = torch.zeros(payload.positions.shape[0], embedder.text_dim)
            return zeros
        image_embeddings = embedder.encode_text(payload.image_texts).cpu()
        mapped = image_embeddings[payload.image_indices]
        return mapped

    unique_texts = sorted(set(payload.texts))
    text_to_idx: Dict[str, int] = {text: i for i, text in enumerate(unique_texts)}
    embeddings = embedder.encode_text(unique_texts).cpu()
    mapped = torch.stack([embeddings[text_to_idx[text]] for text in payload.texts])
    return mapped


def _select_image_index(payload: RaysPayload, override: Optional[int] = None) -> Optional[int]:
    if payload.image_indices is None or "num_images" not in payload.meta:
        return None
    num_images = payload.meta["num_images"]
    if override is not None:
        if override < 0 or override >= num_images:
            raise ValueError("clip-image-index out of range.")
        return override
    return int(torch.randint(0, num_images, (1,)).item())


def _get_near_far(payload: RaysPayload, args: argparse.Namespace) -> Tuple[float, float]:
    near = float(args.near)
    far = float(args.far)
    if "near" in payload.meta:
        near = float(payload.meta["near"])
    if "far" in payload.meta:
        far = float(payload.meta["far"])
    return near, far


def _compute_clip_loss(
    payload: RaysPayload,
    model: SEERModel,
    embedder: CLIPEmbedder,
    device: torch.device,
    num_samples: int,
    near: float,
    far: float,
    chunk_rays: int,
    prompt: str,
    image_index: Optional[int],
    downsample: int,
    num_importance: int,
    model_fine: Optional[SEERModel],
) -> torch.Tensor:
    if payload.image_indices is None or "H" not in payload.meta or "W" not in payload.meta:
        return torch.tensor(0.0, device=device)

    image_idx = _select_image_index(payload, image_index)
    if image_idx is None:
        return torch.tensor(0.0, device=device)

    mask = payload.image_indices == image_idx
    if mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    if payload.image_texts is not None:
        text = payload.image_texts[image_idx]
    else:
        text = prompt.strip()
    if not text:
        return torch.tensor(0.0, device=device)

    rays_o = payload.positions[mask].to(device)
    rays_d = payload.directions[mask].to(device)

    H = int(payload.meta["H"])
    W = int(payload.meta["W"])
    if rays_o.shape[0] != H * W:
        return torch.tensor(0.0, device=device)

    if downsample > 1:
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        idx = idx.view(H, W)[::downsample, ::downsample].reshape(-1)
        rays_o = payload.positions[idx].to(device)
        rays_d = payload.directions[idx].to(device)
        H = (H + downsample - 1) // downsample
        W = (W + downsample - 1) // downsample

    text_embedding = embedder.encode_text(text).to(device)
    result = render_rays(
        model,
        rays_o,
        rays_d,
        num_samples=num_samples,
        near=near,
        far=far,
        perturb=False,
        chunk_rays=chunk_rays,
        clip_embedding=text_embedding,
        white_bkgd=True,
        num_importance=num_importance,
        model_fine=model_fine,
    )

    image = result.rgb.view(H, W, 3).clamp(0.0, 1.0)
    image_tensor = image.permute(2, 0, 1).unsqueeze(0)
    image_tensor = embedder.preprocess_tensor(image_tensor)
    image_embedding = embedder.encode_image(image_tensor, no_grad=False)

    text_embedding = F.normalize(text_embedding, dim=-1)
    image_embedding = F.normalize(image_embedding, dim=-1)
    return 1.0 - (image_embedding * text_embedding).sum(dim=-1).mean()


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    payload = load_rays_payload(args.data)
    embedder = CLIPEmbedder(
        CLIPConfig(
            backend=args.clip_backend,
            model_name=args.clip_model,
            pretrained=args.clip_pretrained,
            device=args.device,
            freeze=True,
        )
    )
    payload.text_embeddings = _build_text_embeddings(payload, embedder)

    dataset = RaysDataset(payload)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

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
    model_coarse = SEERModel(nerf_coarse, clip_dim=embedder.text_dim, hidden_dim=args.hidden_dim).to(device)
    model_fine = SEERModel(nerf_fine, clip_dim=embedder.text_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(
        list(model_coarse.parameters()) + list(model_fine.parameters()),
        lr=args.lr,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    near, far = _get_near_far(payload, args)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model_coarse.train()
        model_fine.train()
        total_loss = 0.0
        clip_loss_value = 0.0
        clip_loss_count = 0
        for batch in loader:
            positions = batch["positions"].to(device)
            directions = batch["directions"].to(device)
            rgbs = batch.get("rgbs")
            if rgbs is None:
                raise ValueError("Training requires rgbs in the dataset payload.")
            rgbs = rgbs.to(device)
            text_embeddings = batch["text_embeddings"].to(device)

            result = render_rays(
                model_coarse,
                positions,
                directions,
                num_samples=args.num_samples,
                near=near,
                far=far,
                perturb=args.perturb,
                chunk_rays=args.chunk_rays,
                clip_embedding=text_embeddings,
                white_bkgd=True,
                num_importance=args.num_importance,
                model_fine=model_fine,
            )
            loss = F.mse_loss(result.rgb, rgbs)
            if args.coarse_loss_weight > 0.0 and result.rgb_coarse is not None:
                loss = loss + args.coarse_loss_weight * F.mse_loss(result.rgb_coarse, rgbs)

            if args.consistency_weight > 0.0:
                sample_count = min(args.consistency_samples, positions.shape[0])
                idx = torch.randperm(positions.shape[0], device=device)[:sample_count]
                cons_loss = multiview_consistency_loss(
                    nerf_fine,
                    positions[idx],
                    directions[idx],
                    num_samples=args.num_samples,
                    near=near,
                    far=far,
                    color_weight=args.consistency_color_weight,
                )
                loss = loss + args.consistency_weight * cons_loss

            if args.clip_loss_weight > 0.0 and args.clip_every > 0 and global_step % args.clip_every == 0:
                clip_loss = _compute_clip_loss(
                    payload=payload,
                    model=model_coarse,
                    embedder=embedder,
                    device=device,
                    num_samples=args.num_samples,
                    near=near,
                    far=far,
                    chunk_rays=args.chunk_rays,
                    prompt=args.clip_prompt,
                    image_index=args.clip_image_index,
                    downsample=args.clip_downsample,
                    num_importance=args.num_importance,
                    model_fine=model_fine,
                )
                clip_loss_value += float(clip_loss.item())
                clip_loss_count += 1
                loss = loss + args.clip_loss_weight * clip_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * positions.shape[0]
            global_step += 1

        avg_loss = total_loss / len(dataset)
        if clip_loss_count > 0:
            clip_loss_value = clip_loss_value / clip_loss_count
        print(
            f"epoch={epoch} loss={avg_loss:.6f} clip_loss={clip_loss_value:.6f} "
            f"near={near} far={far}"
        )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = {
                "model_coarse": model_coarse.state_dict(),
                "model_fine": model_fine.state_dict(),
                "nerf_coarse": nerf_coarse.state_dict(),
                "nerf_fine": nerf_fine.state_dict(),
                "clip_config": embedder.config.__dict__,
                "epoch": epoch,
            }
            ckpt_path = output_dir / f"seer_epoch_{epoch}.pt"
            torch.save(ckpt, ckpt_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to rays .pt or .npz")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
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
    parser.add_argument("--perturb", action="store_true")
    parser.add_argument("--chunk-rays", type=int, default=8192)
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--consistency-samples", type=int, default=1024)
    parser.add_argument("--consistency-color-weight", type=float, default=0.0)
    parser.add_argument("--coarse-loss-weight", type=float, default=0.5)
    parser.add_argument("--clip-loss-weight", type=float, default=0.0)
    parser.add_argument("--clip-every", type=int, default=1)
    parser.add_argument("--clip-prompt", type=str, default="")
    parser.add_argument("--clip-image-index", type=int, default=None)
    parser.add_argument("--clip-downsample", type=int, default=1)
    parser.add_argument("--clip-backend", type=str, default="open_clip")
    parser.add_argument("--clip-model", type=str, default="ViT-B-32")
    parser.add_argument("--clip-pretrained", type=str, default="openai")
    parser.add_argument("--save-every", type=int, default=1)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
