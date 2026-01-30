from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image


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


def _resolve_image_path(dataset_dir: Path, file_path: str) -> Path:
    path = dataset_dir / file_path
    if path.exists():
        return path
    for ext in (".png", ".jpg", ".jpeg"):
        alt = path.with_suffix(ext)
        if alt.exists():
            return alt
    raise FileNotFoundError(f"Image not found for frame path: {file_path}")


def _build_rays(H: int, W: int, focal: float, c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    dirs = np.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -np.ones_like(i)], axis=-1)
    rays_d = (dirs[..., None, :] * c2w[:3, :3]).sum(axis=-1)
    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape)
    return rays_o, rays_d


def preprocess_blender(
    dataset_dir: Path,
    split: str,
    output_path: Path,
    near: float,
    far: float,
    text_prompt: str | None,
) -> None:
    data = _load_transforms(dataset_dir, split)
    camera_angle_x = data["camera_angle_x"]
    frames = data["frames"]

    all_positions = []
    all_directions = []
    all_rgbs = []
    all_indices = []

    for image_idx, frame in enumerate(frames):
        image_path = _resolve_image_path(dataset_dir, frame["file_path"])
        image = Image.open(image_path).convert("RGB")
        W, H = image.size
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        rays_o, rays_d = _build_rays(H, W, focal, c2w)
        rgbs = np.asarray(image, dtype=np.float32) / 255.0

        all_positions.append(rays_o.reshape(-1, 3))
        all_directions.append(rays_d.reshape(-1, 3))
        all_rgbs.append(rgbs.reshape(-1, 3))
        all_indices.append(np.full((rgbs.shape[0] * rgbs.shape[1],), image_idx, dtype=np.int64))

    positions = np.concatenate(all_positions, axis=0)
    directions = np.concatenate(all_directions, axis=0)
    rgbs = np.concatenate(all_rgbs, axis=0)
    image_indices = np.concatenate(all_indices, axis=0)

    image_texts = None
    if text_prompt:
        image_texts = [text_prompt for _ in range(len(frames))]

    payload = {
        "positions": torch.from_numpy(positions),
        "directions": torch.from_numpy(directions),
        "rgbs": torch.from_numpy(rgbs),
        "image_indices": torch.from_numpy(image_indices),
        "image_texts": image_texts,
        "H": H,
        "W": W,
        "num_images": len(frames),
        "near": float(near),
        "far": float(far),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def _load_llff_poses(dataset_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    poses_bounds_path = dataset_dir / "poses_bounds.npy"
    if not poses_bounds_path.exists():
        raise FileNotFoundError("poses_bounds.npy not found for LLFF dataset.")
    poses_arr = np.load(poses_bounds_path)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])
    bounds = poses_arr[:, -2:]
    return poses, bounds


def _collect_image_paths(images_dir: Path) -> List[Path]:
    candidates = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        candidates.extend(sorted(images_dir.glob(ext)))
    if not candidates:
        raise FileNotFoundError(f"No images found in {images_dir}")
    return candidates


def preprocess_llff(
    dataset_dir: Path,
    output_path: Path,
    near: float | None,
    far: float | None,
    text_prompt: str | None,
    images_dir: str,
) -> None:
    poses, bounds = _load_llff_poses(dataset_dir)
    images_root = dataset_dir / images_dir
    image_paths = _collect_image_paths(images_root)
    num_images = len(image_paths)
    if poses.shape[0] != num_images:
        raise ValueError("Number of poses does not match number of images.")

    H, W, focal = poses[0, :, 4]
    H = int(H)
    W = int(W)
    focal = float(focal)

    all_positions = []
    all_directions = []
    all_rgbs = []
    all_indices = []

    for image_idx, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        rgbs = np.asarray(image, dtype=np.float32) / 255.0
        c2w = poses[image_idx, :3, :4].astype(np.float32)
        rays_o, rays_d = _build_rays(H, W, focal, c2w)

        all_positions.append(rays_o.reshape(-1, 3))
        all_directions.append(rays_d.reshape(-1, 3))
        all_rgbs.append(rgbs.reshape(-1, 3))
        all_indices.append(np.full((rgbs.shape[0] * rgbs.shape[1],), image_idx, dtype=np.int64))

    positions = np.concatenate(all_positions, axis=0)
    directions = np.concatenate(all_directions, axis=0)
    rgbs = np.concatenate(all_rgbs, axis=0)
    image_indices = np.concatenate(all_indices, axis=0)

    image_texts = None
    if text_prompt:
        image_texts = [text_prompt for _ in range(num_images)]

    near_val = float(near) if near is not None else float(bounds.min())
    far_val = float(far) if far is not None else float(bounds.max())

    payload = {
        "positions": torch.from_numpy(positions),
        "directions": torch.from_numpy(directions),
        "rgbs": torch.from_numpy(rgbs),
        "image_indices": torch.from_numpy(image_indices),
        "image_texts": image_texts,
        "H": H,
        "W": W,
        "num_images": num_images,
        "near": near_val,
        "far": far_val,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--near", type=float, default=None)
    parser.add_argument("--far", type=float, default=None)
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--dataset-type", type=str, choices=["blender", "llff"], default="blender")
    parser.add_argument("--images-dir", type=str, default="images")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_path = Path(args.output)
    text_prompt = args.text.strip() or None
    if args.dataset_type == "blender":
        near = args.near if args.near is not None else 2.0
        far = args.far if args.far is not None else 6.0
        preprocess_blender(dataset_dir, args.split, output_path, near, far, text_prompt)
    else:
        preprocess_llff(
            dataset_dir,
            output_path,
            near=args.near,
            far=args.far,
            text_prompt=text_prompt,
            images_dir=args.images_dir,
        )


if __name__ == "__main__":
    main()
