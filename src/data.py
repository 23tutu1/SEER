from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class RaysPayload:
    positions: torch.Tensor
    directions: torch.Tensor
    rgbs: Optional[torch.Tensor]
    texts: Optional[List[str]]
    text_embeddings: Optional[torch.Tensor]
    image_indices: Optional[torch.Tensor]
    image_texts: Optional[List[str]]
    meta: Dict[str, int]


def _load_np_or_pt(path: Path) -> Dict:
    if path.suffix in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu")
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def load_rays_payload(path: str | Path) -> RaysPayload:
    path = Path(path)
    data = _load_np_or_pt(path)

    positions = torch.as_tensor(data.get("positions"), dtype=torch.float32)
    directions = torch.as_tensor(data.get("directions"), dtype=torch.float32)

    rgbs_raw = data.get("rgbs")
    rgbs = torch.as_tensor(rgbs_raw, dtype=torch.float32) if rgbs_raw is not None else None

    texts_raw = data.get("texts")
    texts = list(texts_raw) if texts_raw is not None else None

    text_emb_raw = data.get("text_embeddings")
    text_embeddings = (
        torch.as_tensor(text_emb_raw, dtype=torch.float32) if text_emb_raw is not None else None
    )

    image_indices_raw = data.get("image_indices")
    image_indices = (
        torch.as_tensor(image_indices_raw, dtype=torch.long) if image_indices_raw is not None else None
    )

    image_texts_raw = data.get("image_texts")
    image_texts = list(image_texts_raw) if image_texts_raw is not None else None

    meta = {}
    for key in ("H", "W", "num_images"):
        if key in data:
            meta[key] = int(data[key])
    if "near" in data:
        meta["near"] = float(data["near"])
    if "far" in data:
        meta["far"] = float(data["far"])

    return RaysPayload(
        positions=positions,
        directions=directions,
        rgbs=rgbs,
        texts=texts,
        text_embeddings=text_embeddings,
        image_indices=image_indices,
        image_texts=image_texts,
        meta=meta,
    )


class RaysDataset(Dataset):
    def __init__(self, payload: RaysPayload) -> None:
        self.positions = payload.positions
        self.directions = payload.directions
        self.rgbs = payload.rgbs
        self.texts = payload.texts
        self.text_embeddings = payload.text_embeddings
        self.image_indices = payload.image_indices
        self.image_texts = payload.image_texts
        self.meta = payload.meta

        if self.positions.shape != self.directions.shape:
            raise ValueError("positions and directions must have the same shape.")
        if self.rgbs is not None and self.rgbs.shape[0] != self.positions.shape[0]:
            raise ValueError("rgbs and positions must have the same first dimension.")
        if self.text_embeddings is not None and self.text_embeddings.shape[0] != self.positions.shape[0]:
            raise ValueError("text embeddings and positions must have the same first dimension.")
        if self.texts is not None and len(self.texts) != self.positions.shape[0]:
            raise ValueError("texts and positions must have the same length.")
        if self.image_indices is not None and self.image_indices.shape[0] != self.positions.shape[0]:
            raise ValueError("image_indices and positions must have the same length.")

    def __len__(self) -> int:
        return self.positions.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        item = {
            "positions": self.positions[idx],
            "directions": self.directions[idx],
        }
        if self.rgbs is not None:
            item["rgbs"] = self.rgbs[idx]
        if self.text_embeddings is not None:
            item["text_embeddings"] = self.text_embeddings[idx]
        if self.texts is not None:
            item["texts"] = self.texts[idx]
        if self.image_indices is not None:
            item["image_indices"] = self.image_indices[idx]
        return item
