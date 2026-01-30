from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CLIPConfig:
    backend: str = "open_clip"  # "open_clip" or "clip"
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    device: str = "cpu"
    freeze: bool = True
    image_size: int = 224


def _as_list(texts: Iterable[str] | str) -> List[str]:
    if isinstance(texts, str):
        return [texts]
    return list(texts)


def _infer_embed_dim(model: nn.Module) -> int:
    if hasattr(model, "text_projection") and model.text_projection is not None:
        proj = model.text_projection
        if hasattr(proj, "shape") and len(proj.shape) >= 2:
            return int(proj.shape[1])
    if hasattr(model, "embed_dim"):
        return int(model.embed_dim)
    if hasattr(model, "visual") and hasattr(model.visual, "output_dim"):
        return int(model.visual.output_dim)
    raise RuntimeError("Unable to infer CLIP embedding dimension from the model.")


class CLIPEmbedder(nn.Module):
    def __init__(self, config: Optional[CLIPConfig] = None) -> None:
        super().__init__()
        self.config = config or CLIPConfig()
        self.device = torch.device(self.config.device)
        self.model, self.preprocess, self.tokenizer = self._load_backend(self.config)
        self.text_dim = _infer_embed_dim(self.model)
        self.image_size = self._infer_image_size(self.config.image_size)

        if self.config.freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

        self.register_buffer(
            "_clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
            persistent=False,
        )

    def _infer_image_size(self, fallback: int) -> int:
        if hasattr(self.model, "visual") and hasattr(self.model.visual, "image_size"):
            size = self.model.visual.image_size
            if isinstance(size, (tuple, list)):
                return int(size[0])
            return int(size)
        return fallback

    def _load_backend(
        self, config: CLIPConfig
    ) -> Tuple[nn.Module, Optional[nn.Module], callable]:
        backend = config.backend.lower().strip()
        if backend == "open_clip":
            try:
                import open_clip
            except Exception as exc:  # pragma: no cover - dependency optional
                raise ImportError(
                    "open_clip is required for backend='open_clip'. "
                    "Install with: pip install open_clip_torch"
                ) from exc
            model, _, preprocess = open_clip.create_model_and_transforms(
                config.model_name, pretrained=config.pretrained
            )
            tokenizer = open_clip.get_tokenizer(config.model_name)
            model.to(self.device)
            return model, preprocess, tokenizer

        if backend == "clip":
            try:
                import clip
            except Exception as exc:  # pragma: no cover - dependency optional
                raise ImportError(
                    "openai-clip is required for backend='clip'. "
                    "Install with: pip install git+https://github.com/openai/CLIP.git"
                ) from exc
            model, preprocess = clip.load(config.model_name, device=self.device, jit=False)
            tokenizer = clip.tokenize
            return model, preprocess, tokenizer

        raise ValueError(f"Unsupported CLIP backend: {config.backend}")

    def encode_text(self, texts: Sequence[str] | str, no_grad: bool = True) -> torch.Tensor:
        text_list = _as_list(texts)
        tokens = self.tokenizer(text_list)
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.to(self.device)
        if no_grad:
            with torch.no_grad():
                embeddings = self.model.encode_text(tokens)
        else:
            embeddings = self.model.encode_text(tokens)
        embeddings = embeddings.float()
        return F.normalize(embeddings, dim=-1)

    def encode_image(self, images: torch.Tensor, no_grad: bool = True) -> torch.Tensor:
        images = images.to(self.device)
        if no_grad:
            with torch.no_grad():
                embeddings = self.model.encode_image(images)
        else:
            embeddings = self.model.encode_image(images)
        embeddings = embeddings.float()
        return F.normalize(embeddings, dim=-1)

    def preprocess_images(self, images: Sequence) -> torch.Tensor:
        if self.preprocess is None:
            raise RuntimeError("No preprocess function available for this backend.")
        return torch.stack([self.preprocess(img) for img in images]).to(self.device)

    def preprocess_tensor(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError("images must be a 4D tensor with shape (B, 3, H, W).")
        images = images.to(self.device)
        images = F.interpolate(
            images,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return (images - self._clip_mean) / self._clip_std
