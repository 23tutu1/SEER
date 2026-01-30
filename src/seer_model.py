from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class SemanticModulator(nn.Module):
    def __init__(self, clip_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 6),
        )

    def forward(self, clip_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self.net(clip_embedding)
        scale, bias = params.chunk(2, dim=-1)
        return scale, bias


class SEERModel(nn.Module):
    def __init__(self, nerf: nn.Module, clip_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.nerf = nerf
        self.modulator = SemanticModulator(clip_dim, hidden_dim)

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor,
        clip_embedding: torch.Tensor,
    ) -> torch.Tensor:
        raw = self.nerf(positions, directions)
        sigma = raw[..., :1]
        rgb = raw[..., 1:4]

        clip_embedding = self._expand_clip(clip_embedding, rgb)
        scale, bias = self.modulator(clip_embedding)
        rgb = rgb * (1 + scale) + bias
        return torch.cat([sigma, rgb], dim=-1)

    @staticmethod
    def _expand_clip(clip_embedding: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if clip_embedding.dim() == target.dim():
            return clip_embedding
        if clip_embedding.dim() == 2 and target.dim() >= 3:
            if clip_embedding.shape[0] != target.shape[0]:
                raise ValueError("Batch size mismatch between clip embedding and target.")
            expand_shape = (clip_embedding.shape[0],) + target.shape[1:-1] + (
                clip_embedding.shape[-1],
            )
            return clip_embedding.view(clip_embedding.shape[0], 1, -1).expand(*expand_shape)
        raise ValueError("Unsupported clip embedding shape for broadcasting.")
