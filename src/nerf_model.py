from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs: int, include_input: bool = True) -> None:
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.register_buffer(
            "freq_bands",
            2.0 ** torch.arange(0, num_freqs),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [x] if self.include_input else []
        for freq in self.freq_bands:
            out.append(torch.sin(x * freq))
            out.append(torch.cos(x * freq))
        return torch.cat(out, dim=-1)


class NeRFModel(nn.Module):
    def __init__(
        self,
        pos_freqs: int = 10,
        dir_freqs: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 8,
        skips: tuple[int, ...] = (4,),
    ) -> None:
        super().__init__()
        self.pos_enc = PositionalEncoding(pos_freqs)
        self.dir_enc = PositionalEncoding(dir_freqs)
        self.skips = set(skips)

        pos_in_dim = 3 * (1 + 2 * pos_freqs)
        dir_in_dim = 3 * (1 + 2 * dir_freqs)

        layers = []
        in_dim = pos_in_dim
        for i in range(num_layers):
            if i in self.skips and i != 0:
                in_dim += pos_in_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.mlp = nn.ModuleList(layers)

        self.sigma_head = nn.Linear(hidden_dim, 1)
        self.feature_head = nn.Linear(hidden_dim, hidden_dim)
        self.rgb_head = nn.Sequential(
            nn.Linear(hidden_dim + dir_in_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, position: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        pos = self.pos_enc(position)
        x = pos
        for i, layer in enumerate(self.mlp):
            if i in self.skips and i != 0:
                x = torch.cat([x, pos], dim=-1)
            x = F.relu(layer(x))

        sigma = self.sigma_head(x)
        features = self.feature_head(x)
        dir_enc = self.dir_enc(direction)
        rgb = self.rgb_head(torch.cat([features, dir_enc], dim=-1))
        return torch.cat([sigma, rgb], dim=-1)
