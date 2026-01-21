from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

Tensor = torch.Tensor


@dataclass(frozen=True)
class MLPHeadConfig:
    in_dim: int
    hidden_dim_1: int
    hidden_dim_2: int
    num_classes: int


class PixelMLPHead(nn.Module):
    """
    input:  (B,H,W,D)
    output: (B,H,W,K)
    """
    def __init__(self, cfg: MLPHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.hidden_dim_1),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim_1, cfg.hidden_dim_2),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim_2, cfg.num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"On attend un batch (B,H,W,D), on a : {tuple(x.shape)}")

        b, h, w, d = x.shape
        if d != self.cfg.in_dim:
            raise ValueError(f"dimension d'entrée inatendue :{d} au lieu de {self.cfg.in_dim}")

        y = self.net(x.reshape(b * h * w, d)) # Application du MLP
        return y.reshape(b, h, w, self.cfg.num_classes)
