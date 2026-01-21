from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

Tensor = torch.Tensor


@dataclass(frozen=True)
class MLPHeadConfig:
    in_dim: int
    hidden_dim: int
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
            nn.Linear(cfg.in_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected embeddings (B,H,W,D), got shape={tuple(x.shape)}")

        b, h, w, d = x.shape
        if d != self.cfg.in_dim:
            raise ValueError(f"in_dim mismatch: expected {self.cfg.in_dim}, got {d}")

        y = self.net(x.reshape(b * h * w, d))
        return y.reshape(b, h, w, self.cfg.num_classes)
