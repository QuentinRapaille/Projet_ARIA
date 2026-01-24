from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn, Tensor


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

    Implémentation "shallow conv head" : ajoute un minimum de contexte spatial
    via quelques convolutions 3x3 sur la carte (B,D,H,W), puis une projection 1x1 vers K classes.
    """
    def __init__(self, cfg: MLPHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg

        dropout_p = getattr(cfg, "dropout", 0.3)
        hidden = cfg.hidden_dim_2        

        # ANCIENNE VERSION : (sur-apprentissage)
        # self.net = nn.Sequential(
        #     nn.Linear(cfg.in_dim, cfg.hidden_dim_1),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(cfg.hidden_dim_1, cfg.hidden_dim_2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(cfg.hidden_dim_2, cfg.num_classes),
        # )

        # ANCIENNE VERSION (MLP 1 sous-couche)
        # self.net = nn.Sequential(
        #     nn.Linear(cfg.in_dim, hidden),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=dropout_p),
        #     nn.Linear(hidden, cfg.num_classes),
        # )

        # NOUVELLE VERSION (module convolutionel : cohérence spatiale)
        #
        # Avantages :
        #  - Très léger comparé au U-Net suggéré par la doc AlphaEarth
        #  - Ajoute de la cohérence spatiale.

        n_convs = 2 # 2 blocs de convolution

        layers: list[nn.Module] = []

        # D -> hidden
        layers += [
            nn.Conv2d(cfg.in_dim, hidden, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
        ]

        # hidden -> hidden
        for _ in range(n_convs - 1):
            layers += [
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout_p),
            ]

        # hidden -> K 
        layers += [
            nn.Conv2d(hidden, cfg.num_classes, kernel_size=1, bias=True),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"On attend un batch (B,H,W,D), on a : {tuple(x.shape)}")

        b, h, w, d = x.shape
        if d != self.cfg.in_dim:
            raise ValueError(f"dimension d'entrée inattendue : {d} au lieu de {self.cfg.in_dim}")


        # (B,H,W,D) -> (B,D,H,W)
        x_cf = x.permute(0, 3, 1, 2).contiguous()
        # (B,D,H,W) -> (B,K,H,W)
        y_cf = self.net(x_cf)
        # (B,K,H,W) -> (B,H,W,K)
        y = y_cf.permute(0, 2, 3, 1).contiguous()
        return y
