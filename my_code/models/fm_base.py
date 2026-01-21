from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch

Layout = Literal["HWD", "DHW"]


@dataclass(frozen=True)
class FMOutput:
    """
    Sortie standardisée pour le pipeline debug.
    """
    embeddings_hwd: torch.Tensor  # (H, W, D) float32
    embedding_dim: int            # D


class FMBase:
    """
    Interface minimale : charger un embedding pour un patch_id.
    """
    def __init__(self) -> None:
        self._embedding_dim: Optional[int] = None

    @property
    def embedding_dim(self) -> Optional[int]:
        return self._embedding_dim

    def load(self, pid: int) -> FMOutput:
        """
        Charge un embedding (H,W,D) float32.
        """
        path = self._path_for_pid(pid)
        arr = np.load(path)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D embedding, got shape={arr.shape} for {path}")

        emb = self._to_hwd(arr, self._layout())
        d = int(emb.shape[-1])

        if self._embedding_dim is None:
            self._embedding_dim = d
        elif self._embedding_dim != d:
            raise ValueError(
                f"Inconsistent embedding_dim within same FM: first={self._embedding_dim}, now={d} (pid={pid}, file={path})"
            )

        return FMOutput(embeddings_hwd=emb, embedding_dim=d)

    def _to_hwd(self, arr: np.ndarray, layout: Layout) -> torch.Tensor:
        """
        Convertit numpy -> torch float32 (H,W,D).
        """
        if layout == "HWD":
            out = arr
        elif layout == "DHW":
            out = np.moveaxis(arr, 0, -1)
        else:
            raise ValueError(f"Unknown layout={layout}")

        out = out.astype(np.float32, copy=False)
        return torch.from_numpy(out)

    # Méthodes à implémenter dans les enfants
    def _path_for_pid(self, pid: int) -> Path:
        raise NotImplementedError

    def _layout(self) -> Layout:
        raise NotImplementedError
