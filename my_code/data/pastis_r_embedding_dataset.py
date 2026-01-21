from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from my_code.models.fm_base import FMBase


def _read_patch_ids(metadata_geojson: Path) -> List[int]:
    """
    Lit metadata.geojson et récupère les ID_PATCH.
    """
    obj = json.loads(metadata_geojson.read_text(encoding="utf-8"))
    feats = obj.get("features", [])
    pids: List[int] = []
    for ft in feats:
        props = ft.get("properties", {})
        if "ID_PATCH" in props:
            pids.append(int(props["ID_PATCH"]))
    if not pids:
        raise RuntimeError(f"No ID_PATCH found in {metadata_geojson}")
    return pids


def _target_path(pastis_root: Path, pid: int) -> Path:
    """
    Chemin unique imposé par l'organisation PASTIS-R :
      PASTIS-R/ANNOTATIONS/TARGET_<pid>.npy
    """
    path = pastis_root / "ANNOTATIONS" / f"TARGET_{pid}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing target for pid={pid}: {path}")
    return path


def _load_mask_hw(target_path: Path) -> torch.Tensor:
    """
    Charge TARGET_<pid>.npy et retourne un masque (H,W) int64.

    Supporte:
      - (H,W)
      - (3,H,W) où canal 0 = labels
    """
    t = np.load(target_path)

    if t.ndim == 2:
        mask = t
    elif t.ndim == 3 and t.shape[0] == 3:
        mask = t[0]
    else:
        raise ValueError(f"Unexpected TARGET shape={t.shape} for {target_path}")

    return torch.from_numpy(mask.astype(np.int64, copy=False))


class PastisEmbeddingDataset(Dataset):
    """
    Dataset minimal :
      - embeddings: (H,W,D) float32
      - masks:      (H,W)   int64
      - pid:        int64
    """
    def __init__(
        self,
        pastis_root: Path,
        fm: FMBase,
        subset_patch_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self.pastis_root = pastis_root
        self.fm = fm

        pids = _read_patch_ids(pastis_root / "metadata.geojson")
        if subset_patch_ids is not None:
            wanted = set(int(x) for x in subset_patch_ids)
            pids = [pid for pid in pids if pid in wanted]
        if not pids:
            raise RuntimeError("Empty dataset after filtering.")
        self.pids = pids

    def __len__(self) -> int:
        return len(self.pids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pid = int(self.pids[idx])

        fm_out = self.fm.load(pid)
        emb = fm_out.embeddings_hwd  # (H,W,D)

        target_path = _target_path(self.pastis_root, pid)
        mask = _load_mask_hw(target_path)  # (H,W)

        if tuple(emb.shape[:2]) != tuple(mask.shape):
            raise ValueError(
                f"HW mismatch pid={pid}: emb HW={tuple(emb.shape[:2])} vs mask HW={tuple(mask.shape)}"
            )

        return {
            "embeddings": emb,
            "masks": mask,
            "pid": torch.tensor(pid, dtype=torch.int64),
        }
