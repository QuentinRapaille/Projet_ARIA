from __future__ import annotations

import lightning as L
from torch.utils.data import DataLoader

from .pastis_r_embedding_dataset import PastisEmbeddingDataset
from my_code.models.fm_base import FMBase
from pathlib import Path


class EmbeddingDataModule(L.LightningDataModule):
    """
    Minimal : un train loader et un val loader.
    Pour rester minimal sans config supplémentaire, val=train (le but est le debug shapes/grads).
    """
    def __init__(self, pastis_root: str, fm: FMBase, batch_size: int, num_workers: int, pin_memory: bool) -> None:
        super().__init__()
        self.pastis_root = Path(pastis_root)
        self.fm = fm
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_ds = None
        self.val_ds = None

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = PastisEmbeddingDataset(self.pastis_root, self.fm)
        self.val_ds = PastisEmbeddingDataset(self.pastis_root, self.fm)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
