from __future__ import annotations

import lightning as L
import torch
from torch import nn
from pathlib import Path


from torchmetrics.classification import (
    MulticlassJaccardIndex,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)

from my_code.models.mlp_head import MLPHeadConfig, PixelMLPHead


class SegmentationMLPModule(L.LightningModule):
    """
    Minimal:
      - logits = head(embeddings)
      - CE loss
      - debug prints au premier batch
      - check gradients sur la head
    """
    def __init__(self, head_cfg: MLPHeadConfig, lr: float, ignore_index: int) -> None:
        super().__init__()
        self.head = PixelMLPHead(head_cfg)
        self.lr = lr
        self.ignore_index = ignore_index

        # À adapter selon le vrai champ de ta config.
        # L'objectif est d'avoir num_classes = K (ici 20).
        self.num_classes = head_cfg.num_classes

        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.val_miou = MulticlassJaccardIndex(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average="macro",
        )
        self.val_f1_macro = MulticlassF1Score(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average="macro",
        )
        self.val_confmat = MulticlassConfusionMatrix(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.head(embeddings)  # (B,H,W,K)

    def _loss(self, logits_bhwk: torch.Tensor, masks_bhw: torch.Tensor) -> torch.Tensor:
        # CrossEntropyLoss attend (B,K,H,W) + target (B,H,W)
        logits_bkhw = logits_bhwk.permute(0, 3, 1, 2).contiguous()
        return self.criterion(logits_bkhw, masks_bhw)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        emb = batch["embeddings"]
        mask = batch["masks"]

        if batch_idx == 0 and self.global_step == 0:
            self.print(f"[DEBUG] embeddings: {tuple(emb.shape)} {emb.dtype}")
            self.print(f"[DEBUG] masks: {tuple(mask.shape)} {mask.dtype}")

        logits = self(emb)

        if batch_idx == 0 and self.global_step == 0:
            self.print(f"[DEBUG] logits: {tuple(logits.shape)} {logits.dtype}")

        loss = self._loss(logits, mask)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_after_backward(self):
        # Vérifie une fois que la head reçoit du gradient
        if self.global_step == 0:
            has_grad = False
            max_grad = 0.0
            for p in self.head.parameters():
                if p.grad is not None:
                    has_grad = True
                    max_grad = max(max_grad, float(p.grad.abs().max()))
            self.print(f"[DEBUG] head_has_grad={has_grad} head_max_abs_grad={max_grad}")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        emb = batch["embeddings"]
        mask = batch["masks"]
        logits = self(emb)

        preds = torch.argmax(logits, dim=-1)  # (B, H, W)

        self.val_miou.update(preds, mask)
        self.val_f1_macro.update(preds, mask)
        self.val_confmat.update(preds, mask)

        loss = self._loss(logits, mask)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        miou = self.val_miou.compute()
        f1 = self.val_f1_macro.compute()
        confmat = self.val_confmat.compute()

        if self.trainer.is_global_zero:
            out_dir = Path(self.trainer.log_dir) if self.trainer.log_dir else Path("logs")
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(confmat.cpu(), out_dir / f"confmat_epoch={self.current_epoch}.pt")

        self.log("val/mIoU", miou, prog_bar=True)
        self.log("val/F1_macro", f1, prog_bar=True)

        if self.trainer.is_global_zero:
            self.print("[VAL] mIoU =", float(miou))
            self.print("[VAL] F1_macro =", float(f1))
            self.print("[VAL] Confusion matrix:")
            self.print(confmat)

        # Reset explicite (optionnel mais robuste)
        self.val_miou.reset()
        self.val_f1_macro.reset()
        self.val_confmat.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
