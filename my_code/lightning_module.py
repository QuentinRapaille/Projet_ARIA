from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
from torch import nn
from torchmetrics.classification import (
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassJaccardIndex,
)

from my_code.models.mlp_head import MLPHeadConfig, PixelMLPHead


class SegmentationMLPModule(L.LightningModule):
    """
    - logits = head(batchs)
    - Train/Val : mIoU & F1 macro loggés par époque (courbes)
    - Test : utilisé uniquement quand on lance trainer.test(...), pour produire confmat_final.pt
    """

    def __init__(self, head_cfg: MLPHeadConfig, lr: float, ignore_index: int) -> None:
        super().__init__()
        self.head = PixelMLPHead(head_cfg)
        self.lr = float(lr)
        self.ignore_index = int(ignore_index)

        self.num_classes = int(head_cfg.num_classes)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # --- Train metrics ---
        self.train_miou = MulticlassJaccardIndex(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average="macro",
        )
        self.train_f1_macro = MulticlassF1Score(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average="macro",
        )

        # --- Val metrics ---
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

        # --- Test (confusion matrix) ---
        self.test_confmat = MulticlassConfusionMatrix(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.head(embeddings)  # (B,H,W,K)

    def _loss(self, logits_bhwk: torch.Tensor, masks_bhw: torch.Tensor) -> torch.Tensor:
        # CrossEntropyLoss attend logits(B,K,H,W) et target(B,H,W)
        logits_bkhw = logits_bhwk.permute(0, 3, 1, 2).contiguous() # Donc on permute les logits
        return self.criterion(logits_bkhw, masks_bhw)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        emb = batch["embeddings"]
        mask = batch["masks"]

        #if batch_idx == 0 and self.global_step == 0:
        #    self.print(f"[DEBUG] embeddings: {tuple(emb.shape)} {emb.dtype}")
        #    self.print(f"[DEBUG] masks: {tuple(mask.shape)} {mask.dtype}")

        logits = self(emb)

        #if batch_idx == 0 and self.global_step == 0:
        #    self.print(f"[DEBUG] logits: {tuple(logits.shape)} {logits.dtype}")

        loss = self._loss(logits, mask)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        preds = torch.argmax(logits, dim=-1)
        self.train_miou.update(preds, mask)
        self.train_f1_macro.update(preds, mask)

        return loss

    def on_train_epoch_end(self) -> None:
        miou = self.train_miou.compute()
        f1 = self.train_f1_macro.compute()

        self.log("train/mIoU", miou, prog_bar=True, sync_dist=True)
        self.log("train/F1_macro", f1, prog_bar=True, sync_dist=True)

        self.train_miou.reset()
        self.train_f1_macro.reset()

    def on_after_backward(self) -> None:
        if self.global_step == 0:
            has_grad = False
            max_grad = 0.0
            for p in self.head.parameters():
                if p.grad is not None:
                    has_grad = True
                    max_grad = max(max_grad, float(p.grad.abs().max()))
    #        self.print(f"[DEBUG] head_has_grad={has_grad} head_max_abs_grad={max_grad}")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        emb = batch["embeddings"]
        mask = batch["masks"]
        logits = self(emb)

        preds = torch.argmax(logits, dim=-1)
        self.val_miou.update(preds, mask)
        self.val_f1_macro.update(preds, mask)

        loss = self._loss(logits, mask)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        miou = self.val_miou.compute()
        f1 = self.val_f1_macro.compute()

        self.log("val/mIoU", miou, prog_bar=True, sync_dist=True)
        self.log("val/F1_macro", f1, prog_bar=True, sync_dist=True)

        if self.trainer is not None and self.trainer.is_global_zero:
            self.print("[VAL] mIoU =", float(miou))
            self.print("[VAL] F1_macro =", float(f1))

        self.val_miou.reset()
        self.val_f1_macro.reset()


    def test_step(self, batch, batch_idx: int) -> None:
        emb = batch["embeddings"]
        mask = batch["masks"]
        logits = self(emb)
        preds = torch.argmax(logits, dim=-1)
        self.test_confmat.update(preds, mask)

    def on_test_epoch_end(self) -> None:
        confmat = self.test_confmat.compute()

        if self.trainer is not None and self.trainer.is_global_zero:
            out_dir = Path(self.trainer.log_dir) if self.trainer.log_dir else Path("logs")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "confmat_final.pt"
            torch.save(confmat.detach().cpu(), out_path)

            self.print("Confmat finale sauvegardée dans:", str(out_path))
            self.print("La voici :")
            self.print(confmat)

        self.test_confmat.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
