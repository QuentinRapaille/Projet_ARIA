from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import CSVLogger

from my_code.config import load_config
from my_code.models import build_fm
from my_code.data.pastis_r_embedding_datamodule import EmbeddingDataModule
from my_code.models.mlp_head import MLPHeadConfig
from my_code.lightning_module import SegmentationMLPModule


def _infer_in_dim(fm, pastis_root: Path) -> int:
    """
    Infère D en chargeant un seul pid depuis metadata.geojson.
    """
    import json
    meta = json.loads((pastis_root / "metadata.geojson").read_text(encoding="utf-8"))
    feats = meta.get("features", [])
    if not feats:
        raise RuntimeError("metadata.geojson has no features.")
    pid0 = int(feats[0]["properties"]["ID_PATCH"])
    out = fm.load(pid0)
    return int(out.embedding_dim)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    cfg = load_config(args.config)

    L.seed_everything(cfg.seed, workers=True)

    pastis_root = Path(cfg.pastis_root)
    fm = build_fm(cfg.name, pastis_root=pastis_root)

    in_dim = _infer_in_dim(fm, pastis_root)

    datamodule = EmbeddingDataModule(
        pastis_root=str(pastis_root),
        fm=fm,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    head_cfg = MLPHeadConfig(in_dim=in_dim, hidden_dim=cfg.hidden_dim, num_classes=cfg.num_classes)
    model = SegmentationMLPModule(head_cfg=head_cfg, lr=cfg.lr, ignore_index=cfg.ignore_index)

    logger = CSVLogger(save_dir=cfg.out_dir, name=cfg.experiment_name)

    trainer = L.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        max_epochs=cfg.max_epochs,
        logger=logger,
        log_every_n_steps=cfg.log_every_n_steps,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
