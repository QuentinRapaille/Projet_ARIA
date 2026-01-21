from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import yaml


@dataclass(frozen=True)
class Config:
    seed: int
    out_dir: str
    experiment_name: str

    pastis_root: str          
    name: str                 

    num_classes: int
    ignore_index: int
    hidden_dim: int

    batch_size: int
    num_workers: int
    pin_memory: bool

    max_epochs: int
    lr: float

    accelerator: str
    devices: Union[str, int]  # "auto" ou int
    log_every_n_steps: int


_REQUIRED_KEYS = {
    "seed",
    "out_dir",
    "experiment_name",
    "pastis_root",
    "name",
    "num_classes",
    "ignore_index",
    "hidden_dim",
    "batch_size",
    "num_workers",
    "pin_memory",
    "max_epochs",
    "lr",
    "accelerator",
    "devices",
    "log_every_n_steps",
}


def load_config(yaml_path: str | Path) -> Config:
    """
    Charge un YAML et vérifie qu'il contient exactement les clés attendues.
    """
    yaml_path = Path(yaml_path)
    cfg: Dict[str, Any] = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a mapping (dict).")

    missing = _REQUIRED_KEYS - set(cfg.keys())
    if missing:
        raise KeyError(f"Missing keys in YAML: {sorted(missing)}")

    extra = set(cfg.keys()) - _REQUIRED_KEYS
    if extra:
        raise KeyError(f"Unexpected keys in YAML (not allowed): {sorted(extra)}")

    return Config(
        seed=int(cfg["seed"]),
        out_dir=str(cfg["out_dir"]),
        experiment_name=str(cfg["experiment_name"]),
        pastis_root=str(cfg["pastis_root"]),
        name=str(cfg["name"]),
        num_classes=int(cfg["num_classes"]),
        ignore_index=int(cfg["ignore_index"]),
        hidden_dim=int(cfg["hidden_dim"]),
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]),
        max_epochs=int(cfg["max_epochs"]),
        lr=float(cfg["lr"]),
        accelerator=str(cfg["accelerator"]),
        devices=cfg["devices"],  # laisser Lightning gérer "auto" vs int
        log_every_n_steps=int(cfg["log_every_n_steps"]),
    )
