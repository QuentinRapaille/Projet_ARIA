"""
Utilisation : 

python plot_metrics.py                
  --metrics_csv logs/mlp_debug/version_3/metrics.csv 
  --confmat logs/mlp_debug/version_3/confmat_epoch=0.pt 
  --out_dir plots


python plot_metrics.py 
  --metrics_csv logs/lightning_logs/version_0/metrics.csv 
  --confmat logs/lightning_logs/version_0/confmat_epoch=4.pt 
  --out_dir plots 
  --normalize_confmat
"""



from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


def _read_lightning_metrics_csv(csv_path: Path) -> pd.DataFrame:
    """
    Lit le metrics.csv de Lightning et renvoie un DataFrame trié par step.
    """
    df = pd.read_csv(csv_path)
    # Lightning écrit souvent un champ 'step' ; si absent, on garde l'ordre.
    if "step" in df.columns:
        df = df.sort_values("step").reset_index(drop=True)
    return df


def _plot_curves(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Trace quelques métriques standards si elles existent.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    def plot_one(metric_name: str, title: str) -> None:
        if metric_name not in df.columns:
            return

        y = df[metric_name].to_numpy()
        x = df["step"].to_numpy() if "step" in df.columns else np.arange(len(df))

        plt.figure()
        plt.plot(x, y)
        plt.xlabel("step")
        plt.ylabel(metric_name)
        plt.title(title)
        plt.tight_layout()

        # Remplace les séparateurs de chemin et caractères gênants
        safe_name = metric_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        plt.savefig(out_dir / f"{safe_name}.png", dpi=200)
        plt.close()

    # Métriques que ton module log
    plot_one("train/loss_step", "Train loss (step)")
    plot_one("train/loss_epoch", "Train loss (epoch)")
    plot_one("val/loss", "Val loss")
    plot_one("val/mIoU", "Val mIoU (macro)")
    plot_one("val/F1_macro", "Val F1 macro")


def _load_confmat(confmat_path: Path) -> np.ndarray:
    """
    Charge une matrice de confusion sauvegardée en .pt ou .npy.
    """
    if confmat_path.suffix == ".pt":
        cm = torch.load(confmat_path, map_location="cpu")
        cm = cm.detach().cpu().numpy()
    elif confmat_path.suffix == ".npy":
        cm = np.load(confmat_path)
    else:
        raise ValueError("Format non supporté. Utiliser .pt ou .npy")
    return cm.astype(np.int64)


def _plot_confmat_table_and_heatmap(
    cm: np.ndarray,
    out_dir: Path,
    normalize: bool = False,
    class_names: list[str] | None = None,
) -> None:
    """
    Affiche la matrice de confusion :
      - en tableau (matplotlib.table)
      - en heatmap simple (imshow)
    Option normalize=True : normalisation par ligne (rappel par classe GT).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_to_show = cm / row_sums
        cm_title = "Confusion matrix (row-normalized)"
        fname = "confmat_normalized"
    else:
        cm_to_show = cm
        cm_title = "Confusion matrix"
        fname = "confmat_raw"

    k = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(k)]

    # Tableau
    plt.figure(figsize=(max(8, 0.45 * k), max(6, 0.45 * k)))
    plt.axis("off")
    tbl = plt.table(
        cellText=np.round(cm_to_show, 3) if normalize else cm_to_show,
        rowLabels=class_names,
        colLabels=class_names,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.2)
    plt.title(cm_title)
    plt.tight_layout()
    plt.savefig(out_dir / f"{fname}_table.png", dpi=200)
    plt.close()

    # Heatmap
    plt.figure(figsize=(8, 7))
    plt.imshow(cm_to_show, interpolation="nearest")
    plt.title(cm_title)
    plt.xlabel("Predicted")
    plt.ylabel("Ground truth")
    plt.xticks(np.arange(k), class_names, rotation=90)
    plt.yticks(np.arange(k), class_names)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir / f"{fname}_heatmap.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", type=str, required=True, help="Chemin vers metrics.csv (Lightning)")
    parser.add_argument("--confmat", type=str, default="", help="Chemin vers confmat_epoch=*.pt ou .npy")
    parser.add_argument("--out_dir", type=str, default="plots", help="Dossier de sortie")
    parser.add_argument("--normalize_confmat", action="store_true", help="Normaliser la confmat par ligne")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    df = _read_lightning_metrics_csv(Path(args.metrics_csv))
    _plot_curves(df, out_dir)

    if args.confmat:
        cm = _load_confmat(Path(args.confmat))
        _plot_confmat_table_and_heatmap(
            cm,
            out_dir,
            normalize=args.normalize_confmat,
            class_names=None,  # remplacer par une liste de noms si besoin
        )


if __name__ == "__main__":
    main()
