"""Visualizations for EchoVerify model evaluation and training diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    cm_dict: Dict[str, int],
    title: str = "Confusion Matrix",
    out_path: Optional[Path | str] = None,
    normalize: bool = True,
) -> plt.Figure:
    """Plot and save a 2x2 confusion matrix with genuine/synthetic speech labels.

    Args:
        cm_dict: Dict with keys 'tn', 'fp', 'fn', 'tp'
        title: Plot title
        out_path: Optional path to save figure (e.g. under results/)
        normalize: If True, display row-normalized percentages alongside raw counts
    """
    tn = cm_dict["tn"]
    fp = cm_dict["fp"]
    fn = cm_dict["fn"]
    tp = cm_dict["tp"]

    matrix = np.array([[tn, fp], [fn, tp]])
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.divide(
        matrix.astype(float),
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums > 0,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(norm_matrix if normalize else matrix, cmap="Blues", vmin=0, vmax=1.0)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    classes = ["Genuine (0)", "Synthetic (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, pad=12, fontweight="bold")

    # Annotate cells
    for i in range(2):
        for j in range(2):
            count = matrix[i, j]
            pct = norm_matrix[i, j] * 100.0
            cell_text = f"{count:,}\n({pct:.1f}%)" if normalize else f"{count:,}"
            text_color = "white" if norm_matrix[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                cell_text,
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
                fontweight="bold",
            )

    plt.tight_layout()

    if out_path:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=150)
        plt.close(fig)

    return fig


def plot_training_curves(
    history: Dict[str, List[Any]],
    out_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Plot training/dev loss and dev metric curves across epochs."""
    epochs = history.get("epoch", list(range(1, len(history.get("train_loss", [])) + 1)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss plot
    ax_loss = axes[0]
    if "train_loss" in history:
        ax_loss.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#2B6CB0", linewidth=2)
    if "dev_loss" in history:
        ax_loss.plot(epochs, history["dev_loss"], "s--", label="Dev Loss", color="#DD6B20", linewidth=2)
    ax_loss.set_title("BCE Loss Across Epochs", fontsize=11, fontweight="bold")
    ax_loss.set_xlabel("Epoch", fontsize=10)
    ax_loss.set_ylabel("Loss", fontsize=10)
    ax_loss.grid(True, linestyle="--", alpha=0.5)
    ax_loss.legend(fontsize=9)

    # Metrics plot (ROC-AUC and EER)
    ax_m = axes[1]
    has_auc = "dev_roc_auc" in history and any(v is not None for v in history["dev_roc_auc"])
    has_eer = "dev_eer" in history and any(v is not None for v in history["dev_eer"])

    if has_auc:
        auc_vals = [v if v is not None else float("nan") for v in history["dev_roc_auc"]]
        ax_m.plot(epochs, auc_vals, "o-", label="Dev ROC-AUC", color="#38A169", linewidth=2)
    if has_eer:
        eer_vals = [v if v is not None else float("nan") for v in history["dev_eer"]]
        ax_m.plot(epochs, eer_vals, "^-", label="Dev EER", color="#E53E3E", linewidth=2)

    ax_m.set_title("Development Performance Across Epochs", fontsize=11, fontweight="bold")
    ax_m.set_xlabel("Epoch", fontsize=10)
    ax_m.set_ylabel("Score / Rate", fontsize=10)
    ax_m.set_ylim(-0.02, 1.02)
    ax_m.grid(True, linestyle="--", alpha=0.5)
    ax_m.legend(fontsize=9)

    plt.tight_layout()

    if out_path:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=150)
        plt.close(fig)

    return fig
