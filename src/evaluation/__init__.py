"""Evaluation metrics and visualization for EchoVerify."""

from src.evaluation.metrics import (
    compute_all_metrics,
    compute_eer,
    find_operating_threshold,
)
from src.evaluation.visualizer import (
    plot_confusion_matrix,
    plot_training_curves,
)

__all__ = [
    "compute_all_metrics",
    "compute_eer",
    "find_operating_threshold",
    "plot_confusion_matrix",
    "plot_training_curves",
]

