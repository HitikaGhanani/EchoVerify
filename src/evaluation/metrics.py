"""Comprehensive evaluation metrics for synthetic audio detection."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger("EchoVerify.Evaluation")


def compute_eer(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER) and the corresponding operating threshold.

    Synthetic speech (spoof) is assumed to be label 1.
    Genuine speech (bonafide) is label 0.

    Returns:
        eer: Equal Error Rate in [0, 1]
        eer_threshold: Score threshold corresponding to EER
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)

    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        logger.warning(f"Only one class present in y_true ({unique_labels}); EER is undefined.")
        return float("nan"), 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr

    # Find the threshold where FPR and FNR are closest
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx])

    return eer, eer_threshold


def find_operating_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    criterion: str = "eer",
) -> float:
    """Find the optimal decision threshold on development data only.

    Supported criteria:
    - "eer": Threshold minimizing |FPR - FNR|
    - "f1": Threshold maximizing F1 score for spoof detection
    - "balanced": Threshold closest to (FPR=0.01 or minimal genuine FPR)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)

    if criterion == "eer":
        _, eer_thresh = compute_eer(y_true, y_scores)
        if not np.isnan(eer_thresh):
            return eer_thresh
        return 0.5

    # Grid search for best threshold
    threshold_grid = np.linspace(0.01, 0.99, 99)
    best_thresh = 0.5
    best_score = -1.0

    for thresh in threshold_grid:
        y_pred = (y_scores >= thresh).astype(int)
        if criterion == "f1":
            score = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        else:
            score = accuracy_score(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_thresh = float(thresh)

    return best_thresh


def compute_all_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute all evaluation metrics with synthetic speech as positive class (1).

    Metrics:
    - accuracy
    - precision
    - recall
    - f1
    - roc_auc
    - eer
    - eer_threshold
    - operating_threshold
    - genuine_fpr (False Positive Rate on bona fide clips: FP / (FP + TN))
    - spoof_fnr (False Negative Rate on spoof clips: FN / (FN + TP))
    - confusion_matrix: dict with tn, fp, fn, tp counts
    - sample_counts: total, bonafide, spoof

    Handles undefined metrics safely without raising ZeroDivisionError.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)

    total_samples = int(len(y_true))
    bonafide_count = int(np.sum(y_true == 0))
    spoof_count = int(np.sum(y_true == 1))

    # Calculate EER and natural EER threshold
    eer, eer_threshold = compute_eer(y_true, y_scores)

    # Use specified operating threshold or EER threshold or 0.5
    if threshold is not None:
        op_thresh = float(threshold)
    elif not np.isnan(eer_threshold):
        op_thresh = eer_threshold
    else:
        op_thresh = 0.5

    # Thresholded predictions
    y_pred = (y_scores >= op_thresh).astype(int)

    # Standard classification metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    # ROC-AUC
    unique_labels = np.unique(y_true)
    if len(unique_labels) >= 2:
        try:
            auc = float(roc_auc_score(y_true, y_scores))
        except Exception as e:
            logger.warning(f"Failed to compute ROC-AUC: {e}")
            auc = float("nan")
    else:
        auc = float("nan")

    # Confusion matrix: tn, fp, fn, tp
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Genuine-speech false-positive rate: FP / (FP + TN)
    genuine_fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Spoof false-negative rate: FN / (FN + TP)
    spoof_fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
        "eer": round(eer, 4) if not np.isnan(eer) else None,
        "eer_threshold": round(eer_threshold, 4) if not np.isnan(eer_threshold) else None,
        "operating_threshold": round(op_thresh, 4),
        "genuine_speech_fpr": round(genuine_fpr, 4),
        "spoof_fnr": round(spoof_fnr, 4),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "sample_counts": {
            "total": total_samples,
            "bonafide": bonafide_count,
            "spoof": spoof_count,
        },
    }

