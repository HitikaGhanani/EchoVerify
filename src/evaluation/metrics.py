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
) -> Tuple[Optional[float], Optional[float]]:
    """Compute Equal Error Rate (EER) and the corresponding operating threshold.

    Synthetic speech (spoof) is assumed to be label 1.
    Genuine speech (bonafide) is label 0.

    Returns:
        eer: Equal Error Rate in [0, 1] or None if undefined
        eer_threshold: Score threshold corresponding to EER or None if undefined
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)

    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        logger.warning(f"Only one class present in y_true ({unique_labels}); EER is undefined.")
        return None, None

    # Handle constant scores
    if len(y_scores) > 0 and np.all(y_scores == y_scores[0]):
        return 0.5, float(y_scores[0])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr

    # Replace any infinite threshold created by roc_curve with maximum finite score
    finite_thresholds = np.where(np.isinf(thresholds), np.max(y_scores), thresholds)

    # Find the threshold where FPR and FNR are closest
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(finite_thresholds[idx])

    if np.isnan(eer_threshold) or np.isinf(eer_threshold):
        eer_threshold = float(np.mean(y_scores))

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
        if eer_thresh is not None and not np.isnan(eer_thresh) and not np.isinf(eer_thresh):
            return float(eer_thresh)
        if len(y_scores) > 0:
            return float(y_scores[0])
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

    return float(best_thresh)


def compute_all_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute all evaluation metrics with synthetic speech as positive class (1).

    Maintains operating thresholds and unrounded metrics at full precision.
    Returns None (null in JSON) for undefined metrics when classes are absent or scores constant.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)

    total_samples = int(len(y_true))
    bonafide_count = int(np.sum(y_true == 0))
    spoof_count = int(np.sum(y_true == 1))

    # Calculate EER and natural EER threshold
    eer, eer_threshold = compute_eer(y_true, y_scores)

    # Use specified operating threshold or EER threshold or fallback
    if threshold is not None:
        op_thresh = float(threshold)
    elif eer_threshold is not None and not np.isnan(eer_threshold) and not np.isinf(eer_threshold):
        op_thresh = float(eer_threshold)
    elif len(y_scores) > 0 and np.isfinite(y_scores[0]):
        op_thresh = float(y_scores[0])
    else:
        op_thresh = 0.5

    # Thresholded predictions
    y_pred = (y_scores >= op_thresh).astype(int)

    # Classification metrics
    acc = float(accuracy_score(y_true, y_pred))

    # Precision, Recall, F1 are defined only when positive class (spoof) is present
    if spoof_count > 0:
        prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    else:
        prec = None
        rec = None
        f1 = None

    # ROC-AUC requires at least two classes and non-constant scores
    unique_labels = np.unique(y_true)
    if len(unique_labels) >= 2 and not np.all(y_scores == y_scores[0]):
        try:
            auc = float(roc_auc_score(y_true, y_scores))
        except Exception as e:
            logger.warning(f"Failed to compute ROC-AUC: {e}")
            auc = None
    else:
        auc = None

    # Confusion matrix: tn, fp, fn, tp
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Genuine-speech false-positive rate: FP / (FP + TN)
    # Undefined (None) if genuine speech (class 0) is absent
    if bonafide_count > 0:
        genuine_fpr = float(fp / (fp + tn))
    else:
        genuine_fpr = None

    # Spoof false-negative rate: FN / (FN + TP)
    # Undefined (None) if spoof speech (class 1) is absent
    if spoof_count > 0:
        spoof_fnr = float(fn / (fn + tp))
    else:
        spoof_fnr = None

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "operating_threshold": float(op_thresh),
        "genuine_speech_fpr": genuine_fpr,
        "spoof_fnr": spoof_fnr,
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

