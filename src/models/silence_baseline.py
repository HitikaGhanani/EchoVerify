"""Silence duration shortcut baseline for ASVspoof 2019 LA.

Evaluates how well leading and trailing silence features alone predict
bona fide vs. spoof labels.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data.dataset import create_stratified_manifest
from src.data.preprocessing import AudioPreprocessor, PreprocessConfig
from src.data.protocol import ASVSpoofProtocol
from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger("EchoVerify.SilenceBaseline")


def extract_silence_features(
    df: pd.DataFrame,
    preprocessor: AudioPreprocessor,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[Tuple[str, str, str]]]:
    """Extract leading and trailing silence features for all clips in DataFrame.

    Uses strictly leading and trailing silence durations (2 features).
    Preserves 1-to-1 alignment between utt_ids, attack_ids, labels, and feature vectors.
    Explicitly reports any unreadable audio files.

    Returns:
        X: Feature matrix of shape (N, 2) [lead_silence_sec, trail_silence_sec]
        y: Binary labels array of shape (N,)
        utt_ids: Utterance IDs aligned with X and y
        attack_ids: Attack IDs aligned with X and y
        unreadable_files: List of (utt_id, path, error_msg)
    """
    feature_rows: List[List[float]] = []
    labels: List[int] = []
    utt_ids: List[str] = []
    attack_ids: List[str] = []
    unreadable_files: List[Tuple[str, str, str]] = []

    for _, row in df.iterrows():
        utt_id = str(row["utt_id"])
        attack_id = str(row.get("attack_id", "-"))
        audio_path = Path(row["audio_path"])
        label = int(row["label"])

        try:
            _, meta = preprocessor.process_waveform(audio_path)
            # Use ONLY leading and trailing silence durations
            features = [
                float(meta["lead_silence_sec"]),
                float(meta["trail_silence_sec"]),
            ]
            feature_rows.append(features)
            labels.append(label)
            utt_ids.append(utt_id)
            attack_ids.append(attack_id)
        except Exception as e:
            logger.warning(f"Unreadable audio file {utt_id} ({audio_path}): {e}")
            unreadable_files.append((utt_id, str(audio_path), str(e)))

    if unreadable_files:
        logger.warning(
            f"Encountered {len(unreadable_files)} unreadable file(s) out of {len(df)} clips."
        )

    X = np.array(feature_rows, dtype=np.float32) if feature_rows else np.empty((0, 2), dtype=np.float32)
    y = np.array(labels, dtype=int) if labels else np.empty((0,), dtype=int)
    return X, y, utt_ids, attack_ids, unreadable_files


def run_silence_baseline(
    data_root: Optional[str] = None,
    max_train_samples: int = 2000,
    max_dev_samples: int = 1000,
    output_dir: str = "results",
    seed: int = 42,
) -> Dict[str, Any]:
    """Train and evaluate silence shortcut baseline."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    protocol = ASVSpoofProtocol(data_root=data_root)
    train_df = protocol.load_split("train")
    dev_df = protocol.load_split("dev")

    manifest_train = create_stratified_manifest(train_df, max_samples=max_train_samples, seed=seed)
    manifest_dev = create_stratified_manifest(dev_df, max_samples=max_dev_samples, seed=seed)

    # Save sampled manifests
    manifest_train.to_csv(out_path / "silence_baseline_manifest_train.csv", index=False)
    manifest_dev.to_csv(out_path / "silence_baseline_manifest_dev.csv", index=False)

    prep_config = PreprocessConfig(mode="raw", trim_top_db=30.0)
    preprocessor = AudioPreprocessor(prep_config)

    logger.info("Extracting silence features for train cohort...")
    X_train, y_train, train_utts, train_attacks, unreadable_train = extract_silence_features(
        manifest_train, preprocessor
    )
    logger.info(f"Train cohort features extracted: {len(X_train)} clips ({len(unreadable_train)} unreadable)")

    logger.info("Extracting silence features for dev cohort...")
    X_dev, y_dev, dev_utts, dev_attacks, unreadable_dev = extract_silence_features(
        manifest_dev, preprocessor
    )
    logger.info(f"Dev cohort features extracted: {len(X_dev)} clips ({len(unreadable_dev)} unreadable)")

    if len(X_train) == 0 or len(X_dev) == 0:
        raise RuntimeError("No usable audio files found to train silence baseline.")

    # Fit scaler on training data ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)

    clf = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=500)
    clf.fit(X_train_scaled, y_train)

    dev_scores = clf.predict_proba(X_dev_scaled)[:, 1]
    metrics = compute_all_metrics(y_dev, dev_scores)
    operating_threshold = float(metrics["operating_threshold"])

    # Build aligned per-clip scores DataFrame
    y_pred = (dev_scores >= operating_threshold).astype(int)
    scores_df = pd.DataFrame({
        "utt_id": dev_utts,
        "attack_id": dev_attacks,
        "true_label": y_dev.tolist(),
        "score": [float(s) for s in dev_scores],
        "pred_label": y_pred.tolist(),
    })
    scores_df.to_csv(out_path / "silence_baseline_dev_scores.csv", index=False)

    # Save configuration and metrics
    config_dict = {
        "preprocessing_config": prep_config.to_dict(),
        "features": ["lead_silence_sec", "trail_silence_sec"],
        "operating_threshold": operating_threshold,
        "unreadable_train_count": len(unreadable_train),
        "unreadable_dev_count": len(unreadable_dev),
    }
    with open(out_path / "silence_baseline_config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    with open(out_path / "silence_baseline_dev_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save fitted model artifacts
    model_artifact = {
        "classifier": clf,
        "scaler": scaler,
        "features": ["lead_silence_sec", "trail_silence_sec"],
        "preprocessing_config": prep_config.to_dict(),
        "operating_threshold": operating_threshold,
        "dev_metrics": metrics,
    }
    torch.save(model_artifact, out_path / "silence_baseline_model.pt")

    logger.info(f"Silence Baseline Dev Results: EER={metrics.get('eer')}, "
                f"ROC-AUC={metrics.get('roc_auc')}, Threshold={operating_threshold:.6f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Silence shortcut baseline for EchoVerify.")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--max-dev", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    run_silence_baseline(
        data_root=args.data_root,
        max_train_samples=args.max_train,
        max_dev_samples=args.max_dev,
        output_dir=args.output_dir,
    )
