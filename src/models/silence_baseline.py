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
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract leading and trailing silence features for all clips in DataFrame."""
    feature_rows = []
    labels = []

    for _, row in df.iterrows():
        audio_path = Path(row["audio_path"])
        label = int(row["label"])

        try:
            _, meta = preprocessor.process_waveform(audio_path)
            features = [
                meta["lead_silence_sec"],
                meta["trail_silence_sec"],
                meta["edge_silence_sec"],
                meta["active_sec"],
                meta["total_sec"],
            ]
            feature_rows.append(features)
            labels.append(label)
        except Exception as e:
            logger.warning(f"Error processing {audio_path}: {e}")

    X = np.array(feature_rows, dtype=np.float32)
    y = np.array(labels, dtype=int)
    return X, y


def run_silence_baseline(
    data_root: Optional[str] = None,
    max_train_samples: int = 2000,
    max_dev_samples: int = 1000,
    output_dir: str = "results",
    seed: int = 42,
) -> Dict[str, Any]:
    """Train and evaluate silence shortcut baseline."""
    protocol = ASVSpoofProtocol(data_root=data_root)
    train_df = protocol.load_split("train")
    dev_df = protocol.load_split("dev")

    manifest_train = create_stratified_manifest(train_df, max_samples=max_train_samples, seed=seed)
    manifest_dev = create_stratified_manifest(dev_df, max_samples=max_dev_samples, seed=seed)

    prep_config = PreprocessConfig(mode="raw", trim_top_db=30.0)
    preprocessor = AudioPreprocessor(prep_config)

    logger.info("Extracting silence features for train cohort...")
    X_train, y_train = extract_silence_features(manifest_train, preprocessor)
    logger.info("Extracting silence features for dev cohort...")
    X_dev, y_dev = extract_silence_features(manifest_dev, preprocessor)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)

    clf = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=500)
    clf.fit(X_train_scaled, y_train)

    dev_scores = clf.predict_proba(X_dev_scaled)[:, 1]
    metrics = compute_all_metrics(y_dev, dev_scores)

    out_path = Path(output_dir) / "silence_baseline_dev_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Silence Baseline Dev Results: EER={metrics.get('eer')}, ROC-AUC={metrics.get('roc_auc')}")
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
