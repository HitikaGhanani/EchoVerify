"""Evaluation script for EchoVerify on ASVspoof 2019 LA partitions."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import ASVSpoofDataset
from src.data.preprocessing import AudioPreprocessor, PreprocessConfig
from src.data.protocol import ASVSpoofProtocol
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.visualizer import plot_confusion_matrix
from src.models.cnn import EchoVerifyCNN
from src.train import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EchoVerify.Evaluate")


def run_evaluation(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """Evaluate a trained EchoVerify checkpoint on dev or held-out eval partition."""
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Loading checkpoint: {checkpoint_path} on {device}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Restore preprocessing configuration
    prep_dict = checkpoint.get("preprocessing_config", {})
    prep_config = PreprocessConfig.from_dict(prep_dict)
    preprocessor = AudioPreprocessor(prep_config)
    logger.info(f"Restored preprocessing settings: mode='{prep_config.mode}', "
                f"sample_rate={prep_config.sample_rate}, trim_top_db={prep_config.trim_top_db}")

    # Restore fixed operating threshold chosen on development data
    operating_threshold = float(checkpoint.get("operating_threshold", 0.5))
    logger.info(f"Applying fixed development operating threshold: {operating_threshold:.4f}")

    # Reconstruct model
    model = EchoVerifyCNN(
        sample_rate=prep_config.sample_rate,
        n_mels=prep_config.n_mels,
        n_fft=prep_config.n_fft,
        hop_length=prep_config.hop_length,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load dataset protocol
    protocol = ASVSpoofProtocol(data_root=args.data_root)
    split = args.split.lower()

    if not protocol.is_split_available(split):
        if split == "eval":
            logger.warning(
                "Official ASVspoof 2019 LA Evaluation partition is not available on disk.\n"
                "Evaluation partition missing does not affect training or development benchmarks."
            )
            return None
        else:
            raise FileNotFoundError(f"Required split '{split}' protocol or audio not found in dataset root.")

    logger.info(f"Loading '{split}' partition protocol...")
    df = protocol.load_split(split, validate_audio=True)
    if args.max_samples and args.max_samples > 0 and args.max_samples < len(df):
        logger.info(f"Subsampling {args.max_samples} clips from {split} partition...")
        df = df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)

    logger.info(f"Evaluating on {len(df)} clips from '{split}' partition...")

    dataset = ASVSpoofDataset(df, preprocessor=preprocessor)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    criterion = nn.BCEWithLogitsLoss()
    mean_loss, metrics, scores_df = evaluate_model(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        fixed_threshold=operating_threshold,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON clearly labeled
    metrics_filename = f"{split}_metrics.json"
    scores_filename = f"{split}_scores.csv"
    cm_filename = f"confusion_matrix_{split}.png"

    metrics_record = {
        "split": split,
        "is_held_out": (split == "eval"),
        "applied_fixed_threshold": operating_threshold,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "metrics": metrics,
    }

    with open(output_dir / metrics_filename, "w", encoding="utf-8") as f:
        json.dump(metrics_record, f, indent=2)

    scores_df.to_csv(output_dir / scores_filename, index=False)

    if "confusion_matrix" in metrics:
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            title=f"EchoVerify {split.upper()} Confusion Matrix (Threshold={operating_threshold:.4f})",
            out_path=output_dir / cm_filename,
        )

    logger.info(f"--- {split.upper()} Benchmark Results ---")
    logger.info(f"Total clips: {metrics['sample_counts']['total']} "
                f"({metrics['sample_counts']['bonafide']} bona fide, {metrics['sample_counts']['spoof']} spoof)")
    logger.info(f"Fixed Operating Threshold: {operating_threshold:.4f}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"ROC-AUC: {metrics.get('roc_auc')}")
    logger.info(f"EER: {metrics.get('eer')}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1: {metrics['f1']:.4f}")
    logger.info(f"Genuine Speech False Positive Rate: {metrics['genuine_speech_fpr']:.4f}")
    logger.info(f"Spoof False Negative Rate: {metrics['spoof_fnr']:.4f}")
    logger.info(f"Saved results to: {output_dir / metrics_filename}")

    return metrics_record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate EchoVerify on ASVspoof 2019 LA partition.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt",
                        help="Path to trained PyTorch checkpoint")
    parser.add_argument("--split", type=str, choices=["dev", "eval"], default="eval",
                        help="Partition to evaluate on ('dev' or 'eval')")
    parser.add_argument("--data-root", type=str, default=None, help="Root folder of ASVspoof 2019 LA")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional sample limit for quick test")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save evaluation results")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_evaluation(args)
