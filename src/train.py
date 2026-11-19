"""Training script for EchoVerify compact CNN on ASVspoof 2019 LA."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import ASVSpoofDataset, create_stratified_manifest, get_manifest_statistics
from src.data.preprocessing import AudioPreprocessor, PreprocessConfig
from src.data.protocol import ASVSpoofProtocol
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.visualizer import plot_confusion_matrix, plot_training_curves
from src.models.cnn import EchoVerifyCNN

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EchoVerify.Train")


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for a single epoch and return mean loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for waveforms, labels, _, _, _ in tqdm(loader, desc="Training", leave=False):
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(waveforms)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    return float(total_loss / max(1, num_batches))


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    fixed_threshold: Optional[float] = None,
) -> Tuple[float, Dict[str, Any], pd.DataFrame]:
    """Evaluate model on a dataset and return (mean_loss, metrics_dict, scores_df)."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_utt_ids: List[str] = []
    all_attack_ids: List[str] = []
    all_true_labels: List[int] = []
    all_logits: List[float] = []
    all_scores: List[float] = []

    for waveforms, labels, utt_ids, attack_ids, _ in tqdm(loader, desc="Evaluating", leave=False):
        waveforms = waveforms.to(device, non_blocking=True)
        labels_t = labels.to(device, non_blocking=True).unsqueeze(1)

        logits = model(waveforms)
        loss = criterion(logits, labels_t)
        total_loss += float(loss.item())
        num_batches += 1

        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        raw_logits = logits.squeeze(1).cpu().numpy()

        all_utt_ids.extend(utt_ids)
        all_attack_ids.extend(attack_ids)
        all_true_labels.extend(labels.cpu().numpy().astype(int).tolist())
        all_logits.extend(raw_logits.tolist())
        all_scores.extend(probs.tolist())

    mean_loss = float(total_loss / max(1, num_batches))

    y_true = np.array(all_true_labels, dtype=int)
    y_scores = np.array(all_scores, dtype=float)

    # Compute all benchmark metrics
    metrics = compute_all_metrics(y_true, y_scores, threshold=fixed_threshold)
    chosen_threshold = metrics["operating_threshold"]

    # Build per-clip scores DataFrame with full precision
    y_pred = (y_scores >= chosen_threshold).astype(int)
    scores_df = pd.DataFrame({
        "utt_id": all_utt_ids,
        "attack_id": all_attack_ids,
        "true_label": all_true_labels,
        "logit": [float(v) for v in all_logits],
        "score": [float(v) for v in all_scores],
        "pred_label": y_pred.tolist(),
    })

    return mean_loss, metrics, scores_df


def run_training(args: argparse.Namespace) -> None:
    """Run full training workflow according to arguments."""
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Load protocol partitions
    logger.info("Resolving ASVspoof 2019 LA protocol partitions...")
    protocol = ASVSpoofProtocol(data_root=args.data_root)

    train_df = protocol.load_split("train", validate_audio=True)
    dev_df = protocol.load_split("dev", validate_audio=True)
    logger.info(f"Found official protocols: Train={len(train_df)} clips, Dev={len(dev_df)} clips.")

    # Smoke run overrides
    if args.smoke_run:
        logger.info("--- SMOKE RUN ENABLED: Truncating to 24 train / 16 dev clips for rapid testing ---")
        max_train = 24
        max_dev = 16
        epochs = 1
        batch_size = 4
    else:
        max_train = args.max_train_samples
        max_dev = args.max_dev_samples
        epochs = args.epochs
        batch_size = args.batch_size

    # Create deterministic stratified manifests
    logger.info(f"Creating deterministic stratified manifests (seed={args.seed})...")
    manifest_train = create_stratified_manifest(train_df, max_samples=max_train, seed=args.seed)
    manifest_dev = create_stratified_manifest(dev_df, max_samples=max_dev, seed=args.seed)

    train_stats = get_manifest_statistics(manifest_train)
    dev_stats = get_manifest_statistics(manifest_dev)
    logger.info(f"Sampled Train Cohort: {train_stats['total_clips']} clips "
                f"({train_stats['bonafide_count']} bonafide, {train_stats['spoof_count']} spoof, "
                f"ratio {train_stats['spoof_ratio']})")
    logger.info(f"Sampled Dev Cohort: {dev_stats['total_clips']} clips "
                f"({dev_stats['bonafide_count']} bonafide, {dev_stats['spoof_count']} spoof, "
                f"ratio {dev_stats['spoof_ratio']})")

    # Save sampled manifests
    train_manifest_path = output_dir / "manifest_train.csv"
    dev_manifest_path = output_dir / "manifest_dev.csv"
    manifest_train.to_csv(train_manifest_path, index=False)
    manifest_dev.to_csv(dev_manifest_path, index=False)
    logger.info(f"Saved manifests to {train_manifest_path} and {dev_manifest_path}")

    # Build preprocessing configuration
    prep_config = PreprocessConfig(
        sample_rate=16000,
        target_duration_sec=4.0,
        mode=args.mode,
        trim_top_db=args.trim_top_db,
        crop_mode="first",
        pad_mode="zero",
        n_mels=64,
        n_fft=512,
        hop_length=160,
    )
    preprocessor = AudioPreprocessor(prep_config)

    # Build Datasets and DataLoaders
    train_dataset = ASVSpoofDataset(manifest_train, preprocessor=preprocessor)
    dev_dataset = ASVSpoofDataset(manifest_dev, preprocessor=preprocessor)

    num_workers = args.num_workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # Class-weighted loss: pos_weight = num_bonafide / num_spoof
    # Spoof (1) is majority in ASVspoof LA
    num_bonafide = train_stats["bonafide_count"]
    num_spoof = train_stats["spoof_count"]
    pos_weight_val = float(num_bonafide / max(1, num_spoof))
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info(f"BCEWithLogitsLoss pos_weight for synthetic speech (label=1): {pos_weight_val:.4f}")

    # Instantiate model
    model = EchoVerifyCNN(
        sample_rate=prep_config.sample_rate,
        n_mels=prep_config.n_mels,
        n_fft=prep_config.n_fft,
        hop_length=prep_config.hop_length,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Training loop
    history: Dict[str, List[Any]] = {
        "epoch": [],
        "train_loss": [],
        "dev_loss": [],
        "dev_accuracy": [],
        "dev_f1": [],
        "dev_roc_auc": [],
        "dev_eer": [],
        "dev_operating_threshold": [],
    }

    best_dev_metric = float("inf")  # Lower EER is better
    best_checkpoint_path = checkpoint_dir / "best_model.pt"
    best_dev_metrics: Dict[str, Any] = {}
    best_scores_df: Optional[pd.DataFrame] = None

    logger.info(f"Beginning training for {epochs} epoch(s)...")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        dev_loss, dev_metrics, dev_scores = evaluate_model(model, dev_loader, criterion, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 5))
        history["dev_loss"].append(round(dev_loss, 5))
        history["dev_accuracy"].append(dev_metrics.get("accuracy"))
        history["dev_f1"].append(dev_metrics.get("f1"))
        history["dev_roc_auc"].append(dev_metrics.get("roc_auc"))
        history["dev_eer"].append(dev_metrics.get("eer"))
        history["dev_operating_threshold"].append(dev_metrics.get("operating_threshold"))

        eer_str = f"{dev_metrics['eer']:.4f}" if dev_metrics.get("eer") is not None else "N/A"
        auc_str = f"{dev_metrics['roc_auc']:.4f}" if dev_metrics.get("roc_auc") is not None else "N/A"
        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
            f"Dev Loss: {dev_loss:.4f} | Dev EER: {eer_str} | Dev ROC-AUC: {auc_str} | "
            f"Dev Acc: {dev_metrics['accuracy']:.4f} | Dev FPR: {dev_metrics['genuine_speech_fpr']:.4f}"
        )

        # Checkpoint selection: choose lowest EER (or lowest dev loss if EER is None)
        eer_val = dev_metrics.get("eer")
        current_metric = eer_val if eer_val is not None else dev_loss

        if current_metric < best_dev_metric:
            best_dev_metric = current_metric
            best_dev_metrics = dev_metrics
            best_scores_df = dev_scores

            # Save best checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "operating_threshold": dev_metrics["operating_threshold"],
                    "eer_threshold": dev_metrics.get("eer_threshold"),
                    "dev_metrics": dev_metrics,
                    "preprocessing_config": prep_config.to_dict(),
                    "seed": args.seed,
                    "training_args": vars(args),
                    "pos_weight": pos_weight_val,
                },
                best_checkpoint_path,
            )
            logger.info(f"Saved new best checkpoint (epoch {epoch}) to {best_checkpoint_path}")

    # Save training history
    history_path = output_dir / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Save development scores and metrics
    if best_scores_df is not None:
        dev_scores_path = output_dir / "dev_scores.csv"
        best_scores_df.to_csv(dev_scores_path, index=False)

    dev_metrics_path = output_dir / "dev_metrics.json"
    with open(dev_metrics_path, "w", encoding="utf-8") as f:
        json.dump(best_dev_metrics, f, indent=2)

    # Save run configuration
    config_record = {
        "args": vars(args),
        "preprocessing": prep_config.to_dict(),
        "train_statistics": train_stats,
        "dev_statistics": dev_stats,
        "best_epoch_metrics": best_dev_metrics,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)

    # Generate and save diagnostic figures
    curves_path = output_dir / "training_curves.png"
    plot_training_curves(history, out_path=curves_path)

    if "confusion_matrix" in best_dev_metrics:
        cm_path = output_dir / "confusion_matrix_dev.png"
        plot_confusion_matrix(
            best_dev_metrics["confusion_matrix"],
            title=f"EchoVerify Dev Confusion Matrix ({prep_config.mode.capitalize()})",
            out_path=cm_path,
        )

    logger.info("Training and evaluation complete!")
    logger.info(f"Best Dev Metrics: EER={best_dev_metrics.get('eer')}, "
                f"ROC-AUC={best_dev_metrics.get('roc_auc')}, "
                f"Operating Threshold={best_dev_metrics.get('operating_threshold')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train EchoVerify compact CNN on ASVspoof 2019 LA.")
    parser.add_argument("--data-root", type=str, default=None, help="Root folder of ASVspoof 2019 LA")
    parser.add_argument("--mode", type=str, choices=["raw", "trimmed"], default="raw",
                        help="Audio preprocessing mode: 'raw' or 'trimmed' (leading/trailing silence removed)")
    parser.add_argument("--trim-top-db", type=float, default=30.0, help="Silence trimming threshold in dB")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--max-train-samples", type=int, default=2000,
                        help="Max clips for stratified train manifest (use 0 for all)")
    parser.add_argument("--max-dev-samples", type=int, default=1000,
                        help="Max clips for stratified dev manifest (use 0 for all)")
    parser.add_argument("--smoke-run", action="store_true", help="Quick smoke test on small subset")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for logs, metrics, figures")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    try:
        run_training(args)
    except FileNotFoundError as e:
        sep = "=" * 65
        msg = (
            f"\n{sep}\n"
            "DATASET CONFIGURATION ERROR:\n"
            f"{e}\n"
            f"{sep}\n"
            "How to fix:\n"
            "  1. Provide dataset path directly:\n"
            "     python src/train.py --data-root /path/to/ASVspoof2019_LA\n"
            "  2. Or set environment variable:\n"
            "     $env:ASVSPOOF2019_LA_ROOT = 'C:\\path\\to\\ASVspoof2019_LA' (PowerShell)\n"
            "     export ASVSPOOF2019_LA_ROOT='/path/to/ASVspoof2019_LA'    (Bash)\n"
            "  3. Or place dataset in 'data/LA' or 'data/ASVspoof2019_LA'.\n"
            f"{sep}"
        )
        logger.error(msg)
        sys.exit(1)
