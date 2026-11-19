"""Unit, integration, and regression test suite for EchoVerify.

All terminal output uses ASCII characters only for Windows console compatibility.
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from src.data.dataset import create_stratified_manifest, get_manifest_statistics
from src.data.preprocessing import AudioPreprocessor, PreprocessConfig, compute_silence_boundaries
from src.data.protocol import ASVSpoofProtocol, get_repo_root, resolve_dataset_root
from src.evaluation.metrics import compute_all_metrics, compute_eer, find_operating_threshold
from src.evaluation.visualizer import plot_confusion_matrix, plot_training_curves
from src.models.cnn import EchoVerifyCNN, SpectrogramExtractor
from src.models.silence_baseline import extract_silence_features
from src.predict import EchoVerifyPredictor


def create_synthetic_wav(
    duration_sec: float = 2.5,
    sample_rate: int = 16000,
    lead_silence_sec: float = 0.5,
    trail_silence_sec: float = 0.5,
) -> np.ndarray:
    """Generate a synthetic sine wave surrounded by silence."""
    total_samples = int(duration_sec * sample_rate)
    lead_samples = int(lead_silence_sec * sample_rate)
    trail_samples = int(trail_silence_sec * sample_rate)
    active_samples = max(0, total_samples - lead_samples - trail_samples)

    t = np.arange(active_samples) / sample_rate
    sine = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    audio = np.zeros(total_samples, dtype=np.float32)
    audio[lead_samples : lead_samples + active_samples] = sine
    return audio


def test_regression_spectrogram_batch_independence():
    """Regression Check 1: Features of a clip must agree whether processed alone or batched."""
    print("[TEST] Running Regression 1: Spectrogram batch composition independence...")
    extractor = SpectrogramExtractor(sample_rate=16000, n_mels=64, n_fft=512, hop_length=160)
    extractor.eval()

    # Generate quiet clip (amplitude 1e-4) and very loud clip (amplitude 10.0, ~100 dB difference)
    quiet_clip = torch.randn(1, 64000) * 0.0001
    loud_clip = torch.randn(1, 64000) * 10.0

    with torch.no_grad():
        spec_alone = extractor(quiet_clip)
        batch = torch.cat([quiet_clip, loud_clip], dim=0)
        spec_batched = extractor(batch)

    spec_quiet_from_batch = spec_batched[0:1]
    max_diff = float((spec_alone - spec_quiet_from_batch).abs().max().item())

    assert max_diff < 1e-4, (
        f"Spectrogram extraction is batch-dependent! Max difference = {max_diff:.4f} dB"
    )
    assert spec_alone.shape == (1, 1, 64, 401), f"Unexpected shape: {spec_alone.shape}"
    print(f"  [PASSED] Spectrogram batch independence verified (max diff: {max_diff:.6e} dB).")


def test_regression_silence_boundaries():
    """Regression Check 2: Ending boundary coverage, continuous signals, all-zero audio, quiet speech."""
    print("[TEST] Running Regression 2: Silence boundary and frame coverage...")

    # Case A: Continuous 16 kHz audio (64,000 samples) must lose ZERO samples
    continuous = np.sin(np.linspace(0, 500, 64000)).astype(np.float32)
    start_c, end_c, meta_c = compute_silence_boundaries(continuous, sample_rate=16000)
    assert start_c == 0, f"Continuous audio start was {start_c}, expected 0"
    assert end_c == 64000, f"Continuous audio lost {64000 - end_c} samples at end!"
    assert meta_c["active_sec"] == 4.0
    assert meta_c["lead_silence_sec"] == 0.0
    assert meta_c["trail_silence_sec"] == 0.0
    dur_sum = meta_c["lead_silence_sec"] + meta_c["active_sec"] + meta_c["trail_silence_sec"]
    assert abs(dur_sum - meta_c["total_sec"]) < 1e-6, "Durations must partition total duration"

    # Case B: All-zero audio must NOT be reported as active speech
    all_zeros = np.zeros(32000, dtype=np.float32)
    start_z, end_z, meta_z = compute_silence_boundaries(all_zeros, sample_rate=16000)
    assert meta_z["active_sec"] == 0.0, "All-zero audio was erroneously reported as active speech!"
    assert meta_z["edge_silence_sec"] == 2.0
    dur_sum_z = meta_z["lead_silence_sec"] + meta_z["active_sec"] + meta_z["trail_silence_sec"]
    assert abs(dur_sum_z - meta_z["total_sec"]) < 1e-6

    # Verify preprocessor rejects all-zero audio with ValueError
    prep = AudioPreprocessor(PreprocessConfig())
    rejected = False
    try:
        prep.process_waveform(all_zeros)
    except ValueError as ve:
        rejected = True
        assert "all zeros" in str(ve).lower() or "silent" in str(ve).lower()
    assert rejected, "AudioPreprocessor did not reject all-zero audio!"

    # Case C: Short clips (fewer samples than frame_length 512)
    short_audio = 0.5 * np.sin(np.linspace(0, 10, 200)).astype(np.float32)
    start_s, end_s, meta_s = compute_silence_boundaries(short_audio, sample_rate=16000)
    assert meta_s["active_sec"] > 0.0, "Short active clip had 0 active speech!"
    assert abs(meta_s["lead_silence_sec"] + meta_s["active_sec"] + meta_s["trail_silence_sec"] - meta_s["total_sec"]) < 1e-6

    # Case D: Known leading and trailing silence
    known = create_synthetic_wav(duration_sec=3.0, sample_rate=16000, lead_silence_sec=0.5, trail_silence_sec=0.7)
    start_k, end_k, meta_k = compute_silence_boundaries(known, sample_rate=16000, top_db=30.0)
    assert abs(meta_k["lead_silence_sec"] - 0.5) < 0.06, f"Lead mismatch: {meta_k['lead_silence_sec']}"
    assert abs(meta_k["trail_silence_sec"] - 0.7) < 0.06, f"Trail mismatch: {meta_k['trail_silence_sec']}"

    # Case E: Quiet speech must NOT be accidentally removed
    quiet_speech = 0.005 * np.sin(np.linspace(0, 200, 32000)).astype(np.float32)
    _, _, meta_q = compute_silence_boundaries(quiet_speech, sample_rate=16000, top_db=30.0)
    assert meta_q["active_sec"] > 1.8, "Quiet speech was accidentally trimmed!"

    print("  [PASSED] Silence boundary and duration coverage checks passed.")


def test_regression_threshold_and_precision():
    """Regression Check 3: Full precision preservation, constant scores, and single-class inputs."""
    print("[TEST] Running Regression 3: Threshold precision and edge-case handling...")

    # Case A: Close score preservation ([0.60001, 0.60004])
    y_true = np.array([0, 1])
    y_scores = np.array([0.60001, 0.60004])
    metrics = compute_all_metrics(y_true, y_scores)

    op_thresh = metrics["operating_threshold"]
    # Check threshold is unrounded full precision
    assert abs(op_thresh - 0.60004) < 1e-7, f"Threshold was truncated: {op_thresh}"

    # Recomputed predictions using saved threshold must match reported metrics
    recomputed_pred = (y_scores >= op_thresh).astype(int)
    assert (recomputed_pred == y_true).all(), (
        f"Recomputed predictions {recomputed_pred} do not match labels {y_true}"
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["genuine_speech_fpr"] == 0.0
    assert metrics["spoof_fnr"] == 0.0

    # Case B: Constant scores
    y_true_const = np.array([0, 1])
    y_scores_const = np.array([0.5, 0.5])
    metrics_const = compute_all_metrics(y_true_const, y_scores_const)

    assert not np.isinf(metrics_const["operating_threshold"]), "Operating threshold was inf!"
    assert not np.isnan(metrics_const["operating_threshold"]), "Operating threshold was nan!"
    # Ensure serializable to standard JSON
    json_str = json.dumps(metrics_const)
    assert "Infinity" not in json_str and "NaN" not in json_str

    # Case C: Single-class inputs (all bonafide, all spoof)
    y_all_bonafide = np.array([0, 0, 0, 0])
    y_scores_b = np.array([0.1, 0.2, 0.3, 0.4])
    metrics_bona = compute_all_metrics(y_all_bonafide, y_scores_b)

    assert metrics_bona["genuine_speech_fpr"] is not None
    assert metrics_bona["spoof_fnr"] is None, "Spoof FNR must be None when spoof class is absent!"
    assert metrics_bona["precision"] is None
    assert metrics_bona["recall"] is None
    assert metrics_bona["f1"] is None
    assert metrics_bona["roc_auc"] is None
    assert metrics_bona["eer"] is None

    y_all_spoof = np.array([1, 1, 1, 1])
    y_scores_s = np.array([0.7, 0.8, 0.85, 0.9])
    metrics_spoof = compute_all_metrics(y_all_spoof, y_scores_s)
    assert metrics_spoof["genuine_speech_fpr"] is None, "Genuine FPR must be None when bonafide class is absent!"
    assert metrics_spoof["spoof_fnr"] is not None
    assert metrics_spoof["roc_auc"] is None
    assert metrics_spoof["eer"] is None

    # JSON round-trip check
    json_bona = json.dumps(metrics_bona)
    assert '"spoof_fnr": null' in json_bona
    assert '"roc_auc": null' in json_bona

    print("  [PASSED] Threshold precision and edge-case metric checks passed.")


def test_regression_silence_baseline_and_alignment():
    """Regression Check 4: Silence baseline uses strictly lead/trail features and preserves alignment."""
    print("[TEST] Running Regression 4: Silence-only baseline feature contract and alignment...")

    temp_dir = Path(tempfile.mkdtemp())
    try:
        # Create mock audio files and dataframe
        rows = []
        for i in range(10):
            audio = create_synthetic_wav(duration_sec=2.0, lead_silence_sec=0.2 + i * 0.05, trail_silence_sec=0.3)
            audio_path = temp_dir / f"clip_{i}.wav"
            sf.write(str(audio_path), audio, 16000)
            rows.append({
                "utt_id": f"utt_{i}",
                "attack_id": "A01" if i >= 5 else "-",
                "label": 1 if i >= 5 else 0,
                "audio_path": audio_path,
            })

        # Add one intentionally unreadable file
        unreadable_path = temp_dir / "corrupted.wav"
        with open(unreadable_path, "wb") as f:
            f.write(b"not an audio file")
        rows.append({
            "utt_id": "utt_corrupt",
            "attack_id": "A02",
            "label": 1,
            "audio_path": unreadable_path,
        })

        df = pd.DataFrame(rows)
        prep = AudioPreprocessor(PreprocessConfig(mode="raw"))
        X, y, utts, attacks, unreadable = extract_silence_features(df, prep)

        # Feature matrix must have exactly 2 columns: lead_silence_sec and trail_silence_sec
        assert X.shape == (10, 2), f"Expected shape (10, 2), got {X.shape}"
        assert len(y) == 10
        assert len(utts) == 10
        assert len(attacks) == 10
        assert len(unreadable) == 1, f"Expected 1 unreadable file, got {len(unreadable)}"
        assert unreadable[0][0] == "utt_corrupt"

        # Verify alignment: utt_i matches clip_i
        for idx, u in enumerate(utts):
            assert u == f"utt_{idx}"

        print("  [PASSED] Silence baseline feature contract and alignment verified.")
    finally:
        shutil.rmtree(temp_dir)


def test_regression_dataset_path_resolution():
    """Regression Check 5: Repository root points to repo root; invalid path raises clear error."""
    print("[TEST] Running Regression 5: Dataset path resolution and error handling...")

    # Check repository root resolution
    root = get_repo_root()
    assert (root / "src").is_dir(), f"Resolved repo root {root} is missing src/!"
    assert (root / "requirements.txt").is_file(), f"Resolved repo root {root} is missing requirements.txt!"
    assert root.name == "EchoVerify", f"Resolved repo root {root} has unexpected name!"

    # Explicit invalid dataset path must raise FileNotFoundError immediately
    invalid_path = Path("C:/non_existent_dataset_path_123456789")
    raised = False
    try:
        resolve_dataset_root(data_root=invalid_path)
    except FileNotFoundError as e:
        raised = True
        assert "Explicitly supplied dataset path does not exist" in str(e)
    assert raised, "Explicitly invalid data-root did not raise FileNotFoundError!"

    # Explicit existing path missing protocols must raise clear FileNotFoundError
    raised_missing = False
    try:
        resolve_dataset_root(data_root=root / "docs")
    except FileNotFoundError as e:
        raised_missing = True
        assert "Explicitly supplied dataset path is invalid" in str(e)
    assert raised_missing, "Explicit invalid directory did not raise protocol error!"

    print("  [PASSED] Dataset path resolution checks passed.")


def test_stratified_manifest():
    print("[TEST] Testing create_stratified_manifest...")
    data = []
    for i in range(20):
        data.append({"speaker_id": "LA_0001", "utt_id": f"bona_{i}", "unused": "-", "attack_id": "-", "label": 0, "audio_path": Path(f"bona_{i}.flac")})
    for i in range(80):
        attack = f"A{1 + (i % 6):02d}"
        data.append({"speaker_id": "LA_0002", "utt_id": f"spoof_{i}", "unused": "-", "attack_id": attack, "label": 1, "audio_path": Path(f"spoof_{i}.flac")})

    df = pd.DataFrame(data)
    sampled = create_stratified_manifest(df, max_samples=30, seed=42)

    assert len(sampled) == 30
    assert (sampled["label"] == 0).sum() > 0
    assert (sampled["label"] == 1).sum() > 0

    attacks_in_sampled = set(sampled["attack_id"].unique())
    expected_attacks = {"-", "A01", "A02", "A03", "A04", "A05", "A06"}
    assert attacks_in_sampled == expected_attacks, f"Missing attacks: {expected_attacks - attacks_in_sampled}"

    stats = get_manifest_statistics(sampled)
    assert stats["total_clips"] == 30
    print("  [PASSED] Stratified manifest tests passed.")


def test_cnn_architecture():
    print("[TEST] Testing EchoVerifyCNN forward pass and shapes...")
    model = EchoVerifyCNN(sample_rate=16000, n_mels=64, n_fft=512, hop_length=160)
    batch_waveform = torch.randn(4, 64000)

    logits = model(batch_waveform)
    assert logits.shape == (4, 1), f"Expected shape (4, 1), got {logits.shape}"

    scores = model.predict_proba(batch_waveform)
    assert scores.shape == (4, 1)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()

    spec = model.spectrogram_extractor(batch_waveform)
    assert spec.shape[1] == 1
    assert spec.shape[2] == 64
    assert spec.shape[3] == 401
    print("  [PASSED] CNN architecture tests passed.")


def test_end_to_end_inference():
    print("[TEST] Testing end-to-end checkpoint saving and EchoVerifyPredictor inference...")
    temp_dir = Path(tempfile.mkdtemp())
    try:
        model = EchoVerifyCNN()
        prep_config = PreprocessConfig(mode="raw", sample_rate=16000)

        ckpt_path = temp_dir / "best_model.pt"
        torch.save(
            {
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "operating_threshold": 0.42000000001,
                "eer_threshold": 0.42000000001,
                "preprocessing_config": prep_config.to_dict(),
                "seed": 42,
                "dev_metrics": {"eer": 0.05, "roc_auc": 0.98, "genuine_speech_fpr": 0.03},
            },
            ckpt_path,
        )

        audio = create_synthetic_wav(duration_sec=2.0)
        audio_file = temp_dir / "sample.wav"
        sf.write(str(audio_file), audio, 16000)

        predictor = EchoVerifyPredictor(checkpoint_path=ckpt_path)
        result = predictor.predict(audio_file)

        assert "prediction" in result
        assert abs(result["operating_threshold"] - 0.42) < 0.01
        assert "processed_waveform" in result
        assert len(result["processed_waveform"]) == 64000
        assert "log_mel_spectrogram" in result
        assert result["log_mel_spectrogram"].shape[0] == 64

        buffer = io.BytesIO()
        sf.write(buffer, audio, 16000, format="WAV")
        buffer.seek(0)
        result_buf = predictor.predict(buffer.getvalue())
        assert result_buf["prediction"] in ["Genuine", "Synthetic"]

        cm_fig_path = temp_dir / "cm.png"
        plot_confusion_matrix({"tn": 10, "fp": 1, "fn": 2, "tp": 20}, out_path=cm_fig_path)
        assert cm_fig_path.exists()

        history = {"epoch": [1, 2], "train_loss": [0.5, 0.3], "dev_loss": [0.6, 0.4], "dev_roc_auc": [0.9, 0.95], "dev_eer": [0.1, 0.05]}
        curves_path = temp_dir / "curves.png"
        plot_training_curves(history, out_path=curves_path)
        assert curves_path.exists()

        print("  [PASSED] End-to-end inference tests passed.")
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_regression_spectrogram_batch_independence()
    test_regression_silence_boundaries()
    test_regression_threshold_and_precision()
    test_regression_silence_baseline_and_alignment()
    test_regression_dataset_path_resolution()
    test_stratified_manifest()
    test_cnn_architecture()
    test_end_to_end_inference()
    print("\nALL ECHOVERIFY PIPELINE & REGRESSION TESTS PASSED SUCCESSFULLY! [OK]")
