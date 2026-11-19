"""Unit and integration tests for EchoVerify core pipeline."""

import io
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
from src.data.protocol import ASVSpoofProtocol, resolve_dataset_root
from src.evaluation.metrics import compute_all_metrics, compute_eer, find_operating_threshold
from src.evaluation.visualizer import plot_confusion_matrix, plot_training_curves
from src.models.cnn import EchoVerifyCNN
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


def test_silence_and_preprocessing():
    print("Testing AudioPreprocessor and silence detection...")
    audio = create_synthetic_wav(duration_sec=3.0, lead_silence_sec=0.6, trail_silence_sec=0.4)

    start_idx, end_idx, meta = compute_silence_boundaries(audio, sample_rate=16000, top_db=30.0)
    assert meta["total_sec"] == 3.0
    assert abs(meta["lead_silence_sec"] - 0.6) < 0.05, f"Lead silence mismatch: {meta['lead_silence_sec']}"
    assert abs(meta["trail_silence_sec"] - 0.4) < 0.05, f"Trail silence mismatch: {meta['trail_silence_sec']}"

    # Test raw preprocessor: crops/pads to 64000
    raw_cfg = PreprocessConfig(mode="raw", sample_rate=16000, target_duration_sec=4.0)
    raw_prep = AudioPreprocessor(raw_cfg)
    processed_raw, raw_meta = raw_prep.process_waveform(audio)
    assert len(processed_raw) == 64000
    assert raw_meta["total_sec"] == 3.0

    # Test trimmed preprocessor: trimmed audio padded to 64000
    trimmed_cfg = PreprocessConfig(mode="trimmed", sample_rate=16000, target_duration_sec=4.0)
    trimmed_prep = AudioPreprocessor(trimmed_cfg)
    processed_trimmed, trim_meta = trimmed_prep.process_waveform(audio)
    assert len(processed_trimmed) == 64000
    print("  -> Preprocessing tests passed!")


def test_stratified_manifest():
    print("Testing create_stratified_manifest...")
    # Create mock ASVspoof protocol dataframe
    data = []
    # 20 bonafide clips
    for i in range(20):
        data.append({"speaker_id": "LA_0001", "utt_id": f"bona_{i}", "unused": "-", "attack_id": "-", "label": 0, "audio_path": Path(f"bona_{i}.flac")})
    # 80 spoof clips across A01-A06
    for i in range(80):
        attack = f"A{1 + (i % 6):02d}"
        data.append({"speaker_id": "LA_0002", "utt_id": f"spoof_{i}", "unused": "-", "attack_id": attack, "label": 1, "audio_path": Path(f"spoof_{i}.flac")})

    df = pd.DataFrame(data)
    sampled = create_stratified_manifest(df, max_samples=30, seed=42)

    assert len(sampled) == 30
    assert (sampled["label"] == 0).sum() > 0
    assert (sampled["label"] == 1).sum() > 0

    # Verify all attack IDs are preserved
    attacks_in_sampled = set(sampled["attack_id"].unique())
    expected_attacks = {"-", "A01", "A02", "A03", "A04", "A05", "A06"}
    assert attacks_in_sampled == expected_attacks, f"Missing attacks: {expected_attacks - attacks_in_sampled}"

    stats = get_manifest_statistics(sampled)
    assert stats["total_clips"] == 30
    print("  -> Stratified manifest tests passed!")


def test_cnn_architecture():
    print("Testing EchoVerifyCNN forward pass and shapes...")
    model = EchoVerifyCNN(sample_rate=16000, n_mels=64, n_fft=512, hop_length=160)
    batch_waveform = torch.randn(4, 64000)

    logits = model(batch_waveform)
    assert logits.shape == (4, 1), f"Expected shape (4, 1), got {logits.shape}"

    scores = model.predict_proba(batch_waveform)
    assert scores.shape == (4, 1)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()

    # Verify spectrogram extractor
    spec = model.spectrogram_extractor(batch_waveform)
    assert spec.shape[1] == 1
    assert spec.shape[2] == 64
    assert spec.shape[3] == 401  # 64000 / 160 + 1
    print("  -> CNN architecture tests passed!")


def test_evaluation_metrics():
    print("Testing evaluation metrics & EER computation...")
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.15, 0.35, 0.4, 0.6, 0.7, 0.85, 0.9, 0.95])

    metrics = compute_all_metrics(y_true, y_scores)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["eer"] == 0.0
    assert metrics["genuine_speech_fpr"] == 0.0

    # Test edge case with division by zero
    y_all_bonafide = np.array([0, 0, 0, 0])
    y_scores_b = np.array([0.1, 0.2, 0.3, 0.4])
    metrics_edge = compute_all_metrics(y_all_bonafide, y_scores_b)
    assert metrics_edge["accuracy"] == 1.0
    assert metrics_edge["precision"] == 0.0  # safe default
    print("  -> Evaluation metrics tests passed!")


def test_end_to_end_inference():
    print("Testing end-to-end checkpoint saving and EchoVerifyPredictor inference...")
    temp_dir = Path(tempfile.mkdtemp())
    try:
        model = EchoVerifyCNN()
        prep_config = PreprocessConfig(mode="raw", sample_rate=16000)

        ckpt_path = temp_dir / "best_model.pt"
        torch.save(
            {
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "operating_threshold": 0.42,
                "eer_threshold": 0.42,
                "preprocessing_config": prep_config.to_dict(),
                "seed": 42,
                "dev_metrics": {"eer": 0.05, "roc_auc": 0.98, "genuine_speech_fpr": 0.03},
            },
            ckpt_path,
        )

        # Create temporary audio file
        audio = create_synthetic_wav(duration_sec=2.0)
        audio_file = temp_dir / "sample.wav"
        sf.write(str(audio_file), audio, 16000)

        # Test Predictor
        predictor = EchoVerifyPredictor(checkpoint_path=ckpt_path)
        result = predictor.predict(audio_file)

        assert "prediction" in result
        assert result["operating_threshold"] == 0.42
        assert "processed_waveform" in result
        assert len(result["processed_waveform"]) == 64000
        assert "log_mel_spectrogram" in result
        assert result["log_mel_spectrogram"].shape[0] == 64

        # Test buffer inference (Streamlit upload simulation)
        buffer = io.BytesIO()
        sf.write(buffer, audio, 16000, format="WAV")
        buffer.seek(0)
        result_buf = predictor.predict(buffer.getvalue())
        assert result_buf["prediction"] in ["Genuine", "Synthetic"]

        # Test visualizer functions
        cm_fig_path = temp_dir / "cm.png"
        plot_confusion_matrix({"tn": 10, "fp": 1, "fn": 2, "tp": 20}, out_path=cm_fig_path)
        assert cm_fig_path.exists()

        history = {"epoch": [1, 2], "train_loss": [0.5, 0.3], "dev_loss": [0.6, 0.4], "dev_roc_auc": [0.9, 0.95], "dev_eer": [0.1, 0.05]}
        curves_path = temp_dir / "curves.png"
        plot_training_curves(history, out_path=curves_path)
        assert curves_path.exists()

        print("  -> End-to-end inference tests passed!")
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_silence_and_preprocessing()
    test_stratified_manifest()
    test_cnn_architecture()
    test_evaluation_metrics()
    test_end_to_end_inference()
    print("\nALL ECHOVERIFY PIPELINE TESTS PASSED SUCCESSFULLY! ✅")
