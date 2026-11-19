"""Single-file audio deepfake prediction and shared inference module."""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from src.data.preprocessing import AudioPreprocessor, PreprocessConfig
from src.models.cnn import EchoVerifyCNN


class EchoVerifyPredictor:
    """Inference engine for EchoVerify audio deepfake detection.

    Loads a trained checkpoint, extracts saved preprocessing parameters and
    decision threshold, and generates uncalibrated prediction scores.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "checkpoints/best_model.pt",
        device: Optional[str] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at: {self.checkpoint_path}\n"
                "Please train a model first using `python src/train.py` or specify --checkpoint."
            )

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load checkpoint
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Restore preprocessing config
        prep_dict = self.checkpoint.get("preprocessing_config", {})
        self.prep_config = PreprocessConfig.from_dict(prep_dict)
        self.preprocessor = AudioPreprocessor(self.prep_config)

        # Restore operating threshold
        self.operating_threshold = float(self.checkpoint.get("operating_threshold", 0.5))

        # Reconstruct and load model
        self.model = EchoVerifyCNN(
            sample_rate=self.prep_config.sample_rate,
            n_mels=self.prep_config.n_mels,
            n_fft=self.prep_config.n_fft,
            hop_length=self.prep_config.hop_length,
        ).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        audio_source: Union[str, Path, bytes, io.BytesIO, np.ndarray],
        custom_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Perform deepfake screening on an audio input.

        Args:
            audio_source: File path, raw bytes, BytesIO buffer, or numpy array.
            custom_threshold: Optional threshold override.

        Returns:
            Dictionary with prediction label, raw score, logit, threshold,
            silence metadata, and arrays for visualization.
        """
        # Run shared preprocessing
        processed_waveform, silence_meta = self.preprocessor.process_waveform(audio_source)

        # Convert to tensor: (1, 64000)
        tensor_wave = torch.from_numpy(processed_waveform).unsqueeze(0).to(self.device)

        # Forward pass
        logit = float(self.model(tensor_wave).item())
        raw_score = float(torch.sigmoid(torch.tensor(logit)).item())

        threshold = custom_threshold if custom_threshold is not None else self.operating_threshold
        is_synthetic = bool(raw_score >= threshold)
        predicted_label = "Synthetic" if is_synthetic else "Genuine"

        # Compute log-Mel spectrogram for visualization
        spectrogram_tensor = self.model.spectrogram_extractor(tensor_wave)
        spectrogram_np = spectrogram_tensor.squeeze().cpu().numpy()

        return {
            "prediction": predicted_label,
            "is_synthetic": is_synthetic,
            "raw_score": float(raw_score),
            "logit": float(logit),
            "operating_threshold": float(threshold),
            "margin_from_threshold": float(raw_score - threshold),
            "preprocessing_mode": self.prep_config.mode,
            "silence_metadata": silence_meta,
            "processed_waveform": processed_waveform,
            "log_mel_spectrogram": spectrogram_np,
            "benchmark_metrics": self.checkpoint.get("dev_metrics"),
        }


def predict_single_file(
    audio_path: Union[str, Path],
    checkpoint_path: Union[str, Path] = "checkpoints/best_model.pt",
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Helper function to run inference on a single audio file."""
    predictor = EchoVerifyPredictor(checkpoint_path=checkpoint_path, device=device)
    return predictor.predict(audio_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="EchoVerify single-file audio deepfake prediction.")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file (WAV, FLAC, MP3)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt",
                        help="Path to trained PyTorch checkpoint")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--threshold", type=float, default=None, help="Optional operating threshold override")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found at {audio_path}")
        return

    predictor = EchoVerifyPredictor(
        checkpoint_path=args.checkpoint,
        device=None if args.device == "auto" else args.device,
    )
    result = predictor.predict(audio_path, custom_threshold=args.threshold)

    print("\n" + "=" * 50)
    print("           EchoVerify Prediction Result           ")
    print("=" * 50)
    print(f"Audio File:            {audio_path.name}")
    print(f"Prediction:            {result['prediction'].upper()}")
    print(f"Raw Synthetic Score:   {result['raw_score']:.5f} (uncalibrated model score)")
    print(f"Raw Logit:             {result['logit']:.5f}")
    print(f"Operating Threshold:   {result['operating_threshold']:.4f}")
    print(f"Margin:                {result['margin_from_threshold']:+.5f}")
    print(f"Preprocessing Mode:    {result['preprocessing_mode']}")

    sil = result["silence_metadata"]
    print(f"Silence Edges:         Leading: {sil['lead_silence_sec']:.3f}s | "
          f"Trailing: {sil['trail_silence_sec']:.3f}s | Active: {sil['active_sec']:.3f}s")

    bm = result.get("benchmark_metrics")
    if bm:
        print("\nModel Benchmark Metrics (Dev partition):")
        print(f"  EER:                 {bm.get('eer')}")
        print(f"  ROC-AUC:             {bm.get('roc_auc')}")
        print(f"  Genuine Speech FPR:  {bm.get('genuine_speech_fpr')}")
    print("=" * 50)
    print("Notice: EchoVerify is an experimental screening tool, not a definitive")
    print("forensic determination. Always corroborate results.\n")


if __name__ == "__main__":
    main()
