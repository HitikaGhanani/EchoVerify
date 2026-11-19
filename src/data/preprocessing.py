"""Shared audio loading and preprocessing pipeline for EchoVerify."""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError:
    librosa = None


@dataclass
class PreprocessConfig:
    """Configuration for EchoVerify audio preprocessing."""

    sample_rate: int = 16000
    target_duration_sec: float = 4.0
    mode: str = "raw"  # "raw" or "trimmed"
    trim_top_db: float = 30.0
    crop_mode: str = "first"  # "first" or "center"
    pad_mode: str = "zero"
    n_mels: int = 64
    n_fft: int = 512
    hop_length: int = 160

    @property
    def target_samples(self) -> int:
        return int(round(self.sample_rate * self.target_duration_sec))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["target_samples"] = self.target_samples
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PreprocessConfig:
        clean = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**clean)


def compute_silence_boundaries(
    audio: np.ndarray,
    sample_rate: int = 16000,
    top_db: float = 30.0,
    frame_length: int = 512,
    hop_length: int = 160,
) -> Tuple[int, int, Dict[str, float]]:
    """Compute leading, trailing, and active speech durations using energy/dB threshold.

    Uses fast vectorized RMS frame energy thresholding, providing identical
    boundary detection without JIT compilation overhead.

    Returns:
        (start_idx, end_idx, metadata_dict)
    """
    total_samples = len(audio)
    total_sec = float(total_samples / sample_rate) if sample_rate > 0 else 0.0

    if total_samples == 0:
        meta = {
            "total_sec": 0.0,
            "lead_silence_sec": 0.0,
            "trail_silence_sec": 0.0,
            "edge_silence_sec": 0.0,
            "active_sec": 0.0,
        }
        return 0, 0, meta

    # Vectorized RMS frame energy
    num_frames = 1 + (total_samples - frame_length) // hop_length
    if num_frames <= 0:
        start_idx, end_idx = 0, total_samples
    else:
        frames = np.lib.stride_tricks.as_strided(
            audio,
            shape=(num_frames, frame_length),
            strides=(audio.strides[0] * hop_length, audio.strides[0]),
        )
        rms = np.sqrt(np.mean(frames**2, axis=-1) + 1e-12)
        peak_rms = np.max(rms) + 1e-12
        rms_db = 20 * np.log10(rms / peak_rms)
        active_frames = np.where(rms_db >= -top_db)[0]

        if len(active_frames) == 0:
            start_idx, end_idx = 0, total_samples
        else:
            start_idx = int(active_frames[0] * hop_length)
            end_idx = min(total_samples, int((active_frames[-1] + 1) * hop_length))

    # Clamp indices
    start_idx = max(0, min(start_idx, total_samples))
    end_idx = max(start_idx, min(end_idx, total_samples))

    lead_silence_sec = float(start_idx / sample_rate)
    trail_silence_sec = float((total_samples - end_idx) / sample_rate)
    active_sec = float((end_idx - start_idx) / sample_rate)
    edge_silence_sec = lead_silence_sec + trail_silence_sec

    meta = {
        "total_sec": total_sec,
        "lead_silence_sec": lead_silence_sec,
        "trail_silence_sec": trail_silence_sec,
        "edge_silence_sec": edge_silence_sec,
        "active_sec": active_sec,
    }
    return start_idx, end_idx, meta


class AudioPreprocessor:
    """Preprocesses audio clips into standardized 4-second mono 16 kHz tensors.

    Supports:
    - Loading audio from file paths or in-memory byte buffers
    - Converting stereo/multichannel to mono
    - Resampling to 16 kHz
    - Leading/trailing silence trimming (`mode='trimmed'`) vs raw baseline (`mode='raw'`)
    - Documented deterministic cropping and padding to 64,000 samples (4.0s)
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()

    def load_audio(
        self,
        audio_source: Union[str, Path, bytes, io.BytesIO],
    ) -> Tuple[np.ndarray, int]:
        """Load audio into a 1D float32 numpy array and return (audio, sample_rate).

        Uses SoundFile as primary reader; falls back to librosa if needed.
        """
        if isinstance(audio_source, (str, Path)):
            audio_path = Path(audio_source)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            data, sr = sf.read(str(audio_path), dtype="float32")
        elif isinstance(audio_source, bytes):
            buffer = io.BytesIO(audio_source)
            data, sr = sf.read(buffer, dtype="float32")
        elif isinstance(audio_source, io.BytesIO):
            audio_source.seek(0)
            data, sr = sf.read(audio_source, dtype="float32")
        else:
            raise TypeError(f"Unsupported audio source type: {type(audio_source)}")

        # Convert to mono
        if data.ndim > 1:
            # Check axis: usually (samples, channels)
            if data.shape[0] < data.shape[1]:
                data = np.mean(data, axis=0)
            else:
                data = np.mean(data, axis=1)

        data = np.asarray(data, dtype=np.float32)
        return data, int(sr)

    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample 1D audio to target sample rate (default 16 kHz)."""
        if orig_sr == self.config.sample_rate:
            return audio
        if librosa is not None:
            return librosa.resample(
                audio,
                orig_sr=orig_sr,
                target_sr=self.config.sample_rate,
            ).astype(np.float32)
        else:
            from scipy import signal

            gcd = np.gcd(orig_sr, self.config.sample_rate)
            up = self.config.sample_rate // gcd
            down = orig_sr // gcd
            resampled = signal.resample_poly(audio, up, down)
            return resampled.astype(np.float32)

    def crop_or_pad(self, audio: np.ndarray) -> np.ndarray:
        """Crop or pad 1D audio to exactly target_samples (default 64,000 samples / 4.0s).

        - If len(audio) > target_samples:
          Deterministic crop ('first': audio[:target_samples]; 'center': centered slice).
        - If len(audio) < target_samples:
          Zero-padding at the end ('zero': np.pad with constant 0).
        """
        target = self.config.target_samples
        current_len = len(audio)

        if current_len == target:
            return audio

        if current_len > target:
            if self.config.crop_mode == "center":
                start = (current_len - target) // 2
                return audio[start : start + target]
            else:  # "first"
                return audio[:target]

        # Padding (current_len < target)
        pad_width = target - current_len
        return np.pad(audio, (0, pad_width), mode="constant", constant_values=0.0)

    def process_waveform(
        self,
        audio_source: Union[str, Path, bytes, io.BytesIO, np.ndarray],
        orig_sr: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Run the full preprocessing pipeline on an audio input.

        Returns:
            processed_audio: 1D float32 numpy array of shape (target_samples,) = (64000,)
            silence_metadata: Dict with silence durations and active speech lengths
        """
        if isinstance(audio_source, np.ndarray):
            audio = audio_source
            sr = orig_sr or self.config.sample_rate
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1 if audio.shape[0] > audio.shape[1] else 0)
            audio = audio.astype(np.float32)
        else:
            audio, sr = self.load_audio(audio_source)

        if len(audio) == 0:
            raise ValueError("Input audio is empty (0 samples).")

        # Resample to 16 kHz if necessary
        audio = self.resample(audio, sr)

        # Measure silence boundaries and metadata on the 16 kHz audio
        start_idx, end_idx, silence_meta = compute_silence_boundaries(
            audio,
            sample_rate=self.config.sample_rate,
            top_db=self.config.trim_top_db,
        )

        # Apply trimming if configured
        if self.config.mode == "trimmed":
            trimmed_audio = audio[start_idx:end_idx]
            if len(trimmed_audio) > 0:
                audio = trimmed_audio
            # If trimming removed everything (all silence), fallback to original audio

        # Crop or pad to exact 4.0 seconds (64,000 samples)
        standardized = self.crop_or_pad(audio)
        return standardized, silence_meta
