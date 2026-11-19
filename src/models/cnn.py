"""Compact log-Mel spectrogram CNN for audio deepfake detection."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchaudio.transforms as T


class SpectrogramExtractor(nn.Module):
    """Computes log-Mel spectrogram from raw 16 kHz audio waveforms.

    Default settings:
    - n_mels: 64
    - n_fft: 512
    - win_length: 512
    - hop_length: 160 (10ms frame shift at 16kHz)
    - sample_rate: 16000
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: Optional[int] = None,
    ):
        super().__init__()
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length or n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.amp_to_db = T.AmplitudeToDB(top_db=80.0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute log-Mel spectrogram.

        Args:
            waveform: Tensor of shape (batch, samples) or (batch, 1, samples)

        Returns:
            log_mel: Tensor of shape (batch, 1, n_mels, time_frames)
        """
        if waveform.dim() == 2:
            # (batch, samples)
            pass
        elif waveform.dim() == 3 and waveform.size(1) == 1:
            # (batch, 1, samples) -> (batch, samples)
            waveform = waveform.squeeze(1)
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        mel = self.mel_transform(waveform)

        # Add explicit channel dimension BEFORE decibel conversion so AmplitudeToDB
        # receives [batch, 1, mel, time]. This ensures each clip's peak reference
        # is computed independently of other clips in the batch.
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)
        elif mel.dim() == 2:
            mel = mel.unsqueeze(0).unsqueeze(1)

        log_mel = self.amp_to_db(mel)

        return log_mel


class ConvBlock(nn.Module):
    """2D Convolutional block with BatchNorm, ReLU, and MaxPool."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EchoVerifyCNN(nn.Module):
    """Compact 4-block 2D CNN on log-Mel spectrograms for synthetic speech detection.

    Architecture:
    - On-the-fly or input log-Mel spectrogram (64 mels, n_fft=512, hop_length=160)
    - 4 Convolutional blocks (16 -> 32 -> 64 -> 128 channels)
    - Adaptive average pooling (2x2)
    - Fully connected classification head with Dropout (0.3 / 0.2)
    - Single binary logit output (positive class = synthetic speech / spoof)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 512,
        hop_length: int = 160,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.spectrogram_extractor = SpectrogramExtractor(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        self.features = nn.Sequential(
            ConvBlock(1, 16, pool=True),     # (B, 16, 32, T/2)
            ConvBlock(16, 32, pool=True),    # (B, 32, 16, T/4)
            ConvBlock(32, 64, pool=True),    # (B, 64, 8, T/8)
            ConvBlock(64, 128, pool=True),   # (B, 128, 4, T/16)
        )

        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(128 * 2 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.7),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        is_spectrogram: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Raw waveform (batch, samples) or precomputed log-Mel (batch, 1, n_mels, time)
            is_spectrogram: Whether x is already a log-Mel spectrogram

        Returns:
            logits: Tensor of shape (batch, 1) representing uncalibrated synthetic speech logits
        """
        if not is_spectrogram:
            x = self.spectrogram_extractor(x)

        feat = self.features(x)
        pooled = self.pool(feat)
        flattened = torch.flatten(pooled, 1)
        logits = self.classifier(flattened)
        return logits

    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
        is_spectrogram: bool = False,
    ) -> torch.Tensor:
        """Compute uncalibrated synthetic speech scores (sigmoid of logits)."""
        logits = self.forward(x, is_spectrogram=is_spectrogram)
        return torch.sigmoid(logits)

