"""PyTorch Dataset and stratified manifest sampling for ASVspoof 2019 LA."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import AudioPreprocessor, PreprocessConfig


def create_stratified_manifest(
    protocol_df: pd.DataFrame,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a deterministic sampled manifest preserving class and attack coverage.

    - Uses `seed` (default 42) for reproducible sampling.
    - Stratifies by `(label, attack_id)` to ensure all attack IDs (e.g. A01-A06)
      and bona fide clips are covered.
    - If `max_samples is None` or `max_samples <= 0` or `max_samples >= len(protocol_df)`,
      returns the full partition.
    - Ensures the same clip IDs are selected regardless of whether raw or trimmed
      preprocessing is subsequently applied.
    """
    total_available = len(protocol_df)
    if max_samples is None or max_samples <= 0 or max_samples >= total_available:
        # Return deterministically sorted full DataFrame
        return protocol_df.sort_values(by=["label", "attack_id", "utt_id"]).reset_index(drop=True)

    # Group by (label, attack_id)
    groups = protocol_df.groupby(["label", "attack_id"], group_keys=False)
    strata_counts = groups.size()
    total_count = len(protocol_df)

    # Proportional allocation with guarantee of at least 1 sample per group
    sampled_dfs = []
    allocated_total = 0

    for (label_val, attack_val), group_df in groups:
        fraction = len(group_df) / total_count
        target_group_n = max(1, int(round(fraction * max_samples)))
        target_group_n = min(target_group_n, len(group_df))

        sampled_group = group_df.sample(n=target_group_n, random_state=seed)
        sampled_dfs.append(sampled_group)
        allocated_total += target_group_n

    combined = pd.concat(sampled_dfs, ignore_index=True)

    # If rounding caused slightly more or fewer than max_samples, adjust deterministically
    if len(combined) > max_samples:
        combined = combined.sample(n=max_samples, random_state=seed)
    elif len(combined) < max_samples:
        shortfall = max_samples - len(combined)
        remaining_pool = protocol_df[~protocol_df["utt_id"].isin(combined["utt_id"])]
        if len(remaining_pool) > 0:
            extra = remaining_pool.sample(n=min(shortfall, len(remaining_pool)), random_state=seed)
            combined = pd.concat([combined, extra], ignore_index=True)

    # Sort deterministically by utt_id
    combined = combined.sort_values(by=["label", "attack_id", "utt_id"]).reset_index(drop=True)
    return combined


def get_manifest_statistics(manifest_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate summary statistics for a manifest (counts, attack coverage, class balance)."""
    total = len(manifest_df)
    bonafide_count = int((manifest_df["label"] == 0).sum())
    spoof_count = int((manifest_df["label"] == 1).sum())

    attack_distribution = manifest_df["attack_id"].value_counts().to_dict()

    return {
        "total_clips": total,
        "bonafide_count": bonafide_count,
        "spoof_count": spoof_count,
        "spoof_ratio": round(spoof_count / max(1, bonafide_count), 4),
        "attack_distribution": attack_distribution,
    }


class ASVSpoofDataset(Dataset):
    """PyTorch Dataset loading audio and applying shared EchoVerify preprocessing."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        preprocessor: Optional[AudioPreprocessor] = None,
        config: Optional[PreprocessConfig] = None,
    ):
        self.df = manifest_df.reset_index(drop=True)
        self.preprocessor = preprocessor or AudioPreprocessor(config)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str, Dict[str, float]]:
        row = self.df.iloc[idx]
        audio_path = Path(row["audio_path"])
        label = int(row["label"])
        utt_id = str(row["utt_id"])
        attack_id = str(row["attack_id"])

        # Preprocess waveform to 64000 samples (4.0s @ 16kHz)
        waveform_np, silence_meta = self.preprocessor.process_waveform(audio_path)
        waveform_tensor = torch.from_numpy(waveform_np).float()
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return waveform_tensor, label_tensor, utt_id, attack_id, silence_meta

