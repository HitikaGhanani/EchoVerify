"""Official ASVspoof 2019 Logical Access (LA) protocol parser and path resolver."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

LABEL_MAPPING = {
    "bonafide": 0,
    "spoof": 1,
}
INVERSE_LABEL_MAPPING = {0: "bonafide", 1: "spoof"}


@dataclass(frozen=True)
class ASVSpoofPaths:
    """Paths to protocols and audio directories for ASVspoof 2019 LA."""

    root: Path
    protocol_dir: Path
    protocols: Dict[str, Path]
    audio_dirs: Dict[str, Path]


def resolve_dataset_root(data_root: Optional[str | Path] = None) -> Path:
    """Resolve the root directory of the ASVspoof 2019 LA dataset.

    Checks in order:
    1. Explicit `data_root` argument
    2. `ASVSPOOF2019_LA_ROOT` environment variable
    3. Common local repository paths: `data/LA`, `data/ASVspoof2019_LA`, `data`

    Raises:
        FileNotFoundError: If no valid root containing protocol directory is found.
    """
    candidates: List[Path] = []
    if data_root is not None:
        candidates.append(Path(data_root))

    env_root = os.environ.get("ASVSPOOF2019_LA_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    # Common local workspace paths
    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend([
        repo_root / "data" / "LA",
        repo_root / "data" / "ASVspoof2019_LA",
        repo_root / "data",
        Path("data/LA"),
        Path("data/ASVspoof2019_LA"),
        Path("data"),
    ])

    # Deduplicate candidate paths while preserving order
    unique_candidates: List[Path] = []
    seen = set()
    for c in candidates:
        res = c.resolve()
        if res not in seen:
            seen.add(res)
            unique_candidates.append(c)

    for candidate in unique_candidates:
        if candidate.exists():
            proto_dir = candidate / "ASVspoof2019_LA_cm_protocols"
            if proto_dir.exists():
                return candidate.resolve()
            # Also check if candidate itself IS the protocol dir parent
            if (candidate / "ASVspoof2019.LA.cm.train.trn.txt").exists():
                return candidate.parent.resolve()

    checked = "\n  - ".join(str(c.resolve()) for c in unique_candidates)
    raise FileNotFoundError(
        "ASVspoof 2019 LA dataset root could not be located.\n"
        "Please provide --data-root or set the ASVSPOOF2019_LA_ROOT environment variable.\n"
        f"Checked candidate locations:\n  - {checked}\n"
        "Expected directory containing 'ASVspoof2019_LA_cm_protocols/' and audio folders."
    )


def get_asvspoof_paths(data_root: Optional[str | Path] = None) -> ASVSpoofPaths:
    """Build and return verified path mappings for ASVspoof 2019 LA."""
    root = resolve_dataset_root(data_root)
    protocol_dir = root / "ASVspoof2019_LA_cm_protocols"

    protocols = {
        "train": protocol_dir / "ASVspoof2019.LA.cm.train.trn.txt",
        "dev": protocol_dir / "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval": protocol_dir / "ASVspoof2019.LA.cm.eval.trl.txt",
    }

    audio_dirs = {
        "train": root / "ASVspoof2019_LA_train" / "flac",
        "dev": root / "ASVspoof2019_LA_dev" / "flac",
        "eval": root / "ASVspoof2019_LA_eval" / "flac",
    }

    return ASVSpoofPaths(
        root=root,
        protocol_dir=protocol_dir,
        protocols=protocols,
        audio_dirs=audio_dirs,
    )


class ASVSpoofProtocol:
    """Loader and validator for official ASVspoof 2019 LA countermeasure protocols."""

    def __init__(self, data_root: Optional[str | Path] = None):
        self.paths = get_asvspoof_paths(data_root)

    def is_split_available(self, split: str) -> bool:
        """Check whether protocol file and audio directory exist for a split."""
        if split not in self.paths.protocols:
            return False
        proto_file = self.paths.protocols[split]
        audio_dir = self.paths.audio_dirs[split]
        return proto_file.exists() and audio_dir.exists()

    def load_split(
        self,
        split: str,
        validate_audio: bool = True,
        max_missing_check: int = 100,
    ) -> pd.DataFrame:
        """Load and parse the protocol for a split (train, dev, or eval).

        Columns returned:
        - speaker_id: str
        - utt_id: str
        - unused: str (recording/transmission environment)
        - attack_id: str ('-' for bonafide; A01-A06 for train/dev; A07-A19 for eval)
        - label: int (0 for bonafide, 1 for spoof)
        - label_name: str ('bonafide' or 'spoof')
        - split: str
        - audio_path: Path
        """
        if split not in self.paths.protocols:
            raise ValueError(f"Unknown split '{split}'. Expected 'train', 'dev', or 'eval'.")

        proto_path = self.paths.protocols[split]
        if not proto_path.exists():
            raise FileNotFoundError(
                f"Protocol file for split '{split}' does not exist at: {proto_path}"
            )

        audio_dir = self.paths.audio_dirs[split]
        if validate_audio and not audio_dir.exists():
            raise FileNotFoundError(
                f"Audio directory for split '{split}' does not exist at: {audio_dir}"
            )

        df = pd.read_csv(
            proto_path,
            sep=r"\s+",
            header=None,
            names=["speaker_id", "utt_id", "unused", "attack_id", "raw_label"],
            dtype=str,
        )

        if len(df.columns) != 5:
            raise ValueError(
                f"Protocol {proto_path} has {len(df.columns)} columns; expected 5 columns."
            )

        # Validate labels
        invalid_labels = set(df["raw_label"].unique()) - set(LABEL_MAPPING.keys())
        if invalid_labels:
            raise ValueError(
                f"Found unexpected labels in {proto_path}: {invalid_labels}. "
                f"Expected one of: {list(LABEL_MAPPING.keys())}"
            )

        df["label_name"] = df["raw_label"]
        df["label"] = df["raw_label"].map(LABEL_MAPPING)
        df["split"] = split
        df["audio_path"] = df["utt_id"].map(lambda u: audio_dir / f"{u}.flac")
        df.drop(columns=["raw_label"], inplace=True)

        if validate_audio and len(df) > 0:
            sample_to_check = df.head(min(max_missing_check, len(df)))
            missing = [str(p) for p in sample_to_check["audio_path"] if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Sampled {len(sample_to_check)} files for split '{split}' but {len(missing)} were missing on disk.\n"
                    f"Example missing file: {missing[0]}\n"
                    f"Expected audio directory: {audio_dir}"
                )

        return df


def load_protocol_split(
    split: str,
    data_root: Optional[str | Path] = None,
    validate_audio: bool = True,
) -> pd.DataFrame:
    """Convenience helper to load a single split protocol DataFrame."""
    protocol = ASVSpoofProtocol(data_root=data_root)
    return protocol.load_split(split, validate_audio=validate_audio)

