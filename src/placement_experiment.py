"""Paired background placement audit of frozen EchoVerify checkpoints.

Noise donors come exclusively from training bona fide recordings. Evaluation
speech, total duration, noise samples, and thresholds stay fixed across positions.
Run from the repository root with ``python -m src.placement_experiment --help``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

from src.data.preprocessing import AudioPreprocessor, compute_silence_boundaries
from src.data.protocol import ASVSpoofProtocol
from src.predict import EchoVerifyPredictor

SAMPLE_RATE = 16000
SPEECH_SAMPLES = 3 * SAMPLE_RATE
BACKGROUND_SAMPLES = SAMPLE_RATE
POSITIONS = (0.1, 0.5, 0.9)


def make_variant(speech, background, leading_seconds):
    """Move a fixed background around identical speech with exact sample counts."""
    if len(speech) != SPEECH_SAMPLES or len(background) != BACKGROUND_SAMPLES:
        raise ValueError("Expected three seconds of speech and one second of background.")
    left = round(leading_seconds * SAMPLE_RATE)
    variant = np.concatenate((background[:left], speech, background[left:])).astype(np.float32)
    assert len(variant) == 4 * SAMPLE_RATE
    assert np.array_equal(variant[left:left + SPEECH_SAMPLES], speech)
    assert np.array_equal(np.concatenate((variant[:left], variant[left + SPEECH_SAMPLES:])), background)
    return variant


def collect_noise(train, preprocessor, output, seed, top_db, noise_dbfs, count):
    """Find real nonzero, low-energy edge regions using training audio only."""
    donors = []
    records = []
    target_rms = 10 ** (noise_dbfs / 20)
    candidates = train[train.label == 0].sample(frac=1, random_state=seed)
    for row in candidates.itertuples():
        audio, sr = preprocessor.load_audio(row.audio_path)
        audio = preprocessor.resample(audio, sr)
        start, end, _ = compute_silence_boundaries(audio, SAMPLE_RATE, top_db)
        for side, edge in (("leading", audio[:start]), ("trailing", audio[end:])):
            if len(edge) < BACKGROUND_SAMPLES:
                continue
            # Take the edge portion furthest from the detected speech boundary.
            noise = edge[:BACKGROUND_SAMPLES] if side == "leading" else edge[-BACKGROUND_SAMPLES:]
            rms = float(np.sqrt(np.mean(noise.astype(np.float64) ** 2)))
            if not np.isfinite(rms) or rms <= 1e-8:
                continue
            scaled = (noise * (target_rms / rms)).astype(np.float32)
            if float(np.abs(scaled).max()) > 0.1:
                continue
            noise_id = len(donors)
            sf.write(output / f"noise_donor_{noise_id}.wav", scaled, SAMPLE_RATE, subtype="FLOAT")
            records.append({"noise_id": noise_id, "utt_id": row.utt_id,
                            "speaker_id": row.speaker_id, "split": "train", "side": side,
                            "original_rms": rms, "target_dbfs": noise_dbfs,
                            "sha256": hashlib.sha256(scaled.tobytes()).hexdigest()})
            donors.append(scaled)
            break
        if len(donors) == count:
            break
    if not donors:
        raise RuntimeError("No usable one-second recorded-background donors found in training audio.")
    pd.DataFrame(records).to_csv(output / "noise_donors.csv", index=False)
    print(f"Found {len(donors)} real-background donors from training bona fide audio.", flush=True)
    return donors


def select_speech(evaluation, preprocessor, per_class, seed, top_db, output):
    """Use seeded clip selection with eligibility decided independently of scores."""
    selected = []
    scanned = []
    for label in (0, 1):
        accepted = 0
        candidates = evaluation[evaluation.label == label].sample(frac=1, random_state=seed)
        for row in candidates.itertuples():
            audio, sr = preprocessor.load_audio(row.audio_path)
            audio = preprocessor.resample(audio, sr)
            start, end, _ = compute_silence_boundaries(audio, SAMPLE_RATE, top_db)
            eligible = end - start >= SPEECH_SAMPLES and np.all(np.isfinite(audio))
            scanned.append({"utt_id": row.utt_id, "label": label, "attack_id": row.attack_id,
                            "eligible": bool(eligible), "retained_seconds": (end - start) / SAMPLE_RATE})
            if not eligible:
                continue
            speech = audio[start:start + SPEECH_SAMPLES].copy()
            selected.append((row, speech, accepted))
            accepted += 1
            if accepted == per_class:
                break
        if accepted < per_class:
            raise RuntimeError(f"Only {accepted} eligible clips for class {label}, requested {per_class}.")
    pd.DataFrame(scanned).to_csv(output / "eligibility_screen.csv", index=False)
    return selected


def summarize(scores):
    keys = ["model", "utt_id", "background_type"]
    reference = scores[scores.leading_seconds == 0.5][keys + ["score", "prediction"]]
    reference = reference.rename(columns={"score": "reference_score", "prediction": "reference_prediction"})
    paired = scores.merge(reference, on=keys, validate="many_to_one")
    paired["score_change"] = paired.score - paired.reference_score
    paired["absolute_score_change"] = paired.score_change.abs()
    paired["prediction_flipped"] = paired.prediction != paired.reference_prediction
    paired["incorrect"] = paired.prediction != paired.label
    summary = paired.groupby(["model", "background_type", "leading_seconds", "label"]).agg(
        clips=("utt_id", "size"), mean_score=("score", "mean"),
        mean_signed_change=("score_change", "mean"),
        mean_absolute_change=("absolute_score_change", "mean"),
        max_absolute_change=("absolute_score_change", "max"),
        flip_rate=("prediction_flipped", "mean"), error_rate=("incorrect", "mean"),
    ).reset_index()
    summary["error_interpretation"] = np.where(summary.label == 0, "genuine_false_positive_rate", "synthetic_false_negative_rate")
    return paired, summary


def plot_summary(summary, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for label, ax in enumerate(axes):
        part = summary[summary.label == label]
        for (model, background), group in part.groupby(["model", "background_type"]):
            group = group.sort_values("leading_seconds")
            ax.plot(group.leading_seconds, 100 * group.flip_rate, marker="o", label=f"{model} / {background}")
        ax.set_title("Genuine speech" if label == 0 else "Synthetic speech")
        ax.set_xlabel("Leading background (seconds)")
        ax.set_xticks(POSITIONS)
        ax.set_ylim(-2, 102)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Prediction flips versus 0.5/0.5 split (%)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "placement_flip_rates.png", dpi=160)
    plt.close(fig)


def run(args):
    if args.per_class < 1 or args.noise_donors < 1:
        raise ValueError("Sample and donor counts must be positive.")
    if args.noise_dbfs > -30:
        raise ValueError("Use a background RMS level of -30 dBFS or lower.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = None if args.device == "auto" else args.device
    predictors = {name: EchoVerifyPredictor(Path(args.artifact_root) / "checkpoints" / name / "best_model.pt", device=device)
                  for name in ("raw", "trimmed")}
    for predictor in predictors.values():
        if predictor.prep_config.sample_rate != SAMPLE_RATE or predictor.prep_config.target_samples != 4 * SAMPLE_RATE:
            raise ValueError("Experiment requires 16 kHz, four-second checkpoints.")
    protocol = ASVSpoofProtocol(args.data_root)
    loader = AudioPreprocessor()
    donors = collect_noise(protocol.load_split("train"), loader, output, args.seed,
                           args.trim_top_db, args.noise_dbfs, args.noise_donors)
    selected = select_speech(protocol.load_split("eval"), loader, args.per_class,
                             args.seed, args.trim_top_db, output)
    records = []
    for row, speech, within_class_index in tqdm(selected, desc="Paired placement audit"):
        noise_id = within_class_index % len(donors)
        backgrounds = {"zeros": np.zeros(BACKGROUND_SAMPLES, dtype=np.float32),
                       "recorded_noise": donors[noise_id]}
        variants = []
        descriptions = []
        for background_type, background in backgrounds.items():
            for leading in POSITIONS:
                variants.append(make_variant(speech, background, leading))
                descriptions.append((background_type, leading))
        for name, predictor in predictors.items():
            # Deliberately bypass normal trimming/cropping: those operations would
            # erase the controlled placement intervention. Weights remain frozen.
            batch = torch.from_numpy(np.stack(variants)).to(predictor.device)
            with torch.no_grad():
                values = torch.sigmoid(predictor.model(batch)).flatten().cpu().numpy()
            for score, (background_type, leading) in zip(values, descriptions):
                records.append({"model": name, "utt_id": row.utt_id, "label": int(row.label),
                                "attack_id": row.attack_id, "speaker_id": row.speaker_id,
                                "background_type": background_type, "noise_id": noise_id if background_type == "recorded_noise" else -1,
                                "leading_seconds": leading, "trailing_seconds": 1 - leading,
                                "score": float(score), "threshold": predictor.operating_threshold,
                                "prediction": int(score >= predictor.operating_threshold),
                                "speech_sha256": hashlib.sha256(speech.tobytes()).hexdigest()})
        if within_class_index == 0:
            for background_type, background in backgrounds.items():
                for leading in POSITIONS:
                    sf.write(output / f"example_class{row.label}_{background_type}_lead{leading:.1f}.wav",
                             make_variant(speech, background, leading), SAMPLE_RATE, subtype="FLOAT")
    paired, summary = summarize(pd.DataFrame(records))
    paired.to_csv(output / "placement_scores.csv", index=False)
    summary.to_csv(output / "placement_summary.csv", index=False)
    type_keys = ["model", "utt_id", "leading_seconds", "label"]
    zeros = paired[paired.background_type == "zeros"][type_keys + ["score", "prediction"]]
    noise = paired[paired.background_type == "recorded_noise"][type_keys + ["score", "prediction"]]
    type_pairs = noise.merge(zeros, on=type_keys, suffixes=("_noise", "_zeros"), validate="one_to_one")
    type_pairs["noise_minus_zero_score"] = type_pairs.score_noise - type_pairs.score_zeros
    type_pairs["prediction_changed"] = type_pairs.prediction_noise != type_pairs.prediction_zeros
    type_pairs.to_csv(output / "noise_vs_zeros_scores.csv", index=False)
    type_pairs.groupby(["model", "leading_seconds", "label"]).agg(
        clips=("utt_id", "size"),
        mean_noise_minus_zero_score=("noise_minus_zero_score", "mean"),
        prediction_change_rate=("prediction_changed", "mean"),
    ).reset_index().to_csv(output / "noise_vs_zeros_summary.csv", index=False)
    metadata = {"settings": vars(args), "speech_seconds": 3, "background_seconds": 1,
                "total_seconds": 4, "sample_rate": SAMPLE_RATE,
                "evaluation_clips": len(selected), "evaluation_class_counts": {"genuine": args.per_class, "synthetic": args.per_class},
                "thresholds": {k: v.operating_threshold for k, v in predictors.items()},
                "noise_selection": "Automatically detected non-speech edges of training bona fide audio. Listen to saved donors to inspect quality.",
                "interpretation": "Controlled input sensitivity audit. Model names identify training preprocessing. All six constructed variants are scored directly without a second trimming pass. Eligibility requires three seconds retained after edge trimming. Balanced sample results are separate from full-corpus benchmark metrics."}
    (output / "placement_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_summary(summary, output)
    print(summary.to_string(index=False), flush=True)
    print(f"Saved {len(records)} scores for {len(selected)} real evaluation clips to {output}.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--artifact-root", required=True, help="Folder containing checkpoints/raw and checkpoints/trimmed")
    parser.add_argument("--output-dir", default="results/placement")
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--noise-donors", type=int, default=5)
    parser.add_argument("--noise-dbfs", type=float, default=-50)
    parser.add_argument("--trim-top-db", type=float, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run(parser.parse_args())
