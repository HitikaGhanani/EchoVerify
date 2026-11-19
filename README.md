# EchoVerify

EchoVerify is an audio deepfake screening framework and research toolkit focused on detecting synthetic speech while auditing acoustic shortcut vulnerabilities in countermeasure systems.

The repository provides end-to-end tooling for training, evaluating, and auditing deep learning classifiers on the ASVspoof 2019 Logical Access (LA) benchmark. Beyond standard classification, EchoVerify investigates whether detectors learn robust speech artifacts or overfit to peripheral cues such as boundary silence durations, padding strategies, and background placement.


## Central Research Question: Background Placement and Silence Shortcuts

Speech anti-spoofing benchmarks like ASVspoof 2019 LA evaluate models under standardized conditions. However, fixed-duration neural classifiers typically require cropping or padding recordings to a uniform input length (e.g., 4.0 seconds). This requirement introduces two potential confounding factors:

1. **Silence Duration as a Shortcut:** Due to synthesis and recording pipeline discrepancies, synthetic clips and bona fide recordings frequently exhibit different distributions of leading and trailing silence. When models exploit silence duration instead of spectral anomalies in speech, apparent benchmark performance collapses in real-world deployment where silence boundaries vary.
2. **Background Placement and Padding Sensitivity:** When an active speech segment is placed at different temporal offsets within the model's receptive window (e.g., shifted early with 0.1s leading background vs. centered at 0.5s vs. late at 0.9s), or padded with artificial zeros instead of real background acoustic noise, does the classification decision remain invariant?

EchoVerify directly audits these effects:
- A silence-only logistic regression baseline measures how much predictive power resides solely in leading and trailing silence lengths.
- Controlled models trained on raw audio versus silence-trimmed audio are evaluated on held-out test data.
- A paired background-placement audit tests whether shifting speech position or altering background noise characteristics flips model decisions on identical active speech segments.

## Key Experimental Results

All reported results derive from deterministic runs (seed 42) evaluated on UF HiPerGator. Checkpoint artifacts, manifests, configuration files, and metrics are preserved in `results/`.

### 1. Official Held-Out Evaluation (71,237 Clips)

Models were trained on a stratified 2,000-clip training cohort and validated on a 1,000-clip development cohort to establish a fixed operating threshold (minimizing $|FPR - FNR|$ on Dev). The fixed threshold was then applied unconditionally to the full official Eval partition:

| Model Configuration | Preprocessing Mode | Dev Operating Threshold | Eval Accuracy | Eval ROC-AUC | Eval EER | Genuine Speech FPR | Spoof FNR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Raw Baseline** | Raw (4s window) | 0.4371 | 84.37% | 0.9423 | 12.26% | 5.22% | 16.83% |
| **Trimmed Model** | Silence-Trimmed (30 dB) | 0.3419 | 68.56% | 0.7583 | 30.86% | 29.49% | 31.66% |

When boundary silence is trimmed prior to feature extraction, apparent performance drops markedly (EER rises from 12.26% to 30.86%, and genuine false positives increase from 5.22% to 29.49%). This performance gap confirms that the raw model relies substantially on boundary artifacts present in the benchmark dataset.

### 2. Silence-Only Shortcut Baseline

A logistic regression classifier trained strictly on two scalar features—`lead_silence_sec` and `trail_silence_sec`—achieved:
- **Dev ROC-AUC:** 0.7665
- **Dev EER:** 34.85% (Threshold = 0.4366)
- **Dev Accuracy:** 65.50%

Silence duration alone carries significant discriminative signal in ASVspoof 2019 LA, confirming that unconstrained classifiers can exploit non-speech silence boundaries as an unintended heuristic.

### 3. Paired Background Placement Audit

Using 100 genuine and 100 synthetic Eval recordings (each with at least 3.0 seconds of speech), identical active speech was combined with 1.0 second of background across three temporal offsets (0.1s, 0.5s, 0.9s leading duration) under two background types (recorded bona fide room noise at -50 dBFS vs. zero-valued silence):

- **Decision Flips on Identical Speech:** Shifting the speech offset within the 4-second window caused up to a **34.0% decision flip rate** on genuine speech for the raw model, with individual clip prediction scores fluctuating by as much as **0.426** out of 1.0.
- **Noise vs. Zero Padding Discrepancy:** Substituting zero padding for recorded ambient noise caused genuine speech false positive rates to jump to 92.0% - 100.0%, indicating severe vulnerability to silence padding assumptions.

## Repository Structure

```text
EchoVerify/
├── checkpoints/              # Checkpoint directory (gitignored; weights reproduced via CLI)
├── data/                     # Local datasets & manifests (gitignored; never uploaded)
├── docs/                     # System documentation and visual evidence
├── notebooks/
│   ├── setup.ipynb           # Environment verification and ASVspoof EDA
│   ├── echoverify_training_eval.ipynb        # Walkthrough of pipeline and metrics
│   ├── echoverify_experiments.ipynb          # Source experiment notebook
│   └── echoverify_experiments_executed.ipynb # Executed notebook with verified outputs
├── results/                  # Real experimental evidence and benchmark outputs
│   ├── raw/                  # 3-epoch raw baseline results, manifests, curves, confusion matrices
│   ├── trimmed/              # 3-epoch trimmed baseline results, manifests, curves, matrices
│   ├── silence_baseline/     # Silence-only logistic regression metrics and manifests
│   ├── smoke/                # Fast smoke test run outputs
│   ├── placement/            # Background placement scores, summaries, and flip rate plots
│   ├── notebook_evidence/    # Evidence files, integration run outputs, and verified metrics
│   ├── logs/                 # Plain-text execution and evaluation logs
│   ├── environment_hipergator.txt # Frozen conda environment package list from HiPerGator
│   └── test_path_fix.patch   # Path portability patch
├── scripts/
│   └── train_hipergator.sbatch  # Slurm GPU batch submission script
├── src/                      # Modular application source code
│   ├── data/
│   │   ├── protocol.py       # ASVspoof 2019 protocol loader and path validator
│   │   ├── preprocessing.py  # 16 kHz mono conversion, silence boundaries, 4s cropping/padding
│   │   └── dataset.py        # PyTorch Dataset and deterministic stratified manifest creation
│   ├── models/
│   │   ├── cnn.py            # Log-Mel spectrogram CNN (64 Mel bands, n_fft=512, hop=160)
│   │   └── silence_baseline.py # Logistic regression silence shortcut baseline
│   ├── evaluation/
│   │   ├── metrics.py        # Metrics: Acc, Prec, Rec, F1, ROC-AUC, EER, Genuine FPR, Spoof FNR
│   │   └── visualizer.py     # Confusion matrix and training curve plotting
│   ├── train.py              # Training pipeline with checkpoint saving and dev threshold locking
│   ├── evaluate.py           # Evaluation pipeline for Dev and held-out Eval partitions
│   ├── predict.py            # Single-clip inference engine and shared prediction core
│   └── placement_experiment.py # Controlled paired placement audit runner
├── tests/
│   └── test_pipeline.py      # Automated regression and unit test suite
├── ui/
│   └── app.py                # Streamlit screening application
├── .gitignore                # Excludes audio, weights, virtual environments, and scratch runs
├── requirements.txt          # Python package dependencies
└── README.md
```
## Implemented vs. Planned Capabilities

### Current Implementation
- Full ASVspoof 2019 LA protocol parser with strict path validation and partition loading (Train, Dev, Eval).
- Shared, reproducible audio preprocessing with SoundFile: mono conversion, 16 kHz resampling, exact silence boundary estimation, and deterministic 4-second cropping/padding.
- PyTorch log-Mel spectrogram CNN with 4D batch-independent decibel normalization, adaptive average pooling, and binary logit output.
- Deterministic stratified manifest generation (seed 42) preserving class balance and attack distributions across raw and trimmed experiments.
- Evaluation metrics suite computing Accuracy, Precision, Recall, F1, ROC-AUC, Equal Error Rate (EER), Genuine-Speech FPR, and Spoof FNR.
- Strict threshold locking: decision thresholds are chosen on Dev data only (bisection minimization of $|FPR - FNR|$) and preserved at full 64-bit precision in checkpoint dictionaries.
- Silence-only shortcut baseline fitting a StandardScaler and LogisticRegression strictly on training silence lengths.
- Controlled paired background placement experiment auditing temporal shifts and noise-vs-zeros padding effects.
- Streamlit web interface with waveform and log-Mel visualizations, silence edge breakdown, uncalibrated score display, and explicit false-positive rate warnings.
- Automated regression test suite covering batch composition independence, silence boundary arithmetic, metric edge cases, feature contracts, and path portability.

### Planned / Future Roadmap
- Additional benchmark evaluation: validation across ASVspoof 2021 DF and in-the-wild audio datasets.
- Neural architecture expansion: benchmarking deeper backbones (RawNet2, AASIST, and self-supervised front-ends like WavLM) under the same placement audit protocol.
- Robust training augmentations: background noise mixing, random temporal jitter, and acoustic codec perturbations to penalize placement shortcuts during training.
- Empirical score calibration: Platt scaling or temperature scaling on validation logits prior to presentation.

## Setup and Environment
### Prerequisites
- Python 3.9 or higher (Python 3.11 recommended)

### Local Installation
```bash
# Clone the repository
git clone https://github.com/HitikaGhanani/EchoVerify.git
cd EchoVerify

# Create and activate a virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

If you plan to process MP3 files in addition to WAV and FLAC, ensure `ffmpeg` is installed and accessible on your system path.

## Execution Guide

### 1. Run Automated Test Suite
Run the regression tests to verify that feature extraction is independent of batch composition, silence boundary calculations preserve full frame coverage, decision thresholds maintain full precision, and path resolution works across checkout directories:
```bash
python tests/test_pipeline.py
```

### 2. Fast Smoke Run
Test the entire training and validation cycle on a small subset (24 train / 16 dev clips) in roughly 20-30 seconds:
```bash
python src/train.py --data-root /path/to/ASVspoof2019_LA --smoke-run
```

### 3. Model Training (Raw vs. Trimmed Modes)
Train the 3-epoch baseline models on the stratified cohorts (2,000 train / 1,000 dev clips) matching the paper's experiments:

```bash
# Experiment 1: Raw audio (preserves leading and trailing silence)
python src/train.py \
    --data-root /path/to/ASVspoof2019_LA \
    --mode raw \
    --epochs 3 \
    --batch-size 32 \
    --lr 0.001 \
    --seed 42 \
    --max-train-samples 2000 \
    --max-dev-samples 1000 \
    --device auto \
    --output-dir results/raw \
    --checkpoint-dir checkpoints/raw

# Experiment 2: Silence-trimmed audio (removes leading/trailing non-speech at 30 dB)
python src/train.py \
    --data-root /path/to/ASVspoof2019_LA \
    --mode trimmed \
    --trim-top-db 30.0 \
    --epochs 3 \
    --batch-size 32 \
    --lr 0.001 \
    --seed 42 \
    --max-train-samples 2000 \
    --max-dev-samples 1000 \
    --device auto \
    --output-dir results/trimmed \
    --checkpoint-dir checkpoints/trimmed
```

### 4. Benchmark Evaluation
Evaluate a trained checkpoint against either the Development partition or the full held-out Evaluation partition using the locked development threshold:

```bash
# Evaluate Raw checkpoint on held-out Eval partition (71,237 clips)
python src/evaluate.py \
    --checkpoint checkpoints/raw/best_model.pt \
    --split eval \
    --data-root /path/to/ASVspoof2019_LA \
    --output-dir results/raw/heldout

# Evaluate Trimmed checkpoint on held-out Eval partition
python src/evaluate.py \
    --checkpoint checkpoints/trimmed/best_model.pt \
    --split eval \
    --data-root /path/to/ASVspoof2019_LA \
    --output-dir results/trimmed/heldout
```

### 5. Silence Duration Baseline
Fit and evaluate the logistic regression baseline relying strictly on leading and trailing silence lengths:
```bash
python src/models/silence_baseline.py \
    --data-root /path/to/ASVspoof2019_LA \
    --max-train 2000 \
    --max-dev 1000 \
    --output-dir results/silence_baseline
```

### 6. Background Placement Experiment
Run the paired sensitivity audit measuring score stability across temporal offsets and background padding types:
```bash
python src/placement_experiment.py \
    --data-root /path/to/ASVspoof2019_LA \
    --artifact-root . \
    --output-dir results/placement \
    --per-class 100 \
    --noise-donors 5 \
    --noise-dbfs -50 \
    --trim-top-db 30 \
    --seed 42 \
    --device auto
```

### 7. Single-Clip CLI Prediction
Screen an individual audio file using a trained checkpoint:
```bash
python src/predict.py \
    --audio data/sample.wav \
    --checkpoint checkpoints/raw/best_model.pt
```

### 8. Interactive Streamlit Interface
Launch the local web screening tool:
```bash
streamlit run ui/app.py
```
Open `http://localhost:8501` in your browser. Users can enter the checkpoint path, upload an audio recording, and view screening predictions, raw synthetic likelihood scores, silence edge analyses, and acoustic spectrograms.

## Notebooks and Reproducibility

The repository includes both source and fully executed notebooks under `notebooks/`:

- `notebooks/echoverify_experiments_executed.ipynb`: A complete, self-contained execution record with 8 executed code cells and zero errors. It runs an end-to-end integration training pass on a small subset, then imports and analyzes the full 3-epoch baseline models (`raw` and `trimmed`), recomputes held-out metrics from saved per-clip predictions, executes the placement experiment, and audits training noise donors.
- **Distinction between Notebook Run and Baselines:** The integration training run inside the notebook is a lightweight 1-epoch smoke run (16 clips) designed to verify live execution within an interactive kernel. The full quantitative benchmark results (71,237 Eval clips) were generated by the 3-epoch baseline runs on UF HiPerGator using the CLI scripts and preserved in `results/`.
- `notebooks/setup.ipynb`: Interactive exploratory data analysis verifying the ASVspoof 2019 protocol structure, audio waveforms, spectrograms, and silence distributions.

### Checkpoint Reproduction Notice
Model weight files (`best_model.pt`) exceed recommended Git storage limits and are gitignored. Because training is fully deterministic, checkpoints can be reproduced exactly by running the training commands above with `--seed 42` and the manifests preserved in `results/raw/` and `results/trimmed/`.


## Author Details:
- Hitika Ghanani
- MS in Applied Data Science student at UF
