# EchoVerify

**Robust Deep Learning for Audio Deepfake Detection: Silence-Shortcut Audit, Compression Robustness, and Practical Screening**

EchoVerify investigates how audio deepfake detectors (log-Mel spectrogram CNNs and pretrained representations) perform on the **ASVspoof 2019 Logical Access (LA)** benchmark. In particular, EchoVerify provides tools to audit whether classifiers rely on non-speech shortcuts (leading and trailing silence duration), measures robustness when silence is trimmed, and provides an accessible, honest screening interface with development-calibrated error rates and genuine-speech false-positive reporting.

> **Scope & Limitations:** EchoVerify is an **experimental screening tool** designed to support researchers, journalists, and audio analysts. It is **not** a real-time telephony defense, speaker verification system, or definitive forensic certification tool. All scores are uncalibrated model outputs and must be corroborated with contextual evidence.

---

## Repository Structure

```text
EchoVerify/
├── checkpoints/              # Saved model weights, preprocessing configs, and operating thresholds
├── data/                     # Local datasets & manifests (gitignored; never committed)
├── docs/                     # System documentation, screenshots, and architecture diagrams
├── notebooks/
│   ├── setup.ipynb           # Environment verification, protocol checks, and silence EDA
│   └── echoverify_training_eval.ipynb  # End-to-end training and evaluation walkthrough
├── results/                  # Metrics JSON, per-clip CSV scores, confusion matrices, loss curves
├── scripts/
│   └── train_hipergator.sbatch  # Configurable Slurm GPU job template for HiPerGator
├── src/                      # Modular production source code
│   ├── data/
│   │   ├── protocol.py       # ASVspoof 2019 LA protocol parser & path validator
│   │   ├── preprocessing.py  # SoundFile loading, 16 kHz mono conversion, silence trimming, 4s crop/pad
│   │   └── dataset.py        # PyTorch Dataset and deterministic stratified manifest sampling (seed 42)
│   ├── models/
│   │   ├── cnn.py            # Log-Mel spectrogram CNN (64 Mel bands, n_fft=512, hop=160, binary logit)
│   │   └── silence_baseline.py  # Logistic regression shortcut baseline on silence durations
│   ├── evaluation/
│   │   ├── metrics.py        # Acc, Precision, Recall, F1, ROC-AUC, EER, Genuine-speech FPR
│   │   └── visualizer.py     # Diagnostic confusion matrix and training curve plotting
│   ├── train.py              # CLI training pipeline with checkpoint saving & dev threshold locking
│   ├── evaluate.py           # CLI evaluation on Dev or held-out Eval partitions
│   └── predict.py            # CLI single-clip inference engine and shared prediction core
├── tests/
│   └── test_pipeline.py      # Automated unit and integration test suite
├── ui/
│   └── app.py                # Streamlit screening application with acoustic & silence visualizations
├── .gitignore                # Excludes datasets, model weights, checkpoints, logs, and venvs
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Dataset Directory Layout (ASVspoof 2019 LA)

EchoVerify expects the official ASVspoof 2019 Logical Access directory structure. Set the `ASVSPOOF2019_LA_ROOT` environment variable or pass `--data-root`:

```text
ASVspoof2019_LA/
├── ASVspoof2019_LA_cm_protocols/
│   ├── ASVspoof2019.LA.cm.train.trn.txt
│   ├── ASVspoof2019.LA.cm.dev.trl.txt
│   └── ASVspoof2019.LA.cm.eval.trl.txt
├── ASVspoof2019_LA_train/
│   └── flac/
│       ├── LA_T_1000137.flac
│       └── ...
├── ASVspoof2019_LA_dev/
│   └── flac/
│       ├── LA_D_1000265.flac
│       └── ...
└── ASVspoof2019_LA_eval/
    └── flac/
        ├── LA_E_1000147.flac
        └── ...
```

---

## Environment Setup

### 1. Local Environment (CPU / CUDA)

```bash
# Clone and enter directory
cd EchoVerify

# Create and activate Python virtual environment
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Ensure `ffmpeg` is available on your system path if working with MP3 audio files.

---

## Quickstart & Verification Commands

### 1. Verify Pipeline with Unit Tests
Run the automated test suite verifying audio loading, silence edge detection, 4-second cropping/padding, stratified sampling, CNN forward passes, and threshold locking:
```bash
python tests/test_pipeline.py
```

### 2. Fast Smoke Run
Test the end-to-end training pipeline on a small subset (24 train / 16 dev clips) in under 30 seconds:
```bash
python src/train.py --data-root /path/to/ASVspoof2019_LA --smoke-run
```

### 3. Full Training (Raw vs. Silence-Trimmed Modes)
Train the compact log-Mel CNN (64 Mel bands, `n_fft=512`, `hop_length=160`) with class-weighted BCE loss:

```bash
# Experiment A: Raw Baseline (Untrimmed audio)
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

# Experiment B: Silence-Trimmed Audio (Removes leading & trailing silence shortcuts)
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
Evaluate a saved checkpoint against the Development partition or held-out Evaluation partition using the **fixed operating threshold** selected strictly during development:

```bash
# Evaluate on Development partition:
python src/evaluate.py \
    --checkpoint checkpoints/raw/best_model.pt \
    --split dev \
    --data-root /path/to/ASVspoof2019_LA \
    --output-dir results/raw

# Evaluate on Held-out Evaluation partition:
python src/evaluate.py \
    --checkpoint checkpoints/raw/best_model.pt \
    --split eval \
    --data-root /path/to/ASVspoof2019_LA \
    --output-dir results/raw
```

### 5. Single-Clip CLI Prediction
Screen any single audio file (WAV, FLAC, or MP3):
```bash
python src/predict.py \
    --audio data/sample.wav \
    --checkpoint checkpoints/best_model.pt
```

### 6. Silence Shortcut Baseline (Optional Logistic Regression)
Evaluate how well leading and trailing silence durations alone classify genuine vs. spoof audio:
```bash
python src/models/silence_baseline.py \
    --data-root /path/to/ASVspoof2019_LA \
    --max-train 2000 \
    --max-dev 1000 \
    --output-dir results
```

---

## Interactive Streamlit Interface

Launch the interactive screening interface:
```bash
streamlit run ui/app.py
```
Open `http://localhost:8501` in your browser.

### Capturing Interface Screenshots
1. Launch `streamlit run ui/app.py`.
2. Select or enter the path to `checkpoints/best_model.pt` in the sidebar.
3. Upload a sample audio clip (`.wav`, `.flac`, or `.mp3`).
4. Inspect the resulting prediction banner, decision threshold, silence edge breakdown, waveform, and log-Mel spectrogram.
5. Capture a screenshot using Windows Snipping Tool (`Win + Shift + S`) or macOS (`Cmd + Shift + 4`), and save it to `docs/ui_screening_screenshot.png`.

---

## Running on UF HiPerGator (Slurm GPU)

A pre-configured Slurm batch script is provided at [scripts/train_hipergator.sbatch](file:///c:/Development/Projects/EchoVerify/scripts/train_hipergator.sbatch).

1. Edit the header fields in `scripts/train_hipergator.sbatch`:
   - `#SBATCH --account=<YOUR_ACCOUNT>`
   - `#SBATCH --qos=<YOUR_QOS>`
   - `#SBATCH --mail-user=<YOUR_EMAIL>`
   - `export ASVSPOOF2019_LA_ROOT="/blue/<GROUP>/<USER>/data/ASVspoof2019_LA"`
2. Submit the job:
   ```bash
   sbatch scripts/train_hipergator.sbatch
   ```
3. Monitor progress:
   ```bash
   squeue -u $USER
   tail -f logs/slurm-*.out
   ```

---

## Evaluation Metrics & Threshold Locking

- **Positive Class:** Synthetic speech (spoof = 1); genuine speech (bona fide = 0).
- **Reported Metrics:**
  - Equal Error Rate (**EER**)
  - Receiver Operating Characteristic Area Under Curve (**ROC-AUC**)
  - Accuracy, Precision, Recall, F1 Score
  - **Genuine-Speech False Positive Rate (FPR):** $\frac{FP}{FP + TN}$ (measures the frequency of genuine human speech mistakenly flagged as synthetic).
- **Threshold Locking:** Decision thresholds are selected on **development data only** (minimizing $|FPR - FNR|$ for EER) and saved inside `best_model.pt`. Held-out evaluation and Streamlit inference use this frozen threshold to prevent threshold gaming.
