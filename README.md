# EchoVerify

**Robust Deep Learning for Audio Deepfake Detection: Silence-Shortcut Audit, Compression Robustness, and Practical Screening**

EchoVerify investigates how audio deepfake detectors (log-Mel spectrogram CNNs and pretrained `wav2vec 2.0` models) behave on **ASVspoof 2019 Logical Access (LA)** and **ASVspoof 2021 Deepfake (DF)**, specifically measuring leading/trailing silence shortcuts, compression robustness (MP3 64/128 kb/s), and layer-wise representation shifts.

> **Scope:** EchoVerify is an experimental research and screening tool—not real-time call protection, speaker verification, or forensic certification.

## Repository Structure

```text
EchoVerify/
├── data/          # Local datasets & manifests (gitignored; never pushed to GitHub)
├── docs/          # Documentation, experiment notes, and architecture references
├── notebooks/     # Exploratory analysis, probing, and visualization notebooks
├── results/       # Evaluation metrics, tables, and generated plots
├── src/           # Preprocessing, dataset manifests, models, training, and eval code
├── ui/            # Streamlit screening application
│   └── app.py     # Placeholder Streamlit interface
├── .gitignore
├── requirements.txt
└── README.md
```

## Quick Setup

1. **Create and activate a virtual environment** (Python 3.10+ recommended):
   ```bash
   python -m venv .venv
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # Linux / macOS
   source .venv/bin/activate
   ```

2. **Install dependencies** (also ensure `ffmpeg` is installed on your system for MP3 encoding/decoding):
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the placeholder Streamlit UI**:
   ```bash
   streamlit run ui/app.py
   ```
