"""EchoVerify: Streamlit Audio Deepfake Screening Interface."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.predict import EchoVerifyPredictor


@st.cache_resource
def _get_cached_predictor(checkpoint_str: str) -> EchoVerifyPredictor:
    """Internal helper to cache loaded model predictors."""
    return EchoVerifyPredictor(checkpoint_path=Path(checkpoint_str))


def load_predictor(checkpoint_path: str) -> Optional[EchoVerifyPredictor]:
    """Load the EchoVerify predictor without caching None when missing."""
    cp = Path(checkpoint_path)
    if not cp.exists():
        return None
    try:
        return _get_cached_predictor(str(cp.resolve()))
    except Exception as e:
        st.error(f"Failed to load checkpoint at `{checkpoint_path}`: {e}")
        return None


def plot_waveform_and_spectrogram(
    waveform: np.ndarray,
    spectrogram: np.ndarray,
    sr: int = 16000,
) -> plt.Figure:
    """Generate waveform and log-Mel spectrogram plots."""
    fig, (ax_wave, ax_spec) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=False)

    # Waveform plot
    time_axis = np.arange(len(waveform)) / sr
    ax_wave.plot(time_axis, waveform, color="#2B6CB0", linewidth=0.8)
    ax_wave.set_title("Standardized Audio Waveform (16 kHz Mono, 4.0s)", fontsize=11, fontweight="bold")
    ax_wave.set_xlabel("Time (seconds)", fontsize=9)
    ax_wave.set_ylabel("Amplitude", fontsize=9)
    ax_wave.set_xlim(0, len(waveform) / sr)
    ax_wave.grid(True, linestyle="--", alpha=0.4)

    # Spectrogram plot
    cax = ax_spec.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[0, len(waveform) / sr, 0, sr / 2],
    )
    fig.colorbar(cax, ax=ax_spec, format="%+.0f dB", pad=0.02)
    ax_spec.set_title("Log-Mel Spectrogram (64 Mel Bands, n_fft=512, hop=160)", fontsize=11, fontweight="bold")
    ax_spec.set_xlabel("Time (seconds)", fontsize=9)
    ax_spec.set_ylabel("Frequency (Hz)", fontsize=9)

    plt.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(
        page_title="EchoVerify: Audio Deepfake Screening",
        page_icon="🎙️",
        layout="wide",
    )

    st.title("EchoVerify 🎙️")
    st.markdown(
        "**Robust Deep Learning for Audio Deepfake Screening: "
        "Silence-Shortcut Audit & Practical Detection**"
    )

    st.caption(
        "⚠️ **Screening Disclaimer:** EchoVerify is an experimental research and screening tool. "
        "Scores are uncalibrated model outputs and do not constitute definitive forensic certification. "
        "Always corroborate results with independent source and context verification."
    )

    # Sidebar settings
    st.sidebar.header("Model Configuration")
    checkpoint_default = "checkpoints/best_model.pt"
    checkpoint_input = st.sidebar.text_input("Checkpoint Path", value=checkpoint_default)

    predictor = load_predictor(checkpoint_input)

    if predictor is None:
        st.sidebar.warning(f"No checkpoint found at `{checkpoint_input}`.")
        st.info(
            f"ℹ️ **No trained checkpoint loaded (`{checkpoint_input}`).**\n\n"
            "To train a model checkpoint on ASVspoof 2019 LA, run:\n"
            "```bash\n"
            "python src/train.py --data-root /path/to/ASVspoof2019_LA --mode raw\n"
            "```\n"
            "Or run a fast smoke test:\n"
            "```bash\n"
            "python src/train.py --data-root /path/to/ASVspoof2019_LA --smoke-run\n"
            "```"
        )
    else:
        st.sidebar.success("Model checkpoint active")
        st.sidebar.markdown(f"**Operating Threshold:** `{predictor.operating_threshold:.4f}`")
        st.sidebar.markdown(f"**Preprocessing Mode:** `{predictor.prep_config.mode}`")
        st.sidebar.markdown(f"**Target Duration:** `{predictor.prep_config.target_duration_sec}s`")

    # Audio input
    st.subheader("1. Audio Input")
    uploaded_file = st.file_uploader(
        "Upload audio clip (WAV, FLAC, or MP3)",
        type=["wav", "flac", "mp3"],
        help="Audio is processed in memory and never retained on disk.",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        if len(file_bytes) == 0:
            st.error("Uploaded file is empty (0 bytes). Please upload a valid audio file.")
            return

        # Safe audio playback (handles unknown MIME types robustly)
        st.audio(file_bytes)
        st.write(f"📁 **File:** `{uploaded_file.name}` | **Size:** {len(file_bytes):,} bytes")

        if predictor is None:
            st.warning("Cannot run screening without a loaded model checkpoint. Please configure a valid checkpoint in the sidebar.")
            return

        # Run inference using shared predictor
        with st.spinner("Analyzing audio with EchoVerify..."):
            try:
                result = predictor.predict(file_bytes)
            except ValueError as ve:
                st.error(f"Audio processing error: {ve}")
                return
            except Exception as e:
                st.error(f"Unreadable audio file or unsupported encoding: {e}")
                return

        # Display Screening Results
        st.subheader("2. Screening Result")

        is_synth = result["is_synthetic"]
        score = result["raw_score"]
        threshold = result["operating_threshold"]
        margin = result["margin_from_threshold"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if is_synth:
                st.error(f"### {result['prediction'].upper()}")
            else:
                st.success(f"### {result['prediction'].upper()}")
        with col2:
            st.metric(
                label="Raw Synthetic Score",
                value=f"{score:.4f}",
                delta=f"{margin:+.4f} vs threshold",
                help="Uncalibrated model score in [0.0, 1.0]. Higher values indicate synthetic patterns.",
            )
        with col3:
            st.metric(
                label="Decision Threshold",
                value=f"{threshold:.4f}",
                help="Fixed operating threshold determined strictly on Development data.",
            )
        with col4:
            st.metric(
                label="Raw Logit",
                value=f"{result['logit']:.4f}",
            )

        # Context & Corroboration banner
        bm = result.get("benchmark_metrics")
        genuine_fpr_text = ""
        if bm and "genuine_speech_fpr" in bm:
            genuine_fpr_text = (
                f" On the development benchmark, this model's genuine-speech false-positive rate is "
                f"**{bm['genuine_speech_fpr'] * 100:.1f}%** (genuine clips flagged as synthetic)."
            )

        if is_synth:
            st.warning(
                f"🚨 **Screening Note:** The model flagged synthetic acoustic cues.{genuine_fpr_text} "
                "Do not rely solely on this automated score; verify contextual authenticity."
            )
        else:
            st.info(
                f"✅ **Screening Note:** The model classified this recording as genuine speech.{genuine_fpr_text} "
                "Synthetic attacks using unseen techniques or heavy compression may evade detection."
            )

        # Silence Analysis
        sil = result["silence_metadata"]
        with st.expander("🔍 Silence Edge Audit & Metadata", expanded=False):
            scol1, scol2, scol3, scol4 = st.columns(4)
            scol1.metric("Leading Silence", f"{sil['lead_silence_sec']:.3f} s")
            scol2.metric("Trailing Silence", f"{sil['trail_silence_sec']:.3f} s")
            scol3.metric("Edge Silence", f"{sil['edge_silence_sec']:.3f} s")
            scol4.metric("Active Speech", f"{sil['active_sec']:.3f} s")
            st.caption(
                "In ASVspoof 2019 LA, silence duration can act as an unintended shortcut. "
                "EchoVerify audits these silence edges."
            )

        # Visualizations
        st.subheader("3. Acoustic Visualizations")
        fig = plot_waveform_and_spectrogram(
            result["processed_waveform"],
            result["log_mel_spectrogram"],
            sr=predictor.prep_config.sample_rate,
        )
        st.pyplot(fig)
        plt.close(fig)

        # Benchmark Metrics
        if bm:
            with st.expander("📊 Benchmark Evidence (Dev Partition)", expanded=False):
                bcol1, bcol2, bcol3, bcol4 = st.columns(4)
                bcol1.metric("Equal Error Rate (EER)", f"{bm.get('eer')}")
                bcol2.metric("ROC-AUC", f"{bm.get('roc_auc')}")
                bcol3.metric("Accuracy", f"{bm.get('accuracy')}")
                bcol4.metric("Genuine Speech FPR", f"{bm.get('genuine_speech_fpr')}")
                st.json(bm)


if __name__ == "__main__":
    # Check if executed under Streamlit runner
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        has_context = get_script_run_ctx() is not None
    except Exception:
        has_context = False

    if not has_context:
        # User executed `python ui/app.py` directly (e.g. from IDE Run button)
        # Automatically launch under `streamlit run` for a seamless experience
        app_file = str(Path(__file__).resolve())
        print(f"Starting Streamlit server for EchoVerify: {app_file}")
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", app_file]))
    else:
        main()
