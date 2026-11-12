import streamlit as st


def main() -> None:
    """Render the placeholder EchoVerify screening interface."""
    st.set_page_config(page_title="EchoVerify", page_icon="🎙️", layout="centered")

    st.title("EchoVerify")
    st.subheader("Experimental Audio Deepfake Screening Tool")
    st.info(
        "Placeholder UI: model inference, silence-trimmed preprocessing, and "
        "waveform/spectrogram views will be wired here once models are trained."
    )

    uploaded_file = st.file_uploader(
        "Upload an audio clip (WAV or MP3)",
        type=["wav", "mp3"],
    )

    if uploaded_file is not None:
        st.audio(uploaded_file)
        st.write(f"Selected file: `{uploaded_file.name}` ({uploaded_file.size} bytes)")
        st.warning("Model checkpoint not yet connected.")

    st.caption(
        "Disclaimer: EchoVerify is an experimental research and screening tool, "
        "not a forensic certification or definitive determination."
    )


if __name__ == "__main__":
    main()

