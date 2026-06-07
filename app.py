"""Polished Gradio demo for Speech Emotion Recognition."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

from inference import EMOTIONS, extract_mel, load_model, predict, preprocess_audio


PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOJI = {
    "neutral": "😐",
    "calm": "😌",
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "fearful": "😨",
    "disgust": "🤢",
    "surprised": "😲",
}

POSITIVE_COLORS = {
    "neutral": "#66BB6A",
    "calm": "#4CAF50",
    "happy": "#2E7D32",
    "surprised": "#81C784",
}
NEGATIVE_COLORS = {
    "angry": "#F44336",
    "fearful": "#FF7043",
    "disgust": "#D84315",
    "sad": "#E57373",
}
COLORS = {**POSITIVE_COLORS, **NEGATIVE_COLORS}


try:
    MODEL = load_model(device=DEVICE)
    MODEL_ERROR = None
except Exception as exc:  # Keep the UI available with a helpful runtime message.
    MODEL = None
    MODEL_ERROR = str(exc)
    print(f"Model load warning: {MODEL_ERROR}")


def _empty_figure(title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=12)
    ax.set_axis_off()
    plt.tight_layout()
    return fig


def plot_waveform(audio_path: str | Path | None) -> plt.Figure:
    if audio_path is None:
        return _empty_figure("Upload or record audio to view waveform")
    y, sr = preprocess_audio(audio_path)
    fig, ax = plt.subplots(figsize=(7, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color="#3949AB")
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    return fig


def plot_mel_spectrogram(audio_path: str | Path | None) -> plt.Figure:
    if audio_path is None:
        return _empty_figure("Upload or record audio to view Mel spectrogram")
    y, sr = preprocess_audio(audio_path)
    mel_stack = extract_mel(y, sr)
    fig, ax = plt.subplots(figsize=(7, 3))
    img = librosa.display.specshow(
        mel_stack[0],
        sr=sr,
        hop_length=160,
        x_axis="time",
        y_axis="mel",
        fmax=8000,
        cmap="viridis",
        ax=ax,
    )
    ax.set_title("Log-Mel Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    return fig


def plot_probabilities(probabilities: dict[str, float]) -> plt.Figure:
    values = np.asarray([probabilities[emotion] for emotion in EMOTIONS], dtype=np.float32)
    predicted_index = int(values.argmax())
    predicted = EMOTIONS[predicted_index]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_positions = np.arange(len(EMOTIONS))
    bars = ax.barh(
        y_positions,
        values * 100.0,
        color=[COLORS[emotion] for emotion in EMOTIONS],
        alpha=0.86,
    )
    bars[predicted_index].set_edgecolor("#111111")
    bars[predicted_index].set_linewidth(2.8)

    labels = [f"{EMOJI[emotion]} {emotion}" for emotion in EMOTIONS]
    ax.set_yticks(y_positions, labels=labels)
    for tick, emotion in zip(ax.get_yticklabels(), EMOTIONS):
        if emotion == predicted:
            tick.set_fontweight("bold")

    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Emotion probabilities")
    ax.grid(axis="x", alpha=0.18)
    ax.bar_label(bars, labels=[f"{value * 100:.1f}%" for value in values], padding=4)
    plt.tight_layout()
    return fig


def format_prediction(probabilities: dict[str, float]) -> str:
    emotion = max(probabilities, key=probabilities.get)
    confidence = probabilities[emotion] * 100.0
    return (
        "<div class='prediction-card'>"
        f"<span class='prediction-label'>Predicted:</span> "
        f"<span class='prediction-emotion'>{EMOJI[emotion]} {emotion.upper()}</span> "
        f"<span class='prediction-confidence'>({confidence:.1f}%)</span>"
        "</div>"
    )


def format_top3(probabilities: dict[str, float]) -> str:
    top3 = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:3]
    lines = [
        f"<div class='top-row'><b>{rank}. {EMOJI[emotion]} {emotion.title()}</b>"
        f"<span>{score * 100:.1f}%</span></div>"
        for rank, (emotion, score) in enumerate(top3, start=1)
    ]
    return "<div class='top3-box'>" + "".join(lines) + "</div>"


def analyze(audio_path: str | None):
    waveform = plot_waveform(audio_path)
    mel_plot = plot_mel_spectrogram(audio_path)

    if MODEL_ERROR is not None or MODEL is None:
        error = MODEL_ERROR or "Model is not loaded."
        empty_probs = {emotion: 0.0 for emotion in EMOTIONS}
        return (
            waveform,
            mel_plot,
            f"<div class='prediction-card error'>Model unavailable: {error}</div>",
            plot_probabilities(empty_probs),
            "<div class='top3-box'>Prediction unavailable.</div>",
        )

    try:
        probabilities = predict(audio_path, model=MODEL, device=DEVICE)
    except Exception as exc:
        empty_probs = {emotion: 0.0 for emotion in EMOTIONS}
        return (
            waveform,
            mel_plot,
            f"<div class='prediction-card error'>Inference error: {exc}</div>",
            plot_probabilities(empty_probs),
            "<div class='top3-box'>Prediction unavailable.</div>",
        )

    return (
        waveform,
        mel_plot,
        format_prediction(probabilities),
        plot_probabilities(probabilities),
        format_top3(probabilities),
    )


def find_examples() -> list[list[str]]:
    example_paths: list[str] = []
    for actor in ["Actor_23", "Actor_24"]:
        actor_dir = PROJECT_ROOT / "data/raw/ravdess" / actor
        if actor_dir.exists():
            example_paths.extend(str(path) for path in sorted(actor_dir.glob("*.wav"))[:2])
    return [[path] for path in example_paths[:3]]


CSS = """
.app-title {
    font-size: 2.1rem;
    font-weight: 800;
    margin-bottom: 0.1rem;
}
.app-subtitle {
    color: #666;
    font-size: 1rem;
    margin-bottom: 1rem;
}
.prediction-card {
    border-radius: 12px;
    padding: 18px 20px;
    background: linear-gradient(135deg, #fff7e6 0%, #f4fbff 100%);
    border: 1px solid #e6e6e6;
    font-size: 1.25rem;
}
.prediction-label {
    color: #555;
    font-weight: 600;
}
.prediction-emotion {
    font-size: 1.65rem;
    font-weight: 900;
    letter-spacing: 0.02em;
}
.prediction-confidence {
    color: #333;
    font-weight: 700;
}
.error {
    color: #b00020;
    background: #fff0f0;
}
.top3-box {
    border: 1px solid #e8e8e8;
    border-radius: 10px;
    padding: 12px 14px;
    background: #ffffff;
}
.top-row {
    display: flex;
    justify-content: space-between;
    padding: 7px 0;
    border-bottom: 1px solid #f0f0f0;
}
.top-row:last-child {
    border-bottom: 0;
}
"""


with gr.Blocks(theme=gr.themes.Soft(), css=CSS) as demo:
    gr.HTML(
        "<div class='app-title'>🎙️ Speech Emotion Recognition</div>"
        "<div class='app-subtitle'>CNN-Transformer · RAVDESS Dataset</div>"
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Upload / Record",
            )
            analyze_button = gr.Button("Analyze Emotion", variant="primary")
            examples = find_examples()
            if examples:
                gr.Examples(examples=examples, inputs=audio_input, label="Example test audio")

        with gr.Column(scale=2):
            waveform_plot = gr.Plot(label="Waveform Plot")
            mel_plot = gr.Plot(label="Mel Spectrogram Plot")

    prediction_html = gr.HTML(
        "<div class='prediction-card'>Predicted: waiting for audio...</div>"
    )
    probability_plot = gr.Plot(label="Emotion probability chart")
    top3_html = gr.HTML("<div class='top3-box'>Top 3 predictions will appear here.</div>")

    with gr.Accordion("Model Details", open=False):
        gr.Markdown(
            """
            - **Architecture:** Dual-stream CNN-Transformer
            - **Dataset:** RAVDESS (24 actors, 8 emotions, 1440 samples)
            - **Val Accuracy:** 72.22% | **Macro-F1:** 71.79%
            - **Training:** Mixup augmentation (alpha=0.4), AdamW, CosineAnnealingLR
            - **Speaker-disjoint test accuracy:** 48.33%
            """
        )

    analyze_button.click(
        fn=analyze,
        inputs=audio_input,
        outputs=[waveform_plot, mel_plot, prediction_html, probability_plot, top3_html],
    )


if __name__ == "__main__":
    demo.launch(share=True, debug=True)
