"""Feature extraction pipeline for RAVDESS speech emotion recognition."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd


SR = 16_000
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MFCC = 40
N_MELS = 128
TARGET_FRAMES = 300
EPS = 1e-8
N_CHROMA = 12
N_AUX_FEATURES = N_CHROMA + 2
TRANSFORMER_FEATURES = (N_MFCC * 3) + N_AUX_FEATURES


def load_audio(
    path: str | Path,
    sr: int = SR,
    top_db: int = 30,
    trim_silence: bool = True,
) -> tuple[np.ndarray, int]:
    """Load audio as mono 16 kHz with optional silence trimming."""
    y, loaded_sr = librosa.load(path, sr=sr, mono=True)
    if trim_silence:
        trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        if trimmed.size > 0:
            y = trimmed
    return y.astype(np.float32, copy=False), loaded_sr


def sliding_window_starts(
    num_samples: int,
    sr: int = SR,
    window_duration: float = 3.0,
    hop_duration: float = 1.0,
    min_window_duration: float = 1.5,
) -> list[int]:
    """Return window start samples, including a valid padded tail window."""
    window_samples = int(round(window_duration * sr))
    hop_samples = int(round(hop_duration * sr))
    min_samples = int(round(min_window_duration * sr))

    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("window_duration and hop_duration must be positive")
    if num_samples <= window_samples:
        return [0]

    starts = list(range(0, num_samples - window_samples + 1, hop_samples))
    if not starts:
        return [0]

    last_full_end = starts[-1] + window_samples
    if last_full_end < num_samples:
        tail_start = starts[-1] + hop_samples
        if num_samples - tail_start >= min_samples:
            starts.append(tail_start)
    return starts


def sliding_audio_windows(
    y: np.ndarray,
    sr: int = SR,
    window_duration: float = 3.0,
    hop_duration: float = 1.0,
    min_window_duration: float = 1.5,
) -> list[np.ndarray]:
    """Split arbitrary-length audio into fixed-size overlapping windows."""
    window_samples = int(round(window_duration * sr))
    windows: list[np.ndarray] = []
    for start in sliding_window_starts(
        len(y),
        sr=sr,
        window_duration=window_duration,
        hop_duration=hop_duration,
        min_window_duration=min_window_duration,
    ):
        window = y[start : start + window_samples]
        if len(window) < window_samples:
            window = np.pad(window, (0, window_samples - len(window)), mode="constant")
        windows.append(window.astype(np.float32, copy=False))
    return windows


def _ensure_min_frames(feature: np.ndarray, min_frames: int = 9) -> np.ndarray:
    if feature.shape[1] >= min_frames:
        return feature
    return np.pad(feature, ((0, 0), (0, min_frames - feature.shape[1])), mode="constant")


def extract_mfcc(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Extract 40 MFCCs plus delta and delta-delta features as (120, T)."""
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
    )
    mfcc = _ensure_min_frames(mfcc)
    delta = librosa.feature.delta(mfcc)
    delta_delta = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta, delta_delta]).astype(np.float32, copy=False)


def extract_mel(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Extract a 128-bin log-Mel spectrogram as (128, T)."""
    mel_power = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel_power, ref=np.max)
    return log_mel.astype(np.float32, copy=False)


def _match_frame_count(feature: np.ndarray, frames: int) -> np.ndarray:
    """Pad or crop a feature matrix to match a target frame count."""
    if feature.shape[1] == frames:
        return feature
    if feature.shape[1] > frames:
        return feature[:, :frames]
    return np.pad(feature, ((0, 0), (0, frames - feature.shape[1])), mode="constant")


def mel_stack_from_mel(mel_spec: np.ndarray) -> np.ndarray:
    """Build a 3-channel Mel stack: Mel, delta Mel, delta-delta Mel."""
    mel_spec = _ensure_min_frames(mel_spec)
    delta = librosa.feature.delta(mel_spec)
    delta_delta = librosa.feature.delta(mel_spec, order=2)
    return np.stack([mel_spec, delta, delta_delta]).astype(np.float32, copy=False)


def extract_mel_stack(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Extract log-Mel plus delta and delta-delta channels as (3, 128, T)."""
    return mel_stack_from_mel(extract_mel(y, sr))


def extract_transformer_features(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Extract MFCC stack plus Chroma, ZCR, and RMSE as (134, T)."""
    mfcc_stack = extract_mfcc(y, sr)
    frames = mfcc_stack.shape[1]

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
    )
    zcr = librosa.feature.zero_crossing_rate(
        y=y,
        frame_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
    )
    rmse = librosa.feature.rms(
        y=y,
        frame_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
    )

    extras = np.vstack(
        [
            _match_frame_count(chroma, frames),
            _match_frame_count(zcr, frames),
            _match_frame_count(rmse, frames),
        ]
    )
    return np.vstack([mfcc_stack, extras]).astype(np.float32, copy=False)


def add_gaussian_noise_snr(
    y: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add Gaussian noise at a target SNR in decibels."""
    rng = rng or np.random.default_rng()
    signal_power = np.mean(np.square(y)) + EPS
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=y.shape)
    return (y + noise).astype(np.float32, copy=False)


def augment_audio(
    y: np.ndarray,
    sr: int = SR,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply optional raw-audio augmentation before feature extraction."""
    rng = rng or np.random.default_rng()
    augmented = y.astype(np.float32, copy=True)

    if rng.random() < 0.5:
        augmented = add_gaussian_noise_snr(
            augmented,
            snr_db=float(rng.uniform(15.0, 30.0)),
            rng=rng,
        )
    if rng.random() < 0.3:
        augmented = librosa.effects.time_stretch(
            augmented,
            rate=float(rng.uniform(0.8, 1.2)),
        ).astype(np.float32, copy=False)
    if rng.random() < 0.3:
        augmented = librosa.effects.pitch_shift(
            augmented,
            sr=sr,
            n_steps=float(rng.uniform(-2.0, 2.0)),
        ).astype(np.float32, copy=False)

    return augmented


def pad_or_truncate(feat: np.ndarray, T: int = TARGET_FRAMES) -> np.ndarray:
    """Pad or truncate a feature matrix to a fixed number of time frames."""
    if feat.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {feat.shape}")

    channels, frames = feat.shape
    if frames == T:
        return feat.astype(np.float32, copy=False)
    if frames > T:
        return feat[:, :T].astype(np.float32, copy=False)

    padded = np.zeros((channels, T), dtype=np.float32)
    padded[:, :frames] = feat
    return padded


def _resolve_audio_path(path_value: str | Path, metadata_csv: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = metadata_csv.parent / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Audio file not found: {path_value}")


def batch_extract(
    metadata_csv: str | Path,
    out_path: str | Path,
    T: int = TARGET_FRAMES,
) -> Path:
    """Extract fixed-size MFCC and log-Mel features for all metadata rows."""
    metadata_path = Path(metadata_csv)
    metadata = pd.read_csv(metadata_path)

    mfcc_features: list[np.ndarray] = []
    mel_features: list[np.ndarray] = []
    labels: list[int] = []
    actor_ids: list[int] = []

    for row in metadata.itertuples(index=False):
        audio_path = _resolve_audio_path(row.path, metadata_path)
        y, sr = load_audio(audio_path)
        mfcc_features.append(pad_or_truncate(extract_mfcc(y, sr), T=T))
        mel_features.append(pad_or_truncate(extract_mel(y, sr), T=T))
        labels.append(int(row.emotion_id) - 1)
        actor_ids.append(int(row.actor))

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_file,
        mfcc_stack=np.stack(mfcc_features).astype(np.float32, copy=False),
        mel_spec=np.stack(mel_features).astype(np.float32, copy=False),
        labels=np.asarray(labels, dtype=np.int64),
        actor_ids=np.asarray(actor_ids, dtype=np.int64),
    )
    return out_file


def compute_and_save_normalizer(
    train_indices: Iterable[int] | np.ndarray,
    npz_path: str | Path,
) -> Path:
    """Compute per-channel normalization stats on training samples only."""
    feature_path = Path(npz_path)
    indices = np.asarray(list(train_indices), dtype=np.int64)
    if indices.size == 0:
        raise ValueError("train_indices must contain at least one sample index")

    with np.load(feature_path) as data:
        mfcc_train = data["mfcc_stack"][indices]
        mel_train = data["mel_spec"][indices]

    stats = {
        "mfcc_mean": mfcc_train.mean(axis=(0, 2), keepdims=False)[:, None].astype(np.float32),
        "mfcc_std": (mfcc_train.std(axis=(0, 2), keepdims=False)[:, None] + EPS).astype(
            np.float32
        ),
        "mel_mean": mel_train.mean(axis=(0, 2), keepdims=False)[:, None].astype(np.float32),
        "mel_std": (mel_train.std(axis=(0, 2), keepdims=False)[:, None] + EPS).astype(
            np.float32
        ),
        "train_indices": indices,
    }

    out_path = feature_path.with_name("norm_stats.npz")
    np.savez_compressed(out_path, **stats)
    return out_path


if __name__ == "__main__":
    features_path = batch_extract(
        "data/processed/metadata.csv",
        "data/processed/ravdess_features.npz",
    )
    print(f"Saved features to {features_path}")
