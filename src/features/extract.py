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


def load_audio(path: str | Path, sr: int = SR, top_db: int = 30) -> tuple[np.ndarray, int]:
    """Load audio as mono 16 kHz and trim leading/trailing silence."""
    y, loaded_sr = librosa.load(path, sr=sr, mono=True)
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    if trimmed.size == 0:
        trimmed = y
    return trimmed.astype(np.float32, copy=False), loaded_sr


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
