"""Inference pipeline for the CNN-Transformer speech emotion recognizer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.models.cnn_transformer_model import CNNTransformerSER  # noqa: E402


EMOTIONS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]

SR = 16_000
DURATION = 3.0
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MELS = 128
N_MFCC = 40
TARGET_FRAMES = 300
TRANSFORMER_FEATURES = 134
MFCC_STACK_CHANNELS = 120
EPS = 1e-8

DEFAULT_CHECKPOINTS = [
    PROJECT_ROOT / "best_model.pth",
    PROJECT_ROOT / "outputs/p2_cnn_transformer_best.pt",
]


def resolve_checkpoint(checkpoint_path: str | Path | None = None) -> Path:
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    for path in DEFAULT_CHECKPOINTS:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No checkpoint found. Expected best_model.pth or outputs/p2_cnn_transformer_best.pt"
    )


def preprocess_audio(path: str | Path, sr: int = SR, duration: float = DURATION) -> tuple[np.ndarray, int]:
    """Load mono audio, resample, trim silence, then pad/truncate to duration."""
    y, loaded_sr = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)

    target_len = int(sr * duration)
    if y.size >= target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - y.size), mode="constant")

    return y.astype(np.float32, copy=False), loaded_sr


def _pad_or_truncate(feature: np.ndarray, frames: int = TARGET_FRAMES) -> np.ndarray:
    if feature.shape[-1] == frames:
        return feature.astype(np.float32, copy=False)
    if feature.shape[-1] > frames:
        return feature[..., :frames].astype(np.float32, copy=False)
    pad_width = [(0, 0)] * feature.ndim
    pad_width[-1] = (0, frames - feature.shape[-1])
    return np.pad(feature, pad_width, mode="constant").astype(np.float32, copy=False)


def _safe_delta(feature: np.ndarray, order: int = 1) -> np.ndarray:
    if feature.shape[-1] < 9:
        feature = _pad_or_truncate(feature, frames=9)
    return librosa.feature.delta(feature, order=order).astype(np.float32, copy=False)


def _standardize_sample(feature: np.ndarray) -> np.ndarray:
    mean = feature.mean(axis=-1, keepdims=True)
    std = feature.std(axis=-1, keepdims=True) + EPS
    return (feature - mean) / std


def _load_npy_stats(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape[0] == 2:
        return data[0].astype(np.float32), data[1].astype(np.float32)
    if isinstance(data, np.ndarray) and data.shape == ():
        item = data.item()
        if isinstance(item, dict) and {"mean", "std"}.issubset(item):
            return item["mean"].astype(np.float32), item["std"].astype(np.float32)
    return None


def load_normalization_stats() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load normalizer stats, preferring explicit NPY files then existing NPZ."""
    stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    mel_npy = _load_npy_stats(PROJECT_ROOT / "mel_norm_stats.npy")
    mfcc_npy = _load_npy_stats(PROJECT_ROOT / "mfcc_norm_stats.npy")
    if mel_npy is not None:
        stats["mel"] = mel_npy
    if mfcc_npy is not None:
        stats["mfcc"] = mfcc_npy

    npz_path = PROJECT_ROOT / "data/processed/norm_stats.npz"
    if npz_path.exists():
        with np.load(npz_path) as npz:
            stats.setdefault("mel", (npz["mel_mean"].astype(np.float32), npz["mel_std"].astype(np.float32)))
            stats.setdefault("mfcc", (npz["mfcc_mean"].astype(np.float32), npz["mfcc_std"].astype(np.float32)))

    return stats


def normalize(tensor: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
    """Normalize with precomputed stats or per-sample standardization fallback."""
    if mean is None or std is None:
        return _standardize_sample(tensor).astype(np.float32, copy=False)
    return ((tensor - mean) / (std + EPS)).astype(np.float32, copy=False)


def _extract_log_mel(y: np.ndarray, sr: int) -> np.ndarray:
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
    return _pad_or_truncate(log_mel, TARGET_FRAMES)


def extract_mel(y: np.ndarray, sr: int) -> np.ndarray:
    """Return raw log-Mel, delta, and delta-delta stack shaped (3, 128, 300)."""
    log_mel = _extract_log_mel(y, sr)
    delta = _safe_delta(log_mel)
    delta2 = _safe_delta(log_mel, order=2)
    return np.stack([log_mel, delta, delta2]).astype(np.float32, copy=False)


def _build_mel_stream(y: np.ndarray, sr: int, stats: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    log_mel = _extract_log_mel(y, sr)
    mel_mean, mel_std = stats.get("mel", (None, None))
    mel = normalize(log_mel, mel_mean, mel_std)
    delta = _safe_delta(mel)
    delta2 = _safe_delta(mel, order=2)
    return np.stack([mel, delta, delta2]).astype(np.float32, copy=False)


def _extract_mfcc_stack(y: np.ndarray, sr: int) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
    )
    mfcc = _pad_or_truncate(mfcc, TARGET_FRAMES)
    delta = _safe_delta(mfcc)
    delta2 = _safe_delta(mfcc, order=2)
    return np.vstack([mfcc, delta, delta2]).astype(np.float32, copy=False)


def extract_mfcc(y: np.ndarray, sr: int) -> np.ndarray:
    """Return MFCC stream shaped (300, 134), matching training code."""
    mfcc_stack = _extract_mfcc_stack(y, sr)
    aux = np.zeros((TRANSFORMER_FEATURES - MFCC_STACK_CHANNELS, TARGET_FRAMES), dtype=np.float32)
    return np.vstack([mfcc_stack, aux]).T.astype(np.float32, copy=False)


def _build_mfcc_stream(y: np.ndarray, sr: int, stats: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    mfcc_stack = _extract_mfcc_stack(y, sr)
    mfcc_mean, mfcc_std = stats.get("mfcc", (None, None))
    mfcc = normalize(mfcc_stack, mfcc_mean, mfcc_std)
    aux = np.zeros((TRANSFORMER_FEATURES - MFCC_STACK_CHANNELS, TARGET_FRAMES), dtype=np.float32)
    return np.vstack([mfcc, aux]).T.astype(np.float32, copy=False)


def _clean_state_dict(checkpoint: Any, model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a valid state dict")

    stripped = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
        if isinstance(value, torch.Tensor)
    }

    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in stripped.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    skipped = sorted(set(stripped) - set(compatible))
    if skipped:
        print(f"Warning: skipped {len(skipped)} incompatible checkpoint tensors")
    if not compatible:
        raise RuntimeError("No compatible checkpoint tensors found for CNNTransformerSER")
    return compatible


def load_model(checkpoint_path: str | Path | None = None, device: torch.device | str | None = None) -> torch.nn.Module:
    """Load CNNTransformerSER, handling DataParallel and checkpoint wrappers."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CNNTransformerSER(n_classes=len(EMOTIONS))
    checkpoint = torch.load(resolve_checkpoint(checkpoint_path), map_location=device)
    compatible = _clean_state_dict(checkpoint, model)
    model_state = model.state_dict()
    model_state.update(compatible)
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()
    return model


def predict(
    audio_path: str | Path,
    model: torch.nn.Module | None = None,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """Run full preprocessing, feature extraction, normalization, and inference."""
    if audio_path is None:
        raise ValueError("Please upload or record an audio file first.")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if model is None:
        model = load_model(device=device)

    y, sr = preprocess_audio(audio_path, sr=SR, duration=DURATION)
    stats = load_normalization_stats()
    mel = _build_mel_stream(y, sr, stats)
    mfcc = _build_mfcc_stream(y, sr, stats)

    mel_tensor = torch.from_numpy(mel).unsqueeze(0).float().to(device)
    mfcc_tensor = torch.from_numpy(mfcc).unsqueeze(0).float().to(device)

    if tuple(mel_tensor.shape) != (1, 3, 128, 300):
        raise ValueError(f"mel_stream shape mismatch: {tuple(mel_tensor.shape)}")
    if tuple(mfcc_tensor.shape) != (1, 300, 134):
        raise ValueError(f"mfcc_stream shape mismatch: {tuple(mfcc_tensor.shape)}")

    model.eval()
    with torch.no_grad():
        logits = model(mel_tensor, mfcc_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    return {emotion: float(probabilities[index]) for index, emotion in enumerate(EMOTIONS)}
