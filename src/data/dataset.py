"""Dataset utilities for the CNN-Transformer SER model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

try:
    from src.features.extract import (
        SR,
        extract_mel,
        extract_mfcc,
        load_audio,
        pad_or_truncate,
        sliding_window_starts,
    )
except ImportError:
    from features.extract import (
        SR,
        extract_mel,
        extract_mfcc,
        load_audio,
        pad_or_truncate,
        sliding_window_starts,
    )


N_CLASSES = 8
TARGET_FRAMES = 300
MEL_BINS = 128
MFCC_STACK_CHANNELS = 120
TRANSFORMER_FEATURES = 134
EPS = 1e-8


@dataclass(frozen=True)
class FeatureStats:
    mfcc_mean: np.ndarray
    mfcc_std: np.ndarray
    mel_mean: np.ndarray
    mel_std: np.ndarray


def split_indices(
    actor_ids: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split data with actors 23-24 reserved as speaker-disjoint test."""
    actor_ids = np.asarray(actor_ids)
    labels = np.asarray(labels)

    test_mask = np.isin(actor_ids, [23, 24])
    test_idx = np.flatnonzero(test_mask)
    train_val_idx = np.flatnonzero(~test_mask)

    target_val_count = int(round(0.15 * len(labels)))
    val_fraction = target_val_count / len(train_val_idx)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction,
        random_state=seed,
        stratify=labels[train_val_idx],
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def _np_delta(feature: np.ndarray) -> np.ndarray:
    return np.gradient(feature, axis=-1).astype(np.float32, copy=False)


def _compute_stats(
    mfcc_stack: np.ndarray,
    mel_spec: np.ndarray,
    train_indices: np.ndarray,
) -> FeatureStats:
    mfcc_train = mfcc_stack[train_indices]
    mel_train = mel_spec[train_indices]
    return FeatureStats(
        mfcc_mean=mfcc_train.mean(axis=(0, 2))[:, None].astype(np.float32),
        mfcc_std=(mfcc_train.std(axis=(0, 2))[:, None] + EPS).astype(np.float32),
        mel_mean=mel_train.mean(axis=(0, 2))[:, None].astype(np.float32),
        mel_std=(mel_train.std(axis=(0, 2))[:, None] + EPS).astype(np.float32),
    )


def _load_stats(
    norm_stats_path: str | Path | None,
    mfcc_stack: np.ndarray,
    mel_spec: np.ndarray,
    train_indices: np.ndarray,
) -> FeatureStats:
    saved_stats = _load_saved_stats(norm_stats_path)
    if saved_stats is not None:
        return saved_stats
    return _compute_stats(mfcc_stack, mel_spec, train_indices)


def _load_saved_stats(norm_stats_path: str | Path | None) -> FeatureStats | None:
    if norm_stats_path is None or not Path(norm_stats_path).exists():
        return None
    with np.load(norm_stats_path) as stats:
        return FeatureStats(
            mfcc_mean=stats["mfcc_mean"].astype(np.float32),
            mfcc_std=stats["mfcc_std"].astype(np.float32),
            mel_mean=stats["mel_mean"].astype(np.float32),
            mel_std=stats["mel_std"].astype(np.float32),
        )


def _normalize_feature(
    feature: np.ndarray,
    mean: np.ndarray | None,
    std: np.ndarray | None,
) -> np.ndarray:
    if mean is None or std is None:
        sample_mean = feature.mean(axis=-1, keepdims=True)
        sample_std = feature.std(axis=-1, keepdims=True) + EPS
        return ((feature - sample_mean) / sample_std).astype(np.float32)
    return ((feature - mean) / (std + EPS)).astype(np.float32)


class RAVDESSCNNTransformerDataset(Dataset):
    """NPZ-backed dataset for CNN-Transformer SER training.

    Returns:
        mel3: Tensor shaped (3, 128, 300)
        seq: Tensor shaped (300, 134)
        label: LongTensor scalar
    """

    _feature_cache: dict[str, dict[str, np.ndarray]] = {}

    def __init__(
        self,
        npz_path: str | Path,
        split: str,
        norm_stats_path: str | Path | None = None,
        augment: bool = False,
        augment_repeats: int = 5,
        seed: int = 42,
        indices: Iterable[int] | np.ndarray | None = None,
        stats_indices: Iterable[int] | np.ndarray | None = None,
    ) -> None:
        if split not in {"train", "val", "test", "custom"}:
            raise ValueError("split must be one of: train, val, test, custom")

        self.npz_path = str(npz_path)
        self.split = split
        self.augment = bool(augment)
        self.augment_repeats = max(1, int(augment_repeats)) if self.augment else 1

        features = self._load_features(self.npz_path)
        self.mfcc_stack = features["mfcc_stack"]
        self.mel_spec = features["mel_spec"]
        self.labels = features["labels"]
        self.actor_ids = features["actor_ids"]

        train_idx, val_idx, test_idx = split_indices(self.actor_ids, self.labels, seed=seed)
        if indices is None:
            split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
            self.base_indices = split_map[split]
            stat_idx = train_idx
        else:
            self.base_indices = np.asarray(list(indices), dtype=np.int64)
            stat_idx = (
                np.asarray(list(stats_indices), dtype=np.int64)
                if stats_indices is not None
                else self.base_indices
            )
        self.stats = _load_stats(
            norm_stats_path,
            self.mfcc_stack,
            self.mel_spec,
            stat_idx,
        )

    @classmethod
    def _load_features(cls, npz_path: str) -> dict[str, np.ndarray]:
        if npz_path not in cls._feature_cache:
            with np.load(npz_path) as data:
                cls._feature_cache[npz_path] = {key: data[key] for key in data.files}
        return cls._feature_cache[npz_path]

    @property
    def base_labels(self) -> np.ndarray:
        return self.labels[self.base_indices]

    @property
    def base_actor_ids(self) -> np.ndarray:
        return self.actor_ids[self.base_indices]

    def __len__(self) -> int:
        return len(self.base_indices) * self.augment_repeats

    def _base_item(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_idx = self.base_indices[item % len(self.base_indices)]

        mel = (self.mel_spec[base_idx].astype(np.float32) - self.stats.mel_mean) / self.stats.mel_std
        mel_delta = _np_delta(mel)
        mel_delta2 = _np_delta(mel_delta)
        mel3 = np.stack([mel, mel_delta, mel_delta2]).astype(np.float32, copy=False)

        mfcc = (
            self.mfcc_stack[base_idx].astype(np.float32) - self.stats.mfcc_mean
        ) / self.stats.mfcc_std
        aux = np.zeros((TRANSFORMER_FEATURES - MFCC_STACK_CHANNELS, TARGET_FRAMES), dtype=np.float32)
        seq = np.vstack([mfcc, aux]).T.astype(np.float32, copy=False)

        return (
            torch.from_numpy(mel3).float(),
            torch.from_numpy(seq).float(),
            torch.tensor(int(self.labels[base_idx]), dtype=torch.long),
        )

    def _add_noise_snr(self, tensor: torch.Tensor) -> torch.Tensor:
        snr_db = float(torch.empty(1).uniform_(15.0, 30.0).item())
        signal_power = tensor.pow(2).mean().clamp_min(EPS)
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        return tensor + torch.randn_like(tensor) * torch.sqrt(noise_power)

    def _time_stretch(self, mel3: torch.Tensor, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rate = float(torch.empty(1).uniform_(0.8, 1.2).item())
        stretched_frames = max(16, int(round(TARGET_FRAMES / rate)))

        mel = F.interpolate(
            mel3.unsqueeze(0),
            size=(MEL_BINS, stretched_frames),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        seq_t = F.interpolate(
            seq.T.unsqueeze(0),
            size=stretched_frames,
            mode="linear",
            align_corners=False,
        ).squeeze(0).T

        if stretched_frames >= TARGET_FRAMES:
            mel = mel[..., :TARGET_FRAMES]
            seq_t = seq_t[:TARGET_FRAMES]
        else:
            mel = F.pad(mel, (0, TARGET_FRAMES - stretched_frames))
            seq_t = F.pad(seq_t.T, (0, TARGET_FRAMES - stretched_frames)).T
        return mel, seq_t

    def _pitch_shift_approx(self, mel3: torch.Tensor, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        semitones = int(torch.randint(-2, 3, (1,)).item())
        if semitones == 0:
            return mel3, seq
        mel_bins_shift = int(round(semitones * 2))
        mfcc_shift = semitones
        mel3 = torch.roll(mel3, shifts=mel_bins_shift, dims=1)
        seq_mfcc = torch.roll(seq[:, :MFCC_STACK_CHANNELS], shifts=mfcc_shift, dims=1)
        seq = torch.cat([seq_mfcc, seq[:, MFCC_STACK_CHANNELS:]], dim=1)
        return mel3, seq

    def _spec_augment(self, mel3: torch.Tensor) -> torch.Tensor:
        time_width = int(torch.randint(0, 31, (1,)).item())
        if time_width > 0:
            start = int(torch.randint(0, TARGET_FRAMES - time_width + 1, (1,)).item())
            mel3[:, :, start : start + time_width] = 0.0

        freq_width = int(torch.randint(0, 21, (1,)).item())
        if freq_width > 0:
            start = int(torch.randint(0, MEL_BINS - freq_width + 1, (1,)).item())
            mel3[:, start : start + freq_width, :] = 0.0

        return mel3

    def _augment(self, mel3: torch.Tensor, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() < 0.5:
            mel3 = self._add_noise_snr(mel3)
            seq = self._add_noise_snr(seq)
        if torch.rand(1).item() < 0.3:
            mel3, seq = self._time_stretch(mel3, seq)
        if torch.rand(1).item() < 0.3:
            mel3, seq = self._pitch_shift_approx(mel3, seq)
        mel3 = self._spec_augment(mel3)
        return mel3, seq

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mel3, seq, label = self._base_item(item)
        if self.augment:
            mel3, seq = self._augment(mel3, seq)
        return mel3, seq, label


class SlidingWindowRAVDESSDataset(RAVDESSCNNTransformerDataset):
    """Raw-audio dataset that expands each utterance into overlapping windows."""

    def __init__(
        self,
        metadata_csv: str | Path,
        split: str,
        norm_stats_path: str | Path | None = None,
        audio_root: str | Path | None = None,
        window_duration: float = 3.0,
        hop_duration: float = 1.0,
        min_window_duration: float = 1.5,
        augment: bool = False,
        augment_repeats: int = 5,
        seed: int = 42,
        indices: Iterable[int] | np.ndarray | None = None,
    ) -> None:
        if split not in {"train", "val", "test", "custom"}:
            raise ValueError("split must be one of: train, val, test, custom")

        self.metadata_path = Path(metadata_csv)
        self.audio_root = Path(audio_root) if audio_root is not None else None
        self.metadata = pd.read_csv(self.metadata_path)
        self.labels = (self.metadata["emotion_id"].to_numpy(dtype=np.int64) - 1)
        self.actor_ids = self.metadata["actor"].to_numpy(dtype=np.int64)
        self.split = split
        self.augment = bool(augment)
        self.augment_repeats = max(1, int(augment_repeats)) if self.augment else 1
        self.window_duration = float(window_duration)
        self.hop_duration = float(hop_duration)
        self.min_window_duration = float(min_window_duration)
        self.window_samples = int(round(self.window_duration * SR))
        self.stats = _load_saved_stats(norm_stats_path)

        train_idx, val_idx, test_idx = split_indices(self.actor_ids, self.labels, seed=seed)
        if indices is None:
            split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
            utterance_indices = split_map[split]
        else:
            utterance_indices = np.asarray(list(indices), dtype=np.int64)

        self.utterance_indices = np.asarray(utterance_indices, dtype=np.int64)
        self.window_index: list[tuple[int, int]] = []
        for row_idx in self.utterance_indices:
            row = self.metadata.iloc[int(row_idx)]
            num_samples = int(round(float(row["duration"]) * SR))
            starts = sliding_window_starts(
                num_samples,
                sr=SR,
                window_duration=self.window_duration,
                hop_duration=self.hop_duration,
                min_window_duration=self.min_window_duration,
            )
            self.window_index.extend((int(row_idx), start) for start in starts)

        if not self.window_index:
            raise ValueError(f"No valid windows found for split={split}")

    @property
    def base_indices(self) -> np.ndarray:
        return np.arange(len(self.window_index), dtype=np.int64)

    @property
    def base_labels(self) -> np.ndarray:
        return np.asarray(
            [self.labels[row_idx] for row_idx, _ in self.window_index],
            dtype=np.int64,
        )

    @property
    def base_actor_ids(self) -> np.ndarray:
        return np.asarray(
            [self.actor_ids[row_idx] for row_idx, _ in self.window_index],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.window_index) * self.augment_repeats

    def _resolve_audio_path(self, path_value: str) -> Path:
        path = Path(path_value)
        candidates = [path]
        try:
            project_root = self.metadata_path.resolve().parents[2]
            candidates.append(project_root / path)
        except IndexError:
            pass
        if self.audio_root is not None:
            candidates.extend(
                [
                    self.audio_root / path.name,
                    self.audio_root / path.parent.name / path.name,
                ]
            )
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        raise FileNotFoundError(f"Audio file not found for metadata path: {path_value}")

    @lru_cache(maxsize=64)
    def _load_utterance(self, row_idx: int) -> np.ndarray:
        path = self._resolve_audio_path(str(self.metadata.iloc[row_idx]["path"]))
        y, _ = load_audio(path, sr=SR, trim_silence=False)
        return y

    def _base_item(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row_idx, start = self.window_index[item % len(self.window_index)]
        y = self._load_utterance(row_idx)
        window = y[start : start + self.window_samples]
        if len(window) < self.window_samples:
            window = np.pad(window, (0, self.window_samples - len(window)), mode="constant")

        mel = pad_or_truncate(extract_mel(window, SR), T=TARGET_FRAMES)
        if self.stats is None:
            mel = _normalize_feature(mel, None, None)
        else:
            mel = _normalize_feature(mel, self.stats.mel_mean, self.stats.mel_std)
        mel_delta = _np_delta(mel)
        mel_delta2 = _np_delta(mel_delta)
        mel3 = np.stack([mel, mel_delta, mel_delta2]).astype(np.float32)

        mfcc_stack = pad_or_truncate(extract_mfcc(window, SR), T=TARGET_FRAMES)
        if self.stats is None:
            mfcc = _normalize_feature(mfcc_stack, None, None)
        else:
            mfcc = _normalize_feature(
                mfcc_stack,
                self.stats.mfcc_mean,
                self.stats.mfcc_std,
            )
        aux = np.zeros(
            (TRANSFORMER_FEATURES - MFCC_STACK_CHANNELS, TARGET_FRAMES),
            dtype=np.float32,
        )
        seq = np.vstack([mfcc, aux]).T.astype(np.float32)

        return (
            torch.from_numpy(mel3).float(),
            torch.from_numpy(seq).float(),
            torch.tensor(int(self.labels[row_idx]), dtype=torch.long),
        )
