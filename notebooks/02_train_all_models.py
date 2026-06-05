# %%
"""Kaggle training notebook for RAVDESS SER models.

This percent-cell script is designed for Kaggle GPU notebooks. It expects:

    /kaggle/input/ser-ravdess-features/ravdess_features.npz
    /kaggle/input/ser-ravdess-features/norm_stats.npz
    /kaggle/input/ser-ravdess-features/metadata.csv
    /kaggle/input/ser-ravdess-features/src/
"""

# Cell 0 - Setup

from __future__ import annotations

import copy
import os
import random
import sys
from pathlib import Path

sys.path.append("/kaggle/input/ser-ravdess-features/src")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset


INPUT_DIR = Path("/kaggle/input/datasets/nhatphatnguyen/ser-ravdess-features") 
NPZ_PATH = INPUT_DIR / "ravdess_features.npz"
NORM_STATS_PATH = INPUT_DIR / "norm_stats.npz"
METADATA_CSV = INPUT_DIR / "metadata.csv"
MODEL_DIR = Path("/kaggle/working/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_CLASSES = 8
EMOTION_NAMES = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


seed_everything()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_COUNT = torch.cuda.device_count()

print(f"Device: {device}")
print(f"CUDA GPUs: {GPU_COUNT}")
print(f"Checkpoints: {MODEL_DIR}")

# %% [markdown]
# # Cell 1 - src/data/dataset.py

# %%
def split_indices(
    actor_ids: np.ndarray,
    labels: np.ndarray,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/val/test indices with actors 23-24 reserved for test."""
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


class RAVDESSDataset(Dataset):
    """RAVDESS tensor dataset backed by precomputed NPZ features."""

    _feature_cache: dict[str, dict[str, np.ndarray]] = {}
    _stats_cache: dict[str, dict[str, np.ndarray]] = {}

    def __init__(
        self,
        npz_path: str | Path,
        split: str,
        norm_stats_path: str | Path,
        augment: bool = False,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        self.npz_path = str(npz_path)
        self.norm_stats_path = str(norm_stats_path)
        self.split = split
        self.augment = augment and split == "train"

        features = self._load_features(self.npz_path)
        self.mfcc_stack = features["mfcc_stack"]
        self.mel_spec = features["mel_spec"]
        self.labels = features["labels"]
        self.actor_ids = features["actor_ids"]

        stats = self._load_stats(self.norm_stats_path)
        self.mfcc_mean = stats["mfcc_mean"].astype(np.float32)
        self.mfcc_std = stats["mfcc_std"].astype(np.float32)
        self.mel_mean = stats["mel_mean"].astype(np.float32)
        self.mel_std = stats["mel_std"].astype(np.float32)

        train_idx, val_idx, test_idx = split_indices(self.actor_ids, self.labels)
        split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
        self.indices = split_map[split]

    @classmethod
    def _load_features(cls, npz_path: str) -> dict[str, np.ndarray]:
        if npz_path not in cls._feature_cache:
            with np.load(npz_path) as data:
                cls._feature_cache[npz_path] = {key: data[key] for key in data.files}
        return cls._feature_cache[npz_path]

    @classmethod
    def _load_stats(cls, norm_stats_path: str) -> dict[str, np.ndarray]:
        if norm_stats_path not in cls._stats_cache:
            with np.load(norm_stats_path) as data:
                cls._stats_cache[norm_stats_path] = {key: data[key] for key in data.files}
        return cls._stats_cache[norm_stats_path]

    def __len__(self) -> int:
        return len(self.indices)

    def _augment(self, mel: torch.Tensor, mfcc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mel = mel.clone()
        mfcc = mfcc.clone()

        if torch.rand(1).item() < 0.5:
            mel = mel + 0.01 * torch.randn_like(mel)
            mfcc = mfcc + 0.01 * torch.randn_like(mfcc)

        if torch.rand(1).item() < 0.5:
            width = int(torch.randint(10, 41, (1,)).item())
            start = int(torch.randint(0, mel.shape[-1] - width + 1, (1,)).item())
            mel[..., start : start + width] = 0.0
            mfcc[..., start : start + width] = 0.0

        if torch.rand(1).item() < 0.5:
            mel_width = int(torch.randint(8, 25, (1,)).item())
            mel_start = int(torch.randint(0, mel.shape[-2] - mel_width + 1, (1,)).item())
            mel[:, mel_start : mel_start + mel_width, :] = 0.0

            mfcc_width = int(torch.randint(8, 25, (1,)).item())
            mfcc_start = int(torch.randint(0, mfcc.shape[-2] - mfcc_width + 1, (1,)).item())
            mfcc[mfcc_start : mfcc_start + mfcc_width, :] = 0.0

        return mel, mfcc

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self.indices[item]

        mfcc = (self.mfcc_stack[idx].astype(np.float32) - self.mfcc_mean) / self.mfcc_std
        mel = (self.mel_spec[idx].astype(np.float32) - self.mel_mean) / self.mel_std

        mfcc_tensor = torch.from_numpy(mfcc).float()
        mel_tensor = torch.from_numpy(mel).float().unsqueeze(0)
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        if self.augment:
            mel_tensor, mfcc_tensor = self._augment(mel_tensor, mfcc_tensor)

        return mel_tensor, mfcc_tensor, label


def make_loaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_dataset = RAVDESSDataset(NPZ_PATH, "train", NORM_STATS_PATH, augment=True)
    val_dataset = RAVDESSDataset(NPZ_PATH, "val", NORM_STATS_PATH, augment=False)

    loader_kwargs = {
        "num_workers": 2,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


with np.load(NPZ_PATH) as feature_file:
    print({key: feature_file[key].shape for key in feature_file.files})

train_idx, val_idx, test_idx = split_indices(
    RAVDESSDataset._load_features(str(NPZ_PATH))["actor_ids"],
    RAVDESSDataset._load_features(str(NPZ_PATH))["labels"],
)
print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

sample_dataset = RAVDESSDataset(NPZ_PATH, "train", NORM_STATS_PATH, augment=True)
sample_mel, sample_mfcc, sample_label = sample_dataset[0]
print("Sample shapes:", sample_mel.shape, sample_mfcc.shape, sample_label.shape)

# %% [markdown]
# # Cell 2 - src/models/cnn_branch.py

# %%
class CNNClassifier(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_fc = nn.Linear(128, 128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, n_classes)

    def extract_feature_map(self, mel_spec: torch.Tensor) -> torch.Tensor:
        return self.backbone(mel_spec)

    def get_features(
        self,
        mel_spec: torch.Tensor,
        mfcc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.extract_feature_map(mel_spec)
        x = self.pool(x).flatten(1)
        return self.relu(self.feature_fc(x))

    def forward(
        self,
        mel_spec: torch.Tensor,
        mfcc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.get_features(mel_spec)
        return self.classifier(self.dropout(features))

# %% [markdown]
# # Cell 3 - src/models/bilstm_branch.py

# %%
class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 120,
        hidden: int = 128,
        n_layers: int = 2,
        n_classes: int = N_CLASSES,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if n_layers > 1 else 0.0,
        )
        self.feature_fc = nn.Linear(hidden * 2, 256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(256, n_classes)

    def encode(self, mfcc: torch.Tensor) -> torch.Tensor:
        sequence = mfcc.transpose(1, 2)
        outputs, _ = self.lstm(sequence)
        return outputs

    def get_features(
        self,
        mel_spec: torch.Tensor | None = None,
        mfcc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mfcc is None:
            if mel_spec is None:
                raise ValueError("BiLSTMClassifier requires MFCC input")
            mfcc = mel_spec
        outputs = self.encode(mfcc)
        pooled = outputs.max(dim=1).values
        return self.relu(self.feature_fc(pooled))

    def forward(
        self,
        mel_spec: torch.Tensor,
        mfcc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.get_features(mel_spec, mfcc)
        return self.classifier(self.dropout(features))

# %% [markdown]
# # Cell 4 - src/models/attention.py

# %%
class TemporalSelfAttention(nn.Module):
    """Bahdanau-style soft attention over time."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        scores = self.v(torch.tanh(self.W(sequence))).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = torch.sum(sequence * weights, dim=1)
        return context


class TimeFrequencyAttention(nn.Module):
    """Channel-wise attention for a CNN time-frequency feature map."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        reduction = max(channels // 4, 1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        return feature_map * self.attention(feature_map)

# %% [markdown]
# # Cell 5 - src/models/hybrid_model.py

# %%
def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


class HybridSERModel(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.cnn_branch = CNNClassifier(n_classes=n_classes)
        self.tfa = TimeFrequencyAttention(channels=128)

        self.bilstm = nn.LSTM(
            input_size=120,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.temporal_attention = TemporalSelfAttention(hidden_dim=256)

        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )
        print(f"HybridSERModel total parameters: {count_parameters(self):,}")

    def cnn_features(self, mel_spec: torch.Tensor) -> torch.Tensor:
        feature_map = self.cnn_branch.extract_feature_map(mel_spec)
        feature_map = self.tfa(feature_map)
        pooled = self.cnn_branch.pool(feature_map).flatten(1)
        return self.cnn_branch.relu(self.cnn_branch.feature_fc(pooled))

    def bilstm_features(self, mfcc: torch.Tensor) -> torch.Tensor:
        sequence = mfcc.transpose(1, 2)
        outputs, _ = self.bilstm(sequence)
        return self.temporal_attention(outputs)

    def forward(self, mel_spec: torch.Tensor, mfcc: torch.Tensor) -> torch.Tensor:
        cnn_vec = self.cnn_features(mel_spec)
        bilstm_vec = self.bilstm_features(mfcc)
        fused = torch.cat([cnn_vec, bilstm_vec], dim=1)
        return self.classifier(fused)

# %% [markdown]
# # Cell 6 - src/training/train.py

# %%
def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def maybe_data_parallel(model: nn.Module) -> nn.Module:
    model = model.to(device)
    if GPU_COUNT > 1:
        print(f"Using DataParallel across {GPU_COUNT} GPUs")
        return nn.DataParallel(model)
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

    for mel_spec, mfcc, labels in loader:
        mel_spec = mel_spec.to(device, non_blocking=True)
        mfcc = mfcc.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(mel_spec, mfcc)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    patience: int,
    save_path: str | Path,
) -> dict[str, list[float] | int]:
    model = maybe_data_parallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    history: dict[str, list[float] | int] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "epochs_trained": 0,
    }
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(model, val_loader, criterion)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["epochs_trained"] = epoch

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(unwrap_model(model).state_dict())
            torch.save(
                {
                    "model_state_dict": best_state,
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "history": history,
                },
                save_path,
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        unwrap_model(model).load_state_dict(best_state)

    return history


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    model = model.to(device)
    model.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for mel_spec, mfcc, labels in loader:
        mel_spec = mel_spec.to(device, non_blocking=True)
        mfcc = mfcc.to(device, non_blocking=True)
        logits = model(mel_spec, mfcc)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")


def plot_history(history: dict[str, list[float] | int], title: str) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title(f"{title} loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Val")
    axes[1].set_title(f"{title} accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def smoke_test_forward(model: nn.Module, batch_size: int = 4) -> None:
    model = model.to(device)
    mel = torch.randn(batch_size, 1, 128, 300, device=device)
    mfcc = torch.randn(batch_size, 120, 300, device=device)
    logits = model(mel, mfcc)
    print(f"{model.__class__.__name__} logits:", tuple(logits.shape))


smoke_test_forward(CNNClassifier())
smoke_test_forward(BiLSTMClassifier())
smoke_test_forward(HybridSERModel())
torch.cuda.empty_cache()

# %% [markdown]
# # Cell 7 - Train Baseline B1 (CNN)

# %%
train_loader_32, val_loader_32 = make_loaders(batch_size=32)

b1_model = CNNClassifier(n_classes=N_CLASSES)
b1_params = count_parameters(b1_model)
b1_history = train_model(
    b1_model,
    train_loader_32,
    val_loader_32,
    epochs=50,
    lr=1e-3,
    patience=10,
    save_path=MODEL_DIR / "b1_cnn_best.pt",
)
plot_history(b1_history, "B1 CNN")
b1_acc, b1_f1 = evaluate_model(b1_model, val_loader_32)
print(f"B1 CNN | Val Accuracy: {b1_acc:.4f} | Val Macro-F1: {b1_f1:.4f}")
torch.cuda.empty_cache()

# %% [markdown]
# # Cell 8 - Train Baseline B2 (BiLSTM)

# %%
b2_model = BiLSTMClassifier(input_size=120, hidden=128, n_layers=2, n_classes=N_CLASSES)
b2_params = count_parameters(b2_model)
b2_history = train_model(
    b2_model,
    train_loader_32,
    val_loader_32,
    epochs=50,
    lr=1e-3,
    patience=10,
    save_path=MODEL_DIR / "b2_bilstm_best.pt",
)
plot_history(b2_history, "B2 BiLSTM")
b2_acc, b2_f1 = evaluate_model(b2_model, val_loader_32)
print(f"B2 BiLSTM | Val Accuracy: {b2_acc:.4f} | Val Macro-F1: {b2_f1:.4f}")
torch.cuda.empty_cache()

# %% [markdown]
# # Cell 9 - Train Main Model P1 (CNN-BiLSTM-Attention)

# %%
def class_weights_from_loader(train_loader: DataLoader) -> torch.Tensor:
    train_dataset = train_loader.dataset
    train_labels = train_dataset.labels[train_dataset.indices]
    class_counts = np.bincount(train_labels, minlength=N_CLASSES).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    weights = class_counts.sum() / (N_CLASSES * class_counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_p1_v2(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-4,
    patience: int = 15,
    save_path: str | Path = MODEL_DIR / "p1_hybrid_v2_best.pt",
) -> dict[str, list[float] | int]:
    model = maybe_data_parallel(model)
    criterion = nn.CrossEntropyLoss(weight=class_weights_from_loader(train_loader))
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=50,
        eta_min=1e-6,
    )

    history: dict[str, list[float] | int] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "epochs_trained": 0,
    }
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_total = 0

        for mel_spec, mfcc, labels in train_loader:
            mel_spec = mel_spec.to(device, non_blocking=True)
            mfcc = mfcc.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(mel_spec, mfcc)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = labels.size(0)
            train_loss_total += loss.item() * batch_size
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += batch_size

        train_loss = train_loss_total / train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for mel_spec, mfcc, labels in val_loader:
                mel_spec = mel_spec.to(device, non_blocking=True)
                mfcc = mfcc.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(mel_spec, mfcc)
                loss = criterion(logits, labels)

                batch_size = labels.size(0)
                val_loss_total += loss.item() * batch_size
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += batch_size

        val_loss = val_loss_total / val_total
        val_acc = val_correct / val_total
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["epochs_trained"] = epoch

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f} | "
            f"LR {scheduler.get_last_lr()[0]:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(unwrap_model(model).state_dict())
            torch.save(
                {
                    "model_state_dict": best_state,
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "history": history,
                    "class_weights": criterion.weight.detach().cpu(),
                },
                save_path,
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        unwrap_model(model).load_state_dict(best_state)

    return history


train_loader_p1, val_loader_p1 = make_loaders(batch_size=32)

p1_model = HybridSERModel(n_classes=N_CLASSES)
p1_params = count_parameters(p1_model)
p1_history = train_p1_v2(
    p1_model,
    train_loader_p1,
    val_loader_p1,
    epochs=100,
    lr=1e-4,
    patience=15,
    save_path=MODEL_DIR / "p1_hybrid_v2_best.pt",
)
plot_history(p1_history, "P1 Hybrid v2")
p1_model = maybe_data_parallel(p1_model)
checkpoint = torch.load(MODEL_DIR / "p1_hybrid_v2_best.pt", map_location=device)
unwrap_model(p1_model).load_state_dict(checkpoint["model_state_dict"])
p1_acc, p1_f1 = evaluate_model(p1_model, val_loader_p1)
print(f"P1 Hybrid v2 | Val Accuracy: {p1_acc:.4f} | Val Macro-F1: {p1_f1:.4f}")
torch.cuda.empty_cache()

# %% [markdown]
# # Cell 10 - Quick comparison table

# %%
comparison = pd.DataFrame(
    [
        {
            "Model": "B1 CNN",
            "Val Accuracy": b1_acc,
            "Val Macro-F1": b1_f1,
            "#Params": b1_params,
            "Epochs trained": b1_history["epochs_trained"],
        },
        {
            "Model": "B2 BiLSTM",
            "Val Accuracy": b2_acc,
            "Val Macro-F1": b2_f1,
            "#Params": b2_params,
            "Epochs trained": b2_history["epochs_trained"],
        },
        {
            "Model": "P1 Hybrid",
            "Val Accuracy": p1_acc,
            "Val Macro-F1": p1_f1,
            "#Params": p1_params,
            "Epochs trained": p1_history["epochs_trained"],
        },
    ]
)

comparison
