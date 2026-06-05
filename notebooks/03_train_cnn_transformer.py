# %%
"""Single-cell Kaggle runner for P2 CNN-Transformer SER training."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

INPUT_DIR = Path("/kaggle/input/datasets/nhatphatnguyen/ser-ravdess-features")
SRC_DIR = INPUT_DIR / "src"
sys.path.append(str(SRC_DIR))

from data.dataset import RAVDESSCNNTransformerDataset  # noqa: E402
from models.cnn_transformer_model import CNNTransformerSER, count_parameters  # noqa: E402
from training.train import (  # noqa: E402
    class_weights_from_labels,
    confusion_matrix_frame,
    evaluate_model,
    plot_training_curves,
    train_cnn_transformer,
    unwrap_model,
)

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
NPZ_PATH = INPUT_DIR / "ravdess_features.npz"
NORM_STATS_PATH = INPUT_DIR / "norm_stats.npz"
MODEL_DIR = Path("/kaggle/working/models")
SAVE_PATH = MODEL_DIR / "p2_cnn_transformer_best.pt"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


seed_everything()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"CUDA GPUs: {torch.cuda.device_count()}")

train_ds = RAVDESSCNNTransformerDataset(
    NPZ_PATH,
    split="train",
    norm_stats_path=NORM_STATS_PATH,
    augment=True,
    augment_repeats=3,
    seed=SEED,
)
val_ds = RAVDESSCNNTransformerDataset(
    NPZ_PATH,
    split="val",
    norm_stats_path=NORM_STATS_PATH,
    augment=False,
    seed=SEED,
)
test_ds = RAVDESSCNNTransformerDataset(
    NPZ_PATH,
    split="test",
    norm_stats_path=NORM_STATS_PATH,
    augment=False,
    seed=SEED,
)

print(f"Split sizes: train={len(train_ds)} ({len(train_ds.base_indices)} base), val={len(val_ds)}, test={len(test_ds)}")
print("Train/val actor leakage check:", set(train_ds.base_actor_ids) & {23, 24}, set(val_ds.base_actor_ids) & {23, 24})
mel3, seq, label = train_ds[0]
print("Sample:", tuple(mel3.shape), tuple(seq.shape), int(label))

loader_kwargs = {
    "num_workers": 2,
    "pin_memory": torch.cuda.is_available(),
}
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=False, **loader_kwargs)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, drop_last=False, **loader_kwargs)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, drop_last=False, **loader_kwargs)

model = CNNTransformerSER(n_classes=N_CLASSES)
print(f"Trainable parameters: {count_parameters(model):,}")
with torch.no_grad():
    logits = model(
        torch.randn(2, 3, 128, 300),
        torch.randn(2, 300, 134),
    )
print("Smoke-test logits:", tuple(logits.shape))

class_weights = class_weights_from_labels(train_ds.base_labels, n_classes=N_CLASSES)
model, history = train_cnn_transformer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    class_weights=class_weights,
    save_path=SAVE_PATH,
    device=device,
    max_epochs=120,
    lr=1e-4,
    patience=15,
    label_smoothing=0.1,
)

checkpoint = torch.load(SAVE_PATH, map_location=device)
unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])

val_metrics = evaluate_model(model, val_loader, device=device, n_classes=N_CLASSES)
test_metrics = evaluate_model(model, test_loader, device=device, n_classes=N_CLASSES)

print("\nValidation metrics")
print(f"Accuracy:    {val_metrics['accuracy']:.4f}")
print(f"Macro-F1:    {val_metrics['macro_f1']:.4f}")
print(f"Weighted-F1: {val_metrics['weighted_f1']:.4f}")
print("\nValidation confusion matrix")
val_cm = confusion_matrix_frame(val_metrics["confusion_matrix"], EMOTION_NAMES)
display(val_cm)

print("\nSpeaker-disjoint test metrics")
print(f"Accuracy:    {test_metrics['accuracy']:.4f}")
print(f"Macro-F1:    {test_metrics['macro_f1']:.4f}")
print(f"Weighted-F1: {test_metrics['weighted_f1']:.4f}")

plot_training_curves(history, title="P2 CNN-Transformer")

plt.figure(figsize=(8, 6))
sns.heatmap(val_cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Validation Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
