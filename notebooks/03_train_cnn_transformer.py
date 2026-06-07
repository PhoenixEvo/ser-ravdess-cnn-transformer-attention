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
from sklearn.model_selection import StratifiedKFold
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
    augment_repeats=5,
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
print(f"\nSpeaker-disjoint accuracy (actors 23-24): {test_metrics['accuracy']:.4f}")

plot_training_curves(history, title="P2 CNN-Transformer")

plt.figure(figsize=(8, 6))
sns.heatmap(val_cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Validation Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

print("\nStarting stratified 5-fold CV without speaker-disjoint constraint")
features = RAVDESSCNNTransformerDataset._load_features(str(NPZ_PATH))
all_labels = features["labels"]
all_indices = np.arange(len(all_labels))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_rows = []

for fold, (fold_train_idx, fold_val_idx) in enumerate(skf.split(all_indices, all_labels), start=1):
    print(f"\nFold {fold}/5")
    fold_train_ds = RAVDESSCNNTransformerDataset(
        NPZ_PATH,
        split="custom",
        norm_stats_path=None,
        augment=True,
        augment_repeats=5,
        seed=SEED,
        indices=fold_train_idx,
        stats_indices=fold_train_idx,
    )
    fold_val_ds = RAVDESSCNNTransformerDataset(
        NPZ_PATH,
        split="custom",
        norm_stats_path=None,
        augment=False,
        seed=SEED,
        indices=fold_val_idx,
        stats_indices=fold_train_idx,
    )

    fold_train_loader = DataLoader(
        fold_train_ds,
        batch_size=32,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    fold_val_loader = DataLoader(
        fold_val_ds,
        batch_size=32,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    fold_model = CNNTransformerSER(n_classes=N_CLASSES)
    fold_weights = class_weights_from_labels(fold_train_ds.base_labels, n_classes=N_CLASSES)
    fold_model, fold_history = train_cnn_transformer(
        model=fold_model,
        train_loader=fold_train_loader,
        val_loader=fold_val_loader,
        class_weights=fold_weights,
        save_path=MODEL_DIR / f"p2_cnn_transformer_fold{fold}_best.pt",
        device=device,
        max_epochs=120,
        lr=1e-4,
        patience=15,
        label_smoothing=0.1,
    )
    fold_metrics = evaluate_model(fold_model, fold_val_loader, device=device, n_classes=N_CLASSES)
    cv_rows.append(
        {
            "fold": fold,
            "accuracy": fold_metrics["accuracy"],
            "macro_f1": fold_metrics["macro_f1"],
            "weighted_f1": fold_metrics["weighted_f1"],
            "epochs": fold_history["epochs_trained"],
        }
    )
    print(
        f"Fold {fold} | "
        f"Accuracy {fold_metrics['accuracy']:.4f} | "
        f"Macro-F1 {fold_metrics['macro_f1']:.4f} | "
        f"Weighted-F1 {fold_metrics['weighted_f1']:.4f}"
    )
    torch.cuda.empty_cache()

cv_results = pd.DataFrame(cv_rows)
display(cv_results)
cv_mean = cv_results["accuracy"].mean()
cv_std = cv_results["accuracy"].std(ddof=1)
print(f"\n5-fold CV accuracy: {cv_mean:.4f} +/- {cv_std:.4f}")
print(f"Speaker-disjoint accuracy: {test_metrics['accuracy']:.4f}")
