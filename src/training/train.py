"""Training helpers for the CNN-Transformer SER model."""

from __future__ import annotations

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


def class_weights_from_labels(labels: np.ndarray, n_classes: int = 8) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def maybe_data_parallel(model: nn.Module, device: torch.device) -> nn.Module:
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
        return nn.DataParallel(model)
    return model


def train_cnn_transformer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    save_path: str | Path,
    device: torch.device,
    max_epochs: int = 120,
    lr: float = 1e-4,
    patience: int = 15,
    label_smoothing: float = 0.1,
) -> tuple[nn.Module, dict[str, list[float] | int]]:
    model = maybe_data_parallel(model, device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=label_smoothing,
    )
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-6)

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

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_total = 0

        for mel3, seq, labels in train_loader:
            mel3 = mel3.to(device, non_blocking=True)
            seq = seq.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(mel3, seq)
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
            for mel3, seq, labels in val_loader:
                mel3 = mel3.to(device, non_blocking=True)
                seq = seq.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(mel3, seq)
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
                    "class_weights": class_weights.cpu(),
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
    return model, history


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_classes: int = 8,
) -> dict[str, float | np.ndarray]:
    model = model.to(device)
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []

    for mel3, seq, labels in loader:
        mel3 = mel3.to(device, non_blocking=True)
        seq = seq.to(device, non_blocking=True)
        logits = model(mel3, seq)
        y_pred.append(logits.argmax(dim=1).cpu().numpy())
        y_true.append(labels.numpy())

    true = np.concatenate(y_true)
    pred = np.concatenate(y_pred)
    return {
        "accuracy": accuracy_score(true, pred),
        "macro_f1": f1_score(true, pred, average="macro"),
        "weighted_f1": f1_score(true, pred, average="weighted"),
        "confusion_matrix": confusion_matrix(true, pred, labels=np.arange(n_classes)),
    }


def plot_training_curves(history: dict[str, list[float] | int], title: str = "CNN-Transformer") -> None:
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


def confusion_matrix_frame(cm: np.ndarray, labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(cm, index=labels, columns=labels)
