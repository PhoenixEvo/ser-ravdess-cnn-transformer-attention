# %%
"""EDA and feature extraction for the RAVDESS SER project.

This file is a percent-cell notebook script. Open it in VS Code, JupyterLab
with Jupytext, or another editor that supports # %% notebook cells.
"""

from pathlib import Path
import sys

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def find_project_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "data/raw/ravdess").exists():
            return candidate
    raise FileNotFoundError("Could not find project root containing data/raw/ravdess")


PROJECT_ROOT = find_project_root()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.parse_ravdess import EMOTION_MAP, build_metadata  # noqa: E402
from features.extract import (  # noqa: E402
    batch_extract,
    compute_and_save_normalizer,
    extract_mel,
    extract_mfcc,
    load_audio,
    pad_or_truncate,
)

RAW_DIR = PROJECT_ROOT / "data/raw/ravdess"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
METADATA_CSV = PROCESSED_DIR / "metadata.csv"
FEATURES_NPZ = PROCESSED_DIR / "ravdess_features.npz"

sns.set_theme(style="whitegrid", context="notebook")

# %% [markdown]
# # Section A - EDA

# %%
if METADATA_CSV.exists():
    metadata = pd.read_csv(METADATA_CSV)
else:
    metadata = build_metadata(RAW_DIR, METADATA_CSV)

metadata.head()

# %%
emotion_order = [EMOTION_MAP[i] for i in sorted(EMOTION_MAP)]

plt.figure(figsize=(10, 5))
sns.countplot(data=metadata, x="emotion", order=emotion_order, color="#4C78A8")
plt.title("RAVDESS sample count per emotion")
plt.xlabel("Emotion")
plt.ylabel("Samples")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(8, 5))
sns.histplot(metadata["duration"], bins=25, kde=True, color="#59A14F")
plt.title("Audio duration distribution")
plt.xlabel("Duration (seconds)")
plt.ylabel("Samples")
plt.tight_layout()
plt.show()

# %%
gender_counts = metadata["gender"].value_counts().reindex(["male", "female"])
plt.figure(figsize=(5, 5))
plt.pie(
    gender_counts,
    labels=gender_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=["#4C78A8", "#F58518"],
)
plt.title("Male vs female balance")
plt.tight_layout()
plt.show()

# %%
duration_stats = metadata["duration"].agg(["min", "mean", "max"])
print(
    "Duration seconds: "
    f"min={duration_stats['min']:.3f}, "
    f"mean={duration_stats['mean']:.3f}, "
    f"max={duration_stats['max']:.3f}"
)

# %% [markdown]
# # Section B - Feature Visualization

# %%
samples = (
    metadata.sort_values(["emotion_id", "actor", "path"])
    .groupby("emotion_id", as_index=False)
    .head(1)
    .sort_values("emotion_id")
)

sample_features = []
for row in samples.itertuples(index=False):
    y, sr = load_audio(PROJECT_ROOT / row.path)
    mfcc_stack = pad_or_truncate(extract_mfcc(y, sr))
    mel_spec = pad_or_truncate(extract_mel(y, sr))
    sample_features.append(
        {
            "emotion": row.emotion,
            "path": row.path,
            "y": y,
            "sr": sr,
            "mfcc_stack": mfcc_stack,
            "mel_spec": mel_spec,
        }
    )

print("Shape check for one sample:")
print("mfcc_stack =", sample_features[0]["mfcc_stack"].shape)
print("mel_spec   =", sample_features[0]["mel_spec"].shape)

# %%
fig, axes = plt.subplots(4, 2, figsize=(14, 10))
for ax, item in zip(axes.ravel(), sample_features):
    librosa.display.waveshow(item["y"], sr=item["sr"], ax=ax, color="#4C78A8")
    ax.set_title(item["emotion"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
fig.suptitle("Waveform - one sample per emotion", y=1.02)
plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(4, 2, figsize=(14, 10))
for ax, item in zip(axes.ravel(), sample_features):
    im = ax.imshow(item["mfcc_stack"][:40], aspect="auto", origin="lower", cmap="magma")
    ax.set_title(item["emotion"])
    ax.set_xlabel("Frame")
    ax.set_ylabel("MFCC coefficient")
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="MFCC")
fig.suptitle("MFCC heatmaps - one sample per emotion", y=1.02)
plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(4, 2, figsize=(14, 10))
for ax, item in zip(axes.ravel(), sample_features):
    im = ax.imshow(item["mel_spec"], aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(item["emotion"])
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mel bin")
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="dB")
fig.suptitle("Log-Mel spectrograms - one sample per emotion", y=1.02)
plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(4, 2, figsize=(14, 10))
for ax, item in zip(axes.ravel(), sample_features):
    delta_block = item["mfcc_stack"][40:120]
    im = ax.imshow(delta_block, aspect="auto", origin="lower", cmap="coolwarm")
    ax.set_title(item["emotion"])
    ax.set_xlabel("Frame")
    ax.set_ylabel("Delta + delta-delta coefficient")
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="Value")
fig.suptitle("Delta and delta-delta MFCC - one sample per emotion", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Section C - Batch Extraction

# %%
features_path = batch_extract(METADATA_CSV, FEATURES_NPZ)
print(f"Saved batch features to: {features_path}")

# %%
train_indices = metadata.index[metadata["actor"] <= 22].to_numpy()
stats_path = compute_and_save_normalizer(train_indices, FEATURES_NPZ)
print(f"Saved normalizer stats to: {stats_path}")
print(f"Normalizer fit samples: {len(train_indices)}")

# %%
with np.load(FEATURES_NPZ) as features:
    print("NPZ summary")
    print("mfcc_stack:", features["mfcc_stack"].shape)
    print("mel_spec:  ", features["mel_spec"].shape)
    print("labels:    ", features["labels"].shape)
    print("actor_ids: ", features["actor_ids"].shape)

    label_names = pd.Series(features["labels"]).map(lambda label: EMOTION_MAP[int(label) + 1])
    label_distribution = label_names.value_counts().reindex(emotion_order)

print("\nLabel distribution:")
print(label_distribution)
