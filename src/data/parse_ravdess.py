"""Utilities for parsing RAVDESS speech filenames and metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import wave

import pandas as pd


EMOTION_MAP = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}

INTENSITY_MAP = {
    1: "normal",
    2: "strong",
}

DEFAULT_METADATA_CSV = Path("data/processed/metadata.csv")


def parse_filename(path: str | Path) -> dict[str, Any]:
    """Parse a RAVDESS filename into metadata fields.

    Expected filename format: MM-VC-EE-II-SS-RR-AA.wav.
    """
    file_path = Path(path)
    if file_path.suffix.lower() != ".wav":
        raise ValueError(f"Expected a .wav file, got: {file_path.name}")

    parts = file_path.stem.split("-")
    if len(parts) != 7 or not all(part.isdigit() for part in parts):
        raise ValueError(
            "Expected RAVDESS filename format MM-VC-EE-II-SS-RR-AA.wav, "
            f"got: {file_path.name}"
        )

    modality, vocal_channel, emotion_id, intensity_id, statement, repetition, actor = (
        int(part) for part in parts
    )

    if modality != 3:
        raise ValueError(f"Expected audio-only modality 03, got {modality:02d}")
    if vocal_channel != 1:
        raise ValueError(f"Expected speech vocal channel 01, got {vocal_channel:02d}")
    if emotion_id not in EMOTION_MAP:
        raise ValueError(f"Unknown emotion id {emotion_id:02d} in {file_path.name}")
    if intensity_id not in INTENSITY_MAP:
        raise ValueError(f"Unknown intensity id {intensity_id:02d} in {file_path.name}")
    if not 1 <= actor <= 24:
        raise ValueError(f"Actor id must be in 01-24, got {actor:02d}")

    return {
        "emotion": EMOTION_MAP[emotion_id],
        "emotion_id": emotion_id,
        "actor": actor,
        "gender": "male" if actor % 2 == 1 else "female",
        "intensity": INTENSITY_MAP[intensity_id],
        "statement": statement,
        "repetition": repetition,
    }


def _portable_path(path: Path) -> str:
    """Return a stable path string that prefers repo-relative paths."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return float(wav_file.getnframes() / wav_file.getframerate())


def build_metadata(
    root_dir: str | Path,
    out_csv: str | Path = DEFAULT_METADATA_CSV,
) -> pd.DataFrame:
    """Build and save RAVDESS metadata from a raw actor-folder directory."""
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"RAVDESS root directory not found: {root_path}")

    wav_paths = sorted(root_path.glob("Actor_*/*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No WAV files found under: {root_path}")

    rows: list[dict[str, Any]] = []
    for wav_path in wav_paths:
        parsed = parse_filename(wav_path)
        rows.append(
            {
                "path": _portable_path(wav_path),
                "emotion": parsed["emotion"],
                "emotion_id": parsed["emotion_id"],
                "actor": parsed["actor"],
                "gender": parsed["gender"],
                "intensity": parsed["intensity"],
                "duration": _duration_seconds(wav_path),
            }
        )

    metadata = pd.DataFrame(
        rows,
        columns=[
            "path",
            "emotion",
            "emotion_id",
            "actor",
            "gender",
            "intensity",
            "duration",
        ],
    )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(out_path, index=False)
    return metadata


if __name__ == "__main__":
    df = build_metadata("data/raw/ravdess")
    print(f"Saved {len(df)} rows to {DEFAULT_METADATA_CSV}")
