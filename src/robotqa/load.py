"""Read a downloaded LeRobot v3.0 dataset from disk into pandas DataFrames."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_FILES = ["meta/info.json", "meta/tasks.parquet"]
REQUIRED_FRAME_COLUMNS = ["index", "episode_index", "frame_index", "timestamp"]


class DatasetError(Exception):
    """Raised when a dataset folder is incomplete or not in the expected format."""


@dataclass
class Dataset:
    name: str
    fps: int
    info: dict
    episodes: pd.DataFrame
    frames: pd.DataFrame


def _read_parquet_folder(folder: Path) -> pd.DataFrame:
    files = sorted(folder.rglob("*.parquet"))
    if not files:
        raise DatasetError(f"No parquet files found in {folder}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_dataset(root: str | Path) -> Dataset:
    root = Path(root)

    for rel_path in REQUIRED_FILES:
        if not (root / rel_path).exists():
            raise DatasetError(f"Missing {rel_path} in {root}. Was the download interrupted?")

    info = json.loads((root / "meta" / "info.json").read_text())
    version = info.get("codebase_version")
    if version != "v3.0":
        raise DatasetError(f"Expected codebase_version v3.0, got {version}")

    episodes = _read_parquet_folder(root / "meta" / "episodes")
    frames = _read_parquet_folder(root / "data")

    missing = [c for c in REQUIRED_FRAME_COLUMNS if c not in frames.columns]
    if missing:
        raise DatasetError(f"Frame data is missing columns: {missing}")

    return Dataset(name=root.name, fps=info["fps"], info=info, episodes=episodes, frames=frames)