"""Print what's actually in a downloaded LeRobot dataset. Exploration only."""
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
info = json.loads((root / "meta" / "info.json").read_text())

print("codebase_version:", info["codebase_version"])
print("robot_type:", info.get("robot_type"), "| fps:", info["fps"])
print("episodes / frames:", info["total_episodes"], "/", info["total_frames"])
print("features:")
for name, spec in info["features"].items():
    print(f"  {name}: {spec['dtype']} {spec.get('shape')}")

ep_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
episodes = pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)
print("\nepisodes table shape:", episodes.shape)
print("non-stats columns:", [c for c in episodes.columns if not c.startswith("stats/")])

data_files = sorted((root / "data").rglob("*.parquet"))
frames = pd.read_parquet(data_files[0])
print(f"\ndata files: {len(data_files)} | rows in first file: {len(frames)}")
print("frame columns:", list(frames.columns))
print(frames[["index", "episode_index", "frame_index", "timestamp"]].head())