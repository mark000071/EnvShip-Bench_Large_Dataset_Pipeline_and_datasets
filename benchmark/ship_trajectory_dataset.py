#!/usr/bin/env python3
"""PyTorch dataset for exported ship trajectory benchmark shards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def _load_json_array(value: str) -> list[float]:
    return json.loads(value)


class ShipTrajectoryDataset(torch.utils.data.Dataset):
    """Reads exported `benchmark/{core,full}/{train,val,test}/*.csv.gz` shards."""

    def __init__(self, root: str | Path, version: str = "core", split: str = "train") -> None:
        self.root = Path(root)
        self.version = version
        self.split = split
        self.files = sorted((self.root / version / split).glob("*.csv.gz"))
        if not self.files:
            raise FileNotFoundError(f"No shard files found under {self.root / version / split}")

        # Shards are stored row-wise; concatenating here keeps the loader simple
        # for preview use and baseline experiments.
        frames = [pd.read_csv(path, compression="gzip") for path in self.files]
        self.df = pd.concat(frames, ignore_index=True)

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _kinematics(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        vx = np.diff(x, prepend=x[0])
        vy = np.diff(y, prepend=y[0])
        ax = np.diff(vx, prepend=vx[0])
        ay = np.diff(vy, prepend=vy[0])
        return vx, vy, ax, ay

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        hist_x = np.asarray(_load_json_array(row["hist_x_json"]), dtype=np.float32)
        hist_y = np.asarray(_load_json_array(row["hist_y_json"]), dtype=np.float32)
        fut_x = np.asarray(_load_json_array(row["fut_x_json"]), dtype=np.float32)
        fut_y = np.asarray(_load_json_array(row["fut_y_json"]), dtype=np.float32)
        vx, vy, ax, ay = self._kinematics(hist_x, hist_y)

        # The default baseline view uses position, velocity, and acceleration
        # derived from the released local metric coordinates.
        hist = np.stack([hist_x, hist_y, vx, vy, ax, ay], axis=-1)
        future = np.stack([fut_x, fut_y], axis=-1)
        neighbor = np.full((hist.shape[0] + future.shape[0], 1, hist.shape[-1]), 1e9, dtype=np.float32)

        return {
            "sample_id": row["sample_id"],
            "mmsi": int(row["mmsi"]),
            "hist": torch.from_numpy(hist),
            "future": torch.from_numpy(future),
            "neighbor": torch.from_numpy(neighbor),
            "ship_type_id": int(row["ship_type_id"]),
            "quality_tier": row["quality_tier"],
        }
