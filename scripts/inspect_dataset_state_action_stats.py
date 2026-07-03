"""Inspect raw observation.state and action statistics from a LeRobot dataset.

Examples:
  uv run scripts/inspect_dataset_state_action_stats.py --config-name pi05_piper_pick_and_place
  uv run scripts/inspect_dataset_state_action_stats.py --config-name pi05_piper_pick_and_place --max-samples 2000
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import tyro

import openpi.training.config as _config
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset


def _to_numpy_list(column: list[Any]) -> np.ndarray:
    arrays = [np.asarray(x, dtype=np.float32) for x in column]
    shapes = Counter(tuple(a.shape) for a in arrays)
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent shapes in column: {shapes}")
    return np.stack(arrays, axis=0)


def _print_stats(name: str, values: np.ndarray) -> None:
    flat = values.reshape(values.shape[0], -1)
    print(f"\\n{name}")
    print(f"  samples: {values.shape[0]}")
    print(f"  shape:   {values.shape[1:]}")
    print(f"  min:     {np.round(flat.min(axis=0), 6).tolist()}")
    print(f"  max:     {np.round(flat.max(axis=0), 6).tolist()}")
    print(f"  mean:    {np.round(flat.mean(axis=0), 6).tolist()}")
    print(f"  std:     {np.round(flat.std(axis=0), 6).tolist()}")


def main(config_name: str, max_samples: int | None = 1000) -> None:
    train_config = _config.get_config(config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.repo_id is None:
        raise ValueError("Config does not define a LeRobot repo_id")

    dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id)
    hf_dataset = dataset.hf_dataset
    total = len(hf_dataset)
    limit = min(total, max_samples) if max_samples is not None else total
    print(f"repo_id: {data_config.repo_id}")
    print(f"total samples: {total}")
    print(f"inspecting samples: {limit}")

    sample_slice = hf_dataset.select(range(limit))
    state = _to_numpy_list(sample_slice["observation.state"])
    action = _to_numpy_list(sample_slice["action"])

    _print_stats("observation.state", state)
    _print_stats("action", action)

    first = {
        "observation.state": np.round(state[0], 6).tolist(),
        "action": np.round(action[0], 6).tolist(),
    }
    print("\\nfirst sample:")
    print(json.dumps(first, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
