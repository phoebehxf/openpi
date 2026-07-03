"""Inspect raw and transformed state/action/prompt information from a LeRobot dataset.

Examples:
  uv run scripts/inspect_dataset_state_action_stats.py --config-name pi05_piper_pick_and_place
  uv run scripts/inspect_dataset_state_action_stats.py --config-name pi05_piper_pick_and_place --max-samples 2000
"""

from __future__ import annotations

from collections import Counter
import json
from typing import Any

import numpy as np
import tyro

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def _to_numpy_list(column: list[Any]) -> np.ndarray:
    arrays = [np.asarray(x, dtype=np.float32) for x in column]
    shapes = Counter(tuple(a.shape) for a in arrays)
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent shapes in column: {shapes}")
    return np.stack(arrays, axis=0)




def _to_serializable(value: Any):
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, np.generic):
        return value.item()
    return value

def _print_stats(name: str, values: np.ndarray) -> None:
    flat = values.reshape(values.shape[0], -1)
    print(f"\n{name}")
    print(f"  samples: {values.shape[0]}")
    print(f"  shape:   {values.shape[1:]}")
    print(f"  min:     {np.round(flat.min(axis=0), 6).tolist()}")
    print(f"  max:     {np.round(flat.max(axis=0), 6).tolist()}")
    print(f"  mean:    {np.round(flat.mean(axis=0), 6).tolist()}")
    print(f"  std:     {np.round(flat.std(axis=0), 6).tolist()}")


def main(config_name: str, max_samples: int | None = 1000, transformed_index: int = 0) -> None:
    train_config = _config.get_config(config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.repo_id is None:
        raise ValueError("Config does not define a LeRobot repo_id")

    raw_dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id)
    hf_dataset = raw_dataset.hf_dataset
    total = len(hf_dataset)
    limit = min(total, max_samples) if max_samples is not None else total

    print(f"repo_id: {data_config.repo_id}")
    print(f"total samples: {total}")
    print(f"inspecting raw samples: {limit}")
    print(f"raw columns: {hf_dataset.column_names}")

    sample_slice = hf_dataset.select(range(limit))
    state = _to_numpy_list(sample_slice["observation.state"])
    action = _to_numpy_list(sample_slice["action"])

    _print_stats("raw observation.state", state)
    _print_stats("raw action", action)

    raw_first = sample_slice[0]
    raw_summary = {
        "keys": sorted(raw_first.keys()),
        "observation.state": np.round(np.asarray(raw_first["observation.state"], dtype=np.float32), 6).tolist(),
        "action": np.round(np.asarray(raw_first["action"], dtype=np.float32), 6).tolist(),
        "prompt": raw_first.get("prompt"),
        "tasks": raw_first.get("tasks"),
        "task_index": raw_first.get("task_index"),
    }
    print("\nraw first sample:")
    print(json.dumps(_to_serializable(raw_summary), indent=2, ensure_ascii=False))

    transformed_dataset = _data_loader.create_torch_dataset(data_config, train_config.model.action_horizon, train_config.model)
    transformed_index = max(0, min(len(transformed_dataset) - 1, transformed_index))
    transformed_first = transformed_dataset[transformed_index]

    transformed_summary = {
        "index": transformed_index,
        "keys": sorted(transformed_first.keys()),
        "prompt": transformed_first.get("prompt"),
        "tasks": transformed_first.get("tasks"),
        "task_index": transformed_first.get("task_index"),
    }
    if "observation.state" in transformed_first:
        transformed_summary["observation.state"] = np.round(
            np.asarray(transformed_first["observation.state"], dtype=np.float32), 6
        ).tolist()
    if "action" in transformed_first:
        transformed_summary["action"] = np.round(
            np.asarray(transformed_first["action"], dtype=np.float32), 6
        ).tolist()

    print("\ntransformed sample (after create_torch_dataset):")
    print(json.dumps(_to_serializable(transformed_summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    tyro.cli(main)
