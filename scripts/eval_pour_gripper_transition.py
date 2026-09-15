#!/usr/bin/env python3
"""Evaluate gripper predictions around real closing transitions in the pour dataset."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_LEROBOT_HOME", str(ROOT / "local_datasets"))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402


def image(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.shape[0] == 3:
        value = value.transpose(1, 2, 0)
    if np.issubdtype(value.dtype, np.floating):
        value = np.clip(value * 255, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    repo_id = "local/piper-pour-water-cleaned-merged-v1"
    dataset = LeRobotDataset(repo_id)
    episode_rows = [
        json.loads(line)
        for line in (ROOT / "local_datasets" / repo_id / "meta/episodes.jsonl").read_text().splitlines()
    ]
    selected = np.linspace(0, len(episode_rows) - 1, min(args.episodes, len(episode_rows)), dtype=int)
    offsets = np.cumsum([0] + [int(row["length"]) for row in episode_rows])
    client = websocket_client_policy.WebsocketClientPolicy(args.server_host, args.server_port)

    predictions: dict[int, list[float]] = {offset: [] for offset in (-10, -5, 0, 5, 10)}
    print("episode close_frame offset current_label predicted_min predicted_first predicted_last")
    for episode_index in selected:
        start, stop = int(offsets[episode_index]), int(offsets[episode_index + 1])
        actions = np.asarray(dataset.hf_dataset[start:stop]["action"], dtype=np.float32)
        closed = actions[:, 6] < 0.5
        transitions = np.flatnonzero((~closed[:-1]) & closed[1:]) + 1
        if not len(transitions):
            continue
        close_frame = int(transitions[0])
        for relative in predictions:
            frame = int(np.clip(close_frame + relative, 0, stop - start - 1))
            item = dataset[start + frame]
            observation = {
                "observation/state": np.asarray(item["observation.state"], dtype=np.float32),
                "observation/image": image(item["observation.images.rgb"]),
                "observation/wrist_image": image(item["observation.images.wrist"]),
                "prompt": item.get("task", "Pour the water from the bottle into the red cup"),
            }
            predicted = np.asarray(client.infer(observation)["actions"])[:, 6]
            predictions[relative].append(float(predicted.min()))
            print(
                episode_index,
                close_frame,
                relative,
                f"{actions[frame, 6]:.3f}",
                f"{predicted.min():.3f}",
                f"{predicted[0]:.3f}",
                f"{predicted[-1]:.3f}",
            )

    print("\noffset samples mean_predicted_min close_chunk_rate")
    for relative, values in predictions.items():
        array = np.asarray(values)
        print(relative, len(array), f"{array.mean():.3f}", f"{np.mean(array < 0.4):.1%}")


if __name__ == "__main__":
    main()
