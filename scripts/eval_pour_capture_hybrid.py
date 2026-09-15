#!/usr/bin/env python3
"""Separate image and state domain shift using a captured request and a training close frame."""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_LEROBOT_HOME", str(ROOT / "local_datasets"))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402


def image(value) -> np.ndarray:
    value = np.asarray(value)
    if value.shape[0] == 3:
        value = value.transpose(1, 2, 0)
    if np.issubdtype(value.dtype, np.floating):
        value = np.clip(value * 255, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-capture-index", type=int, default=None)
    parser.add_argument("--train-offset", type=int, default=0, help="Frame offset from first close transition.")
    args = parser.parse_args()

    repo_id = "local/piper-pour-water-cleaned-merged-v1"
    dataset = LeRobotDataset(repo_id)
    episodes = [
        json.loads(line)
        for line in (ROOT / "local_datasets" / repo_id / "meta/episodes.jsonl").read_text().splitlines()
    ]
    offsets = np.cumsum([0] + [int(row["length"]) for row in episodes])

    close_items = []
    for episode_index in range(len(episodes)):
        start, stop = int(offsets[episode_index]), int(offsets[episode_index + 1])
        actions = np.asarray(dataset.hf_dataset[start:stop]["action"], dtype=np.float32)
        closed = actions[:, 6] < 0.5
        transition = int((np.flatnonzero((~closed[:-1]) & closed[1:]) + 1)[0])
        selected_frame = int(np.clip(transition + args.train_offset, 0, stop - start - 1))
        item = dataset[start + selected_frame]
        close_items.append((episode_index, selected_frame, item))

    captures = []
    for directory in sorted(glob.glob(str(args.capture / "request_*"))):
        capture_number = int(Path(directory).name.split("_")[1])
        if args.max_capture_index is not None and capture_number > args.max_capture_index:
            continue
        arrays = np.load(Path(directory) / "arrays.npz")
        captures.append((Path(directory), arrays))

    distances = np.empty((len(captures), len(close_items)), dtype=np.float32)
    for capture_index, (_, arrays) in enumerate(captures):
        capture_state = arrays["observation/state"][:6]
        for close_index, (_, _, item) in enumerate(close_items):
            distances[capture_index, close_index] = np.linalg.norm(
                capture_state - np.asarray(item["observation.state"])[:6]
            )
    capture_index, close_index = np.unravel_index(np.argmin(distances), distances.shape)
    capture_dir, capture_arrays = captures[capture_index]
    episode_index, close_frame, train_item = close_items[close_index]

    real_state = np.asarray(capture_arrays["observation/state"], dtype=np.float32)
    train_state = np.asarray(train_item["observation.state"], dtype=np.float32)
    real_images = (
        np.asarray(capture_arrays["observation/image"]),
        np.asarray(capture_arrays["observation/wrist_image"]),
    )
    train_images = (
        image(train_item["observation.images.rgb"]),
        image(train_item["observation.images.wrist"]),
    )
    prompt = "Pour the water from the bottle into the red cup"
    combinations = {
        "train_images+train_state": (train_images, train_state),
        "real_images+real_state": (real_images, real_state),
        "train_images+real_state": (train_images, real_state),
        "real_images+train_state": (real_images, train_state),
        "train_external+real_wrist+real_state": ((train_images[0], real_images[1]), real_state),
        "real_external+train_wrist+real_state": ((real_images[0], train_images[1]), real_state),
    }

    print(f"capture={capture_dir.name}")
    print(f"training_episode={episode_index} close_frame={close_frame}")
    print(f"joint_state_l2_distance={distances[capture_index, close_index]:.5f} rad")
    print(f"real_state={np.round(real_state, 4).tolist()}")
    print(f"train_state={np.round(train_state, 4).tolist()}")

    client = websocket_client_policy.WebsocketClientPolicy(args.server_host, args.server_port)
    print("\ncombination repeats mean_min median_min close_chunk_rate mean_first mean_last")
    for name, (images, state) in combinations.items():
        predicted = []
        for _ in range(args.repeats):
            observation = {
                "observation/image": images[0],
                "observation/wrist_image": images[1],
                "observation/state": state,
                "prompt": prompt,
            }
            predicted.append(np.asarray(client.infer(observation)["actions"])[:, 6])
        predicted = np.asarray(predicted)
        minima = predicted.min(axis=1)
        print(
            name,
            len(predicted),
            f"{minima.mean():.3f}",
            f"{np.median(minima):.3f}",
            f"{np.mean(minima < 0.4):.1%}",
            f"{predicted[:, 0].mean():.3f}",
            f"{predicted[:, -1].mean():.3f}",
        )


if __name__ == "__main__":
    main()
