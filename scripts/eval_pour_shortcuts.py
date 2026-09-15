#!/usr/bin/env python3
"""Test whether a pour policy relies more on robot state than camera images."""

# ruff: noqa: E402
import argparse
import json
import os
from pathlib import Path
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_LEROBOT_HOME", str(ROOT / "local_datasets"))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import train_pour_openpi_lora

from openpi.policies import policy_config
from openpi.shared import normalize
from openpi_client import websocket_client_policy


def _image(value):
    value = np.asarray(value)
    if value.shape[0] == 3:
        value = value.transpose(1, 2, 0)
    if np.issubdtype(value.dtype, np.floating):
        value = np.clip(value * 255, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(value)


def _observation(item, *, state_item=None, image_item=None):
    state_item = item if state_item is None else state_item
    image_item = item if image_item is None else image_item
    return {
        "observation/state": np.asarray(state_item["observation.state"], dtype=np.float32),
        "observation/image": _image(image_item["observation.images.rgb"]),
        "observation/wrist_image": _image(image_item["observation.images.wrist"]),
        "prompt": item.get("task", "Pour the water from the bottle into the red cup"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path(
            "checkpoints/pi05_piper_pick_and_place_v3/"
            "piper_pour_water_state_norm_openpi_lora_v2/1000"
        ),
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--samples-per-episode", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--server-host", default=None, help="Use an already-running policy server.")
    parser.add_argument("--server-port", type=int, default=8000)
    args = parser.parse_args()

    cfg_args = types.SimpleNamespace(
        steps=30_000,
        batch_size=8,
        peak_lr=5e-5,
        warmup_steps=1_000,
        save_interval=500,
        keep_period=5_000,
        num_workers=0,
        exp_name="offline_eval",
        wandb=False,
        resume=False,
        legacy=False,
    )
    cfg = train_pour_openpi_lora.build_config(cfg_args)
    meta = LeRobotDatasetMetadata(train_pour_openpi_lora.REPO_ID)
    horizon = cfg.model.action_horizon
    dataset = LeRobotDataset(
        train_pour_openpi_lora.REPO_ID,
        delta_timestamps={"action": [i / meta.fps for i in range(horizon)]},
    )
    adapter = args.adapter.resolve()
    if args.server_host is not None:
        policy = websocket_client_policy.WebsocketClientPolicy(args.server_host, args.server_port)
        print(f"server: {args.server_host}:{args.server_port}")
    else:
        norm_stats = normalize.load(adapter / "assets" / train_pour_openpi_lora.REPO_ID)
        policy = policy_config.create_trained_policy(
            cfg,
            train_pour_openpi_lora.BASE_PARAMS.parent,
            adapter_path=adapter,
            norm_stats=norm_stats,
        )

    episode_rows = [
        json.loads(line)
        for line in (
            ROOT / "local_datasets" / train_pour_openpi_lora.REPO_ID / "meta/episodes.jsonl"
        ).read_text().splitlines()
    ]
    offsets = np.cumsum([0] + [int(row["length"]) for row in episode_rows])
    selected = np.linspace(0, len(episode_rows) - 1, min(args.episodes, len(episode_rows)), dtype=int)
    rng = np.random.default_rng(args.seed)
    rows = []
    for episode in selected:
        other_episode = (episode + len(episode_rows) // 2) % len(episode_rows)
        length = int(episode_rows[episode]["length"])
        other_length = int(episode_rows[other_episode]["length"])
        for progress in np.linspace(0.05, 0.9, args.samples_per_episode):
            frame = min(round(progress * (length - 1)), length - horizon)
            other_frame = min(round(progress * (other_length - 1)), other_length - horizon)
            item = dataset[int(offsets[episode] + frame)]
            other = dataset[int(offsets[other_episode] + other_frame)]
            state = np.asarray(item["observation.state"], dtype=np.float32)
            gt = np.asarray(item["action"], dtype=np.float32).reshape(horizon, -1)
            noise = rng.standard_normal((horizon, cfg.model.action_dim), dtype=np.float32)
            if args.server_host is None:
                infer = lambda obs: policy.infer(obs, noise=noise)["actions"]
            else:
                infer = lambda obs: policy.infer(obs)["actions"]
            normal = np.asarray(infer(_observation(item)))[:, :7]
            normal_repeat = np.asarray(infer(_observation(item)))[:, :7]
            image_swap = np.asarray(
                infer(_observation(item, image_item=other))
            )[:, :7]
            state_swap = np.asarray(
                infer(_observation(item, state_item=other))
            )[:, :7]
            gt_move = gt[-1, :6] - state[:6]
            pred_move = normal[-1, :6] - state[:6]
            rows.append(
                {
                    "normal_rmse": np.sqrt(np.mean((normal[:, :6] - gt[:, :6]) ** 2)),
                    "hold_rmse": np.sqrt(np.mean((state[None, :6] - gt[:, :6]) ** 2)),
                    "pred_move": np.linalg.norm(pred_move),
                    "gt_move": np.linalg.norm(gt_move),
                    "direction": np.dot(pred_move, gt_move)
                    / (np.linalg.norm(pred_move) * np.linalg.norm(gt_move) + 1e-8),
                    "resample_drift": np.sqrt(np.mean((normal_repeat[:, :6] - normal[:, :6]) ** 2)),
                    "image_swap_drift": np.sqrt(np.mean((image_swap[:, :6] - normal[:, :6]) ** 2)),
                    "state_swap_drift": np.sqrt(np.mean((state_swap[:, :6] - normal[:, :6]) ** 2)),
                }
            )

    print(f"checkpoint: {'server' if args.server_host is not None else adapter}")
    print(f"samples: {len(rows)} from {len(selected)} episodes")
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows])
        print(f"{key:18s} mean={values.mean():.5f} median={np.median(values):.5f}")
    image_drift = np.mean([row["image_swap_drift"] for row in rows])
    state_drift = np.mean([row["state_swap_drift"] for row in rows])
    print(f"state/image sensitivity ratio: {state_drift / (image_drift + 1e-8):.2f}x")
    print(f"beats hold baseline: {np.mean([row['normal_rmse'] < row['hold_rmse'] for row in rows]):.1%}")


if __name__ == "__main__":
    main()
