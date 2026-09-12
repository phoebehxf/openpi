#!/usr/bin/env python3
"""Evaluate a pour overlay on real training observations, split by episode phase."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_LEROBOT_HOME", str(ROOT / "local_datasets"))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import train_pour_openpi_lora

from openpi.policies import policy_config

PHASES = (
    ("start", 0.00, 0.10),
    ("approach_grasp", 0.10, 0.35),
    ("carry", 0.35, 0.60),
    ("pour", 0.60, 0.85),
    ("finish", 0.85, 1.00),
)


def _image(value) -> np.ndarray:
    value = np.asarray(value)
    if value.shape[0] == 3:
        value = value.transpose(1, 2, 0)
    if np.issubdtype(value.dtype, np.floating):
        value = np.clip(value * 255, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(value)


def _sample_frames(length: int, lo: float, hi: float, count: int, horizon: int) -> list[int]:
    start = round(lo * (length - 1))
    stop = round(hi * (length - 1))
    stop = max(start, min(stop, length - horizon))
    # Include the phase boundary. In particular, "start" must evaluate frame 0,
    # which is the observation on which the real rollout failed to launch.
    return np.linspace(start, stop, count, endpoint=False, dtype=int).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        default="checkpoints/pi05_piper_pick_and_place_v2/"
        "piper_pour_water_cleaned_openpi_lora_v1/15500",
    )
    parser.add_argument("--samples-per-phase", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("pour_phase_eval_15500.csv"))
    args = parser.parse_args()
    if args.samples_per_phase <= 0:
        parser.error("--samples-per-phase must be positive")

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
    )
    train_config = train_pour_openpi_lora.build_config(cfg_args)
    horizon = train_config.model.action_horizon
    repo_id = train_pour_openpi_lora.REPO_ID
    meta = LeRobotDatasetMetadata(repo_id)
    dataset = LeRobotDataset(
        repo_id,
        delta_timestamps={"action": [step / meta.fps for step in range(horizon)]},
    )
    base = train_pour_openpi_lora.BASE_PARAMS.parent
    print(f"Loading base={base}")
    print(f"Applying overlay={Path(args.adapter).resolve()}")
    policy = policy_config.create_trained_policy(
        train_config,
        base,
        adapter_path=args.adapter,
    )

    episode_rows = [json.loads(line) for line in (ROOT / "local_datasets" / repo_id / "meta/episodes.jsonl").read_text().splitlines()]
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float | int | str]] = []
    offset = 0
    for episode in episode_rows:
        length = int(episode["length"])
        for phase, lo, hi in PHASES:
            frames = _sample_frames(length, lo, hi, args.samples_per_phase, horizon)
            rng.shuffle(frames)
            for frame in frames:
                item = dataset[offset + frame]
                state = np.asarray(item["observation.state"], dtype=np.float32)
                gt = np.asarray(item["action"], dtype=np.float32).reshape(horizon, -1)
                obs = {
                    "observation/state": state,
                    "observation/image": _image(item["observation.images.rgb"]),
                    "observation/wrist_image": _image(item["observation.images.wrist"]),
                    "prompt": item.get("task", meta.tasks[0]),
                }
                pred = np.asarray(policy.infer(obs)["actions"], dtype=np.float32)[:, : gt.shape[1]]
                gt_delta = gt[-1, :6] - state[:6]
                pred_delta = pred[-1, :6] - state[:6]
                gt_mag = float(np.linalg.norm(gt_delta))
                pred_mag = float(np.linalg.norm(pred_delta))
                cosine = float(np.dot(gt_delta, pred_delta) / (gt_mag * pred_mag + 1e-8))
                endpoint_rmse = float(np.sqrt(np.mean((pred[-1, :6] - gt[-1, :6]) ** 2)))
                hold_rmse = float(np.sqrt(np.mean((state[:6] - gt[-1, :6]) ** 2)))
                rows.append(
                    {
                        "episode": int(episode["episode_index"]),
                        "frame": frame,
                        "phase": phase,
                        "progress": frame / max(length - 1, 1),
                        "gt_move_l2": gt_mag,
                        "pred_move_l2": pred_mag,
                        "move_ratio": pred_mag / (gt_mag + 1e-8),
                        "direction_cosine": cosine,
                        "endpoint_rmse": endpoint_rmse,
                        "hold_endpoint_rmse": hold_rmse,
                        "beats_hold": float(endpoint_rmse < hold_rmse),
                        "stuck_on_moving_target": float(pred_mag < 0.03 and gt_mag > 0.08),
                        "gripper_accuracy": float(np.mean((pred[:, 6] >= 0.5) == (gt[:, 6] >= 0.5))),
                    }
                )
        offset += length

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nphase          n  gt_move pred_move ratio cosine end_rmse hold_rmse beats_hold stuck gripper")
    for phase, _, _ in PHASES:
        phase_rows = [row for row in rows if row["phase"] == phase]
        means = {
            key: float(np.mean([float(row[key]) for row in phase_rows]))
            for key in rows[0]
            if key not in {"phase"}
        }
        print(
            f"{phase:14s} {len(phase_rows):2d} "
            f"{means['gt_move_l2']:.3f}   {means['pred_move_l2']:.3f}    "
            f"{means['move_ratio']:.2f}  {means['direction_cosine']:+.2f}   "
            f"{means['endpoint_rmse']:.3f}    {means['hold_endpoint_rmse']:.3f}     "
            f"{means['beats_hold']:.0%}      {means['stuck_on_moving_target']:.0%}  "
            f"{means['gripper_accuracy']:.0%}"
        )
    print(f"\nDetailed samples: {args.output.resolve()}")


if __name__ == "__main__":
    main()
