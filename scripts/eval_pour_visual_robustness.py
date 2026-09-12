#!/usr/bin/env python3
"""Measure frame-0 policy sensitivity to small camera perturbations."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import types

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_LEROBOT_HOME", str(ROOT / "local_datasets"))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import train_pour_openpi_lora

from openpi.policies import policy_config


def _image(value) -> np.ndarray:
    value = np.asarray(value)
    if value.shape[0] == 3:
        value = value.transpose(1, 2, 0)
    if np.issubdtype(value.dtype, np.floating):
        value = np.clip(value * 255, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(value)


def _shift(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    return cv2.warpAffine(
        image,
        np.float32([[1, 0, dx], [0, 1, dy]]),
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _brightness(image: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _zoom(image: np.ndarray, fraction: float) -> np.ndarray:
    height, width = image.shape[:2]
    crop_h, crop_w = round(height / fraction), round(width / fraction)
    top, left = (height - crop_h) // 2, (width - crop_w) // 2
    crop = image[top : top + crop_h, left : left + crop_w]
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)


def _variants(rgb: np.ndarray, wrist: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "original": (rgb, wrist),
        "rgb_shift_5px": (_shift(rgb, 5, 5), wrist),
        "wrist_shift_5px": (rgb, _shift(wrist, 5, 5)),
        "both_shift_5px": (_shift(rgb, 5, 5), _shift(wrist, 5, 5)),
        "both_shift_10px": (_shift(rgb, 10, 10), _shift(wrist, 10, 10)),
        "both_brightness_85pct": (_brightness(rgb, 0.85), _brightness(wrist, 0.85)),
        "both_brightness_115pct": (_brightness(rgb, 1.15), _brightness(wrist, 1.15)),
        "both_center_zoom_5pct": (_zoom(rgb, 1.05), _zoom(wrist, 1.05)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        default="checkpoints/pi05_piper_pick_and_place_v2/"
        "piper_pour_water_cleaned_openpi_lora_v1/15500",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("pour_visual_robustness_15500.csv"))
    args = parser.parse_args()

    cfg_args = types.SimpleNamespace(
        steps=30_000, batch_size=8, peak_lr=5e-5, warmup_steps=1_000,
        save_interval=500, keep_period=5_000, num_workers=0,
        exp_name="offline_eval", wandb=False, resume=False,
    )
    train_config = train_pour_openpi_lora.build_config(cfg_args)
    horizon = train_config.model.action_horizon
    action_dim = train_config.model.action_dim
    repo_id = train_pour_openpi_lora.REPO_ID
    meta = LeRobotDatasetMetadata(repo_id)
    dataset = LeRobotDataset(
        repo_id,
        delta_timestamps={"action": [step / meta.fps for step in range(horizon)]},
    )
    policy = policy_config.create_trained_policy(
        train_config,
        train_pour_openpi_lora.BASE_PARAMS.parent,
        adapter_path=args.adapter,
    )
    episodes = [
        json.loads(line)
        for line in (ROOT / "local_datasets" / repo_id / "meta/episodes.jsonl").read_text().splitlines()
    ]
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float | int | str]] = []
    offset = 0
    for episode in episodes:
        item = dataset[offset]
        state = np.asarray(item["observation.state"], dtype=np.float32)
        gt = np.asarray(item["action"], dtype=np.float32).reshape(horizon, -1)
        rgb = _image(item["observation.images.rgb"])
        wrist = _image(item["observation.images.wrist"])
        noise = rng.standard_normal((horizon, action_dim), dtype=np.float32)
        predictions: dict[str, np.ndarray] = {}
        for name, (variant_rgb, variant_wrist) in _variants(rgb, wrist).items():
            output = policy.infer(
                {
                    "observation/state": state,
                    "observation/image": variant_rgb,
                    "observation/wrist_image": variant_wrist,
                    "prompt": item.get("task", meta.tasks[0]),
                },
                noise=noise,
            )
            predictions[name] = np.asarray(output["actions"], dtype=np.float32)[:, : gt.shape[1]]
        baseline = predictions["original"]
        gt_delta = gt[-1, :6] - state[:6]
        gt_mag = float(np.linalg.norm(gt_delta))
        baseline_delta = baseline[-1, :6] - state[:6]
        for name, pred in predictions.items():
            pred_delta = pred[-1, :6] - state[:6]
            pred_mag = float(np.linalg.norm(pred_delta))
            direction = float(np.dot(gt_delta, pred_delta) / (gt_mag * pred_mag + 1e-8))
            base_cosine = float(
                np.dot(baseline_delta, pred_delta)
                / (np.linalg.norm(baseline_delta) * pred_mag + 1e-8)
            )
            rows.append(
                {
                    "episode": int(episode["episode_index"]),
                    "variant": name,
                    "gt_move_l2": gt_mag,
                    "pred_move_l2": pred_mag,
                    "direction_cosine": direction,
                    "endpoint_rmse": float(np.sqrt(np.mean((pred[-1, :6] - gt[-1, :6]) ** 2))),
                    "baseline_direction_cosine": base_cosine,
                    "endpoint_drift_from_original": float(np.linalg.norm(pred[-1, :6] - baseline[-1, :6])),
                    "chunk_rmse_from_original": float(np.sqrt(np.mean((pred[:, :6] - baseline[:, :6]) ** 2))),
                    "stuck_on_moving_target": float(pred_mag < 0.03 and gt_mag > 0.08),
                }
            )
        offset += int(episode["length"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("variant                  dir_gt  end_rmse dir_base endpoint_drift chunk_drift stuck bad_dir")
    for name in _variants(np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2, 3), np.uint8)):
        selected = [row for row in rows if row["variant"] == name]
        means = {
            key: float(np.mean([float(row[key]) for row in selected]))
            for key in rows[0]
            if key not in {"variant"}
        }
        bad_direction = np.mean([float(row["direction_cosine"]) < 0.5 for row in selected])
        print(
            f"{name:24s} {means['direction_cosine']:+.3f}   {means['endpoint_rmse']:.3f}    "
            f"{means['baseline_direction_cosine']:+.3f}    {means['endpoint_drift_from_original']:.3f}          "
            f"{means['chunk_rmse_from_original']:.3f}       {means['stuck_on_moving_target']:.0%}  "
            f"{bad_direction:.0%}"
        )
    print(f"\nDetailed samples: {args.output.resolve()}")


if __name__ == "__main__":
    main()
