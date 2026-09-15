#!/usr/bin/env python3
"""A/B test real versus neutral gripper state on captured three-phase observations."""

# ruff: noqa: E402
import argparse
import json
import os
from pathlib import Path
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_LEROBOT_HOME", str(ROOT / "local_datasets"))

import train_pour_three_phase_lora

from openpi.policies import policy_config
from openpi.shared import normalize


def _config():
    args = types.SimpleNamespace(
        steps=30_000,
        batch_size=8,
        peak_lr=3e-5,
        warmup_steps=1_000,
        save_interval=1_000,
        keep_period=5_000,
        num_workers=0,
        exp_name="capture_gripper_state_eval",
        wandb=False,
        resume=False,
    )
    return train_pour_three_phase_lora.build_config(args)


def _capture_index(directory: Path) -> int:
    return int(directory.name.split("_")[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("checkpoints/pi05_piper_pick_and_place_v3/piper_pour_three_phase_state_lora_v1/15000"),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--stride", type=int, default=1, help="Evaluate every Nth captured request.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top", type=int, default=10, help="Print this many lowest-minimum requests per variant.")
    args = parser.parse_args()
    if args.stride <= 0 or args.top <= 0:
        parser.error("stride and top must be positive")

    adapter = args.adapter.resolve()
    repo_id = train_pour_three_phase_lora.REPO_ID
    stats = normalize.load(adapter / "assets" / repo_id)
    state_stats = stats["state"]
    if state_stats.q01 is None or state_stats.q99 is None:
        raise ValueError("PI0.5 neutral-state test requires q01/q99 normalization statistics")
    neutral_raw = float((state_stats.q01[6] + state_stats.q99[6]) / 2)

    directories = sorted(args.capture.resolve().glob("request_*"), key=_capture_index)[:: args.stride]
    if not directories:
        raise FileNotFoundError(f"No request_* captures under {args.capture}")

    cfg = _config()
    print(f"Loading checkpoint: {adapter}", flush=True)
    policy = policy_config.create_trained_policy(
        cfg,
        train_pour_three_phase_lora.BASE_PARAMS.parent,
        adapter_path=adapter,
        norm_stats=stats,
    )
    print(f"Loaded. requests={len(directories)}, neutral_raw_gripper={neutral_raw:.6f}", flush=True)

    rng = np.random.default_rng(args.seed)
    variants = ("real", "neutral", "open_neutral_closed_real", "forced_closed")
    rows: list[dict[str, float | int | str]] = []
    for position, directory in enumerate(directories, start=1):
        arrays = np.load(directory / "arrays.npz")
        real_state = np.asarray(arrays["observation/state"], dtype=np.float32)
        prompt_path = directory / "prompt.txt"
        prompt = (
            prompt_path.read_text().strip()
            if prompt_path.is_file()
            else "Grasp the bottle and move it next to the red cup"
        )
        noise = rng.standard_normal((cfg.model.action_horizon, cfg.model.action_dim), dtype=np.float32)
        states = {
            "real": real_state,
            "neutral": np.concatenate([real_state[:6], np.asarray([neutral_raw], dtype=np.float32)]),
            "open_neutral_closed_real": np.concatenate(
                [
                    real_state[:6],
                    np.asarray([neutral_raw if real_state[6] >= 0.5 else real_state[6]], dtype=np.float32),
                ]
            ),
            "forced_closed": np.concatenate([real_state[:6], np.asarray([0.0], dtype=np.float32)]),
        }
        for variant in variants:
            observation = {
                "observation/image": arrays["observation/image"],
                "observation/wrist_image": arrays["observation/wrist_image"],
                "observation/state": states[variant],
                "prompt": prompt,
            }
            gripper = np.asarray(policy.infer(observation, noise=noise)["actions"])[:, 6]
            rows.append(
                {
                    "request": _capture_index(directory),
                    "variant": variant,
                    "minimum": float(gripper.min()),
                    "mean": float(gripper.mean()),
                    "last": float(gripper[-1]),
                    "closed_steps": int(np.sum(gripper < args.threshold)),
                }
            )
        if position % 10 == 0 or position == len(directories):
            print(f"Evaluated {position}/{len(directories)} requests", flush=True)

    print("\nvariant         mean_min  median_min  lowest_min  chunks_with_close  closed_steps")
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        minima = np.asarray([row["minimum"] for row in selected])
        chunks = sum(int(row["closed_steps"] > 0) for row in selected)
        closed_steps = sum(int(row["closed_steps"]) for row in selected)
        print(
            f"{variant:15s} {minima.mean():8.3f} {np.median(minima):11.3f} {minima.min():11.3f} "
            f"{chunks:8d}/{len(selected):<8d} {closed_steps:12d}"
        )

    print(f"\nLowest {args.top} requests per variant:")
    for variant in variants:
        selected = sorted((row for row in rows if row["variant"] == variant), key=lambda row: row["minimum"])
        values = ", ".join(f"req{int(row['request']):03d}={float(row['minimum']):.3f}" for row in selected[: args.top])
        print(f"{variant:15s} {values}")

    metadata = {
        "checkpoint": str(adapter),
        "capture": str(args.capture.resolve()),
        "requests": len(directories),
        "threshold": args.threshold,
        "neutral_raw_gripper": neutral_raw,
    }
    print("\n" + json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
