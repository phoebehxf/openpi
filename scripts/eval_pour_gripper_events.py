#!/usr/bin/env python3
"""Evaluate gripper open/close transitions for multiple pour adapters."""

import argparse
import gc
import json
import os
from pathlib import Path
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_LEROBOT_HOME", str(ROOT / "local_datasets"))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
import train_pour_openpi_lora  # noqa: E402
import train_pour_three_phase_lora  # noqa: E402

from openpi.policies import policy_config  # noqa: E402
from openpi.shared import normalize  # noqa: E402


def image(value) -> np.ndarray:
    value = np.asarray(value)
    if value.shape[0] == 3:
        value = value.transpose(1, 2, 0)
    if np.issubdtype(value.dtype, np.floating):
        value = np.clip(value * 255, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(value)


def config(training_module):
    args = types.SimpleNamespace(
        steps=30_000,
        batch_size=8,
        peak_lr=5e-5,
        warmup_steps=1_000,
        save_interval=500,
        keep_period=5_000,
        num_workers=0,
        exp_name="gripper_event_eval",
        wandb=False,
        resume=False,
        legacy=False,
    )
    return training_module.build_config(args)


def first_transition(labels: np.ndarray, *, target: bool) -> int | None:
    indices = np.flatnonzero((labels[1:] == target) & (labels[:-1] != target)) + 1
    return int(indices[0]) if len(indices) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, action="append", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--three-phase",
        action="store_true",
        help="Evaluate close on phase 0 and open on phase 2 of the three-phase dataset.",
    )
    args = parser.parse_args()

    print("[1/3] loading dataset metadata...", flush=True)
    training_module = train_pour_three_phase_lora if args.three_phase else train_pour_openpi_lora
    cfg = config(training_module)
    repo_id = training_module.REPO_ID
    meta = LeRobotDatasetMetadata(repo_id)
    horizon = cfg.model.action_horizon
    dataset = LeRobotDataset(
        repo_id,
        delta_timestamps={"action": [index / meta.fps for index in range(horizon)]},
    )
    print(f"[1/3] dataset ready: {dataset.num_episodes} episodes", flush=True)
    episodes = [
        json.loads(line)
        for line in (ROOT / "local_datasets" / repo_id / "meta/episodes.jsonl").read_text().splitlines()
    ]
    offsets = np.cumsum([0] + [int(row["length"]) for row in episodes])
    if args.three_phase:
        source_count = len(episodes) // 3
        sources = np.linspace(0, source_count - 1, min(args.episodes, source_count), dtype=int)
        selected_events = [(int(source) * 3, "close", True) for source in sources]
        selected_events += [(int(source) * 3 + 2, "open", False) for source in sources]
    else:
        selected = np.linspace(0, len(episodes) - 1, min(args.episodes, len(episodes)), dtype=int)
        selected_events = [
            (int(episode_index), event_name, target)
            for episode_index in selected
            for event_name, target in (("close", True), ("open", False))
        ]

    samples = []
    for selection_index, (episode_index, event_name, target) in enumerate(selected_events, start=1):
        print(
            f"[2/3] decoding episode {episode_index} ({event_name}) ({selection_index}/{len(selected_events)})...",
            flush=True,
        )
        start, stop = int(offsets[episode_index]), int(offsets[episode_index + 1])
        actions = np.asarray(dataset.hf_dataset[start:stop]["action"], dtype=np.float32)
        closed = actions[:, 6] < 0.5
        event = first_transition(closed, target=target)
        if event is None:
            continue
        for relative in range(-args.window, args.window + 1, args.stride):
            frame = int(np.clip(event + relative, 0, stop - start - horizon))
            item = dataset[start + frame]
            samples.append((episode_index, event_name, relative, item))
    print(f"[2/3] prepared {len(samples)} event-window samples", flush=True)

    rng = np.random.default_rng(args.seed)
    noises = [rng.standard_normal((horizon, cfg.model.action_dim), dtype=np.float32) for _ in samples]
    for adapter_arg in args.adapter:
        adapter = adapter_arg.resolve()
        print(f"[3/3] loading checkpoint {adapter}...", flush=True)
        stats = normalize.load(adapter / "assets" / repo_id)
        policy = policy_config.create_trained_policy(
            cfg,
            training_module.BASE_PARAMS.parent,
            adapter_path=adapter,
            norm_stats=stats,
        )
        print("[3/3] checkpoint loaded; running inference...", flush=True)
        rows = []
        for (_episode_index, event_name, relative, item), noise in zip(samples, noises, strict=True):
            observation = {
                "observation/state": np.asarray(item["observation.state"], dtype=np.float32),
                "observation/image": image(item["observation.images.rgb"]),
                "observation/wrist_image": image(item["observation.images.wrist"]),
                "prompt": item.get("task", meta.tasks[0]),
            }
            predicted = np.asarray(policy.infer(observation, noise=noise)["actions"])[:, 6]
            truth = np.asarray(item["action"]).reshape(horizon, -1)[:, 6]
            pred_closed, true_closed = predicted < 0.5, truth < 0.5
            target = event_name == "close"
            true_event = first_transition(true_closed, target=target)
            pred_event = first_transition(pred_closed, target=target)
            rows.append(
                {
                    "event": event_name,
                    "relative": relative,
                    "accuracy": float(np.mean(pred_closed == true_closed)),
                    "true_event": true_event is not None,
                    "pred_event": pred_event is not None,
                    "timing_error": None if true_event is None or pred_event is None else pred_event - true_event,
                }
            )

        print(f"\ncheckpoint={adapter.parent.name}/{adapter.name} samples={len(rows)}")
        for event_name in ("close", "open"):
            event_rows = [row for row in rows if row["event"] == event_name]
            positive = [row for row in event_rows if row["true_event"]]
            detected = [row for row in positive if row["pred_event"]]
            timing = [abs(row["timing_error"]) for row in detected]
            print(
                f"{event_name:5s} step_accuracy={np.mean([row['accuracy'] for row in event_rows]):.1%} "
                f"event_recall={len(detected)}/{len(positive)}={len(detected) / max(len(positive), 1):.1%} "
                f"timing_mae={np.mean(timing) if timing else float('nan'):.2f} frames"
            )
        del policy
        gc.collect()


if __name__ == "__main__":
    main()
