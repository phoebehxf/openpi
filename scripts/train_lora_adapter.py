#!/usr/bin/env python3
"""Train a task-specific LoRA adapter from a fixed OpenPI checkpoint."""

from __future__ import annotations

import argparse
import dataclasses
import pathlib

import flax.nnx as nnx
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import train

import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as train_config
from openpi.training import lora_checkpoints
from openpi.training import optimizer
from openpi.training import weight_loaders

DEFAULT_REPO_ID = "phoebe777777/piper-pour-water"
DEFAULT_BASE_CONFIG = "pi05_piper_pick_and_place_v2"
DEFAULT_BASE_CHECKPOINT = "checkpoints/pi05_piper_pick_and_place_v2/piper_pick_pi05_lora_v4/69999/params"
DEFAULT_NORM_ASSET_ID = "phoebe777777/piper-pick-up-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--init-checkpoint", default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--norm-asset-id", default=DEFAULT_NORM_ASSET_ID)
    parser.add_argument("--exp-name", default="piper_cup_hang_rack_lora_v1")
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--peak-lr", type=float, default=5e-5)
    parser.add_argument("--decay-lr", type=float, default=1e-6)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--keep-period", type=int, default=2_500)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-num-batches", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the checkpoint, shared stats, and dataset metadata without loading the model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        raise SystemExit("--steps and --batch-size must be positive")
    if not 0 <= args.warmup_steps < args.steps:
        raise SystemExit("--warmup-steps must be in [0, --steps)")

    root = pathlib.Path(__file__).resolve().parents[1]
    init_params = pathlib.Path(args.init_checkpoint).expanduser()
    if not init_params.is_absolute():
        init_params = root / init_params
    init_params = init_params.resolve()
    if not args.resume and not init_params.is_dir():
        raise SystemExit(f"Base checkpoint params directory does not exist: {init_params}")

    base = train_config.get_config(args.base_config)
    if not isinstance(base.data, train_config.LeRobotPiperDataConfig):
        raise SystemExit(f"Base config {args.base_config!r} does not use LeRobotPiperDataConfig")

    norm_stats = (root / base.assets_base_dir / base.name / args.norm_asset_id / "norm_stats.json").resolve()
    if not norm_stats.is_file():
        raise SystemExit(f"Shared-base normalization statistics are missing: {norm_stats}")

    metadata = LeRobotDatasetMetadata(args.repo_id)
    episode_count = int(metadata.total_episodes)
    frame_count = int(metadata.total_frames)
    tasks = sorted(str(task) for task in metadata.tasks.values())
    if episode_count == 0:
        raise SystemExit(f"Dataset {args.repo_id!r} contains no episodes")

    adapter_data = dataclasses.replace(
        base.data,
        repo_id=args.repo_id,
        assets=dataclasses.replace(base.data.assets, asset_id=args.norm_asset_id),
    )
    freeze_everything_except_lora = nnx.Not(nnx_utils.PathRegex(".*lora.*"))
    checkpoint_dir = (root / base.checkpoint_base_dir / base.name / args.exp_name).resolve()

    print("[lora-adapter] validation complete", flush=True)
    print(f"[lora-adapter] dataset:        {args.repo_id}", flush=True)
    print(f"[lora-adapter] tasks:          {tasks}", flush=True)
    print(f"[lora-adapter] episodes:       {episode_count}", flush=True)
    print(f"[lora-adapter] frames:         {frame_count}", flush=True)
    print(f"[lora-adapter] base params:    {init_params}", flush=True)
    print(f"[lora-adapter] norm stats:     {norm_stats}", flush=True)
    print("[lora-adapter] trainable:      parameter paths matching .*lora.*", flush=True)
    print(f"[lora-adapter] output:         {checkpoint_dir}", flush=True)
    print(
        f"[lora-adapter] schedule:       steps={args.steps}, batch={args.batch_size}, "
        f"warmup={args.warmup_steps}, peak_lr={args.peak_lr:g}, decay_lr={args.decay_lr:g}",
        flush=True,
    )
    if args.dry_run:
        return

    config = dataclasses.replace(
        base,
        exp_name=args.exp_name,
        data=adapter_data,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(init_params)),
        freeze_filter=freeze_everything_except_lora,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=args.warmup_steps,
            peak_lr=args.peak_lr,
            decay_steps=args.steps,
            decay_lr=args.decay_lr,
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.steps,
        save_interval=args.save_interval,
        keep_period=args.keep_period,
        eval_interval=args.eval_interval,
        eval_num_batches=args.eval_num_batches,
        wandb_enabled=args.wandb,
        overwrite=False,
        resume=args.resume,
    )
    checkpoint_io = lora_checkpoints.CompactLoraCheckpointIO(
        base_checkpoint=str(init_params),
    )
    train.main(config, checkpoint_io=checkpoint_io)


if __name__ == "__main__":
    main()
