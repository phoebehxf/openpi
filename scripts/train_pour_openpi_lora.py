#!/usr/bin/env python3
"""Fine-tune pouring with compact trainable-parameter checkpoints."""

import argparse
import dataclasses
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
# Must be set before importing LeRobot (including indirect imports through train).
os.environ["HF_LEROBOT_HOME"] = str(ROOT / "local_datasets")

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import train

from openpi.training import config
from openpi.training import optimizer
from openpi.training import trainable_checkpoints
from openpi.training import weight_loaders

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "local/piper-pour-water-cleaned-merged-v1"
# REPO_ID = "phoebe777777/piper-pour-water-cleaned-v1"
BASE_NAME = "pi05_piper_pick_and_place_v2"
NORM_ID = "phoebe777777/piper-pick-up-v2"
BASE_PARAMS = ROOT / "checkpoints" / BASE_NAME / "piper_pick_pi05_lora_v4/69999/params"


def build_config(args):
    base = config.get_config(BASE_NAME)
    data = dataclasses.replace(
        base.data,
        repo_id=REPO_ID,
        base_config=dataclasses.replace(base.data.base_config, prompt_from_task=True),
        assets=dataclasses.replace(base.data.assets, asset_id=NORM_ID),
    )
    return dataclasses.replace(
        base,
        data=data,
        exp_name=args.exp_name,
        assets_base_dir=str(ROOT / "assets"),
        checkpoint_base_dir=str(ROOT / "checkpoints"),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(BASE_PARAMS)),
        freeze_filter=base.model.get_freeze_filter(),
        ema_decay=None,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=args.warmup_steps,
            peak_lr=args.peak_lr,
            decay_steps=args.steps,
            decay_lr=1e-6,
        ),
        num_train_steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_interval=args.save_interval,
        keep_period=args.keep_period,
        wandb_enabled=args.wandb,
        resume=args.resume,
        overwrite=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--peak-lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--save-interval", type=int, default=1_000)
    parser.add_argument("--keep-period", type=int, default=5_000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--exp-name", default="piper_pour_water_cleaned_openpi_lora_v1")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or not 0 <= args.warmup_steps < args.steps:
        parser.error("Invalid batch size or warmup/steps")
    if args.save_interval <= 0 or args.keep_period <= 0 or args.peak_lr <= 0:
        parser.error("Invalid save/keep interval or learning rate")

    cfg = build_config(args)
    metadata = LeRobotDatasetMetadata(REPO_ID)
    if metadata.total_episodes == 0 or metadata.total_tasks != 1:
        raise ValueError("Expected one non-empty pouring task")
    norm = ROOT / "assets" / BASE_NAME / NORM_ID / "norm_stats.json"
    if not BASE_PARAMS.is_dir() or not norm.is_file():
        raise FileNotFoundError("Missing base checkpoint or Piper normalization stats")

    print(f"Dataset: {REPO_ID}")
    print(f"Episodes: {metadata.total_episodes}; frames: {metadata.total_frames}; tasks: {metadata.tasks}")
    print(f"OpenPI default LoRA filter: {cfg.freeze_filter}")
    print(f"Base: {BASE_PARAMS.resolve()}")
    print(f"Output: {cfg.checkpoint_dir}")
    print(
        f"steps={args.steps}, batch={args.batch_size}, peak_lr={args.peak_lr}; "
        f"trainable overlay checkpoints, every {args.keep_period} steps preserved"
    )
    if not args.dry_run:
        checkpoint_io = trainable_checkpoints.CompactTrainableCheckpointIO(
            base_checkpoint=str(BASE_PARAMS), parameter_filter=cfg.trainable_filter
        )
        train.main(cfg, checkpoint_io=checkpoint_io)


if __name__ == "__main__":
    main()
