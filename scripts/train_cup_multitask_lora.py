#!/usr/bin/env python3
"""Two-task Piper training with compact trainable-parameter checkpoints."""

# ruff: noqa: E402
import argparse
import dataclasses
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Must be set before importing LeRobot (including indirect imports through train).
os.environ["HF_LEROBOT_HOME"] = str(ROOT / "local_datasets")

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import train

from openpi.training import config
from openpi.training import optimizer
from openpi.training import trainable_checkpoints
from openpi.training import weight_loaders

REPO_ID = "local/piper-cup-rack-multitask-cleaned-v2"
BASE_NAME = "pi05_piper_pick_and_place_v2"
NORM_ID = "phoebe777777/piper-pick-up-v2"


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
        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(ROOT / "checkpoints" / BASE_NAME / "piper_pick_pi05_lora_v4/69999/params")
        ),
        freeze_filter=base.model.get_freeze_filter(),
        ema_decay=None,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=args.warmup_steps, peak_lr=args.peak_lr, decay_steps=args.steps, decay_lr=1e-6
        ),
        num_train_steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_interval=args.save_interval,
        keep_period=None,
        wandb_enabled=args.wandb,
        resume=args.resume,
        overwrite=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--peak-lr", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--exp-name", default="piper_cup_rack_multitask_cleaned_openpi_lora_v1")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or not 0 <= args.warmup_steps < args.steps or args.save_interval <= 0 or args.peak_lr <= 0:
        parser.error("Invalid batch size, warmup/steps, save interval, or learning rate")
    cfg = build_config(args)
    dataset_root = ROOT / "local_datasets" / REPO_ID
    if not (dataset_root / "meta/info.json").is_file():
        raise FileNotFoundError("Run scripts/merge_cup_datasets.py first")
    meta = LeRobotDatasetMetadata(REPO_ID, root=dataset_root)
    if meta.total_tasks != 2 or meta.total_episodes == 0:
        raise ValueError("Expected a non-empty, two-task dataset")
    norm = ROOT / "assets" / BASE_NAME / NORM_ID / "norm_stats.json"
    if not norm.is_file() or not Path(cfg.weight_loader.params_path).is_dir():
        raise FileNotFoundError("Missing base checkpoint or Piper normalization stats")
    print(f"Dataset: {dataset_root}\nEpisodes: {meta.total_episodes}; frames: {meta.total_frames}\nTasks: {meta.tasks}")
    print(f"OpenPI default LoRA filter: {cfg.freeze_filter}\nOutput: {cfg.checkpoint_dir}")
    print(
        f"steps={args.steps}, batch={args.batch_size}, peak_lr={args.peak_lr}; "
        "trainable overlay checkpoints, latest 2 retained"
    )
    if not args.dry_run:
        checkpoint_io = trainable_checkpoints.CompactTrainableCheckpointIO(
            base_checkpoint=cfg.weight_loader.params_path,
            parameter_filter=cfg.trainable_filter,
        )
        train.main(cfg, checkpoint_io=checkpoint_io)


if __name__ == "__main__":
    main()
