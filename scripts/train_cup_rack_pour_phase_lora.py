#!/usr/bin/env python3
"""LoRA fine-tune the merged cup-rack and pouring phase dataset."""

# ruff: noqa: E402
import argparse
import dataclasses
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_LEROBOT_HOME", str(ROOT / "local_datasets"))

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import train

from openpi.training import config
from openpi.training import optimizer
from openpi.training import trainable_checkpoints
from openpi.training import weight_loaders


REPO_ID = "local/piper-cup-rack-pour-phase-merged-v1"
CONFIG_NAME = "pi05_piper_pick_and_place_v3"
ASSET_CONFIG_NAME = "pi05_piper_pick_and_place_v2"
BASE_PARAMS = ROOT / "checkpoints/piper_all_manipulation_full_v1/79999/params"
EXPECTED_TASKS = {
    "Grasp the bottle and move it next to the red cup",
    "Tilt the bottle to pour water into the red cup",
    "Return the bottle upright, place it down, and release it",
    "Grasp the red cup and move it next to the rack",
    "Align the cup handle opening with the rack peg and hang the cup on it",
    "Grasp the red cup on the rack",
    "Lift the red cup diagonally upward to unhook it from the rack",
    "Move the red cup to the table, place it down, and release it",
}


def build_config(args: argparse.Namespace) -> config.TrainConfig:
    base = config.get_config(CONFIG_NAME)
    model = dataclasses.replace(base.model, discrete_state_input=True)
    data = dataclasses.replace(
        base.data,
        repo_id=REPO_ID,
        base_config=dataclasses.replace(base.data.base_config, prompt_from_task=True),
        assets=dataclasses.replace(
            base.data.assets,
            asset_id=REPO_ID,
            assets_dir=str(ROOT / "assets" / ASSET_CONFIG_NAME),
        ),
    )
    return dataclasses.replace(
        base,
        model=model,
        data=data,
        exp_name=args.exp_name,
        assets_base_dir=str(ROOT / "assets"),
        checkpoint_base_dir=str(ROOT / "checkpoints"),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(args.base_params)),
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=args.warmup_steps,
            peak_lr=args.peak_lr,
            decay_steps=args.steps,
            decay_lr=args.decay_lr,
        ),
        num_train_steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_interval=args.save_interval,
        keep_period=args.keep_period,
        eval_interval=args.eval_interval,
        eval_num_batches=args.eval_num_batches,
        policy_metadata={"eval_task_names": sorted(EXPECTED_TASKS)},
        wandb_enabled=args.wandb,
        resume=args.resume,
        overwrite=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-params", type=Path, default=BASE_PARAMS)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--peak-lr", type=float, default=3e-5)
    parser.add_argument("--decay-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--save-interval", type=int, default=1_000)
    parser.add_argument("--keep-period", type=int, default=5_000)
    parser.add_argument("--eval-interval", type=int, default=1_000)
    parser.add_argument("--eval-num-batches", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--exp-name", default="piper_cup_rack_pour_phase_lora_v1")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or not 0 <= args.warmup_steps < args.steps:
        parser.error("Invalid batch size or warmup/steps")
    if args.save_interval <= 0 or args.keep_period <= 0 or args.peak_lr <= 0:
        parser.error("Invalid save interval, keep period, or learning rate")

    dataset_root = Path(os.environ["HF_LEROBOT_HOME"]) / REPO_ID
    metadata = LeRobotDatasetMetadata(REPO_ID, root=dataset_root)
    tasks = set(metadata.tasks.values())
    if metadata.total_episodes != 355 or tasks != EXPECTED_TASKS:
        raise ValueError(
            f"Expected merged 355-episode, 8-task dataset; "
            f"got episodes={metadata.total_episodes}, tasks={sorted(tasks)}"
        )
    cfg = build_config(args)
    norm = ROOT / "assets" / ASSET_CONFIG_NAME / REPO_ID / "norm_stats.json"
    if not args.base_params.is_dir() or not norm.is_file():
        raise FileNotFoundError(f"Missing base params or norm stats: base={args.base_params}, norm={norm}")

    print(f"Dataset: {dataset_root}")
    print(f"Episodes: {metadata.total_episodes}; frames: {metadata.total_frames}; tasks: {metadata.total_tasks}")
    print(f"Initialization: {args.base_params}")
    print(f"Model: PI0.5 LoRA VLM + LoRA action expert; discrete state input=True")
    print(f"Norm stats: {norm}")
    print(f"Output: {cfg.checkpoint_dir}")
    if args.dry_run:
        return
    checkpoint_io = trainable_checkpoints.CompactTrainableCheckpointIO(
        base_checkpoint=str(args.base_params),
        parameter_filter=cfg.trainable_filter,
    )
    train.main(cfg, checkpoint_io=checkpoint_io, eval_config=cfg)


if __name__ == "__main__":
    main()
