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
from openpi.models import parameter_overlays

REPO_ID = "local/piper-cup-rack-multitask-cleaned-v2"
BASE_NAME = "pi05_piper_pick_and_place_v2"
STATE_CONFIG_NAME = "pi05_piper_pick_and_place_v3"
NORM_ID = "phoebe777777/piper-pick-up-v2"
BASE_PARAMS = ROOT / "checkpoints" / BASE_NAME / "piper_pick_pi05_lora_v4/69999/params"
SOURCE_STEP = ROOT / "checkpoints" / BASE_NAME / "piper_cup_rack_multitask_cleaned_openpi_lora_v1/22500"


@dataclasses.dataclass(frozen=True)
class BaseWithOverlayWeightLoader:
    """Initialize a new experiment from a compact overlay on top of its base."""

    params_path: str
    overlay_path: str

    def load(self, params):
        base_params = weight_loaders.CheckpointWeightLoader(self.params_path).load(params)
        return parameter_overlays.apply_overlay(
            base_params, self.overlay_path, source_checkpoint=self.params_path
        )


def build_config(args):
    config_name = BASE_NAME if args.legacy else STATE_CONFIG_NAME
    base = config.get_config(config_name)
    model = dataclasses.replace(base.model, discrete_state_input=not args.legacy)
    data = dataclasses.replace(
        base.data,
        repo_id=REPO_ID,
        base_config=dataclasses.replace(base.data.base_config, prompt_from_task=True),
        assets=dataclasses.replace(
            base.data.assets,
            asset_id=NORM_ID,
            assets_dir=str(ROOT / "assets" / BASE_NAME),
        ),
    )
    return dataclasses.replace(
        base,
        model=model,
        data=data,
        exp_name=args.exp_name
        or (
            "piper_cup_rack_multitask_cleaned_openpi_lora_v1"
            if args.legacy
            else "piper_cup_rack_multitask_state_input_v1"
        ),
        assets_base_dir=str(ROOT / "assets"),
        checkpoint_base_dir=str(ROOT / "checkpoints"),
        weight_loader=(
            weight_loaders.CheckpointWeightLoader(str(BASE_PARAMS))
            if args.legacy
            else BaseWithOverlayWeightLoader(params_path=str(BASE_PARAMS), overlay_path=str(SOURCE_STEP))
        ),
        freeze_filter=model.get_freeze_filter(),
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
    parser.add_argument("--exp-name", default=None)
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use the original v2/no-state experiment; combine with --resume to continue it.",
    )
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
    norm = Path(cfg.data.assets.assets_dir) / NORM_ID / "norm_stats.json"
    source_overlay = SOURCE_STEP / "overlay" / parameter_overlays.OVERLAY_FILENAME
    if not norm.is_file() or not BASE_PARAMS.is_dir() or (not args.legacy and not source_overlay.is_file()):
        raise FileNotFoundError(
            f"Missing base checkpoint, source overlay, or original Piper stats: "
            f"base={BASE_PARAMS}, overlay={source_overlay}, stats={norm}"
        )
    print(f"Dataset: {dataset_root}\nEpisodes: {meta.total_episodes}; frames: {meta.total_frames}\nTasks: {meta.tasks}")
    print(f"OpenPI default LoRA filter: {cfg.freeze_filter}")
    print(f"Mode: {'legacy v2 (no state)' if args.legacy else 'v3 state-input branch'}")
    print(f"State input: {cfg.model.discrete_state_input}; norm stats: {norm.resolve()}")
    if not args.legacy and not args.resume:
        print(f"Initializing weights from: {SOURCE_STEP.resolve()}")
    print(f"Output: {cfg.checkpoint_dir}")
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
