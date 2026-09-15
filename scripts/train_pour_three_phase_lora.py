#!/usr/bin/env python3
"""Fine-tune the three-prompt Piper pouring dataset from the 25k pour adapter."""

# ruff: noqa: E402
import argparse
import dataclasses
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_LEROBOT_HOME"] = str(ROOT / "local_datasets")

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import train

from openpi.models import parameter_overlays
from openpi.training import config
from openpi.training import optimizer
from openpi.training import trainable_checkpoints
from openpi.training import weight_loaders

REPO_ID = "local/piper-pour-water-three-phase-v1"
CONFIG_NAME = "pi05_piper_pick_and_place_v3"
BASE_NAME = "pi05_piper_pick_and_place_v2"
BASE_PARAMS = ROOT / "checkpoints" / BASE_NAME / "piper_pick_pi05_lora_v4/69999/params"
SOURCE_STEP = ROOT / "checkpoints/pi05_piper_pick_and_place_v3" / "piper_pour_water_state_norm_openpi_lora_v2/25000"
EXPECTED_TASKS = {
    "Grasp the bottle and move it next to the red cup",
    "Tilt the bottle to pour water into the red cup",
    "Return the bottle upright, place it down, and release it",
}


@dataclasses.dataclass(frozen=True)
class BaseWithOverlayWeightLoader:
    params_path: str
    overlay_path: str

    def load(self, params):
        base_params = weight_loaders.CheckpointWeightLoader(self.params_path).load(params)
        return parameter_overlays.apply_overlay(
            base_params,
            self.overlay_path,
            source_checkpoint=self.params_path,
        )


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
            assets_dir=str(ROOT / "assets" / BASE_NAME),
        ),
    )
    return dataclasses.replace(
        base,
        model=model,
        data=data,
        exp_name=args.exp_name,
        assets_base_dir=str(ROOT / "assets"),
        checkpoint_base_dir=str(ROOT / "checkpoints"),
        weight_loader=BaseWithOverlayWeightLoader(
            params_path=str(BASE_PARAMS),
            overlay_path=str(SOURCE_STEP),
        ),
        freeze_filter=model.get_freeze_filter(),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--peak-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--save-interval", type=int, default=1_000)
    parser.add_argument("--keep-period", type=int, default=5_000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--exp-name", default="piper_pour_three_phase_state_lora_v1")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or not 0 <= args.warmup_steps < args.steps:
        parser.error("Invalid batch size or warmup/steps")
    if args.save_interval <= 0 or args.keep_period <= 0 or args.peak_lr <= 0:
        parser.error("Invalid save interval, keep period, or learning rate")

    cfg = build_config(args)
    dataset_root = ROOT / "local_datasets" / REPO_ID
    metadata = LeRobotDatasetMetadata(REPO_ID, root=dataset_root)
    tasks = set(metadata.tasks.values())
    if metadata.total_episodes != 150 or tasks != EXPECTED_TASKS:
        raise ValueError(
            f"Expected the 150-episode three-phase dataset; "
            f"got episodes={metadata.total_episodes}, tasks={sorted(tasks)}"
        )

    norm = ROOT / "assets" / BASE_NAME / REPO_ID / "norm_stats.json"
    source_overlay = SOURCE_STEP / "overlay" / parameter_overlays.OVERLAY_FILENAME
    if not BASE_PARAMS.is_dir() or not source_overlay.is_file() or not norm.is_file():
        raise FileNotFoundError(
            "Missing base params, 25k source overlay, or three-phase norm stats:\n"
            f"base={BASE_PARAMS}\nsource={source_overlay}\nnorm={norm}"
        )

    print(f"Dataset: {REPO_ID}")
    print(f"Episodes: {metadata.total_episodes}; frames: {metadata.total_frames}")
    print(f"Tasks: {metadata.tasks}")
    print(f"State input: {cfg.model.discrete_state_input}")
    print(f"Initialization: {BASE_PARAMS.parent} + {SOURCE_STEP}")
    print(f"Norm stats: {norm}")
    print(f"Output: {cfg.checkpoint_dir}")
    print(
        f"steps={args.steps}, batch={args.batch_size}, peak_lr={args.peak_lr}, "
        f"warmup={args.warmup_steps}, resume={args.resume}"
    )
    if not args.dry_run:
        checkpoint_io = trainable_checkpoints.CompactTrainableCheckpointIO(
            base_checkpoint=str(BASE_PARAMS),
            parameter_filter=cfg.trainable_filter,
        )
        train.main(cfg, checkpoint_io=checkpoint_io)


if __name__ == "__main__":
    main()
