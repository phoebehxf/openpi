#!/usr/bin/env python3
"""Full-rank pi0.5 fine-tuning for the unified Piper manipulation dataset."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

import flax.nnx as nnx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_LEROBOT_HOME", str(ROOT / "local_datasets"))

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

try:
    from scripts import compute_norm_stats
    from scripts import train
except ImportError:  # Direct execution adds scripts/ rather than the repo root to sys.path.
    import compute_norm_stats
    import train
from openpi.models import pi0_config
from openpi.shared import normalize
from openpi.training import config
from openpi.training import optimizer
from openpi.training import weight_loaders
import openpi.transforms as transforms

REPO_ID = "local/piper-all-manipulation-cleaned-v1"
CONFIG_NAME = "pi05_piper_multitask_full"
ASSET_ROOT = ROOT / "assets"


@dataclasses.dataclass(frozen=True)
class RandomDropNormalizedState(transforms.DataTransformFn):
    """Replace the complete normalized state with its neutral value for one sample."""

    probability: float

    def __call__(self, data: dict) -> dict:
        if self.probability > 0 and np.random.random() < self.probability:
            result = dict(data)
            # This transform runs after Normalize and before TokenizePrompt. Zero
            # therefore means the normalization midpoint, not a zero joint pose.
            result["state"] = np.zeros_like(np.asarray(data["state"]))
            return result
        return data


@dataclasses.dataclass(frozen=True)
class PiperDataWithStateDropout(config.LeRobotPiperDataConfig):
    state_dropout_probability: float = 0.0

    def create(self, assets_dirs: Path, model_config):
        data = super().create(assets_dirs, model_config)
        if self.state_dropout_probability <= 0:
            return data
        model_transforms = transforms.Group(
            inputs=(
                RandomDropNormalizedState(self.state_dropout_probability),
                *data.model_transforms.inputs,
            ),
            outputs=data.model_transforms.outputs,
        )
        return dataclasses.replace(data, model_transforms=model_transforms)


def build_config(args: argparse.Namespace, *, state_dropout: float) -> config.TrainConfig:
    model = pi0_config.Pi0Config(
        dtype="float32",
        pi05=True,
        action_horizon=10,
        discrete_state_input=True,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
    )
    data = PiperDataWithStateDropout(
        repo_id=args.repo_id,
        base_config=config.DataConfig(prompt_from_task=True),
        assets=config.AssetsConfig(asset_id=args.repo_id, assets_dir=str(ASSET_ROOT / CONFIG_NAME)),
        use_delta_joint_actions=False,
        state_dropout_probability=state_dropout,
    )
    return config.TrainConfig(
        name=CONFIG_NAME,
        model=model,
        data=data,
        exp_name=args.exp_name,
        assets_base_dir=str(ASSET_ROOT),
        checkpoint_base_dir=str(ROOT / "checkpoints"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.steps,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=args.warmup_steps,
            peak_lr=args.peak_lr,
            decay_steps=args.lr_decay_steps,
            decay_lr=args.decay_lr,
        ),
        optimizer=optimizer.AdamW(clip_gradient_norm=args.clip_gradient_norm),
        # No LoRA variants + nnx.Nothing means every parameter is trainable.
        freeze_filter=nnx.Nothing(),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader(args.base_params),
        save_interval=args.save_interval,
        keep_period=args.keep_period,
        eval_interval=args.eval_interval,
        eval_num_batches=args.eval_num_batches,
        fsdp_devices=args.fsdp_devices,
        wandb_enabled=args.wandb,
        resume=args.resume,
        overwrite=False,
    )


def compute_stats(cfg: config.TrainConfig, *, max_frames: int | None) -> None:
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    loader, num_batches = compute_norm_stats.create_torch_dataloader(
        data_config,
        cfg.model.action_horizon,
        cfg.batch_size,
        cfg.model,
        cfg.num_workers,
        max_frames,
    )
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    import tqdm

    for batch in tqdm.tqdm(loader, total=num_batches, desc="Computing normalization stats"):
        for key in stats:
            stats[key].update(np.asarray(batch[key]))
    output = cfg.assets_dirs / cfg.data.repo_id
    normalize.save(output, {key: value.get_statistics() for key, value in stats.items()})
    print(f"Wrote normalization stats: {output / 'norm_stats.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--exp-name", default="piper_all_manipulation_full_v1")
    parser.add_argument("--base-params", default="gs://openpi-assets/checkpoints/pi05_base/params")
    parser.add_argument("--steps", type=int, default=40_000, help="Cumulative final step for this stage.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--peak-lr", type=float, default=5e-6)
    parser.add_argument("--decay-lr", type=float, default=5e-7)
    parser.add_argument("--lr-decay-steps", type=int, default=40_000)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--clip-gradient-norm", type=float, default=0.3)
    parser.add_argument("--state-dropout", type=float, default=0.0)
    parser.add_argument("--save-interval", type=int, default=1_000)
    parser.add_argument("--keep-period", type=int, default=5_000)
    parser.add_argument("--eval-interval", type=int, default=1_000)
    parser.add_argument("--eval-num-batches", type=int, default=4)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--max-norm-frames", type=int, default=50_000)
    parser.add_argument("--compute-norm", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.state_dropout <= 1.0:
        parser.error("--state-dropout must be in [0, 1]")
    if args.batch_size <= 0 or args.steps <= 0 or args.lr_decay_steps <= 0 or args.fsdp_devices <= 0:
        parser.error("batch size, steps, and fsdp-devices must be positive")

    dataset_root = Path(os.environ["HF_LEROBOT_HOME"]) / args.repo_id
    metadata = LeRobotDatasetMetadata(args.repo_id, root=dataset_root)
    cfg = build_config(args, state_dropout=args.state_dropout)
    task_names = [metadata.tasks[index] for index in sorted(metadata.tasks)]
    cfg = dataclasses.replace(cfg, policy_metadata={"eval_task_names": task_names})
    print(f"Dataset: {args.repo_id}; episodes={metadata.total_episodes}; frames={metadata.total_frames}")
    print(f"Tasks ({len(metadata.tasks)}): {metadata.tasks}")
    print("Model: pi0.5 full-rank gemma_2b + gemma_300m; action_horizon=10; discrete state=True")
    print(f"State dropout={args.state_dropout}; final step={args.steps}; resume={args.resume}")
    print(f"Output: {cfg.checkpoint_dir}")

    if args.compute_norm:
        compute_stats(cfg, max_frames=args.max_norm_frames)
        return

    # Configuration/data validation should work before normalization assets
    # exist; it deliberately does not construct the actual training loader.
    if args.dry_run:
        print("Dry run complete (model weights and training batches were not loaded).")
        return

    norm_path = cfg.assets_dirs / args.repo_id / "norm_stats.json"
    if not norm_path.is_file():
        raise FileNotFoundError(f"Missing {norm_path}; run this script once with --compute-norm")

    # Evaluation uses identical data/model settings but never corrupts state.
    eval_cfg = build_config(args, state_dropout=0.0)
    eval_cfg = dataclasses.replace(eval_cfg, policy_metadata=cfg.policy_metadata)
    train.main(cfg, eval_config=eval_cfg)


if __name__ == "__main__":
    main()
