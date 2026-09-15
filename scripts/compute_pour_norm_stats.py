#!/usr/bin/env python3
"""Compute normalization stats for the cleaned, merged pour-water dataset."""

import argparse
import dataclasses
import os
from pathlib import Path

import numpy as np
import tqdm

ROOT = Path(__file__).resolve().parents[1]
# LeRobot reads this variable while its modules are imported.
os.environ["HF_LEROBOT_HOME"] = str(ROOT / "local_datasets")

import compute_norm_stats as _compute  # noqa: E402

import openpi.shared.normalize as normalize  # noqa: E402
from openpi.training import config  # noqa: E402

BASE_NAME = "pi05_piper_pick_and_place_v2"
DEFAULT_REPO_ID = "local/piper-pour-water-cleaned-merged-v1"


def build_data_config(repo_id: str):
    base = config.get_config(BASE_NAME)
    model = dataclasses.replace(base.model, discrete_state_input=True)
    data_factory = dataclasses.replace(
        base.data,
        repo_id=repo_id,
        base_config=dataclasses.replace(base.data.base_config, prompt_from_task=True),
        assets=dataclasses.replace(base.data.assets, asset_id=repo_id),
    )
    return model, data_factory.create(ROOT / "assets" / BASE_NAME, model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers must be non-negative")
    if args.max_frames is not None and args.max_frames < args.batch_size:
        parser.error("max-frames must be at least one batch")

    model, data_config = build_data_config(args.repo_id)
    output_dir = ROOT / "assets" / BASE_NAME / args.repo_id
    output_file = output_dir / "norm_stats.json"
    print(f"Dataset: {args.repo_id}")
    print(f"State input: {model.discrete_state_input}")
    print(f"Output: {output_file}")
    if args.dry_run:
        return
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"{output_file} already exists; pass --overwrite to replace it")

    loader, num_batches = _compute.create_torch_dataloader(
        data_config,
        model.action_horizon,
        args.batch_size,
        model,
        args.num_workers,
        args.max_frames,
    )
    if num_batches == 0:
        raise ValueError("Dataset does not contain one complete batch")

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in tqdm.tqdm(loader, total=num_batches, desc="Computing pour stats"):
        for key, running_stats in stats.items():
            running_stats.update(np.asarray(batch[key]))

    normalize.save(output_dir, {key: value.get_statistics() for key, value in stats.items()})
    print(f"Wrote: {output_file}")


if __name__ == "__main__":
    main()
