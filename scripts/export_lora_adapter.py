#!/usr/bin/env python3
"""Extract the LoRA tensors from an OpenPI training checkpoint."""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import tyro

from openpi.models import lora_adapters
from openpi.models import model as _model


@dataclasses.dataclass(frozen=True)
class Args:
    # Training step directory or its params directory.
    checkpoint_dir: str
    # Destination directory for adapter.npz and adapter.json.
    output_dir: str
    # Full-match regex selecting adapter parameter paths.
    parameter_regex: str = lora_adapters.DEFAULT_PARAMETER_REGEX


def main(args: Args) -> None:
    checkpoint_dir = pathlib.Path(args.checkpoint_dir).expanduser().resolve()
    params_path = checkpoint_dir if checkpoint_dir.name == "params" else checkpoint_dir / "params"
    if not params_path.is_dir():
        raise FileNotFoundError(f"Checkpoint params directory does not exist: {params_path}")

    params = _model.restore_params(params_path, restore_type=np.ndarray)
    adapter_path = lora_adapters.save_adapter(
        params,
        args.output_dir,
        parameter_regex=args.parameter_regex,
        source_checkpoint=str(checkpoint_dir),
    )
    size_mib = adapter_path.stat().st_size / (1024 * 1024)
    print(f"Saved LoRA adapter to {adapter_path} ({size_mib:.1f} MiB)")


if __name__ == "__main__":
    main(tyro.cli(Args))
