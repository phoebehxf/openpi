"""Utilities for exporting and loading small, task-specific LoRA adapters.

OpenPI training checkpoints contain the complete parameter tree, even when only
LoRA parameters are trainable. This module stores just those parameters in a
compressed NumPy archive and applies them on top of a shared base checkpoint at
inference time.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any

import flax.traverse_util
import jax
import numpy as np

import openpi.shared.array_typing as at

FORMAT_VERSION = 1
DEFAULT_PARAMETER_REGEX = ".*lora.*"
ADAPTER_FILENAME = "adapter.npz"
MANIFEST_FILENAME = "adapter.json"


def _resolve_adapter_file(adapter_path: pathlib.Path | str) -> pathlib.Path:
    path = pathlib.Path(adapter_path).expanduser().resolve()
    if not path.is_dir():
        return path
    direct_path = path / ADAPTER_FILENAME
    if direct_path.is_file():
        return direct_path
    return path / "adapter" / ADAPTER_FILENAME


def extract_adapter_params(
    params: at.Params, *, parameter_regex: str = DEFAULT_PARAMETER_REGEX
) -> dict[str, np.ndarray]:
    """Flatten and copy all parameters whose full path matches ``parameter_regex``."""
    pattern = re.compile(parameter_regex)
    flat_params = flax.traverse_util.flatten_dict(params, sep="/")
    adapter = {
        # The explicit copy is important for asynchronous checkpointing: a
        # zero-copy host view may otherwise outlive a donated/deleted JAX
        # buffer when training advances while the checkpoint is compressed.
        path: np.array(jax.device_get(value), copy=True)
        for path, value in flat_params.items()
        if pattern.fullmatch(path)
    }
    if not adapter:
        raise ValueError(f"No parameters matched adapter regex {parameter_regex!r}.")
    return adapter


def save_adapter(
    params: at.Params,
    output_dir: pathlib.Path | str,
    *,
    parameter_regex: str = DEFAULT_PARAMETER_REGEX,
    source_checkpoint: str | None = None,
) -> pathlib.Path:
    """Save matching parameters and a human-readable manifest to ``output_dir``."""
    output_dir = pathlib.Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = extract_adapter_params(params, parameter_regex=parameter_regex)

    adapter_path = output_dir / ADAPTER_FILENAME
    np.savez_compressed(adapter_path, **adapter)
    manifest: dict[str, Any] = {
        "format": "openpi_lora_adapter",
        "format_version": FORMAT_VERSION,
        "parameter_regex": parameter_regex,
        "source_checkpoint": source_checkpoint,
        "parameter_count": len(adapter),
        "parameters": {
            path: {"shape": list(value.shape), "dtype": str(value.dtype)} for path, value in adapter.items()
        },
    }
    (output_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return adapter_path


def apply_adapter(
    params: at.Params,
    adapter_path: pathlib.Path | str,
    *,
    parameter_regex: str = DEFAULT_PARAMETER_REGEX,
) -> at.Params:
    """Strictly overlay an adapter onto an existing base parameter tree."""
    archive_path = _resolve_adapter_file(adapter_path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"LoRA adapter archive does not exist: {archive_path}")

    pattern = re.compile(parameter_regex)
    flat_params = flax.traverse_util.flatten_dict(params, sep="/")
    result = dict(flat_params)
    with np.load(archive_path, allow_pickle=False) as archive:
        adapter_paths = set(archive.files)
        if not adapter_paths:
            raise ValueError(f"LoRA adapter is empty: {archive_path}")
        expected_paths = {path for path in flat_params if pattern.fullmatch(path)}
        missing_paths = expected_paths - adapter_paths
        if missing_paths:
            preview = ", ".join(sorted(missing_paths)[:3])
            raise ValueError(f"Adapter is missing {len(missing_paths)} LoRA parameter(s), including: {preview}")
        for path in sorted(adapter_paths):
            if not pattern.fullmatch(path):
                raise ValueError(f"Adapter contains a non-LoRA parameter: {path}")
            if path not in flat_params:
                raise ValueError(f"Adapter parameter is absent from the base model: {path}")
            value = archive[path]
            reference = flat_params[path]
            if value.shape != reference.shape:
                raise ValueError(
                    f"Shape mismatch for adapter parameter {path}: adapter {value.shape}, base {reference.shape}"
                )
            result[path] = value.astype(reference.dtype, copy=False)

    return flax.traverse_util.unflatten_dict(result, sep="/")
