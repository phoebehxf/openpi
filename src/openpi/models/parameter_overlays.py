"""Save and apply explicit parameter overlays on top of a fixed base checkpoint."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import flax.traverse_util
import jax
import numpy as np

import openpi.shared.array_typing as at

FORMAT_VERSION = 1
OVERLAY_FILENAME = "overlay.npz"
MANIFEST_FILENAME = "overlay.json"


def _resolve_overlay_dir(path: pathlib.Path | str) -> pathlib.Path:
    path = pathlib.Path(path).expanduser().resolve()
    if (path / OVERLAY_FILENAME).is_file():
        return path
    return path / "overlay"


def extract_overlay_params(params: at.Params) -> dict[str, np.ndarray]:
    """Copy every supplied parameter to independent host memory."""
    flat = flax.traverse_util.flatten_dict(params, sep="/")
    overlay = {path: np.array(jax.device_get(value), copy=True) for path, value in flat.items()}
    if not overlay:
        raise ValueError("Parameter overlay is empty.")
    return overlay


def save_overlay(
    flat_params: dict[str, np.ndarray],
    output_dir: pathlib.Path | str,
    *,
    source_checkpoint: str,
) -> pathlib.Path:
    """Write an already snapshotted flat overlay without another array copy."""
    if not flat_params:
        raise ValueError("Parameter overlay is empty.")
    output_dir = pathlib.Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / OVERLAY_FILENAME
    # Uncompressed storage avoids a second large CPU/memory spike during save.
    np.savez(archive_path, **flat_params)
    manifest: dict[str, Any] = {
        "format": "openpi_parameter_overlay",
        "format_version": FORMAT_VERSION,
        "source_checkpoint": str(pathlib.Path(source_checkpoint).expanduser().resolve()),
        "parameter_count": len(flat_params),
        "element_count": sum(value.size for value in flat_params.values()),
        "parameters": {
            path: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for path, value in flat_params.items()
        },
    }
    (output_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return archive_path


def apply_overlay(
    params: at.Params,
    overlay_path: pathlib.Path | str,
    *,
    source_checkpoint: pathlib.Path | str | None = None,
) -> at.Params:
    """Strictly overlay the manifest-listed parameters onto a base tree."""
    overlay_dir = _resolve_overlay_dir(overlay_path)
    archive_path = overlay_dir / OVERLAY_FILENAME
    manifest_path = overlay_dir / MANIFEST_FILENAME
    if not archive_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Parameter overlay is incomplete: {overlay_dir}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "openpi_parameter_overlay" or manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported parameter overlay manifest: {manifest_path}")
    if source_checkpoint is not None:
        expected_source = pathlib.Path(source_checkpoint).expanduser().resolve()
        recorded_source = pathlib.Path(manifest["source_checkpoint"]).expanduser().resolve()
        if expected_source != recorded_source:
            raise ValueError(f"Overlay base mismatch: expected {recorded_source}, got {expected_source}")

    flat_base = flax.traverse_util.flatten_dict(params, sep="/")
    result = dict(flat_base)
    expected_paths = set(manifest.get("parameters", {}))
    with np.load(archive_path, allow_pickle=False) as archive:
        archive_paths = set(archive.files)
        if archive_paths != expected_paths:
            raise ValueError("Overlay archive paths do not match its manifest.")
        for path in sorted(archive_paths):
            if path not in flat_base:
                raise ValueError(f"Overlay parameter is absent from the base model: {path}")
            value = archive[path]
            reference = flat_base[path]
            declared = manifest["parameters"][path]
            if list(value.shape) != declared["shape"] or str(value.dtype) != declared["dtype"]:
                raise ValueError(f"Overlay tensor does not match its manifest: {path}")
            if value.shape != reference.shape:
                raise ValueError(
                    f"Shape mismatch for overlay parameter {path}: overlay {value.shape}, base {reference.shape}"
                )
            result[path] = value.astype(reference.dtype, copy=False)
    return flax.traverse_util.unflatten_dict(result, sep="/")


def is_parameter_overlay(path: pathlib.Path | str) -> bool:
    return (_resolve_overlay_dir(path) / MANIFEST_FILENAME).is_file()
