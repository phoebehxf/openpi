import pathlib

import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import parameter_overlays


def test_overlay_round_trip_and_base_validation(tmp_path: pathlib.Path):
    base = {"frozen": jnp.array([1.0]), "block": {"kernel": jnp.array([2.0, 3.0])}}
    changed = {"block": {"kernel": jnp.array([7.0, 8.0])}}
    flat = parameter_overlays.extract_overlay_params(changed)
    step_dir = tmp_path / "12"
    parameter_overlays.save_overlay(flat, step_dir / "overlay", source_checkpoint="base/params")
    restored = parameter_overlays.apply_overlay(base, step_dir, source_checkpoint="base/params")
    np.testing.assert_array_equal(restored["frozen"], np.array([1.0]))
    np.testing.assert_array_equal(restored["block"]["kernel"], np.array([7.0, 8.0]))
    with pytest.raises(ValueError, match="base mismatch"):
        parameter_overlays.apply_overlay(base, step_dir, source_checkpoint="other/params")


def test_overlay_rejects_incompatible_parameter_tree(tmp_path: pathlib.Path):
    flat = parameter_overlays.extract_overlay_params({"kernel": jnp.ones((2,))})
    parameter_overlays.save_overlay(flat, tmp_path, source_checkpoint="base/params")
    with pytest.raises(ValueError, match="Shape mismatch"):
        parameter_overlays.apply_overlay({"kernel": jnp.ones((3,))}, tmp_path)
