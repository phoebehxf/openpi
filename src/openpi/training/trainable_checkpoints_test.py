import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.training import checkpoints
from openpi.training import trainable_checkpoints
from openpi.training import utils as training_utils


class _TinyModel(nnx.Module):
    def __init__(self, *, frozen: float, kernel: float, lora: float):
        self.frozen = nnx.Param(jnp.asarray([frozen], dtype=jnp.float32))
        self.kernel = nnx.Param(jnp.asarray([kernel], dtype=jnp.float32))
        self.lora_a = nnx.Param(jnp.asarray([lora], dtype=jnp.float32))


class _DataLoader:
    class _Config:
        norm_stats = None
        asset_id = None

    def data_config(self):
        return self._Config()


TRAINABLE = nnx.Any(nnx.PathContains("kernel"), nnx.PathContains("lora_a"))


def _state(*, frozen: float, kernel: float, lora: float, step: int):
    model = _TinyModel(frozen=frozen, kernel=kernel, lora=lora)
    graphdef, params = nnx.split(model)
    tx = optax.adam(1e-3)
    return training_utils.TrainState(
        step=jnp.asarray(step), params=params, model_def=graphdef, tx=tx,
        opt_state=tx.init(params.filter(TRAINABLE)), ema_decay=None, ema_params=None,
    )


def _backend():
    return trainable_checkpoints.CompactTrainableCheckpointIO(
        base_checkpoint="base/params", parameter_filter=TRAINABLE
    )


def test_compact_checkpoint_restores_all_trainable_params(tmp_path: pathlib.Path):
    tmp_path = tmp_path / "checkpoints"
    backend = _backend()
    manager, _ = backend.initialize_checkpoint_dir(tmp_path, keep_period=None, overwrite=False, resume=False)
    trained = _state(frozen=1, kernel=7, lora=8, step=12)
    backend.save_state(manager, trained, _DataLoader(), 12)
    assert (tmp_path / "12/overlay/overlay.npz").is_file()
    assert (tmp_path / "12/train_state").is_dir()
    assert not (tmp_path / "12/params").exists()
    restored = backend.restore_state(manager, _state(frozen=11, kernel=0, lora=0, step=0), _DataLoader())
    pure = restored.params.to_pure_dict()
    assert int(restored.step) == 12
    np.testing.assert_array_equal(pure["frozen"], np.array([11.0]))
    np.testing.assert_array_equal(pure["kernel"], np.array([7.0]))
    np.testing.assert_array_equal(pure["lora_a"], np.array([8.0]))
    manager.close()


def test_save_finishes_before_training_buffers_are_deleted(tmp_path: pathlib.Path):
    tmp_path = tmp_path / "checkpoints"
    backend = _backend()
    manager, _ = backend.initialize_checkpoint_dir(tmp_path, keep_period=None, overwrite=False, resume=False)
    trained = _state(frozen=1, kernel=7, lora=8, step=12)
    backend.save_state(manager, trained, _DataLoader(), 12)
    for value in jax.tree.leaves(trained.params):
        value.delete()
    with np.load(tmp_path / "12/overlay/overlay.npz") as archive:
        np.testing.assert_array_equal(archive["kernel"], np.array([7.0]))
    manager.close()


def test_resume_existing_standard_checkpoint(tmp_path: pathlib.Path):
    tmp_path = tmp_path / "checkpoints"
    standard_manager, _ = checkpoints.initialize_checkpoint_dir(
        tmp_path, keep_period=None, overwrite=False, resume=False
    )
    trained = _state(frozen=1, kernel=7, lora=8, step=13)
    checkpoints.save_state(standard_manager, trained, _DataLoader(), 12)
    standard_manager.wait_until_finished()
    standard_manager.close()
    backend = _backend()
    manager, resuming = backend.initialize_checkpoint_dir(tmp_path, keep_period=None, overwrite=False, resume=True)
    assert resuming
    restored = backend.restore_state(manager, _state(frozen=0, kernel=0, lora=0, step=0), _DataLoader())
    assert int(restored.step) == 13
    np.testing.assert_array_equal(restored.params.to_pure_dict()["kernel"], np.array([7.0]))
    manager.close()
