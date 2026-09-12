import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import lora_adapters
from openpi.training import checkpoints
from openpi.training import lora_checkpoints
from openpi.training import utils as training_utils


class _TinyModel(nnx.Module):
    def __init__(self, *, kernel: float, lora: float):
        self.kernel = nnx.Param(jnp.asarray([kernel], dtype=jnp.float32))
        self.lora_a = nnx.Param(jnp.asarray([lora], dtype=jnp.float32))


class _DataLoader:
    class _Config:
        norm_stats = None
        asset_id = None

    def data_config(self):
        return self._Config()


def _state(*, kernel: float, lora: float, step: int) -> training_utils.TrainState:
    model = _TinyModel(kernel=kernel, lora=lora)
    graphdef, params = nnx.split(model)
    trainable = params.filter(nnx.PathContains("lora_a"))
    tx = optax.adam(1e-3)
    return training_utils.TrainState(
        step=jnp.asarray(step),
        params=params,
        model_def=graphdef,
        tx=tx,
        opt_state=tx.init(trainable),
        ema_decay=None,
        ema_params=None,
    )


def test_compact_checkpoint_is_resumable_and_deployable(tmp_path: pathlib.Path):
    checkpoint_dir = tmp_path / "checkpoints"
    backend = lora_checkpoints.CompactLoraCheckpointIO(base_checkpoint="base/params")
    manager, resuming = backend.initialize_checkpoint_dir(
        checkpoint_dir, keep_period=100, overwrite=False, resume=False
    )
    assert not resuming

    trained = _state(kernel=7.0, lora=3.0, step=12)
    backend.save_state(manager, trained, _DataLoader(), 12)
    manager.wait_until_finished()

    step_dir = checkpoint_dir / "12"
    assert (step_dir / "adapter" / "adapter.npz").is_file()
    assert (step_dir / "adapter" / "adapter.json").is_file()
    assert (step_dir / "train_state").is_dir()
    assert not (step_dir / "params").exists()

    # Resume overlays only LoRA: the fixed base kernel remains untouched.
    base = _state(kernel=11.0, lora=0.0, step=0)
    restored = backend.restore_state(manager, base, _DataLoader())
    assert int(restored.step) == 12
    np.testing.assert_array_equal(restored.params.to_pure_dict()["kernel"], np.array([11.0]))
    np.testing.assert_array_equal(restored.params.to_pure_dict()["lora_a"], np.array([3.0]))

    # The numeric step directory itself is accepted by the deployment loader.
    deployed = lora_adapters.apply_adapter(base.params.to_pure_dict(), step_dir)
    np.testing.assert_array_equal(deployed["kernel"], np.array([11.0]))
    np.testing.assert_array_equal(deployed["lora_a"], np.array([3.0]))
    manager.close()


def test_async_adapter_save_survives_deleted_training_buffers(tmp_path: pathlib.Path):
    checkpoint_dir = tmp_path / "checkpoints"
    backend = lora_checkpoints.CompactLoraCheckpointIO(base_checkpoint="base/params")
    manager, _ = backend.initialize_checkpoint_dir(checkpoint_dir, keep_period=100, overwrite=False, resume=False)

    trained = _state(kernel=7.0, lora=3.0, step=12)
    backend.save_state(manager, trained, _DataLoader(), 12)
    # Model buffer donation can invalidate the state immediately after save()
    # returns. The adapter callback must only retain independent host arrays.
    for value in jax.tree.leaves(trained.params):
        value.delete()
    manager.wait_until_finished()

    with np.load(checkpoint_dir / "12" / "adapter" / "adapter.npz") as archive:
        np.testing.assert_array_equal(archive["lora_a"], np.array([3.0]))
    manager.close()


def test_resume_migrated_standard_checkpoint(tmp_path: pathlib.Path):
    checkpoint_dir = tmp_path / "standard_checkpoints"
    standard_manager, _ = checkpoints.initialize_checkpoint_dir(
        checkpoint_dir, keep_period=100, overwrite=False, resume=False
    )
    trained = _state(kernel=7.0, lora=4.0, step=13)
    checkpoints.save_state(standard_manager, trained, _DataLoader(), 12)
    standard_manager.wait_until_finished()
    standard_manager.close()

    # Migration adds the deployable adapter without modifying the old resume state.
    lora_adapters.save_adapter(trained.params.to_pure_dict(), checkpoint_dir / "12" / "adapter")

    backend = lora_checkpoints.CompactLoraCheckpointIO(base_checkpoint="base/params")
    compact_manager, resuming = backend.initialize_checkpoint_dir(
        checkpoint_dir, keep_period=100, overwrite=False, resume=True
    )
    assert resuming
    assert compact_manager.latest_step() == 12

    base = _state(kernel=11.0, lora=0.0, step=0)
    restored = backend.restore_state(compact_manager, base, _DataLoader())
    assert int(restored.step) == 13
    np.testing.assert_array_equal(restored.params.to_pure_dict()["kernel"], np.array([11.0]))
    np.testing.assert_array_equal(restored.params.to_pure_dict()["lora_a"], np.array([4.0]))
    compact_manager.close()
