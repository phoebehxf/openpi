"""Memory-conscious checkpoints containing only parameters changed by training."""

from __future__ import annotations

import dataclasses
import logging
import pathlib

from etils import epath
import flax.nnx as nnx
import orbax.checkpoint as ocp

from openpi.models import parameter_overlays
from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
from openpi.training import checkpoints as _checkpoints


class CompactTrainableCheckpointIO:
    """Store trainable parameters plus optimizer state, rebuilding from a base."""

    initialize_from_base_on_resume = True

    def __init__(self, *, base_checkpoint: str, parameter_filter: nnx.filterlib.Filter) -> None:
        self._base_checkpoint = str(pathlib.Path(base_checkpoint).expanduser().resolve())
        self._parameter_filter = parameter_filter
        self._checkpoint_dir: pathlib.Path | None = None

    def initialize_checkpoint_dir(self, checkpoint_dir, *, keep_period, overwrite, resume):
        checkpoint_dir = epath.Path(checkpoint_dir).resolve()
        resuming = False
        if checkpoint_dir.exists():
            if overwrite:
                checkpoint_dir.rmtree()
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                logging.info("Wiped checkpoint directory %s", checkpoint_dir)
            elif resume:
                resuming = True
            else:
                raise FileExistsError(
                    f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume."
                )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir = pathlib.Path(checkpoint_dir)
        manager = ocp.CheckpointManager(
            checkpoint_dir,
            item_handlers={
                "assets": _checkpoints.CallbackHandler(),
                "train_state": ocp.PyTreeCheckpointHandler(),
                "overlay": _checkpoints.CallbackHandler(),
            },
            options=ocp.CheckpointManagerOptions(
                max_to_keep=2,
                keep_period=keep_period,
                create=False,
                async_options=ocp.AsyncOptions(timeout_secs=7200),
            ),
        )
        if resuming and tuple(manager.all_steps()) in [(), (0,)]:
            logging.info("Checkpoint directory contains no completed checkpoints; not resuming.")
            resuming = False
        return manager, resuming

    def save_state(self, checkpoint_manager, state, data_loader, step):
        if state.ema_params is not None:
            raise ValueError("Compact trainable checkpoints require ema_decay=None.")
        trainable_state = state.params.filter(self._parameter_filter)
        overlay_params = parameter_overlays.extract_overlay_params(trainable_state.to_pure_dict())
        with at.disable_typechecking():
            compact_state = dataclasses.replace(state, params={})

        def save_assets(directory: epath.Path) -> None:
            data_config = data_loader.data_config()
            if data_config.norm_stats is not None and data_config.asset_id is not None:
                _normalize.save(directory / data_config.asset_id, data_config.norm_stats)

        def save_overlay(directory: epath.Path) -> None:
            parameter_overlays.save_overlay(
                overlay_params,
                pathlib.Path(directory),
                source_checkpoint=self._base_checkpoint,
            )

        checkpoint_manager.save(
            step,
            {"assets": save_assets, "train_state": compact_state, "overlay": save_overlay},
        )
        # Surface failures here and never donate state while Orbax is reading it.
        checkpoint_manager.wait_until_finished()

    def restore_state(self, checkpoint_manager, state, data_loader, step=None):
        del data_loader
        if self._checkpoint_dir is None:
            raise RuntimeError("initialize_checkpoint_dir must be called before restore_state.")
        restore_step = checkpoint_manager.latest_step() if step is None else step
        if restore_step is None:
            raise FileNotFoundError("No completed checkpoint is available to restore.")
        step_dir = self._checkpoint_dir / str(restore_step)

        if (step_dir / "overlay" / parameter_overlays.OVERLAY_FILENAME).is_file():
            with at.disable_typechecking():
                compact_target = dataclasses.replace(state, params={}, ema_params=None)
            restored = checkpoint_manager.restore(
                restore_step, items={"train_state": compact_target}
            )["train_state"]
            merged = parameter_overlays.apply_overlay(
                state.params.to_pure_dict(),
                step_dir,
                source_checkpoint=self._base_checkpoint,
            )
            state.params.replace_by_pure_dict(merged)
            logging.info("Restored trainable overlay checkpoint at step %s", restore_step)
            return dataclasses.replace(restored, params=state.params, ema_params=None)

        if (step_dir / "params").is_dir():
            logging.info("Migrating standard full checkpoint at step %s", restore_step)
            standard_manager = ocp.CheckpointManager(
                self._checkpoint_dir,
                item_handlers={
                    "assets": _checkpoints.CallbackHandler(),
                    "train_state": ocp.PyTreeCheckpointHandler(),
                    "params": ocp.PyTreeCheckpointHandler(),
                },
                options=ocp.CheckpointManagerOptions(create=False),
            )
            try:
                return _checkpoints.restore_state(standard_manager, state, None, step=restore_step)
            finally:
                standard_manager.close()
        raise FileNotFoundError(f"Checkpoint step {restore_step} has neither overlay nor params: {step_dir}")
