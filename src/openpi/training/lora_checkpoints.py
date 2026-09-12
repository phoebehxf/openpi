"""Compact, deployable checkpoints for LoRA-only training runs.

Unlike the standard OpenPI checkpoint backend, this backend never writes the
frozen model parameters.  Each checkpoint contains the optimizer/step state and
an exported LoRA adapter.  On resume, the fixed base checkpoint is initialized
normally and the saved adapter is overlaid on it.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib

from etils import epath
import orbax.checkpoint as ocp

from openpi.models import lora_adapters
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.shared.normalize as _normalize
from openpi.training import checkpoints as _checkpoints
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


class CompactLoraCheckpointIO:
    """Checkpoint backend that stores no frozen model tensors."""

    # train.main uses this to initialize the fixed base before applying a saved
    # adapter. The standard backend keeps its existing shape-only restore path.
    initialize_from_base_on_resume = True

    def __init__(
        self,
        *,
        base_checkpoint: str,
        parameter_regex: str = lora_adapters.DEFAULT_PARAMETER_REGEX,
    ) -> None:
        self._base_checkpoint = base_checkpoint
        self._parameter_regex = parameter_regex
        self._checkpoint_dir: pathlib.Path | None = None

    def initialize_checkpoint_dir(
        self,
        checkpoint_dir: epath.Path | str,
        *,
        keep_period: int | None,
        overwrite: bool,
        resume: bool,
    ) -> tuple[ocp.CheckpointManager, bool]:
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
                    f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                    "to indicate how to handle it."
                )

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir = pathlib.Path(checkpoint_dir)
        manager = ocp.CheckpointManager(
            checkpoint_dir,
            item_handlers={
                "assets": _checkpoints.CallbackHandler(),
                "train_state": ocp.PyTreeCheckpointHandler(),
                "adapter": _checkpoints.CallbackHandler(),
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

    def save_state(
        self,
        checkpoint_manager: ocp.CheckpointManager,
        state: training_utils.TrainState,
        data_loader: _data_loader.DataLoader,
        step: int,
    ) -> None:
        if state.ema_params is not None:
            raise ValueError("Compact LoRA checkpoints require ema_decay=None.")

        adapter_state = state.params.filter(nnx_utils.PathRegex(self._parameter_regex))
        # Snapshot on the training thread before handing work to the async
        # callback. The next donated train step may delete the JAX buffers.
        adapter_params = lora_adapters.extract_adapter_params(
            adapter_state.to_pure_dict(), parameter_regex=self._parameter_regex
        )
        # The optimizer was initialized from the LoRA-only trainable filter, so
        # this contains only step + LoRA optimizer moments, not frozen weights.
        with at.disable_typechecking():
            compact_state = dataclasses.replace(state, params={})

        def save_assets(directory: epath.Path) -> None:
            data_config = data_loader.data_config()
            if data_config.norm_stats is not None and data_config.asset_id is not None:
                _normalize.save(directory / data_config.asset_id, data_config.norm_stats)

        def save_adapter(directory: epath.Path) -> None:
            lora_adapters.save_adapter(
                adapter_params,
                pathlib.Path(directory),
                parameter_regex=self._parameter_regex,
                source_checkpoint=self._base_checkpoint,
            )

        checkpoint_manager.save(
            step,
            {
                "assets": save_assets,
                "train_state": compact_state,
                "adapter": save_adapter,
            },
        )

    def restore_state(
        self,
        checkpoint_manager: ocp.CheckpointManager,
        state: training_utils.TrainState,
        data_loader: _data_loader.DataLoader,
        step: int | None = None,
    ) -> training_utils.TrainState:
        del data_loader
        if self._checkpoint_dir is None:
            raise RuntimeError("initialize_checkpoint_dir must be called before restore_state.")
        restore_step = checkpoint_manager.latest_step() if step is None else step
        if restore_step is None:
            raise FileNotFoundError("No completed LoRA checkpoint is available to restore.")

        with at.disable_typechecking():
            compact_target = dataclasses.replace(state, params={}, ema_params=None)
        restored = checkpoint_manager.restore(
            restore_step,
            items={"train_state": compact_target},
        )["train_state"]

        adapter_dir = self._checkpoint_dir / str(restore_step) / "adapter"
        merged_pure = lora_adapters.apply_adapter(
            state.params.to_pure_dict(),
            adapter_dir,
            parameter_regex=self._parameter_regex,
        )
        state.params.replace_by_pure_dict(merged_pure)
        logging.info("Restored compact LoRA checkpoint at step %s", restore_step)
        return dataclasses.replace(restored, params=state.params, ema_params=None)
