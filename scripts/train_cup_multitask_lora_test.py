import argparse

from flax import nnx
import pytest
import train_cup_multitask_lora as training


def test_default_freeze_filter_and_full_checkpoint_config():
    args = argparse.Namespace(
        exp_name="test",
        warmup_steps=1,
        peak_lr=5e-5,
        steps=10,
        batch_size=32,
        num_workers=0,
        save_interval=5,
        wandb=False,
        resume=False,
    )
    cfg = training.build_config(args)
    frozen = nnx.filterlib.to_predicate(cfg.freeze_filter)
    assert frozen(("PaliGemma", "llm", "kernel"), None)
    assert not frozen(("PaliGemma", "llm", "lora_a"), None)
    assert not frozen(("PaliGemma", "img", "kernel"), None)
    assert not frozen(("action_out_proj", "kernel"), None)
    assert cfg.data.base_config.prompt_from_task
    assert cfg.data.repo_id == training.REPO_ID
    assert cfg.keep_period is None
    assert cfg.ema_decay is None
    assert not cfg.overwrite


def test_save_waits_before_return(monkeypatch):
    events = []

    class Manager:
        def wait_until_finished(self):
            events.append("wait")

    monkeypatch.setattr(training.checkpoints, "save_state", lambda *args: events.append("save"))
    training.SynchronousFullCheckpointIO.save_state(Manager(), None, None, 1)
    assert events == ["save", "wait"]


def test_save_failure_propagates(monkeypatch):
    class Manager:
        def wait_until_finished(self):
            raise RuntimeError("save failed")

    monkeypatch.setattr(training.checkpoints, "save_state", lambda *args: None)
    with pytest.raises(RuntimeError, match="save failed"):
        training.SynchronousFullCheckpointIO.save_state(Manager(), None, None, 1)
