# Task-specific LoRA adapters

OpenPI can train multiple small LoRA adapters against one fixed JAX base
checkpoint. Each adapter must use the same model configuration and Piper
normalization statistics as that base.

## Train

The defaults reproduce the Piper cup-from-rack setup:

```bash
cd /home/huix/bci_robot/openpi

.venv/bin/python scripts/train_lora_adapter.py --dry-run

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_lora_adapter.py \
  --steps 5000 \
  --batch-size 4 \
  --wandb
```

To train another task, give it a different dataset and experiment name while
keeping the same base checkpoint and normalization asset:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_lora_adapter.py \
  --repo-id OWNER/ANOTHER_DATASET \
  --exp-name another_task_lora_v1 \
  --steps 5000 \
  --batch-size 4
```

Resume the exact same experiment with `--resume`. Do not change its base model,
dataset, or normalization settings when resuming.

## Checkpoints and export

The LoRA training wrapper writes compact, atomic checkpoints. Every completed
numeric step contains:

```text
<step>/adapter/adapter.npz   # directly deployable LoRA weights
<step>/adapter/adapter.json  # compatibility manifest
<step>/train_state/          # step and LoRA optimizer state for resume
<step>/assets/               # normalization stats
```

Frozen base parameters are not copied. The standalone exporter is only needed
to convert an older full checkpoint:

```bash
.venv/bin/python scripts/export_lora_adapter.py \
  --checkpoint-dir checkpoints/pi05_piper_pick_and_place_v2/piper_cup_from_rack_lora_v1/4999 \
  --output-dir adapters/piper_cup_from_rack_v1
```

The output contains `adapter.npz` and `adapter.json`. The loader rejects
non-LoRA entries, missing parameter paths, and incompatible tensor shapes.

## Serve one task adapter

Load the shared base checkpoint and select either an exported adapter directory
or a completed numeric training step at server startup:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_piper_pick_and_place_v2 \
  --policy.dir checkpoints/pi05_piper_pick_and_place_v2/piper_pick_pi05_lora_v4/69999 \
  --policy.adapter checkpoints/pi05_piper_pick_and_place_v2/piper_cup_from_rack_lora_v1/500
```

Start the same command with a different `--policy.adapter` to serve another
task. This is startup-time selection, not in-process hot swapping. A production
hot-swap service should cache one compiled policy per task and route requests to
the selected policy instead of mutating parameters used by concurrent requests.
