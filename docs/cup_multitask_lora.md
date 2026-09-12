# Cup rack multitask LoRA

The source datasets are the cleaned local v2.1 `phoebe777777/piper-hang-cup-on-rack-cleaned-v1`
and `phoebe777777/piper-take-cup-from-rack-cleaned-v1`. No additional trimming is applied.

```bash
.venv/bin/python scripts/merge_cup_datasets.py
.venv/bin/python scripts/train_cup_multitask_lora.py --dry-run
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_cup_multitask_lora.py --wandb
```

The merge refuses to overwrite an existing destination. Output lives at
`local_datasets/local/piper-cup-rack-multitask-cleaned-v1`. Every video is copied unchanged
after decoding its frame count; episode/global/task indices and numeric statistics
are rebuilt. Image statistics are retained per episode and aggregated. Original
datasets are untouched. `meta/merge_manifest.json` records source episode mappings.
Sampling uses the standard shuffled frame sampler, not equal task oversampling.

Training defaults: 30,000 steps, batch 32, peak LR 5e-5, warmup 1,000,
decay LR 1e-6, action horizon 10. It initializes from
`piper_pick_pi05_lora_v4/69999/params` and reuses original Piper v2 normalization.
Task prompts are read from each sample's task_index.

This uses `model.get_freeze_filter()`, including trainable vision/projection
parameters, NOT strict LoRA-only freezing. Full standard OpenPI checkpoints are
saved every 1,000 steps; only the latest two are retained. Saving blocks until
completion before the next training step to avoid asynchronous/donated-array
lifetime races. This adds checkpoint pauses; batch 32 GPU fit is not guaranteed.

Output:
`checkpoints/pi05_piper_pick_and_place_v2/piper_cup_rack_multitask_cleaned_openpi_lora_v1`
(the workspace checkpoints symlink resolves under `/media/data1/huix/vla_models`).

Resume with the same training settings and `--resume --wandb`. Do not change the
dataset or scheduler settings when resuming. Omit `--wandb` if not wanted.

Deploy the complete checkpoint, without `--policy.adapter`:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_piper_pick_and_place_v2 \
  --policy.dir checkpoints/pi05_piper_pick_and_place_v2/piper_cup_rack_multitask_cleaned_openpi_lora_v1/STEP
```

Replace STEP with a completed saved step. Choose the task using the client prompt:

- `Pick up the red cup from the table and hang it on the rack`
- `Take the red cup off the rack and place it on the table`

Do not export only LoRA arrays from this run: doing so discards other trained weights.
No held-out episode split is introduced; training eval metrics are not a success-rate test.
