#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
COMMON=(
  --repo-id phoebe777777/piper-all-manipulation-cleaned-v1-filtered-v2
  --exp-name piper_all_manipulation_full_v1
  --batch-size 8
  --num-workers 8
  --peak-lr 5e-6
  --decay-lr 5e-7
  --lr-decay-steps 40000
  --warmup-steps 1000
  --save-interval 1000
  --keep-period 5000
  --eval-interval 1000
  # A deterministic shuffled eval subset. With batch 8 this evaluates 512
  # samples, which is large enough to cover the grouped multitask dataset and
  # substantially more rare open/close transitions than four batches.
  --eval-num-batches 64
  --fsdp-devices 1
  --wandb
)

NORM_STATS="assets/pi05_piper_multitask_full/phoebe777777/piper-all-manipulation-cleaned-v1-filtered-v2/norm_stats.json"
if [[ ! -f "${NORM_STATS}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" scripts/train_piper_multitask_full.py \
    "${COMMON[@]}" \
    --compute-norm \
    --max-norm-frames 50000
fi

# Curriculum endpoints are cumulative. Later stages restore model and optimizer
# state from the same experiment directory and only change input state dropout.
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" scripts/train_piper_multitask_full.py \
  "${COMMON[@]}" --steps 10000 --state-dropout 0.00

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" scripts/train_piper_multitask_full.py \
  "${COMMON[@]}" --steps 20000 --state-dropout 0.10 --resume

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" scripts/train_piper_multitask_full.py \
  "${COMMON[@]}" --steps 30000 --state-dropout 0.25 --resume

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" scripts/train_piper_multitask_full.py \
  "${COMMON[@]}" --steps 40000 --state-dropout 0.50 --resume
