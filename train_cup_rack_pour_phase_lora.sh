#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${TRAIN_GPU_ID:-0,1}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
REPO_ID="local/piper-cup-rack-pour-phase-merged-v1"
NORM_STATS="assets/pi05_piper_pick_and_place_v2/${REPO_ID}/norm_stats.json"

# This job is intended to coexist with another workload on GPU 0. GPU 1 has
# only ~5 GiB free and is deliberately hidden. Limit JAX's preallocated pool to
# 32% of one A100-80GB (~26 GiB) instead of its much larger default allocation.
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.32}"

# Batch 2 is the conservative setting for the capped memory pool. Four times
# as many optimizer steps preserves the sample count of the original
# batch-8/30k schedule.
BATCH_SIZE="${BATCH_SIZE:-2}"
STEPS="${STEPS:-120000}"

if [[ ! -f "${NORM_STATS}" ]]; then
  "${PYTHON_BIN}" scripts/compute_pour_norm_stats.py \
    --repo-id "${REPO_ID}" \
    --batch-size 8 \
    --num-workers 2 \
    --max-frames 50000
fi

"${PYTHON_BIN}" scripts/train_cup_rack_pour_phase_lora.py \
  --wandb \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --peak-lr 3e-5 \
  --warmup-steps 4000 \
  --save-interval 4000 \
  --keep-period 20000 \
  --eval-interval 4000 \
  --eval-num-batches 64
