#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
REPO_ID="local/piper-cup-rack-pour-phase-merged-v1"
NORM_STATS="assets/pi05_piper_pick_and_place_v2/${REPO_ID}/norm_stats.json"

if [[ ! -f "${NORM_STATS}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" scripts/compute_pour_norm_stats.py \
    --repo-id "${REPO_ID}" \
    --batch-size 32 \
    --num-workers 2 \
    --max-frames 50000
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" scripts/train_cup_rack_pour_phase_lora.py \
  --wandb \
  --steps 30000 \
  --batch-size 32 \
  --peak-lr 3e-5 \
  --warmup-steps 1000 \
  --save-interval 1000 \
  --keep-period 5000 \
  --eval-interval 1000 \
  --eval-num-batches 64
