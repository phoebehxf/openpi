#!/usr/bin/env bash
set -euo pipefail

NORM_STATS="assets/pi05_piper_pick_and_place_v2/local/piper-pour-water-three-phase-v1/norm_stats.json"
if [[ ! -f "${NORM_STATS}" ]]; then
  echo "Three-phase norm stats are missing; computing them first..."
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/compute_pour_norm_stats.py \
    --repo-id local/piper-pour-water-three-phase-v1 \
    --batch-size 32 \
    --num-workers 2 \
    --max-frames 10000
fi

CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/train_pour_three_phase_lora.py \
  --wandb \
  --steps 30000 \
  --batch-size 8 \
  --peak-lr 3e-5 \
  --warmup-steps 1000 \
  --save-interval 1000 \
  --keep-period 5000
