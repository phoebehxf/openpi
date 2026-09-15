CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  scripts/train_cup_multitask_lora.py \
  --legacy \
  --resume \
  --steps 30000 \
  --batch-size 8 \
  --wandb