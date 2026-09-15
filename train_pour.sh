CUDA_VISIBLE_DEVICES=2 .venv/bin/python \
  scripts/train_pour_openpi_lora.py \
  --resume \
  --wandb \
  --batch-size 8 \
  --save-interval 500