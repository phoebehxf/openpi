CUDA_VISIBLE_DEVICES=0 uv run python scripts/serve_policy.py \
    --port 8000   policy:checkpoint \
    --policy.config pi05_piper_multitask_full \
    --policy.dir checkpoints/piper_all_manipulation_full_v1/119999 \
    --policy.norm-stats checkpoints/piper_all_manipulation_full_v1/119999/assets/phoebe777777/piper-all-manipulation-cleaned-v1-filtered-v2 