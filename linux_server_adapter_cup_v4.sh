CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/serve_policy.py \
  --capture-dir debug/0915/pour_water_v2_15000_open_neutral_latch_1 \
  --neutralize-open-gripper-state 0.4999 \
  --latch-gripper-after-close \
  --gripper-latch-threshold 0.5 \
  --gripper-latch-min-steps 5 \
  policy:checkpoint \
  --policy.config pi05_piper_pick_and_place_v3 \
  --policy.dir checkpoints/pi05_piper_pick_and_place_v2/piper_pick_pi05_lora_v4/69999 \
  --policy.adapter checkpoints/pi05_piper_pick_and_place_v3/piper_pour_water_state_norm_openpi_lora_v2/15000 \
  --policy.norm-stats checkpoints/pi05_piper_pick_and_place_v3/piper_pour_water_state_norm_openpi_lora_v2/15000/assets/local/piper-pour-water-cleaned-merged-v1