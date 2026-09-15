# CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/serve_policy.py policy:checkpoint  \
#  --policy.config pi05_piper_pick_and_place_v3 --policy.dir checkpoints/pi05_piper_pick_and_place_v2/piper_pick_pi05_lora_v4/69999 \
#  --policy.adapter checkpoints/pi05_piper_pick_and_place_v3/piper_cup_rack_multitask_state_input_v1/1500
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/serve_policy.py \
  --capture-dir debug/0915/piper_pour_three_phase_state_lora_v1_15000_open_neutral_latch_1 \
  --neutralize-open-gripper-state 0.4999 \
  --latch-gripper-after-close \
  --gripper-latch-threshold 0.5 \
  --gripper-latch-min-steps 5 \
  policy:checkpoint \
  --policy.config pi05_piper_pick_and_place_v3 \
  --policy.dir checkpoints/pi05_piper_pick_and_place_v2/piper_pick_pi05_lora_v4/69999 \
  --policy.adapter checkpoints/pi05_piper_pick_and_place_v3/piper_pour_three_phase_state_lora_v1/15000 \
  --policy.norm-stats checkpoints/pi05_piper_pick_and_place_v3/piper_pour_three_phase_state_lora_v1/15000/assets/local/piper-pour-water-three-phase-v1
# CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/serve_policy.py \
#   --capture-dir debug/0914/cup_rack_multitask_cleaned_openpi_lora_v1_24500_7 \
#   policy:checkpoint \
#   --policy.config pi05_piper_pick_and_place_v3 \
#   --policy.dir checkpoints/pi05_piper_pick_and_place_v2/piper_pick_pi05_lora_v4/69999 \
#   --policy.adapter checkpoints/pi05_piper_pick_and_place_v2/piper_cup_rack_multitask_cleaned_openpi_lora_v1/24500 \
#   # --policy.norm-stats checkpoints/pi05_piper_pick_and_place_v3/piper_pour_water_state_norm_openpi_lora_v2/25000/assets/local/piper-pour-water-cleaned-merged-v1 
