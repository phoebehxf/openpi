from openpi.training import config as _config
from openpi.policies import policy_config
import numpy as np

config = _config.get_config("pi05_piper_pick_and_place")
policy = policy_config.create_trained_policy(
    config,
    "checkpoints/pi05_piper_pick_and_place/piper_pick_pi05_lora_v1/9999",
)

example = {
    "observation/state": np.zeros(7, dtype=np.float32),
    "observation/image": np.zeros((480, 640, 3), dtype=np.uint8),
    "observation/wrist_image": np.zeros((480, 640, 3), dtype=np.uint8),
    "prompt": "pick up the blue pen and place it in the large container",
}

out = policy.infer(example)
print(out["actions"].shape)
print(out["actions"][0])