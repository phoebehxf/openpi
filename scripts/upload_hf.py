from pathlib import Path

from huggingface_hub import HfApi
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


repo_id = "phoebe777777/piper-pick-up-v2-v2.0"
root = Path(
    "/home/huix/.cache/huggingface/lerobot/phoebe777777/piper-pick-up-v2"
).expanduser()

api = HfApi()
api.create_repo(
    repo_id=repo_id,
    repo_type="dataset",
    private=True,
    exist_ok=True,
)
api.create_tag(
    repo_id=repo_id,
    tag="v2.1",
    repo_type="dataset",
    exist_ok=True,
)

dataset = LeRobotDataset(repo_id=repo_id, root=root)
dataset.push_to_hub(private=True)