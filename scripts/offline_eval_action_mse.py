"""Offline validation: does the fine-tuned policy fit the *training* data?

Feeds real observations sampled from the training LeRobot dataset through the
exact inference path used at deployment (`policy.infer`), and compares the
predicted action chunk against the ground-truth action chunk stored in the
dataset (frames [t, t+1, ..., t+H-1]).

Interpretation:
  * High MSE on the training set  -> training-side problem
        (delta/abs mismatch, norm stats, under-training, wrong prompt, ...).
  * Low MSE on the training set but the real robot fails -> the model + the
        transform stack are fine; the problem is on the deployment/client side
        (obs it sends, image format, state units, control loop, async, ...).

Because `policy.infer` runs the SAME input/output transforms (repack -> Libero
-> normalize -> model -> unnormalize -> Libero out) that the websocket server
uses, a low MSE here also clears the transform stack itself.

Examples:
  uv run scripts/offline_eval_action_mse.py \
      --config-name pi05_piper_pick_and_place \
      --checkpoint-dir checkpoints/pi05_piper_pick_and_place/piper_pick_pi05_lora_v1/9999 \
      --num-samples 200
"""

from __future__ import annotations

import json

import numpy as np
import tyro

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import openpi.policies.policy_config as policy_config
import openpi.training.config as _config


def _to_hwc_uint8(img) -> np.ndarray:
    """LeRobot returns video frames as float32 CHW in [0,1]; deployment sends HWC uint8."""
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] == 3:  # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _fmt(vec: np.ndarray, p: int = 4) -> list[float]:
    return [round(float(x), p) for x in np.asarray(vec).reshape(-1)]


def main(
    config_name: str = "pi05_piper_pick_and_place",
    checkpoint_dir: str = "checkpoints/pi05_piper_pick_and_place/piper_pick_pi05_lora_v1/9999",
    num_samples: int = 200,
    seed: int = 0,
    num_infer_samples: int = 1,
    default_prompt: str | None = None,
) -> None:
    train_config = _config.get_config(config_name)
    action_horizon = train_config.model.action_horizon
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Config does not define a LeRobot repo_id")

    print(f"config:        {config_name}")
    print(f"checkpoint:    {checkpoint_dir}")
    print(f"repo_id:       {repo_id}")
    print(f"action_horizon:{action_horizon}")
    print(f"prompt_from_task: {data_config.prompt_from_task}")

    # Raw dataset with the SAME action-chunk alignment used in training:
    # action at frames [t, t+1, ..., t+H-1] (see data_loader.create_torch_dataset).
    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    fps = meta.fps
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps={"action": [t / fps for t in range(action_horizon)]},
    )
    total = len(dataset)
    print(f"total frames:  {total}")

    # Per-dim action std (from dataset stats) for a scale-normalized error report.
    action_std = np.asarray(meta.stats["action"]["std"], dtype=np.float32)
    action_dim = action_std.shape[0]

    print("\nLoading policy (this loads model weights + norm stats from the checkpoint)...")
    policy = policy_config.create_trained_policy(
        train_config, checkpoint_dir, default_prompt=default_prompt
    )

    rng = np.random.default_rng(seed)
    indices = rng.choice(total, size=min(num_samples, total), replace=False)
    indices.sort()

    # Accumulators over all (sample, horizon_step) that are NOT padded.
    sq_err_dim = np.zeros(action_dim, dtype=np.float64)          # model prediction
    sq_err_hold = np.zeros(action_dim, dtype=np.float64)         # baseline: hold current state
    sq_err_first = np.zeros(action_dim, dtype=np.float64)        # baseline: repeat gt[0]
    count = 0
    # Per-horizon-step error growth.
    per_step_sq = np.zeros(action_horizon, dtype=np.float64)
    per_step_cnt = np.zeros(action_horizon, dtype=np.float64)

    n_done = 0
    for idx in indices:
        item = dataset[int(idx)]

        state = np.asarray(item["observation.state"], dtype=np.float32).reshape(-1)
        gt = np.asarray(item["action"], dtype=np.float32).reshape(action_horizon, action_dim)
        pad = np.asarray(item.get("action_is_pad", np.zeros(action_horizon, dtype=bool))).reshape(-1)

        prompt = item.get("task")
        if prompt is None and default_prompt is not None:
            prompt = default_prompt

        infer_input = {
            "observation/state": state,
            "observation/image": _to_hwc_uint8(item["observation.images.rgb"]),
            "observation/wrist_image": _to_hwc_uint8(item["observation.images.wrist"]),
        }
        if prompt is not None:
            infer_input["prompt"] = prompt

        # Average over noise samples if requested (flow matching is stochastic).
        preds = []
        for _ in range(num_infer_samples):
            out = policy.infer(infer_input)
            preds.append(np.asarray(out["actions"], dtype=np.float32).reshape(action_horizon, action_dim))
        pred = np.mean(preds, axis=0)

        valid = ~pad  # (H,)
        if not valid.any():
            continue
        diff = (pred - gt) ** 2                      # (H, D)
        hold = (np.broadcast_to(state, gt.shape) - gt) ** 2
        first = (np.broadcast_to(gt[0], gt.shape) - gt) ** 2

        vmask = valid[:, None]
        sq_err_dim += (diff * vmask).sum(axis=0)
        sq_err_hold += (hold * vmask).sum(axis=0)
        sq_err_first += (first * vmask).sum(axis=0)
        count += int(valid.sum())

        per_step_sq += (diff.mean(axis=1) * valid)
        per_step_cnt += valid

        n_done += 1
        if n_done <= 3:
            print(f"\n[sample {n_done}] frame {int(idx)}  prompt={prompt!r}")
            print(f"  gt[0]  : {_fmt(gt[0])}")
            print(f"  pred[0]: {_fmt(pred[0])}")
            print(f"  state  : {_fmt(state)}")

    if count == 0:
        print("No valid (non-padded) steps evaluated.")
        return

    mse_dim = sq_err_dim / count
    rmse_dim = np.sqrt(mse_dim)
    hold_rmse_dim = np.sqrt(sq_err_hold / count)
    first_rmse_dim = np.sqrt(sq_err_first / count)
    # Scale-normalized: RMSE as a fraction of that dim's action std.
    nrmse_dim = rmse_dim / (action_std + 1e-8)

    print("\n" + "=" * 72)
    print(f"Evaluated {n_done} observations, {count} valid action steps.")
    print("=" * 72)
    print(f"\nOverall MSE (all dims):        {mse_dim.mean():.6f}")
    print(f"Overall RMSE (all dims):       {np.sqrt(mse_dim.mean()):.6f}")
    print(f"Baseline RMSE hold-state:      {np.sqrt((sq_err_hold/count).mean()):.6f}")
    print(f"Baseline RMSE repeat-gt[0]:    {np.sqrt((sq_err_first/count).mean()):.6f}")

    print("\nPer-dimension (dims 0-5 = joints [rad], dim 6 = gripper):")
    header = "  dim   model_RMSE   nRMSE(/std)   hold_RMSE   repeat0_RMSE   action_std"
    print(header)
    for d in range(action_dim):
        print(
            f"  {d:>3}   {rmse_dim[d]:>9.4f}   {nrmse_dim[d]:>10.3f}   "
            f"{hold_rmse_dim[d]:>9.4f}   {first_rmse_dim[d]:>11.4f}   {action_std[d]:>9.4f}"
        )

    per_step_rmse = np.sqrt(per_step_sq / np.maximum(per_step_cnt, 1))
    print("\nPer-horizon-step RMSE (does error grow along the chunk?):")
    print("  step:  " + "  ".join(f"{s}" for s in range(action_horizon)))
    print("  rmse:  " + "  ".join(f"{v:.3f}" for v in per_step_rmse))

    summary = {
        "config": config_name,
        "checkpoint": checkpoint_dir,
        "num_obs": n_done,
        "num_steps": count,
        "overall_mse": float(mse_dim.mean()),
        "overall_rmse": float(np.sqrt(mse_dim.mean())),
        "per_dim_rmse": _fmt(rmse_dim, 5),
        "per_dim_nrmse": _fmt(nrmse_dim, 4),
        "baseline_hold_state_rmse": float(np.sqrt((sq_err_hold / count).mean())),
        "baseline_repeat0_rmse": float(np.sqrt((sq_err_first / count).mean())),
        "per_step_rmse": _fmt(per_step_rmse, 4),
    }
    print("\nJSON summary:")
    print(json.dumps(summary))


if __name__ == "__main__":
    tyro.cli(main)
