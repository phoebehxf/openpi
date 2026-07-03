"""Model-space offline check: reconcile 'low training loss' with 'bad action prediction'.

Unlike offline_eval_action_mse.py (which goes through the deployment path
`policy.infer`), this script pulls batches from the EXACT training data loader
(create_torch_data_loader -> repack -> Libero -> normalize -> model transforms)
and runs the JAX model directly. It reports:

  1. compute_loss on the loaded checkpoint  -> should reproduce the wandb loss
     (~0.017). If it does, the weights loaded correctly.
  2. sample_actions vs ground-truth action chunk, in BOTH normalized model space
     and unnormalized raw space (rad).

Reading:
  * loss ~0.017 AND sample RMSE low  -> the model is fine; the earlier high RMSE
        from policy.infer means the DEPLOYMENT path differs (transform/mapping bug).
  * loss ~0.017 BUT sample RMSE high -> low denoising loss coexists with bad
        sampling (genuine model/sampling issue), matching the policy.infer result.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import tyro

import openpi.models.model as _model
import openpi.shared.download as download
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


def main(
    config_name: str = "pi05_piper_pick_and_place",
    checkpoint_dir: str = "checkpoints/pi05_piper_pick_and_place/piper_pick_pi05_lora_v1/9999",
    num_batches: int = 25,
    batch_size: int = 8,
    num_sample_steps: int = 10,
    seed: int = 0,
) -> None:
    print(f"jax devices: {jax.devices()}")
    train_config = _config.get_config(config_name)
    action_horizon = train_config.model.action_horizon
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    ckpt = download.maybe_download(str(checkpoint_dir))
    print("loading model weights (bfloat16)...")
    model = train_config.model.load(_model.restore_params(ckpt / "params", dtype=jnp.bfloat16))

    # Same norm stats the policy would use, to unnormalize predictions back to rad.
    norm_stats = _checkpoints.load_norm_stats(ckpt / "assets", data_config.asset_id)
    use_q = data_config.use_quantile_norm
    print(f"use_quantile_norm: {use_q}")

    def unnorm(x, key):  # x: (..., D) normalized; return first 7 real dims in rad
        st = norm_stats[key]
        x = np.asarray(x, dtype=np.float32)[..., :7]
        if use_q:
            q01 = np.asarray(st.q01)[:7]
            q99 = np.asarray(st.q99)[:7]
            return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        mean = np.asarray(st.mean)[:7]
        std = np.asarray(st.std)[:7]
        return x * (std + 1e-6) + mean

    # Force everything onto a single device so the eval is independent of how many
    # GPUs are present (the default loader shards the batch across all devices).
    single_dev = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    loader = _data_loader.create_torch_data_loader(
        data_config,
        train_config.model,
        action_horizon,
        batch_size=batch_size,
        sharding=single_dev,
        shuffle=True,
        num_batches=num_batches,
        seed=seed,
    )

    rng = jax.random.key(seed)
    losses = []
    sq_norm = None      # normalized-space squared error, per real dim
    sq_raw = None       # raw-space (rad) squared error, per real dim
    sq_raw_hold = None  # baseline: predict current (unnormalized) state
    n = 0
    action_dim_real = 7

    for bi, (obs, actions) in enumerate(loader):
        if bi >= num_batches:
            break
        rng, lrng, srng = jax.random.split(rng, 3)

        # 1) reproduce the training loss on this checkpoint
        loss = model.compute_loss(lrng, obs, actions, train=False)
        losses.append(float(np.asarray(loss).mean()))

        # 2) sample actions and compare (normalized model space)
        pred = model.sample_actions(srng, obs, num_steps=num_sample_steps)  # (b, ah, ad_model)
        pred = np.asarray(pred, dtype=np.float32)
        gt = np.asarray(actions, dtype=np.float32)                          # (b, ah, ad_model)

        pr = pred[..., :action_dim_real]
        gr = gt[..., :action_dim_real]
        d_norm = (pr - gr) ** 2
        sq_norm = d_norm.sum(axis=(0, 1)) if sq_norm is None else sq_norm + d_norm.sum(axis=(0, 1))

        # unnormalize to rad (raw space)
        pred_raw = unnorm(pred, "actions")
        gt_raw = unnorm(gt, "actions")
        state_raw = unnorm(np.asarray(obs.state, dtype=np.float32), "state")
        d_raw = (pred_raw - gt_raw) ** 2
        d_hold = (state_raw[:, None, :] - gt_raw) ** 2
        sq_raw = d_raw.sum(axis=(0, 1)) if sq_raw is None else sq_raw + d_raw.sum(axis=(0, 1))
        sq_raw_hold = d_hold.sum(axis=(0, 1)) if sq_raw_hold is None else sq_raw_hold + d_hold.sum(axis=(0, 1))

        n += pr.shape[0] * pr.shape[1]

        if bi == 0:
            print("\n[batch 0, sample 0]")
            print("  gt_raw[0]  :", [round(float(x), 4) for x in gt_raw[0, 0]])
            print("  pred_raw[0]:", [round(float(x), 4) for x in pred_raw[0, 0]])
            print("  state_raw  :", [round(float(x), 4) for x in state_raw[0]])

    rmse_norm = np.sqrt(sq_norm / n)
    rmse_raw = np.sqrt(sq_raw / n)
    rmse_hold = np.sqrt(sq_raw_hold / n)

    print("\n" + "=" * 72)
    print(f"batches: {len(losses)}   steps: {n}")
    print(f"mean compute_loss (should match wandb ~0.017): {np.mean(losses):.5f}")
    print("=" * 72)
    print(f"\nnormalized-space RMSE (all dims): {rmse_norm.mean():.4f}")
    print(f"raw-space (rad) RMSE (all dims):  {rmse_raw.mean():.4f}")
    print(f"raw-space baseline hold-state:    {rmse_hold.mean():.4f}")
    print("\nper-dim RMSE (0-5 joints rad, 6 gripper):")
    print("  dim   norm_RMSE   raw_RMSE   hold_RMSE")
    for d in range(action_dim_real):
        print(f"  {d:>3}   {rmse_norm[d]:>8.4f}   {rmse_raw[d]:>8.4f}   {rmse_hold[d]:>8.4f}")

    print("\nJSON summary:")
    print(json.dumps({
        "config": config_name,
        "checkpoint": checkpoint_dir,
        "mean_loss": float(np.mean(losses)),
        "rmse_norm": float(rmse_norm.mean()),
        "rmse_raw": float(rmse_raw.mean()),
        "rmse_hold": float(rmse_hold.mean()),
        "per_dim_raw_rmse": [round(float(x), 5) for x in rmse_raw],
    }))


if __name__ == "__main__":
    tyro.cli(main)
