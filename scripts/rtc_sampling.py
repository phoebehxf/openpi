"""Real-Time Chunking (RTC) guided sampling for openpi pi0 / pi05 models.

Self-contained port of LeRobot's RTC (`lerobot/policies/rtc/modeling_rtc.py`, which
implements https://www.physicalintelligence.company/download/real_time_chunking.pdf)
onto openpi's JAX pi0 sampler. NOTHING in the existing openpi codebase is modified --
this module only *reads* public attributes/methods of a loaded `Pi0` model and rebuilds
the flow-matching denoise loop with RTC prefix guidance layered on top.

Why RTC
-------
openpi's stock deployment (`ActionChunkBroker` / the hand-rolled `open_loop_horizon`
loop in `scripts/piper_remote_client.py`) predicts a chunk, executes it open-loop, then
predicts a fresh chunk with NO continuity across the boundary -> a visible jerk every
`open_loop_horizon` steps, and inference latency is not hidden. RTC treats the next chunk
as an inpainting problem: the still-unexecuted tail of the previous chunk is used as a
soft prefix that guides denoising, so consecutive chunks join smoothly and the model can
keep moving while the next chunk is being computed.

The math (identical to LeRobot, adapted to openpi's time convention where t goes 1->0):
  * base velocity            v_t = denoise(x_t, t)          (one pi0 flow step)
  * clean-action estimate    x1_t = x_t - t * v_t           (== x1 in linear flow matching)
  * prefix error             err  = (prev_leftover - x1_t) * weights   (weights: get_prefix_weights)
  * guidance correction      corr = vjp_x( x1_t )(err)       (== torch.autograd.grad(x1_t, x_t, err))
  * guidance weight          gw   = clamp( c * inv_r2, max ) with tau = 1 - t
  * guided velocity          v_t' = v_t - gw * corr

Usage
-----
  * one-shot:            actions = rtc_sample_actions(model, rng, obs, prev_chunk_left_over=..., ...)
  * stateful real-time:  gen = RTCActionGenerator(model, cfg); chunk = gen.infer(rng, obs)
  * self-check / demo:   uv run scripts/rtc_sampling.py --checkpoint-dir <ckpt>

Integrating into serving is left to the caller (serve_policy.py is intentionally NOT
touched): construct an `RTCActionGenerator`, and in your control loop call `.infer(...)`
every `execution_horizon` steps, feeding back the observation each time.
"""

from __future__ import annotations

import dataclasses
import enum

import einops
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
from openpi.models.pi0 import make_attn_mask


class PrefixAttentionSchedule(str, enum.Enum):
    ZEROS = "zeros"
    ONES = "ones"
    LINEAR = "linear"
    EXP = "exp"


@dataclasses.dataclass(frozen=True)
class RTCConfig:
    """Mirror of LeRobot's RTCConfig, defaults tuned for a short (10-step) horizon.

    execution_horizon: how many actions you actually execute before re-planning. The
        prefix used to guide the next chunk is the unexecuted tail (length
        action_horizon - execution_horizon). Must be <= action_horizon.
    inference_delay: how many of the leading prefix steps are treated as "already
        committed" (hard weight 1.0). In a real loop set this to roughly the number of
        control steps that elapse while one inference runs.
    """

    prefix_attention_schedule: PrefixAttentionSchedule = PrefixAttentionSchedule.LINEAR
    max_guidance_weight: float = 10.0
    execution_horizon: int = 8
    inference_delay: int = 1
    num_steps: int = 10


# --------------------------------------------------------------------------------------
# Prefix weights (static, numpy -> jnp). Verbatim port of LeRobot get_prefix_weights.
# --------------------------------------------------------------------------------------
def _linweights(start: int, end: int, total: int) -> np.ndarray:
    skip_steps_at_end = max(total - end, 0)
    linspace_steps = total - skip_steps_at_end - start
    if end <= start or linspace_steps <= 0:
        return np.array([], dtype=np.float32)
    return np.linspace(1.0, 0.0, linspace_steps + 2, dtype=np.float32)[1:-1]


def _add_trailing_zeros(weights: np.ndarray, total: int, end: int) -> np.ndarray:
    zeros_len = total - end
    if zeros_len <= 0:
        return weights
    return np.concatenate([weights, np.zeros(zeros_len, dtype=np.float32)])


def _add_leading_ones(weights: np.ndarray, start: int, total: int) -> np.ndarray:
    ones_len = min(start, total)
    if ones_len <= 0:
        return weights
    return np.concatenate([np.ones(ones_len, dtype=np.float32), weights])


def get_prefix_weights(start: int, end: int, total: int, schedule: PrefixAttentionSchedule) -> np.ndarray:
    """Per-timestep guidance weights over the chunk (length `total`).

    start == inference_delay (leading hard-committed steps, weight 1),
    end   == execution_horizon (beyond this, weight 0 -- the fresh part of the chunk).
    """
    start = min(start, end)
    if schedule == PrefixAttentionSchedule.ZEROS:
        weights = np.zeros(total, dtype=np.float32)
        weights[:start] = 1.0
    elif schedule == PrefixAttentionSchedule.ONES:
        weights = np.ones(total, dtype=np.float32)
        weights[end:] = 0.0
    elif schedule == PrefixAttentionSchedule.LINEAR:
        lin = _linweights(start, end, total)
        weights = _add_leading_ones(_add_trailing_zeros(lin, total, end), start, total)
    elif schedule == PrefixAttentionSchedule.EXP:
        lin = _linweights(start, end, total)
        lin = lin * np.expm1(lin) / (np.e - 1.0)
        weights = _add_leading_ones(_add_trailing_zeros(lin, total, end), start, total)
    else:
        raise ValueError(f"unknown prefix_attention_schedule: {schedule}")
    # Length can drift by rounding in the LINEAR/EXP branches; clamp to `total`.
    if weights.shape[0] < total:
        weights = np.concatenate([weights, np.zeros(total - weights.shape[0], dtype=np.float32)])
    return weights[:total]


# --------------------------------------------------------------------------------------
# Core: rebuild the pi0 denoise closure and run the RTC-guided loop.
# --------------------------------------------------------------------------------------
def _make_denoiser(model, observation):
    """Return a pure `denoise(x_t, time) -> v_t` closure, mirroring Pi0.sample_actions.

    The prefix KV cache is computed once here and captured, exactly like the stock
    sampler; only the suffix (action) branch runs per step.
    """
    batch_size = observation.state.shape[0]
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def denoise(x_t, time):
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
        pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (prefix_out, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=pos,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        return model.action_out_proj(suffix_out[:, -model.action_horizon :])

    return denoise


def _guided_velocity(denoise, x_t, time, prev, weights, max_guidance_weight):
    """RTC prefix guidance for one denoise step (vectorized, batched).

    Reproduces LeRobot RTCProcessor.denoise_step: correction = J_{x1_t}^T(err) via VJP,
    then v_t' = v_t - guidance_weight * correction.
    """

    def x1_fn(x):
        v = denoise(x, time)
        return x - time * v, v  # (primal x1_t, aux v_t)

    x1_t, vjp_fn, v_t = jax.vjp(x1_fn, x_t, has_aux=True)
    err = (prev - x1_t) * weights  # treated as a constant cotangent, like grad_outputs.detach()
    (correction,) = vjp_fn(err)

    tau = 1.0 - time
    squared_one_minus_tau = (1.0 - tau) ** 2
    inv_r2 = (squared_one_minus_tau + tau**2) / squared_one_minus_tau
    c = jnp.where(tau > 0, (1.0 - tau) / jnp.where(tau > 0, tau, 1.0), max_guidance_weight)
    guidance_weight = jnp.nan_to_num(c * inv_r2, posinf=max_guidance_weight)
    guidance_weight = jnp.minimum(guidance_weight, max_guidance_weight)
    return v_t - guidance_weight * correction


def rtc_sample_actions(
    model,
    rng: jax.Array,
    observation: _model.Observation,
    *,
    prev_chunk_left_over: jax.Array | None = None,
    config: RTCConfig = RTCConfig(),
    noise: jax.Array | None = None,
) -> jax.Array:
    """Sample an action chunk with RTC prefix guidance.

    prev_chunk_left_over: (b, T_prev, ad) unexecuted tail of the previously issued chunk,
        already ALIGNED to the new chunk's timeline (i.e. index 0 is the action for the
        current step). Pass None on the very first chunk -> reduces exactly to the stock
        Pi0 sampler (verified in `main`).
    """
    observation = _model.preprocess_observation(None, observation, train=False)
    num_steps = config.num_steps
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    action_horizon = model.action_horizon
    action_dim = model.action_dim

    if noise is None:
        noise = jax.random.normal(rng, (batch_size, action_horizon, action_dim))

    denoise = _make_denoiser(model, observation)

    prev = None
    weights = None
    if prev_chunk_left_over is not None:
        prev = jnp.asarray(prev_chunk_left_over, dtype=noise.dtype)
        if prev.ndim == 2:
            prev = prev[None]
        # Right-pad the (shorter) leftover to the full chunk shape with zeros.
        pad_t = action_horizon - prev.shape[1]
        pad_d = action_dim - prev.shape[2]
        if pad_t > 0 or pad_d > 0:
            prev = jnp.pad(prev, ((0, 0), (0, max(pad_t, 0)), (0, max(pad_d, 0))))
        prev = prev[:, :action_horizon, :action_dim]
        execution_horizon = min(config.execution_horizon, prev_chunk_left_over.shape[-2])
        w = get_prefix_weights(config.inference_delay, execution_horizon, action_horizon, config.prefix_attention_schedule)
        weights = jnp.asarray(w)[None, :, None]

    x_t = noise
    time = 1.0
    for _ in range(num_steps):
        if prev is None:
            v_t = denoise(x_t, time)
        else:
            v_t = _guided_velocity(denoise, x_t, time, prev, weights, config.max_guidance_weight)
        x_t = x_t + dt * v_t
        time = time + dt
    return x_t


# --------------------------------------------------------------------------------------
# Stateful helper for a real-time control loop.
# --------------------------------------------------------------------------------------
class RTCActionGenerator:
    """Keeps the previous chunk so a control loop can call `.infer` every replan.

    Real-time loop sketch (control runs at some Hz; replan every execution_horizon steps):

        gen = RTCActionGenerator(model, RTCConfig(execution_horizon=8, inference_delay=2))
        chunk = gen.infer(rng, obs)              # first chunk (no guidance)
        # execute chunk[0 : execution_horizon] on the robot ...
        chunk = gen.infer(rng, obs)              # next chunk, guided by the unexecuted tail
        ...

    The leftover handed to the next call is the previous chunk shifted forward by
    execution_horizon (index 0 == the action for the upcoming control step), which is what
    RTC expects as the aligned prefix.
    """

    def __init__(self, model, config: RTCConfig = RTCConfig()):
        self._model = model
        self._config = config
        self._prev_chunk = None  # (b, ah, ad) last issued chunk

    def reset(self) -> None:
        self._prev_chunk = None

    def infer(self, rng: jax.Array, observation: _model.Observation, *, execution_horizon: int | None = None) -> jax.Array:
        eh = int(execution_horizon if execution_horizon is not None else self._config.execution_horizon)
        leftover = None
        if self._prev_chunk is not None:
            leftover = self._prev_chunk[:, eh:, :]  # unexecuted tail, aligned to new timeline
        chunk = rtc_sample_actions(
            self._model, rng, observation, prev_chunk_left_over=leftover, config=self._config
        )
        self._prev_chunk = np.asarray(chunk)
        return chunk


# --------------------------------------------------------------------------------------
# Self-check / demo (offline, no robot). Validates the port against the stock sampler.
# --------------------------------------------------------------------------------------
def main(
    config_name: str = "pi05_piper_pick_and_place",
    checkpoint_dir: str = "checkpoints/pi05_piper_pick_and_place/piper_pick_pi05_lora_v2/29999",
    num_steps: int = 10,
    execution_horizon: int = 8,
    inference_delay: int = 1,
    seed: int = 0,
) -> None:
    import openpi.shared.download as download
    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader

    print(f"jax devices: {jax.devices()}")
    train_config = _config.get_config(config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    ckpt = download.maybe_download(str(checkpoint_dir))
    model = train_config.model.load(_model.restore_params(ckpt / "params", dtype=jnp.bfloat16))

    single_dev = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    loader = _data_loader.create_torch_data_loader(
        data_config, train_config.model, train_config.model.action_horizon,
        batch_size=4, sharding=single_dev, shuffle=True, num_batches=1, seed=seed,
    )
    obs, gt_actions = next(iter(loader))
    gt = np.asarray(gt_actions, dtype=np.float32)
    ah = train_config.model.action_horizon
    ad = model.action_dim
    rng = jax.random.key(seed)
    noise = jax.random.normal(rng, (obs.state.shape[0], ah, ad))
    cfg = RTCConfig(execution_horizon=execution_horizon, inference_delay=inference_delay, num_steps=num_steps)

    # 1) Equivalence check: prev=None must reproduce the stock sampler bit-for-bit (same noise).
    base = np.asarray(model.sample_actions(rng, obs, num_steps=num_steps, noise=noise), dtype=np.float32)
    ported = np.asarray(
        rtc_sample_actions(model, rng, obs, prev_chunk_left_over=None, config=cfg, noise=noise), dtype=np.float32
    )
    max_abs = float(np.abs(base - ported).max())
    print("\n[1] prev=None vs stock Pi0.sample_actions")
    print(f"    max|Δ| = {max_abs:.2e}  ->  {'OK (port matches)' if max_abs < 1e-3 else 'MISMATCH!'}")

    # 2) RTC guidance runs and is finite; measure chunk-boundary continuity.
    # Simulate: previous chunk == ground truth; we already executed `execution_horizon`
    # steps, so the aligned leftover is gt shifted forward by execution_horizon.
    leftover = gt[:, execution_horizon:, :]
    guided = np.asarray(
        rtc_sample_actions(model, rng, obs, prev_chunk_left_over=leftover, config=cfg, noise=noise), dtype=np.float32
    )
    finite = bool(np.isfinite(guided).all())
    # Continuity: agreement on the overlapping (still-unexecuted) region with the prefix.
    overlap = ah - execution_horizon
    base_gap = float(np.abs(base[:, :overlap, :7] - leftover[:, :overlap, :7]).mean())
    guided_gap = float(np.abs(guided[:, :overlap, :7] - leftover[:, :overlap, :7]).mean())
    print("\n[2] RTC-guided sampling (prev = shifted ground truth)")
    print(f"    finite: {finite}   shape: {guided.shape}")
    print(f"    prefix weights: {np.round(get_prefix_weights(inference_delay, execution_horizon, ah, cfg.prefix_attention_schedule), 3).tolist()}")
    print(f"    overlap L1 vs prefix (real dims 0-6):  plain={base_gap:.4f}  guided={guided_gap:.4f}"
          f"   -> {'guided is closer (smoother join)' if guided_gap < base_gap else 'no improvement'}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
