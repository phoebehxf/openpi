"""RTC-enabled policy server for openpi pi0 / pi05 (remote GPU side).

Drop-in alternative to `scripts/serve_policy.py`: same websocket protocol / port, but it
applies Real-Time Chunking (RTC) prefix guidance so consecutive action chunks join
smoothly. NOTHING existing is modified -- this reuses `create_trained_policy` (for the
model + the exact input/output transforms) and the JAX RTC sampler in `rtc_sampling.py`.

Where RTC sits (see rtc_sampling.py for the math):
    raw obs --①input_transform--> model space --②RTC sample--> model space --③output_transform--> raw actions
RTC guidance happens in ② (normalized model space). The still-unexecuted tail of the
PREVIOUS chunk is kept here in model space (no re-normalization needed) and used as the
guiding prefix for the next chunk.

Protocol additions (all optional, read from the obs dict, popped before transforms):
    obs["rtc_reset"]        : bool  -- start of a new episode; drop the stored prev chunk.
    obs["rtc_executed"]     : int   -- how many actions from the previous server chunk had
                                       already been executed when this observation was captured.
    obs["inference_delay"]  : int   -- measured #control-steps elapsed during inference on
                                       the client; sets how many leading prefix steps are
                                       hard-committed (weight 1.0). Falls back to config.

The server keeps the previous chunk and slices its tail at `execution_horizon`, matching
the client that executes exactly `execution_horizon` steps per chunk. A stock (non-RTC)
client still works -- without the extra keys it just never resets and uses the default
inference_delay; but for correct alignment use the matching async client
`scripts/piper_rtc_client.py`.

Run (remote GPU):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/serve_policy_rtc.py \
        --config pi05_piper_pick_and_place \
        --dir checkpoints/pi05_piper_pick_and_place/piper_pick_pi05_lora_v2/29999 \
        --execution-horizon 8 --port 8000
"""

from __future__ import annotations

import dataclasses
import logging
import socket

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import tyro
from typing_extensions import override

import openpi.models.model as _model
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

try:
    import rtc_sampling
except ImportError:
    from scripts import rtc_sampling


@dataclasses.dataclass
class Args:
    config: str = "pi05_piper_pick_and_place_v2"
    dir: str = "checkpoints/pi05_piper_pick_and_place/piper_pick_pi05_lora_v2/29999"
    port: int = 8000
    default_prompt: str | None = None
    # RTC knobs (mirror rtc_sampling.RTCConfig).
    rtc_enabled: bool = True
    execution_horizon: int = 8
    inference_delay: int = 1
    max_guidance_weight: float = 10.0
    num_steps: int = 10
    prefix_attention_schedule: rtc_sampling.PrefixAttentionSchedule = rtc_sampling.PrefixAttentionSchedule.LINEAR


class RTCPolicy(_base_policy.BasePolicy):
    """Wraps a trained openpi Policy, swapping stock sampling for stateful RTC sampling.

    Reuses the wrapped policy's input/output transforms and model. Keeps the previous
    chunk in model space so the guiding prefix needs no re-normalization. The heavy
    sampling is JIT-compiled once via the `module_jit`/`nnx.split` pattern.
    """

    def __init__(self, policy, cfg: rtc_sampling.RTCConfig, *, rtc_enabled: bool):
        self._policy = policy
        self._cfg = cfg
        self._rtc_enabled = rtc_enabled
        self._model = policy._model
        self._rng = jax.random.key(0)
        self._prev_chunk = None  # (b, ah, ad) previous chunk, MODEL space

        num_steps = cfg.num_steps
        dt = -1.0 / num_steps
        max_gw = cfg.max_guidance_weight
        graphdef, state = nnx.split(self._model)

        def _run(state, srng, observation, *, prev, weights):
            model = nnx.merge(graphdef, state)
            observation = _model.preprocess_observation(None, observation, train=False)
            b = observation.state.shape[0]
            noise = jax.random.normal(srng, (b, model.action_horizon, model.action_dim))
            denoise = rtc_sampling._make_denoiser(model, observation)
            x_t, time = noise, 1.0
            for _ in range(num_steps):
                if prev is None:
                    v_t = denoise(x_t, time)
                else:
                    v_t = rtc_sampling._guided_velocity(denoise, x_t, time, prev, weights, max_gw)
                x_t = x_t + dt * v_t
                time = time + dt
            return x_t

        # Two traces: plain (first chunk) and guided (prev + weights are arrays).
        self._sample_plain = jax.jit(lambda st, r, o: _run(st, r, o, prev=None, weights=None))
        self._sample_guided = jax.jit(lambda st, r, o, p, w: _run(st, r, o, prev=p, weights=w))
        self._state = state

    @property
    def metadata(self):
        return self._policy.metadata

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        obs = dict(obs)
        if bool(obs.pop("rtc_reset", False)):
            self._prev_chunk = None
        executed = int(obs.pop("rtc_executed", self._cfg.execution_horizon))
        inf_delay = obs.pop("inference_delay", None)

        inputs = self._policy._input_transform(obs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)

        self._rng, srng = jax.random.split(self._rng)
        eh = self._cfg.execution_horizon
        use_guidance = self._rtc_enabled and self._prev_chunk is not None

        if not use_guidance:
            chunk = self._sample_plain(self._state, srng, observation)
        else:
            ah = self._model.action_horizon
            executed = int(np.clip(executed, 0, ah))
            leftover = self._prev_chunk[:, executed:, :]
            if leftover.shape[1] == 0:
                chunk = self._sample_plain(self._state, srng, observation)
                self._prev_chunk = np.asarray(chunk)
                outputs = {
                    "state": np.asarray(inputs["state"][0]),
                    "actions": np.asarray(chunk[0]),
                }
                return self._policy._output_transform(outputs)
            delay = self._cfg.inference_delay if inf_delay is None else int(inf_delay)
            prefix_end = min(leftover.shape[1], ah)
            w = rtc_sampling.get_prefix_weights(delay, prefix_end, ah, self._cfg.prefix_attention_schedule)
            # Pad leftover to full chunk shape (model space).
            prev = jnp.asarray(np.pad(np.asarray(leftover), ((0, 0), (0, ah - leftover.shape[1]), (0, 0))))
            chunk = self._sample_guided(self._state, srng, observation, prev, jnp.asarray(w)[None, :, None])

        self._prev_chunk = np.asarray(chunk)  # model space, for next call's prefix
        outputs = {
            "state": np.asarray(inputs["state"][0]),
            "actions": np.asarray(chunk[0]),
        }
        outputs = self._policy._output_transform(outputs)
        return outputs


def main(args: Args) -> None:
    logging.info("Loading policy (config=%s, dir=%s)...", args.config, args.dir)
    policy = _policy_config.create_trained_policy(
        _config.get_config(args.config), args.dir, default_prompt=args.default_prompt
    )
    cfg = rtc_sampling.RTCConfig(
        prefix_attention_schedule=args.prefix_attention_schedule,
        max_guidance_weight=args.max_guidance_weight,
        execution_horizon=args.execution_horizon,
        inference_delay=args.inference_delay,
        num_steps=args.num_steps,
    )
    rtc_policy = RTCPolicy(policy, cfg, rtc_enabled=args.rtc_enabled)
    logging.info(
        "RTC server ready: rtc_enabled=%s execution_horizon=%d num_steps=%d schedule=%s",
        args.rtc_enabled, args.execution_horizon, args.num_steps, args.prefix_attention_schedule.value,
    )

    hostname = socket.gethostname()
    logging.info("Serving on %s:%d", socket.gethostbyname(hostname), args.port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=rtc_policy, host="0.0.0.0", port=args.port, metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
