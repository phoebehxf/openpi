import dataclasses
import enum
import logging
from pathlib import Path
import socket

import numpy as np
from typing_extensions import override
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import normalize as _normalize
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str

    # Optional exported LoRA adapter directory or adapter.npz file.
    adapter: str | None = None

    # Optional directory containing norm_stats.json (or the file itself).
    # Use this when an adapter was trained with stats different from the base checkpoint.
    norm_stats: str | None = None


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False
    # Save each exact websocket request plus decoded inputs and policy outputs.
    capture_dir: str | None = None
    # Test-only raw gripper state override. For PI0.5 quantile normalization,
    # use the midpoint of q01/q99 (0.4999 for the three-phase pour stats) to
    # present a neutral gripper state to the model while leaving robot control unchanged.
    gripper_state_override: float | None = None
    # Test-only one-sided override: replace the model input only while the real
    # gripper state is open (>= 0.5). Once feedback reports closed, pass the
    # real closed state through so the model can maintain its grasp.
    neutralize_open_gripper_state: float | None = None
    # Latch the model input to closed after a predicted close chunk. A new client
    # connection or an explicit _reset_gripper_latch request clears the latch.
    latch_gripper_after_close: bool = False
    gripper_latch_threshold: float = 0.5
    gripper_latch_min_steps: int = 5

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


class GripperStateOverridePolicy(_policy.BasePolicy):
    """Override only the model input's final state dimension for a real-robot A/B test."""

    def __init__(
        self,
        policy: _policy.BasePolicy,
        value: float,
        *,
        open_only: bool = False,
        latch_after_close: bool = False,
        latch_threshold: float = 0.5,
        latch_min_steps: int = 5,
    ):
        self._policy = policy
        self._value = value
        self._open_only = open_only
        self._latch_after_close = latch_after_close
        self._latch_threshold = latch_threshold
        self._latch_min_steps = latch_min_steps
        self._grasp_latched = False

    def reset_gripper_latch(self) -> None:
        """Clear deployment-only state at the start of/retry within an episode."""
        self._grasp_latched = False

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        modified = dict(obs)
        reset_requested = bool(modified.pop("_reset_gripper_latch", False))
        if reset_requested:
            self.reset_gripper_latch()
        if "observation/state" not in obs:
            raise KeyError("gripper-state override requires observation/state")
        state = np.asarray(obs["observation/state"], dtype=np.float32).copy()
        if state.ndim != 1 or state.shape[0] < 7:
            raise ValueError(f"Expected observation/state shaped [>=7], got {state.shape}")
        original = float(state[6])
        latched_before = self._grasp_latched
        # A manual retry must escape the closed-state attractor even though the
        # client's last commanded gripper state is still 0 (closed).
        applied = reset_requested or latched_before or not self._open_only or original >= 0.5
        if latched_before:
            state[6] = 0.0
        elif applied:
            state[6] = self._value
        modified["observation/state"] = state
        result = self._policy.infer(modified)
        gripper_actions = np.asarray(result["actions"])[..., 6]
        close_steps = int(np.sum(gripper_actions < self._latch_threshold))
        if not reset_requested and self._latch_after_close and close_steps >= self._latch_min_steps:
            self._grasp_latched = True
        result["gripper_state_override"] = {
            "original": original,
            "model_input": float(state[6]),
            "applied": applied,
            "open_only": self._open_only,
            "close_steps": close_steps,
            "latch_threshold": self._latch_threshold,
            "latch_min_steps": self._latch_min_steps,
            "latched_before": latched_before,
            "latched_after": self._grasp_latched,
            "reset_requested": reset_requested,
        }
        return result

    @property
    def metadata(self) -> dict:
        return self._policy.metadata


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        # dir="gs://openpi-assets/checkpoints/pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            norm_stats = None
            if args.policy.norm_stats is not None:
                norm_path = Path(args.policy.norm_stats).expanduser().resolve()
                norm_stats = _normalize.load(norm_path.parent if norm_path.is_file() else norm_path)
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                adapter_path=args.policy.adapter,
                norm_stats=norm_stats,
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    if args.gripper_state_override is not None and args.neutralize_open_gripper_state is not None:
        raise ValueError("Use only one of --gripper-state-override and --neutralize-open-gripper-state")
    if args.latch_gripper_after_close and args.neutralize_open_gripper_state is None:
        raise ValueError("--latch-gripper-after-close requires --neutralize-open-gripper-state")
    if not np.isfinite(args.gripper_latch_threshold):
        raise ValueError("--gripper-latch-threshold must be finite")
    if args.gripper_latch_min_steps <= 0:
        raise ValueError("--gripper-latch-min-steps must be positive")
    if args.gripper_state_override is not None:
        if not np.isfinite(args.gripper_state_override):
            raise ValueError("--gripper-state-override must be finite")
        logging.warning(
            "TEST MODE: overriding model gripper state input with %.6f; robot feedback/control is unchanged",
            args.gripper_state_override,
        )
        policy = GripperStateOverridePolicy(policy, args.gripper_state_override)
    if args.neutralize_open_gripper_state is not None:
        if not np.isfinite(args.neutralize_open_gripper_state):
            raise ValueError("--neutralize-open-gripper-state must be finite")
        logging.warning(
            "TEST MODE: using %.6f as model gripper input only while real state is open; "
            "closed feedback passes through unchanged",
            args.neutralize_open_gripper_state,
        )
        policy = GripperStateOverridePolicy(
            policy,
            args.neutralize_open_gripper_state,
            open_only=True,
            latch_after_close=args.latch_gripper_after_close,
            latch_threshold=args.gripper_latch_threshold,
            latch_min_steps=args.gripper_latch_min_steps,
        )
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
        capture_dir=args.capture_dir,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
