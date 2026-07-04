"""Move the Piper arm to the training start ("home") pose before running the policy.

The pi0/pi05 policy expects to start from roughly the same joint configuration seen at the
beginning of training episodes. If the real arm starts far from that pose (e.g. joint1/joint2
out of the training range), the policy sees an out-of-distribution state and hedges / does
nothing useful. Run this first so every experiment starts from the same, in-distribution home.

Default home target is the median start/end pose across all 182 training episodes:
    [-1.536, 0.000, 0.000, 2.296, 21.716, -1.269] degrees

IMPORTANT SAFETY NOTES:
- The pyAgxArm backend expects joint targets to be refreshed continuously. This script
  interpolates from the measured pose to home and sends bounded steps at a fixed rate.
- The arm has no mechanical brake. `RobotArm.close()` calls `disable()`, which cuts motor
  torque and makes the arm DROP under gravity. This script does NOT disable on exit by default;
  it leaves the arm enabled and holding position. Support the arm before you power it down.
  Pass --disable-on-exit only when the arm is in a safe/supported position.
"""

import dataclasses
import importlib
import os
import pathlib
import sys
import time
from typing import Optional

import numpy as np
import tyro


DATASET_HOME_RAD = (
    -0.02680826,
    0.0,
    0.0,
    0.04007276,
    0.37901668,
    -0.02214823,
)


@dataclasses.dataclass
class Args:
    # Auto-detect $BCI_PIPER_ROOT, a sibling bci_piper checkout, or the Linux default.
    bci_piper_root: Optional[str] = None
    robot_model: str = "piper"
    real: bool = False
    speed_percent: int = 10
    # Median start/end joint pose across all 182 training episodes, in radians.
    home: tuple[float, float, float, float, float, float] = DATASET_HOME_RAD
    # Tolerance (rad) to consider a joint "arrived".
    tol_rad: float = 0.02
    # How long to keep refreshing the final target before giving up (seconds).
    arrive_timeout_s: float = 30.0
    # Refresh rate and interpolation velocity for pyAgxArm joint commands.
    command_rate_hz: float = 50.0
    max_joint_vel_rad_s: float = 0.15
    # Keep refreshing the final target briefly after reaching it.
    hold_s: float = 0.5
    # Skip the interactive confirmation before moving.
    yes: bool = False
    # Cut motor torque on exit (arm will DROP if unsupported). Off by default for safety.
    disable_on_exit: bool = False


def _find_bci_piper_root(explicit: Optional[str]) -> pathlib.Path:
    candidates: list[pathlib.Path] = []
    if explicit:
        candidates.append(pathlib.Path(explicit))
    if env_root := os.environ.get("BCI_PIPER_ROOT"):
        candidates.append(pathlib.Path(env_root))
    candidates.append(pathlib.Path(__file__).resolve().parents[2] / "bci_piper")
    candidates.append(pathlib.Path("/home/huix/bci_robot/bci_piper"))

    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "robot_arm_side").is_dir():
            return root

    tried = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find bci_piper (expected robot_arm_side/). Tried:\n  "
        f"{tried}\nPass --bci-piper-root or set BCI_PIPER_ROOT."
    )


def _load_robot(args: Args):
    root = _find_bci_piper_root(args.bci_piper_root)
    robot_side = root / "robot_arm_side"
    print(f"[home] using bci_piper at {root}")
    if str(robot_side) not in sys.path:
        sys.path.insert(0, str(robot_side))

    module = importlib.import_module("eeg_robot_arm_control")
    cfg = module.ControlConfig(
        robot_model=args.robot_model,
        speed_percent=args.speed_percent,
    )
    robot = module.RobotArm(real=args.real, cfg=cfg)
    robot.connect()
    return robot


def _stream_move_to_home(robot, start: np.ndarray, target: np.ndarray, args: Args) -> None:
    if args.command_rate_hz <= 0:
        raise ValueError("command_rate_hz must be greater than zero")
    if args.max_joint_vel_rad_s <= 0:
        raise ValueError("max_joint_vel_rad_s must be greater than zero")

    delta = target - start
    duration = max(float(np.abs(delta).max()) / args.max_joint_vel_rad_s, 1.0 / args.command_rate_hz)
    steps = max(1, int(np.ceil(duration * args.command_rate_hz)))
    period = 1.0 / args.command_rate_hz
    print(
        f"[home] interpolating {steps} commands over {duration:.1f}s "
        f"(limit={args.max_joint_vel_rad_s:.3f} rad/s)"
    )

    next_send = time.monotonic()
    for step in range(1, steps + 1):
        fraction = step / steps
        command = start + fraction * delta
        robot.robot.move_j(command.tolist())
        next_send += period
        sleep_s = next_send - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)

    hold_deadline = time.monotonic() + max(0.0, args.hold_s)
    while time.monotonic() < hold_deadline:
        robot.robot.move_j(target.tolist())
        time.sleep(period)


def _safe_teardown(robot, args: Args) -> None:
    """Release the connection WITHOUT cutting torque, unless explicitly asked to disable."""
    if robot.robot is None:
        return
    try:
        if args.disable_on_exit:
            print("[home] disabling motors (arm will go limp) ...")
            robot.robot.disable()
        else:
            print("[home] leaving motors ENABLED and holding. Support the arm before power-off.")
        robot.robot.disconnect()
    except Exception as exc:  # noqa: BLE001
        print(f"[home] teardown warning: {exc!r}")


def main(args: Args) -> None:
    target = np.asarray(args.home, dtype=np.float32)
    if target.shape != (6,):
        raise ValueError(f"home must have 6 values, got {target.shape}")

    robot = _load_robot(args)
    try:
        if not robot.real:
            print("[home] dry-run (no --real): would move to", np.round(target, 3).tolist())
            return

        current = robot._read_joints(timeout=0.5)
        if current is None:
            raise RuntimeError("Failed to read Piper joint angles.")
        current = np.asarray(current[:6], dtype=np.float32)
        err = target - current
        print("[home] current :", np.round(current, 3).tolist())
        print("[home] target  :", np.round(target, 3).tolist())
        print("[home] delta   :", np.round(err, 3).tolist())
        print(f"[home] max |delta| = {np.abs(err).max():.3f} rad")

        if not args.yes:
            reply = input("[home] move to home? type 'yes' to move: ").strip().lower()
            if reply != "yes":
                print("[home] aborted.")
                return

        robot._ensure_control_mode()
        robot.robot.set_motion_mode("j")
        robot.robot.set_speed_percent(args.speed_percent)
        enable_deadline = time.monotonic() + 2.0
        enabled = False
        while time.monotonic() < enable_deadline:
            enabled = bool(robot.robot.enable())
            if enabled:
                break
            time.sleep(0.05)
        if not enabled:
            raise RuntimeError("Piper did not report enabled; refusing to send home motion.")
        _stream_move_to_home(robot, current, target, args)
        print("[home] trajectory sent; polling and refreshing target until arrived ...")

        deadline = time.monotonic() + args.arrive_timeout_s
        period = 1.0 / args.command_rate_hz
        while time.monotonic() < deadline:
            robot.robot.move_j(target.tolist())
            current = robot._read_joints(timeout=min(0.05, period))
            if current is not None:
                current = np.asarray(current[:6], dtype=np.float32)
                err = target - current
                if np.abs(err).max() <= args.tol_rad:
                    print("[home] arrived, joints=", np.round(current, 3).tolist())
                    break
            time.sleep(period)
        else:
            print("[home] WARNING: did not reach tolerance within timeout; last joints=",
                  np.round(current, 3).tolist() if current is not None else None)
    finally:
        _safe_teardown(robot, args)


if __name__ == "__main__":
    main(tyro.cli(Args))
