"""Move the Piper arm to the training start ("home") pose before running the policy.

The pi0/pi05 policy expects to start from roughly the same joint configuration seen at the
beginning of training episodes. If the real arm starts far from that pose (e.g. joint1/joint2
out of the training range), the policy sees an out-of-distribution state and hedges / does
nothing useful. Run this first so every experiment starts from the same, in-distribution home.

Default home target is episode 0 frame 0 of phoebe777777/piper-pick-up-repaired:
    [-0.028, 0.403, 0.000, 0.041, 0.379, -0.021]   (gripper open)

The move is done by slew-limited interpolation from the current joints to the target, so it
crawls there instead of snapping. Keep a hand on the e-stop.
"""

import dataclasses
import importlib
import pathlib
import sys
import time

import numpy as np
import tyro


@dataclasses.dataclass
class Args:
    bci_piper_root: str = "/home/huix/bci_robot/bci_piper"
    robot_model: str = "piper"
    real: bool = False
    speed_percent: int = 10
    # Training home = ep0 frame0 state of piper-pick-up-repaired (6 joints, radians).
    home: tuple[float, float, float, float, float, float] = (-0.028, 0.403, 0.000, 0.041, 0.379, -0.021)
    # Max joint change per control step (rad) -- safety slew-rate limit.
    step_rad: float = 0.03
    control_hz: float = 30.0
    # Tolerance (rad) to consider a joint "arrived".
    tol_rad: float = 0.01
    max_steps: int = 2000
    # Skip the interactive confirmation before moving.
    yes: bool = False


def _load_robot(args: Args):
    root = pathlib.Path(args.bci_piper_root).expanduser().resolve()
    robot_side = root / "robot_arm_side"
    if not robot_side.exists():
        raise FileNotFoundError(f"Could not find bci_piper robot_arm_side at {robot_side}")
    if str(robot_side) not in sys.path:
        sys.path.insert(0, str(robot_side))

    module = importlib.import_module("eeg_robot_arm_control")
    cfg = module.ControlConfig(
        robot_model=args.robot_model,
        speed_percent=args.speed_percent,
        joint_step_rad=args.step_rad,
    )
    robot = module.RobotArm(real=args.real, cfg=cfg)
    robot.connect()
    return robot


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
        print(f"[home] max |delta| = {np.abs(err).max():.3f} rad, "
              f"est. steps ~ {int(np.ceil(np.abs(err).max() / args.step_rad))} @ {args.control_hz}Hz")

        if not args.yes:
            reply = input("[home] proceed with slew move? type 'yes' to move: ").strip().lower()
            if reply != "yes":
                print("[home] aborted.")
                return

        dt = 1.0 / args.control_hz
        robot._ensure_control_mode()
        robot.robot.set_motion_mode("j")
        for step in range(args.max_steps):
            start = time.perf_counter()
            current = robot._read_joints(timeout=0.2)
            if current is None:
                raise RuntimeError("Failed to read Piper joint angles mid-move.")
            current = np.asarray(current[:6], dtype=np.float32)
            err = target - current
            if np.abs(err).max() <= args.tol_rad:
                print(f"[home] arrived at step {step}, joints=", np.round(current, 3).tolist())
                break
            step_cmd = np.clip(err, -args.step_rad, args.step_rad)
            cmd = current + step_cmd
            robot.robot.move_j(cmd.tolist())
            if step % 10 == 0:
                print(f"[home] step={step} joints={np.round(current,3).tolist()} "
                      f"max|err|={np.abs(err).max():.3f}")
            elapsed = time.perf_counter() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)
        else:
            print("[home] WARNING: hit max_steps before reaching tolerance.")
    finally:
        robot.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
