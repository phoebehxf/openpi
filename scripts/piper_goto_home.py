"""Move the Piper arm to the training start ("home") pose before running the policy.

The pi0/pi05 policy expects to start from roughly the same joint configuration seen at the
beginning of training episodes. If the real arm starts far from that pose (e.g. joint1/joint2
out of the training range), the policy sees an out-of-distribution state and hedges / does
nothing useful. Run this first so every experiment starts from the same, in-distribution home.

Default home target is episode 0 frame 0 of phoebe777777/piper-pick-up-repaired:
    [-0.028, 0.403, 0.000, 0.041, 0.379, -0.021]   (gripper open)

IMPORTANT SAFETY NOTES:
- Piper `move_j` is a point-to-point command; this sends it ONCE and polls until arrived.
  Do NOT stream move_j at high rate -- overlapping P2P commands make the arm move erratically.
- The arm has no mechanical brake. `RobotArm.close()` calls `disable()`, which cuts motor
  torque and makes the arm DROP under gravity. This script does NOT disable on exit by default;
  it leaves the arm enabled and holding position. Support the arm before you power it down.
  Pass --disable-on-exit only when the arm is in a safe/supported position.
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
    # Tolerance (rad) to consider a joint "arrived".
    tol_rad: float = 0.02
    # How long to wait for the single move_j to reach the target before giving up (seconds).
    arrive_timeout_s: float = 30.0
    # Skip the interactive confirmation before moving.
    yes: bool = False
    # Cut motor torque on exit (arm will DROP if unsupported). Off by default for safety.
    disable_on_exit: bool = False


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
    )
    robot = module.RobotArm(real=args.real, cfg=cfg)
    robot.connect()
    return robot


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
            reply = input("[home] send ONE move_j to home? type 'yes' to move: ").strip().lower()
            if reply != "yes":
                print("[home] aborted.")
                return

        robot._ensure_control_mode()
        robot.robot.set_motion_mode("j")
        robot.robot.set_speed_percent(args.speed_percent)
        # Single point-to-point move; the controller plans the trajectory at speed_percent.
        robot.robot.move_j(target.tolist())
        print("[home] move_j sent; polling until arrived ...")

        deadline = time.time() + args.arrive_timeout_s
        while time.time() < deadline:
            current = robot._read_joints(timeout=0.2)
            if current is not None:
                current = np.asarray(current[:6], dtype=np.float32)
                err = target - current
                if np.abs(err).max() <= args.tol_rad:
                    print("[home] arrived, joints=", np.round(current, 3).tolist())
                    break
            time.sleep(0.1)
        else:
            print("[home] WARNING: did not reach tolerance within timeout; last joints=",
                  np.round(current, 3).tolist() if current is not None else None)
    finally:
        _safe_teardown(robot, args)


if __name__ == "__main__":
    main(tyro.cli(Args))
