"""Open-loop replay of a LeRobot dataset episode onto a Piper arm.

Goal
----
Send the recorded ``action`` sequence of a single demo episode straight to the
robot, with no policy and no closed-loop smoothing. If this cannot reproduce the
demo motion, the problem is in data collection or the v3->v2 format conversion,
not in the policy.

The dataset ``phoebe777777/piper-pick-up-repaired`` stores 7-d vectors:
``[joint1..joint6 (absolute radians), gripper (0..1 fraction)]`` at 30 fps. So the
replay just moves the arm to each absolute joint target in order and drives the
gripper from the fraction. This deliberately BYPASSES the aggressive smoothing in
``piper_remote_client.py`` (``action_alpha``, ``max_joint_delta_rad=0.03``), which
exists for jittery policy output and would prevent the arm from following a
recorded trajectory.

Typical usage
-------------
Dry-run (no hardware) to sanity check the data and pacing::

    python scripts/piper_replay_dataset.py --episode 0

Export one episode's actions to a .npy (run on any machine that has pyarrow),
then replay that file on the robot machine (numpy only, no pyarrow needed)::

    python scripts/piper_replay_dataset.py --episode 0 --dump-actions ep0.npy
    python scripts/piper_replay_dataset.py --actions-file ep0.npy --real

Replay on the real robot::

    python scripts/piper_replay_dataset.py --episode 0 --real --speed-percent 30
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import pathlib
import sys
import time
from typing import Optional

import numpy as np
import tyro


@dataclasses.dataclass
class Args:
    # --- dataset selection ---
    repo_id: str = "phoebe777777/piper-pick-up-repaired"
    episode: int = 0
    # Root of the LeRobot cache holding <repo_id>. Defaults to the HF lerobot cache.
    dataset_root: Optional[str] = None
    # Which column to replay: "action" (default) or "state" (recorded observations).
    # Replaying "state" is a useful cross-check: if state replays fine but action
    # does not, the action column itself is suspect.
    source: str = "action"

    # --- pre-exported actions (numpy only, skips pyarrow) ---
    # If set for replay, load the Nx7 sequence from this .npy/.csv instead of parquet.
    actions_file: Optional[str] = None
    # If set, just export the selected episode sequence here and exit (no robot).
    dump_actions: Optional[str] = None

    # --- robot connection (mirrors piper_remote_client.py) ---
    # Root of the bci_piper checkout (the one containing robot_arm_side/). Leave as
    # None to auto-detect: $BCI_PIPER_ROOT, then a sibling ../bci_piper of this repo,
    # then the Linux deployment path.
    bci_piper_root: Optional[str] = None
    robot_model: str = "piper"
    real: bool = False
    speed_percent: int = 30

    # --- replay pacing ---
    # Playback rate in Hz. Dataset is 30 fps; keep at 30 for real-time replay.
    replay_hz: float = 30.0
    # Take every Nth frame (1 = all frames). Increases per-step joint delta.
    downsample: int = 1
    start_frame: int = 0
    end_frame: int = -1  # -1 = to the end of the episode

    # --- gripper ---
    disable_gripper: bool = False

    # --- safety ---
    # Move gradually to the first frame's joint pose before starting replay.
    move_to_start: bool = True
    # Per-step approach increment (rad) while moving to the start pose.
    approach_step_rad: float = 0.02
    approach_hz: float = 20.0
    settle_sec: float = 1.0
    # Safety cap on the per-step commanded joint delta during replay. A jump larger
    # than this is clamped and warned about (protects against corrupt frames /
    # aggressive downsampling). Set very high to effectively disable.
    max_step_rad: float = 0.35
    # Abort if a commanded per-step delta exceeds this (rad). 0 disables.
    abort_step_rad: float = 1.2

    print_every: int = 30


def _default_cache_root() -> pathlib.Path:
    env = os.environ.get("HF_LEROBOT_HOME")
    if env:
        return pathlib.Path(env).expanduser()
    return pathlib.Path.home() / ".cache" / "huggingface" / "lerobot"


def load_episode_sequence(args: Args) -> tuple[np.ndarray, str]:
    """Return (Nx7 float32 array, task_string) for the requested episode column."""
    import pyarrow.parquet as pq  # local import so the robot side can skip pyarrow

    root = pathlib.Path(args.dataset_root).expanduser() if args.dataset_root else _default_cache_root() / args.repo_id
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root}. Pass --dataset-root or pre-export with --dump-actions."
        )

    column = "action" if args.source == "action" else "observation.state"

    # Locate the data file for this episode via the episodes metadata.
    meta_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not meta_files:
        raise FileNotFoundError(f"No episode metadata parquet under {root/'meta'/'episodes'}")

    file_index = chunk_index = None
    task = ""
    for mf in meta_files:
        m = pq.read_table(
            mf, columns=["episode_index", "tasks", "data/chunk_index", "data/file_index"]
        ).to_pydict()
        for i, ep in enumerate(m["episode_index"]):
            if int(ep) == args.episode:
                chunk_index = int(m["data/chunk_index"][i])
                file_index = int(m["data/file_index"][i])
                tasks = m["tasks"][i]
                task = tasks[0] if isinstance(tasks, (list, tuple)) and tasks else str(tasks)
                break
        if file_index is not None:
            break
    if file_index is None:
        raise ValueError(f"Episode {args.episode} not found in dataset metadata.")

    data_file = root / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file for episode {args.episode} not found: {data_file}")

    tbl = pq.read_table(data_file, columns=[column, "episode_index", "frame_index"]).to_pydict()
    rows = [
        (int(fi), np.asarray(v, dtype=np.float32))
        for fi, ep, v in zip(tbl["frame_index"], tbl["episode_index"], tbl[column])
        if int(ep) == args.episode
    ]
    if not rows:
        raise ValueError(f"No rows for episode {args.episode} in {data_file}")
    rows.sort(key=lambda r: r[0])
    seq = np.stack([r[1] for r in rows], axis=0)
    if seq.shape[1] != 7:
        raise ValueError(f"Expected 7-d {column}, got shape {seq.shape}")
    return seq, task


def load_actions_file(path: str) -> np.ndarray:
    p = pathlib.Path(path).expanduser()
    if p.suffix == ".npy":
        seq = np.load(p)
    else:
        seq = np.loadtxt(p, delimiter=",")
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != 7:
        raise ValueError(f"actions-file must be Nx7, got {seq.shape}")
    return seq


def slice_sequence(seq: np.ndarray, args: Args) -> np.ndarray:
    end = seq.shape[0] if args.end_frame < 0 else min(args.end_frame, seq.shape[0])
    seq = seq[args.start_frame:end]
    if args.downsample > 1:
        seq = seq[:: args.downsample]
    return seq


def _find_bci_piper_root(explicit: Optional[str]) -> pathlib.Path:
    candidates = []
    if explicit:
        candidates.append(pathlib.Path(explicit))
    env = os.environ.get("BCI_PIPER_ROOT")
    if env:
        candidates.append(pathlib.Path(env))
    # Sibling of this repo, e.g. .../piper/bci_piper next to .../piper/openpi
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    candidates.append(repo_root.parent / "bci_piper")
    # Linux deployment default.
    candidates.append(pathlib.Path("/home/huix/bci_robot/bci_piper"))

    for c in candidates:
        if (c.expanduser() / "robot_arm_side").exists():
            return c.expanduser().resolve()
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not find bci_piper (a dir containing robot_arm_side/). Tried:\n  "
        f"{tried}\nPass --bci-piper-root explicitly or set $BCI_PIPER_ROOT."
    )


def connect_robot(args: Args):
    root = _find_bci_piper_root(args.bci_piper_root)
    robot_side = root / "robot_arm_side"
    print(f"[replay] using bci_piper at {root}")
    if str(robot_side) not in sys.path:
        sys.path.insert(0, str(robot_side))

    module = importlib.import_module("eeg_robot_arm_control")
    cfg = module.ControlConfig(robot_model=args.robot_model, speed_percent=args.speed_percent)
    robot = module.RobotArm(real=args.real, cfg=cfg)
    robot.connect()
    return robot


def send_absolute(robot, joints6, gripper: Optional[float]) -> None:
    """Send an absolute joint target (list of 6 radians) and optional gripper fraction."""
    target = [float(x) for x in joints6]
    if not robot.real:
        msg = f"[replay] dry-run move_j={[round(v, 4) for v in target]}"
        if gripper is not None:
            msg += f" gripper={gripper:.2f}"
        print(msg)
    else:
        robot._ensure_control_mode()
        robot.robot.set_motion_mode("j")
        robot.robot.move_j(target)
    if gripper is not None:
        robot._set_gripper_fraction(float(gripper))


def move_to_start(robot, start_joints6, args: Args) -> None:
    """Gradually walk the arm from its current pose to the episode start pose."""
    target = np.asarray(start_joints6, dtype=np.float32)
    if not robot.real:
        print(f"[replay] dry-run approach to start={np.round(target, 4).tolist()}")
        send_absolute(robot, target, None)
        return

    current = robot._read_joints(timeout=1.0)
    if current is None:
        raise RuntimeError("Failed to read Piper joints before approach.")
    current = np.asarray(current[:6], dtype=np.float32)

    dist = float(np.max(np.abs(target - current)))
    n_steps = max(1, int(np.ceil(dist / max(1e-6, args.approach_step_rad))))
    print(f"[replay] approaching start pose over {n_steps} steps (max joint dist {dist:.3f} rad)")
    dt = 1.0 / max(1e-6, args.approach_hz)
    for i in range(1, n_steps + 1):
        interp = current + (target - current) * (i / n_steps)
        send_absolute(robot, interp, None)
        time.sleep(dt)
    time.sleep(args.settle_sec)

    reached = robot._read_joints(timeout=1.0)
    if reached is not None:
        err = float(np.max(np.abs(np.asarray(reached[:6], dtype=np.float32) - target)))
        print(f"[replay] reached start pose, max joint error {err:.4f} rad")


def replay(robot, seq: np.ndarray, task: str, args: Args) -> None:
    joints = seq[:, :6]
    grip = seq[:, 6]
    n = seq.shape[0]
    print(
        f"[replay] task={task!r} frames={n} source={args.source} "
        f"replay_hz={args.replay_hz} downsample={args.downsample} gripper={'off' if args.disable_gripper else 'on'}"
    )

    if args.move_to_start:
        move_to_start(robot, joints[0], args)

    dt = 1.0 / max(1e-6, args.replay_hz)
    prev = np.asarray(joints[0], dtype=np.float32)
    max_seen = 0.0
    clipped_count = 0
    start_t = time.perf_counter()
    for i in range(n):
        loop_start = time.perf_counter()
        raw = np.asarray(joints[i], dtype=np.float32)
        step = raw - prev
        step_mag = float(np.max(np.abs(step)))
        max_seen = max(max_seen, step_mag)

        if args.abort_step_rad > 0 and step_mag > args.abort_step_rad:
            raise RuntimeError(
                f"[replay] ABORT at frame {i}: per-step joint delta {step_mag:.3f} rad exceeds "
                f"--abort-step-rad {args.abort_step_rad}. Data likely corrupt/misordered."
            )
        if step_mag > args.max_step_rad:
            clipped_count += 1
            step = np.clip(step, -args.max_step_rad, args.max_step_rad)
            target = prev + step
        else:
            target = raw

        gripper = None if args.disable_gripper else float(np.clip(grip[i], 0.0, 1.0))
        send_absolute(robot, target, gripper)
        prev = np.asarray(target, dtype=np.float32)

        if i % max(1, args.print_every) == 0:
            print(
                f"[replay] frame {i}/{n} joints={np.round(target, 4).tolist()} "
                f"grip={'-' if gripper is None else round(gripper, 2)} step={step_mag:.4f}"
            )

        elapsed = time.perf_counter() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

    wall = time.perf_counter() - start_t
    print(
        f"[replay] done: {n} frames in {wall:.1f}s (target {n * dt:.1f}s). "
        f"max per-step joint delta={max_seen:.4f} rad, clipped {clipped_count} frames "
        f"(cap {args.max_step_rad})."
    )
    if clipped_count:
        print(
            "[replay] WARNING: some steps were clipped. Raise --max-step-rad or lower "
            "--downsample for a faithful replay."
        )


def main(args: Args) -> None:
    if args.source not in ("action", "state"):
        raise ValueError("--source must be 'action' or 'state'")

    # Load the sequence.
    if args.actions_file:
        seq = load_actions_file(args.actions_file)
        task = f"<from {args.actions_file}>"
        print(f"[replay] loaded {seq.shape[0]} frames from {args.actions_file}")
    else:
        seq, task = load_episode_sequence(args)
        print(f"[replay] loaded episode {args.episode}: {seq.shape[0]} frames, task={task!r}")

    seq = slice_sequence(seq, args)
    if seq.shape[0] == 0:
        raise ValueError("Empty sequence after start/end/downsample slicing.")

    # Report a quick summary useful for spotting bad conversions.
    j = seq[:, :6]
    print(
        "[replay] joint range (rad): min="
        f"{np.round(j.min(axis=0), 3).tolist()} max={np.round(j.max(axis=0), 3).tolist()}"
    )
    print(
        f"[replay] gripper range: [{seq[:,6].min():.3f}, {seq[:,6].max():.3f}]; "
        f"max frame-to-frame joint delta={np.max(np.abs(np.diff(j, axis=0))) if len(j) > 1 else 0:.4f} rad"
    )

    if args.dump_actions:
        out = pathlib.Path(args.dump_actions).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix == ".csv":
            np.savetxt(out, seq, delimiter=",")
        else:
            np.save(out, seq)
        print(f"[replay] exported {seq.shape[0]}x7 {args.source} sequence to {out}. Not connecting to robot.")
        return

    robot = connect_robot(args)
    try:
        replay(robot, seq, task, args)
    finally:
        robot.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
