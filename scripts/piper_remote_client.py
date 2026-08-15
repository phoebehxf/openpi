import dataclasses
from datetime import datetime, timezone
import importlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tyro

from openpi_client import image_tools
from openpi_client import websocket_client_policy

try:
    import piper_goto_home as home_utils
except ImportError:
    from scripts import piper_goto_home as home_utils


@dataclasses.dataclass
class Args:
    server_host: str
    server_port: int = 8000
    prompt: str = "pick up the object and place it in the basket"
    bci_piper_root: Optional[str] = None
    robot_model: str = "piper"
    real: bool = False
    # Requires --real. Reads real joints/cameras/policy and computes the target, but never sends motion.
    dry_run: bool = False
    # Cut motor torque on exit (arm has no brake -> it DROPS if unsupported). Off by default for safety.
    disable_on_exit: bool = False
    speed_percent: int = 10
    # Move the real arm to the training dataset's common start/end pose before policy control.
    home_on_start: bool = True
    home: tuple[float, float, float, float, float, float] = home_utils.DATASET_HOME_RAD
    home_yes: bool = True
    home_tol_rad: float = 0.02
    home_timeout_s: float = 30.0
    home_command_rate_hz: float = 50.0
    home_max_joint_vel_rad_s: float = 0.15
    home_hold_s: float = 0.5
    global_camera_model: str = "D435"
    wrist_camera_model: str = "D405"
    global_camera_serial: Optional[str] = None
    wrist_camera_serial: Optional[str] = None
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 15
    image_size: int = 224
    show_preview: bool = True
    control_hz: float = 3.0
    open_loop_horizon: int = 1
    # Training episodes run ~740 frames (median, ~25s) up to ~1940 (~65s) at 30fps.
    # 300 steps (~10s @30hz) cuts off before a pick-and-place finishes -> give it room.
    max_steps: int = 1200
    # Unified experiment registry. Logging is enabled when all four IDs are provided.
    participant_id: Optional[str] = None
    session_id: Optional[str] = None
    method_id: Optional[str] = None
    task_id: Optional[str] = None
    calibration_id: Optional[str] = None
    # Disable for quick setup/debug runs that must not enter the experiment registry.
    log_experiment: bool = True
    # Continuously record both RealSense color streams while a logged trial is running.
    record_videos: bool = True
    max_joint_delta_rad: float = 0.03
    action_alpha: float = 0.15
    action_mode: str = "absolute"
    disable_gripper: bool = True
    print_state_debug: bool = True
    print_every: int = 1
    freeze_observation: bool = False
    gripper_open_fraction: float = 1.0
    gripper_closed_fraction: float = 0.0
    # The training gripper signal is effectively binary (0 or 1; ~0% of frames land
    # in between), but the model emits a continuous fraction that hovers mid-range near
    # open/close transitions -> sending it raw at control_hz makes the gripper chatter.
    # Binarize with hysteresis: flip to closed only below close_threshold, back to open
    # only above open_threshold, hold in the dead-band between.
    gripper_binarize: bool = True
    gripper_open_threshold: float = 0.6
    gripper_close_threshold: float = 0.4
    # Optional hard floor for the TCP, expressed in the robot base frame (meters).
    # When set, every joint command is checked with the driver's forward kinematics
    # before it is sent. The policy is stopped if any sampled point on the short
    # joint-space segment would put the TCP below this height.
    table_safety_min_tcp_z_m: Optional[float] = None
    table_safety_path_samples: int = 8
    # Optional taught workspace fence JSON (x/y/z bounds in meters). When supplied,
    # this is checked in addition to the standalone table-height limit above.
    workspace_box: Optional[str] = None
    # Override only the workspace floor margin while preserving the taught table-touch
    # height. For example, 0.005 keeps the TCP 5 mm above the taught touch pose.
    workspace_floor_margin_m: Optional[float] = None
    # Numerical tolerance for FK/feedback noise at a workspace boundary. If the arm is
    # already outside by more than this, commands that reduce (or do not worsen) the
    # violation are still allowed so it can recover instead of deadlocking.
    workspace_safety_tolerance_m: float = 0.001


class RealSenseCamera:
    def __init__(self, serial: str, *, width: int, height: int, fps: int):
        import pyrealsense2 as rs

        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._pipeline.start(config)
        self.serial = serial
        self._width = width
        self._height = height
        self._fps = fps
        self._latest_rgb: Optional[np.ndarray] = None
        self._frame_ready = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._writer = None
        self._recorded_frames = 0
        self._capture_error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._capture_loop, name=f"realsense-{serial}", daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                except RuntimeError:
                    if self._stop.is_set():
                        break
                    continue
                color = frames.get_color_frame()
                if not color:
                    continue
                image_bgr = np.asanyarray(color.get_data()).copy()
                image_rgb = image_bgr[..., ::-1].copy()
                with self._lock:
                    self._latest_rgb = image_rgb
                    if self._writer is not None:
                        self._writer.write(image_bgr)
                        self._recorded_frames += 1
                self._frame_ready.set()
        except BaseException as exc:
            self._capture_error = exc
            self._frame_ready.set()

    def get_rgb(self, timeout_s: float = 5.0) -> np.ndarray:
        if not self._frame_ready.wait(timeout_s):
            raise TimeoutError(f"Timed out waiting for RealSense {self.serial}")
        if self._capture_error is not None:
            raise RuntimeError(f"RealSense capture failed for {self.serial}") from self._capture_error
        with self._lock:
            if self._latest_rgb is None:
                raise RuntimeError(f"RealSense {self.serial} produced no color frame")
            return self._latest_rgb.copy()

    def start_recording(self, path: Path) -> None:
        import cv2

        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite video: {path}")
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(self._fps),
            (self._width, self._height),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Could not open video writer: {path}")
        with self._lock:
            if self._writer is not None:
                writer.release()
                raise RuntimeError(f"RealSense {self.serial} is already recording")
            self._writer = writer
            self._recorded_frames = 0

    def stop_recording(self) -> int:
        with self._lock:
            writer = self._writer
            self._writer = None
            frames = self._recorded_frames
            self._recorded_frames = 0
            if writer is not None:
                writer.release()
        return frames

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.stop_recording()
        self._pipeline.stop()


class CameraRig:
    def __init__(self, args: Args):
        import pyrealsense2 as rs

        context = rs.context()
        devices = {}
        for device in context.query_devices():
            name = device.get_info(rs.camera_info.name)
            serial = device.get_info(rs.camera_info.serial_number)
            devices[name] = serial
            print(f"Detected camera: {name} serial={serial}")

        global_serial = args.global_camera_serial or self._find_serial(devices, args.global_camera_model)
        wrist_serial = args.wrist_camera_serial or self._find_serial(devices, args.wrist_camera_model)

        self.global_camera = RealSenseCamera(
            global_serial,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
        )
        self.wrist_camera = RealSenseCamera(
            wrist_serial,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
        )
        print(f"Using global camera serial={global_serial}, wrist camera serial={wrist_serial}")

    @staticmethod
    def _find_serial(devices: dict, model_substr: str) -> str:
        for name, serial in devices.items():
            if model_substr in name:
                return serial
        raise RuntimeError(f"Could not find RealSense camera containing '{model_substr}'.")

    def get_global_image(self) -> np.ndarray:
        return self.global_camera.get_rgb()

    def get_wrist_image(self) -> np.ndarray:
        return self.wrist_camera.get_rgb()

    def start_recording(self, external_path: Path, wrist_path: Path) -> None:
        self.global_camera.start_recording(external_path)
        try:
            self.wrist_camera.start_recording(wrist_path)
        except BaseException:
            self.global_camera.stop_recording()
            raise

    def stop_recording(self) -> tuple[int, int]:
        return self.global_camera.stop_recording(), self.wrist_camera.stop_recording()

    def close(self) -> None:
        self.global_camera.close()
        self.wrist_camera.close()


class PiperRobotClient:
    def __init__(self, args: Args):
        root = home_utils._find_bci_piper_root(args.bci_piper_root)
        robot_side = root / "robot_arm_side"
        print(f"[robot] using bci_piper at {root}")
        if str(robot_side) not in sys.path:
            sys.path.insert(0, str(robot_side))

        module = importlib.import_module("eeg_robot_arm_control")
        cfg = module.ControlConfig(
            robot_model=args.robot_model,
            speed_percent=args.speed_percent,
            joint_step_rad=args.max_joint_delta_rad,
        )
        self._robot = module.RobotArm(real=args.real, cfg=cfg)
        self._robot.connect()
        self._gripper_fraction = float(args.gripper_open_fraction)
        self._gripper_open_fraction = float(args.gripper_open_fraction)
        self._gripper_closed_fraction = float(args.gripper_closed_fraction)
        self._gripper_binarize = bool(args.gripper_binarize)
        self._gripper_open_threshold = float(args.gripper_open_threshold)
        self._gripper_close_threshold = float(args.gripper_close_threshold)
        # Latched binary gripper state for hysteresis (True == open). Seed from the start pose.
        self._gripper_open = float(args.gripper_open_fraction) >= 0.5
        self._max_joint_delta_rad = float(args.max_joint_delta_rad)
        self._action_alpha = float(args.action_alpha)
        self._action_mode = str(args.action_mode).lower()
        self._disable_gripper = bool(args.disable_gripper)
        self._dry_run = bool(args.dry_run)
        self._disable_on_exit = bool(args.disable_on_exit)
        self._table_safety_min_tcp_z_m = args.table_safety_min_tcp_z_m
        self._table_safety_path_samples = int(args.table_safety_path_samples)
        self._workspace_box = None
        self._workspace_safety_tolerance_m = float(args.workspace_safety_tolerance_m)
        if self._action_mode not in ("absolute", "delta"):
            raise ValueError(f"Unsupported action_mode: {args.action_mode}")
        if self._table_safety_path_samples < 2:
            raise ValueError("table_safety_path_samples must be at least 2")
        if (
            not np.isfinite(self._workspace_safety_tolerance_m)
            or self._workspace_safety_tolerance_m < 0.0
        ):
            raise ValueError("workspace_safety_tolerance_m must be finite and non-negative")
        if self._dry_run and not self._robot.real:
            raise ValueError(
                "--dry_run requires --real: it reads real joint feedback but skips motion. "
                "Without --real the state is faked to zeros and tells you nothing about safety."
            )
        if self._dry_run:
            print("[dry-run] real feedback ON, motion OFF: move_j/gripper commands will NOT be sent.")
        if self._table_safety_min_tcp_z_m is not None:
            if not np.isfinite(self._table_safety_min_tcp_z_m):
                raise ValueError("table_safety_min_tcp_z_m must be finite")
            if self._robot.real and not hasattr(self._robot.robot, "fk"):
                raise RuntimeError(
                    "Table safety requires the robot driver's fk(joints) method, but it is unavailable."
                )
            print(
                "[safety] table floor enabled: "
                f"minimum TCP z={self._table_safety_min_tcp_z_m:.4f} m, "
                f"path_samples={self._table_safety_path_samples}"
            )

        if args.workspace_box is not None:
            box_path = Path(args.workspace_box).expanduser().resolve()
            with box_path.open("r", encoding="utf-8") as handle:
                raw_box = json.load(handle)
            box = {
                key: float(raw_box[key])
                for key in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
            }
            box["r_min"] = float(raw_box.get("r_min", 0.0))
            box["margin_m"] = float(raw_box.get("margin_m", 0.0))
            if args.workspace_floor_margin_m is not None:
                requested_margin = float(args.workspace_floor_margin_m)
                if not np.isfinite(requested_margin) or requested_margin < 0.0:
                    raise ValueError("workspace_floor_margin_m must be finite and non-negative")
                taught_touch_z = box["z_min"] - box["margin_m"]
                box["z_min"] = taught_touch_z + requested_margin
                box["margin_m"] = requested_margin
            if not all(np.isfinite(value) for value in box.values()):
                raise ValueError(f"Workspace box contains non-finite values: {box_path}")
            for axis in ("x", "y", "z"):
                if box[f"{axis}_min"] >= box[f"{axis}_max"]:
                    raise ValueError(f"Invalid workspace {axis} bounds in {box_path}")
            if box["r_min"] < 0.0:
                raise ValueError(f"Invalid negative r_min in {box_path}")
            if self._robot.real and not hasattr(self._robot.robot, "fk"):
                raise RuntimeError(
                    "Workspace safety requires the robot driver's fk(joints) method, but it is unavailable."
                )
            self._workspace_box = box
            print(
                f"[safety] workspace box enabled: {box_path} "
                f"x=[{box['x_min']:.3f},{box['x_max']:.3f}] "
                f"y=[{box['y_min']:.3f},{box['y_max']:.3f}] "
                f"z=[{box['z_min']:.3f},{box['z_max']:.3f}] "
                f"r_min={box['r_min']:.3f} m"
            )

    def _apply_workspace_safety(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Clamp a joint target to the safe prefix of its predicted Cartesian path.

        The pyAgxArm ``fk()`` result is a flange pose, while the taught workspace box
        was recorded from ``get_tcp_pose``. Align the FK path to the live TCP position
        on every command before comparing it with the taught box. Over one bounded
        joint step this local translation is substantially safer than treating flange
        xyz as TCP xyz.
        """
        min_z = self._table_safety_min_tcp_z_m
        box = self._workspace_box
        if (min_z is None and box is None) or not self._robot.real:
            return target

        lowest_z = float("inf")
        violation: Optional[str] = None
        violation_xyz: Optional[np.ndarray] = None
        violation_fraction = 0.0
        last_safe_fraction = 0.0

        try:
            current_fk = np.asarray(self._robot.robot.fk(current.tolist()), dtype=np.float64).reshape(-1)
        except Exception as exc:
            raise RuntimeError(
                "WORKSPACE SAFETY STOP: current-pose forward kinematics failed; refusing motion."
            ) from exc
        measured_tcp = np.asarray(
            self._robot._read_pose_once("get_tcp_pose"), dtype=np.float64
        ).reshape(-1)
        if (
            current_fk.size < 3
            or measured_tcp.size < 3
            or not np.all(np.isfinite(current_fk[:3]))
            or not np.all(np.isfinite(measured_tcp[:3]))
        ):
            raise RuntimeError(
                "WORKSPACE SAFETY STOP: cannot align FK flange pose with live get_tcp_pose; "
                "refusing motion."
            )
        flange_to_tcp = getattr(self._robot.robot, "get_flange2tcp_pose", None)
        if callable(flange_to_tcp):
            current_tcp_model = np.asarray(flange_to_tcp(current_fk.tolist()), dtype=np.float64).reshape(-1)
            if current_tcp_model.size < 3 or not np.all(np.isfinite(current_tcp_model[:3])):
                raise RuntimeError(
                    "WORKSPACE SAFETY STOP: flange-to-TCP conversion returned an invalid pose."
                )
        else:
            current_tcp_model = current_fk
        # Correct any small MDH/calibration residual so fraction=0 exactly matches
        # the live pose used when the workspace was taught.
        tcp_model_residual_xyz = measured_tcp[:3] - current_tcp_model[:3]

        def violation_at(xyz: np.ndarray) -> tuple[float, Optional[str]]:
            """Return the largest Cartesian fence violation in meters."""
            x, y, z = (float(v) for v in xyz[:3])
            violations: list[tuple[float, str]] = []
            if min_z is not None and z < min_z:
                violations.append((min_z - z, f"z below table limit {min_z:.4f} m"))
            if box is not None:
                if x < box["x_min"]:
                    violations.append((box["x_min"] - x, f"x below {box['x_min']:.4f} m"))
                elif x > box["x_max"]:
                    violations.append((x - box["x_max"], f"x above {box['x_max']:.4f} m"))
                if y < box["y_min"]:
                    violations.append((box["y_min"] - y, f"y below {box['y_min']:.4f} m"))
                elif y > box["y_max"]:
                    violations.append((y - box["y_max"], f"y above {box['y_max']:.4f} m"))
                if z < box["z_min"]:
                    violations.append((box["z_min"] - z, f"z below {box['z_min']:.4f} m"))
                elif z > box["z_max"]:
                    violations.append((z - box["z_max"], f"z above {box['z_max']:.4f} m"))
                radius = float(np.hypot(x, y))
                if radius < box["r_min"]:
                    violations.append(
                        (box["r_min"] - radius, f"inside base keep-out radius {box['r_min']:.4f} m")
                    )
            return max(violations, default=(0.0, None), key=lambda item: item[0])

        current_violation_m, _ = violation_at(measured_tcp[:3])
        allowed_violation_m = max(
            self._workspace_safety_tolerance_m,
            current_violation_m,
        )

        def predict_tcp_xyz(joints: np.ndarray) -> np.ndarray:
            try:
                pose = np.asarray(self._robot.robot.fk(joints.tolist()), dtype=np.float64).reshape(-1)
            except Exception as exc:
                raise RuntimeError(
                    "TABLE SAFETY STOP: forward kinematics failed; refusing to send motion."
                ) from exc
            if pose.size < 3 or not np.all(np.isfinite(pose[:3])):
                raise RuntimeError(
                    f"TABLE SAFETY STOP: invalid FK pose {pose.tolist()}; refusing to send motion."
                )
            if callable(flange_to_tcp):
                predicted_tcp = np.asarray(flange_to_tcp(pose.tolist()), dtype=np.float64).reshape(-1)
                if predicted_tcp.size < 3 or not np.all(np.isfinite(predicted_tcp[:3])):
                    raise RuntimeError(
                        "WORKSPACE SAFETY STOP: predicted flange-to-TCP conversion failed."
                    )
            else:
                predicted_tcp = pose
            return predicted_tcp[:3] + tcp_model_residual_xyz

        # Project only the downward component out of the VLA joint action. This is a
        # local numerical-Jacobian correction: among joint-space changes that lift the
        # target back to the floor, it makes the smallest change to the original target,
        # so lateral motion is retained as much as possible.
        floor_z = max(
            value for value in (
                min_z,
                box["z_min"] if box is not None else None,
            ) if value is not None
        )
        effective_floor_z = floor_z - allowed_violation_m
        projected_target = target.astype(np.float64, copy=True)
        projection_used = False
        jacobian_eps = 1e-4
        for _ in range(4):
            target_xyz = predict_tcp_xyz(projected_target)
            deficit = effective_floor_z - float(target_xyz[2])
            if deficit <= 1e-5:
                break
            dz_dq = np.empty(6, dtype=np.float64)
            for joint_index in range(6):
                plus = projected_target.copy()
                minus = projected_target.copy()
                plus[joint_index] += jacobian_eps
                minus[joint_index] -= jacobian_eps
                dz_dq[joint_index] = (
                    predict_tcp_xyz(plus)[2] - predict_tcp_xyz(minus)[2]
                ) / (2.0 * jacobian_eps)
            norm_sq = float(dz_dq @ dz_dq)
            if norm_sq < 1e-10:
                break
            projected_target += (deficit / norm_sq) * dz_dq
            projected_target = np.clip(
                projected_target,
                current - self._max_joint_delta_rad,
                current + self._max_joint_delta_rad,
            )
            projection_used = True
        if projection_used and predict_tcp_xyz(projected_target)[2] >= effective_floor_z - 1e-5:
            print(
                "[safety] removed downward action component; "
                f"target_z={predict_tcp_xyz(target)[2]:.4f} -> "
                f"projected_z={predict_tcp_xyz(projected_target)[2]:.4f} m"
            )
            target = projected_target.astype(target.dtype, copy=False)

        for fraction in np.linspace(0.0, 1.0, self._table_safety_path_samples):
            joints = current + float(fraction) * (target - current)
            predicted_tcp_xyz = predict_tcp_xyz(joints)
            z = float(predicted_tcp_xyz[2])
            if z < lowest_z:
                lowest_z = z
            violation_m, candidate_violation = violation_at(predicted_tcp_xyz)
            if violation_m > allowed_violation_m + 1e-9:
                violation = candidate_violation
                violation_xyz = predicted_tcp_xyz.copy()
                violation_fraction = float(fraction)
                break
            last_safe_fraction = float(fraction)

        if violation is not None:
            clearance_text = ""
            if box is not None:
                touch_z = box["z_min"] - box["margin_m"]
                clearance_text = (
                    f" estimated_clearance_above_taught_touch="
                    f"{float(violation_xyz[2]) - touch_z:.4f} m;"
                )
            safe_target = current + last_safe_fraction * (target - current)
            print(
                "[safety] clamped joint action: predicted TCP path violates the fence: "
                f"{violation}; xyz={np.round(violation_xyz, 4).tolist()} m "
                f"at path_fraction={violation_fraction:.2f}; "
                f"lowest_z_seen={lowest_z:.4f} m;{clearance_text} "
                f"using safe_path_fraction={last_safe_fraction:.2f}."
            )
            return safe_target

        return target

    def move_to_home(self, args: Args) -> None:
        if not self._robot.real:
            print("[home] simulated robot: startup home motion skipped")
            return
        if self._dry_run:
            print("[home] dry-run: startup home motion skipped")
            return

        target = np.asarray(args.home, dtype=np.float32)
        if target.shape != (6,):
            raise ValueError(f"home must have 6 values, got {target.shape}")
        current = self._robot._read_joints(timeout=0.5)
        if current is None:
            raise RuntimeError("Failed to read Piper joints before startup home.")
        current = np.asarray(current[:6], dtype=np.float32)
        print("[home] current:", np.round(current, 4).tolist())
        print("[home] target :", np.round(target, 4).tolist())
        print(f"[home] max |delta|={np.abs(target - current).max():.4f} rad")

        if not args.home_yes:
            reply = input("[home] move to home before policy control? type 'yes': ").strip().lower()
            if reply != "yes":
                raise RuntimeError("Startup home aborted; policy control was not started.")

        self._robot._ensure_control_mode()
        self._robot.robot.set_motion_mode("j")
        self._robot.robot.set_speed_percent(args.speed_percent)
        enable_deadline = time.monotonic() + 2.0
        while not bool(self._robot.robot.enable()):
            if time.monotonic() >= enable_deadline:
                raise RuntimeError("Piper did not report enabled; policy control was not started.")
            time.sleep(0.05)

        home_args = home_utils.Args(
            command_rate_hz=args.home_command_rate_hz,
            max_joint_vel_rad_s=args.home_max_joint_vel_rad_s,
            hold_s=args.home_hold_s,
        )
        home_utils._stream_move_to_home(self._robot, current, target, home_args)

        deadline = time.monotonic() + args.home_timeout_s
        period = 1.0 / args.home_command_rate_hz
        last = current
        while time.monotonic() < deadline:
            self._robot.robot.move_j(target.tolist())
            measured = self._robot._read_joints(timeout=min(0.05, period))
            if measured is not None:
                last = np.asarray(measured[:6], dtype=np.float32)
                if np.abs(target - last).max() <= args.home_tol_rad:
                    print("[home] arrived:", np.round(last, 4).tolist())
                    return
            time.sleep(period)
        raise RuntimeError(
            f"Startup home timed out; last joints={np.round(last, 4).tolist()}. "
            "Policy control was not started."
        )

    def _resolve_gripper(self, action_gripper: float) -> float:
        """Map the model's raw gripper output to the value actually commanded.

        Locked at the current fraction when --disable-gripper. Otherwise binarize with
        hysteresis (the training signal is effectively 0/1, so a raw continuous command
        chatters near transitions), or pass the clipped fraction through if binarize off.
        """
        if self._disable_gripper:
            return self._gripper_fraction
        frac = float(np.clip(action_gripper, 0.0, 1.0))
        if not self._gripper_binarize:
            return frac
        if self._gripper_open:
            if frac < self._gripper_close_threshold:
                self._gripper_open = False
        elif frac > self._gripper_open_threshold:
            self._gripper_open = True
        return self._gripper_open_fraction if self._gripper_open else self._gripper_closed_fraction

    def get_state(self) -> np.ndarray:
        if not self._robot.real:
            return np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self._gripper_fraction], dtype=np.float32)

        joints = self._robot._read_joints(timeout=0.2)
        if joints is None:
            raise RuntimeError("Failed to read Piper joint angles.")
        state = np.asarray(list(joints[:6]) + [self._gripper_fraction], dtype=np.float32)
        if state.shape != (7,):
            raise RuntimeError(f"Expected Piper state shape (7,), got {state.shape}")
        return state

    def send_action(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,):
            raise ValueError(f"Expected action shape (7,), got {action.shape}")

        if not self._robot.real:
            gripper = self._resolve_gripper(action[6])
            sent = np.asarray(list(action[:6]) + [gripper], dtype=np.float32)
            self._gripper_fraction = gripper
            return None, action.copy(), sent

        current = self._robot._read_joints(timeout=0.2)
        if current is None:
            raise RuntimeError("Failed to read Piper joint angles before action send.")
        current = np.asarray(current[:6], dtype=np.float32)

        raw_action = np.asarray(action[:6], dtype=np.float32)
        if self._action_mode == "delta":
            delta = np.clip(raw_action, -1.0, 1.0) * self._max_joint_delta_rad
            clipped_target = current + delta
        else:
            interpolated = current + self._action_alpha * (raw_action - current)
            clipped_target = np.clip(
                interpolated,
                current - self._max_joint_delta_rad,
                current + self._max_joint_delta_rad,
            )
        gripper = self._resolve_gripper(action[6])

        if not self._dry_run:
            clipped_target = self._apply_workspace_safety(current, clipped_target)
            self._robot._ensure_control_mode()
            self._robot.robot.set_motion_mode("j")
            self._robot.robot.move_j(clipped_target.tolist())
            if not self._disable_gripper:
                self._robot._set_gripper_fraction(gripper)

        self._gripper_fraction = gripper
        sent = np.asarray(list(clipped_target) + [gripper], dtype=np.float32)
        current_state = np.asarray(list(current) + [self._gripper_fraction], dtype=np.float32)
        return current_state, action.copy(), sent

    def close(self) -> None:
        if self._robot.robot is None:
            return
        if self._disable_on_exit:
            print("[robot] disabling motors (arm will go limp)")
            self._robot.robot.disable()
        else:
            print("[robot] leaving motors enabled; support arm before power-off")
        self._robot.robot.disconnect()


def preprocess_image(image: np.ndarray, image_size: int) -> np.ndarray:
    image = image_tools.resize_with_pad(image, image_size, image_size)
    return image_tools.convert_to_uint8(image)


def draw_preview(global_raw: np.ndarray, wrist_raw: np.ndarray, global_model: np.ndarray, wrist_model: np.ndarray) -> None:
    import cv2

    def _bgr(image: np.ndarray) -> np.ndarray:
        return image[..., ::-1].copy()

    def _label(image: np.ndarray, text: str) -> np.ndarray:
        out = image.copy()
        cv2.putText(out, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def _pad_to_width(image: np.ndarray, width: int) -> np.ndarray:
        padding = width - image.shape[1]
        if padding <= 0:
            return image
        left = padding // 2
        right = padding - left
        return cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    raw_h = min(global_raw.shape[0], wrist_raw.shape[0])
    raw_w_global = int(global_raw.shape[1] * raw_h / global_raw.shape[0])
    raw_w_wrist = int(wrist_raw.shape[1] * raw_h / wrist_raw.shape[0])
    raw_global = cv2.resize(_bgr(global_raw), (raw_w_global, raw_h))
    raw_wrist = cv2.resize(_bgr(wrist_raw), (raw_w_wrist, raw_h))
    raw_panel = np.hstack((_label(raw_global, "global raw"), _label(raw_wrist, "wrist raw")))

    model_global = _bgr(global_model)
    model_wrist = _bgr(wrist_model)
    model_panel = np.hstack((_label(model_global, "global model"), _label(model_wrist, "wrist model")))

    preview_width = max(raw_panel.shape[1], model_panel.shape[1])
    raw_panel = _pad_to_width(raw_panel, preview_width)
    model_panel = _pad_to_width(model_panel, preview_width)
    preview = np.vstack((raw_panel, model_panel))
    cv2.imshow("openpi piper preview", preview)
    cv2.waitKey(1)


def build_observation(cameras: CameraRig, robot: PiperRobotClient, prompt: str, image_size: int, *, show_preview: bool):
    global_raw = cameras.get_global_image()
    wrist_raw = cameras.get_wrist_image()
    global_model = preprocess_image(global_raw, image_size)
    wrist_model = preprocess_image(wrist_raw, image_size)

    if show_preview:
        draw_preview(global_raw, wrist_raw, global_model, wrist_model)

    obs = {
        "observation/image": global_model,
        "observation/wrist_image": wrist_model,
        "observation/state": robot.get_state(),
        "prompt": prompt,
    }
    return obs, global_raw, wrist_raw


def print_debug(
    step: int,
    obs_state: np.ndarray,
    exec_state: Optional[np.ndarray],
    model_action: np.ndarray,
    sent_action: np.ndarray,
) -> None:
    prefix = f"step={step}"
    print(prefix, "obs_state=", np.round(obs_state, 4).tolist())
    if exec_state is not None:
        print(prefix, "exec_state=", np.round(exec_state, 4).tolist())
    print(prefix, "model_action=", np.round(model_action, 4).tolist())
    print(prefix, "sent_action=", np.round(sent_action, 4).tolist())


def main(args: Args) -> None:
    client = websocket_client_policy.WebsocketClientPolicy(args.server_host, args.server_port)
    server_metadata = client.get_server_metadata()
    print("Connected to policy server:", server_metadata)

    cameras = CameraRig(args)
    robot = PiperRobotClient(args)

    action_chunk = None
    action_index = 0
    dt = 1.0 / args.control_hz
    frozen_obs = None
    experiment_logger = None
    trial = None
    trial_started_at = None
    completed_steps = 0
    trial_paths = None
    recording_active = False
    metadata_path = None

    experiment_ids = (args.participant_id, args.session_id, args.method_id, args.task_id)
    if args.log_experiment and any(experiment_ids) and not all(experiment_ids):
        raise ValueError(
            "Experiment logging requires --participant-id, --session-id, --method-id and --task-id together"
        )
    logging_enabled = args.log_experiment and all(experiment_ids)
    if args.log_experiment and not any(experiment_ids):
        print("Experiment logging disabled: no participant/session/method/task IDs were provided")
    elif not args.log_experiment:
        print("Experiment logging disabled by --no-log-experiment")

    def finalize_videos() -> None:
        nonlocal recording_active
        if not recording_active or trial_paths is None or experiment_logger is None or trial is None:
            return
        external_frames, wrist_frames = cameras.stop_recording()
        recording_active = False
        videos = (
            ("video_external", trial_paths.external_video, external_frames),
            ("video_wrist", trial_paths.wrist_video, wrist_frames),
        )
        for file_type, path, frame_count in videos:
            if frame_count <= 0 or not path.is_file() or path.stat().st_size <= 0:
                print(f"WARNING: {file_type} produced no usable video: {path}")
                continue
            experiment_logger.attach_file(trial_id=trial.trial_id, file_type=file_type, path=path)
            print(f"Saved {file_type}: {path} ({frame_count} frames)")

    def write_trial_metadata(*, termination_reason: Optional[str] = None) -> None:
        if metadata_path is None or trial is None:
            return
        payload = {
            "schema_version": 1,
            "trial_id": trial.trial_id,
            "started_at": trial.started_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "completed_steps": completed_steps,
            "termination_reason": termination_reason,
            "args": dataclasses.asdict(args),
            "policy_server_metadata": server_metadata,
        }
        temporary = metadata_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, metadata_path)

    try:
        if args.home_on_start:
            robot.move_to_home(args)

        if logging_enabled:
            from experiment_logger import ExperimentLogger

            experiment_logger = ExperimentLogger()
            trial = experiment_logger.start_trial(
                participant_id=args.participant_id,
                session_id=args.session_id,
                method_id=args.method_id,
                task_id=args.task_id,
                calibration_id=args.calibration_id,
            )
            trial_started_at = time.monotonic()
            print(f"Started experiment trial: {trial.trial_id}")
            trial_paths = experiment_logger.get_trial_paths(trial.trial_id)
            metadata_path = trial_paths.metadata
            write_trial_metadata()
            experiment_logger.attach_file(trial_id=trial.trial_id, file_type="metadata", path=metadata_path)
            if args.record_videos:
                cameras.start_recording(trial_paths.external_video, trial_paths.wrist_video)
                recording_active = True
                print(f"Recording external video: {trial_paths.external_video}")
                print(f"Recording wrist video: {trial_paths.wrist_video}")

        for step in range(args.max_steps):
            start = time.perf_counter()

            if action_chunk is None or action_index >= min(args.open_loop_horizon, len(action_chunk)):
                if args.freeze_observation and frozen_obs is not None:
                    obs = frozen_obs
                else:
                    obs, _, _ = build_observation(
                        cameras,
                        robot,
                        args.prompt,
                        args.image_size,
                        show_preview=args.show_preview,
                    )
                    if args.freeze_observation and frozen_obs is None:
                        frozen_obs = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in obs.items()}
                obs_state = np.asarray(obs["observation/state"], dtype=np.float32)
                action_chunk = np.asarray(client.infer(obs)["actions"], dtype=np.float32)
                action_index = 0
                print(f"step={step} fetched action chunk shape={action_chunk.shape}")

            model_action = action_chunk[action_index]
            action_index += 1
            current_state, model_action_dbg, sent_action = robot.send_action(model_action)

            if args.print_state_debug and step % max(1, args.print_every) == 0:
                print_debug(step, obs_state, current_state, model_action_dbg, sent_action)

            elapsed = time.perf_counter() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)
            completed_steps = step + 1

        if experiment_logger is not None and trial is not None and trial_started_at is not None:
            finalize_videos()
            write_trial_metadata(termination_reason="step_limit")
            experiment_logger.end_trial(
                trial_id=trial.trial_id,
                success=None,
                duration_sec=time.monotonic() - trial_started_at,
                notes=f"Reached step limit; completed_steps={completed_steps}; max_steps={args.max_steps}",
            )
            print(f"Completed experiment trial: {trial.trial_id} (result pending manual review)")
    except KeyboardInterrupt:
        if experiment_logger is not None and trial is not None and trial_started_at is not None:
            duration_sec = time.monotonic() - trial_started_at
            write_trial_metadata(termination_reason="user_interrupt")
            experiment_logger.interrupt_trial(
                trial_id=trial.trial_id,
                duration_sec=duration_sec,
                reason="user_interrupt",
            )
            print(f"Interrupted experiment trial: {trial.trial_id} after {duration_sec:.2f}s")
        raise
    finally:
        try:
            finalize_videos()
        except Exception as exc:
            print(f"WARNING: Failed to finalize trial videos: {exc}")
        if args.show_preview:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass
        cameras.close()
        robot.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
