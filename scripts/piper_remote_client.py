import dataclasses
import importlib
import pathlib
import sys
import time

import numpy as np
import tyro

from openpi_client import image_tools
from openpi_client import websocket_client_policy


@dataclasses.dataclass
class Args:
    server_host: str
    server_port: int = 8000
    prompt: str = "pick up the object and place it in the basket"
    bci_piper_root: str = "/home/huix/bci_robot/bci_piper"
    robot_model: str = "piper"
    real: bool = False
    speed_percent: int = 10
    global_camera_model: str = "D435"
    wrist_camera_model: str = "D405"
    global_camera_serial: str | None = None
    wrist_camera_serial: str | None = None
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 15
    image_size: int = 224
    show_preview: bool = True
    control_hz: float = 5.0
    open_loop_horizon: int = 2
    max_steps: int = 300
    max_joint_delta_rad: float = 0.08
    gripper_open_fraction: float = 1.0
    gripper_closed_fraction: float = 0.0


class RealSenseCamera:
    def __init__(self, serial: str, *, width: int, height: int, fps: int):
        import pyrealsense2 as rs

        self._rs = rs
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._pipeline.start(config)
        self.serial = serial

    def get_rgb(self) -> np.ndarray:
        while True:
            frames = self._pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if color:
                image_bgr = np.asanyarray(color.get_data())
                return image_bgr[..., ::-1].copy()

    def close(self) -> None:
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
    def _find_serial(devices: dict[str, str], model_substr: str) -> str:
        for name, serial in devices.items():
            if model_substr in name:
                return serial
        raise RuntimeError(f"Could not find RealSense camera containing '{model_substr}'.")

    def get_global_image(self) -> np.ndarray:
        return self.global_camera.get_rgb()

    def get_wrist_image(self) -> np.ndarray:
        return self.wrist_camera.get_rgb()

    def close(self) -> None:
        self.global_camera.close()
        self.wrist_camera.close()


class PiperRobotClient:
    def __init__(self, args: Args):
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
            joint_step_rad=args.max_joint_delta_rad,
        )
        self._robot = module.RobotArm(real=args.real, cfg=cfg)
        self._robot.connect()
        self._gripper_fraction = float(args.gripper_open_fraction)
        self._max_joint_delta_rad = float(args.max_joint_delta_rad)

    def get_state(self) -> np.ndarray:
        if not self._robot.real:
            return np.asarray([0, 0, 0, 0, 0, 0, self._gripper_fraction], dtype=np.float32)

        joints = self._robot._read_joints(timeout=0.2)
        if joints is None:
            raise RuntimeError("Failed to read Piper joint angles.")
        state = np.asarray(list(joints[:6]) + [self._gripper_fraction], dtype=np.float32)
        if state.shape != (7,):
            raise RuntimeError(f"Expected Piper state shape (7,), got {state.shape}")
        return state

    def send_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,):
            raise ValueError(f"Expected action shape (7,), got {action.shape}")

        if not self._robot.real:
            gripper = float(np.clip(action[6], 0.0, 1.0))
            print(
                "dry-run action:",
                {"target": action[:6].round(4).tolist(), "gripper": round(gripper, 3)},
            )
            self._gripper_fraction = gripper
            return

        current = self._robot._read_joints(timeout=0.2)
        if current is None:
            raise RuntimeError("Failed to read Piper joint angles before action send.")
        current = np.asarray(current[:6], dtype=np.float32)

        target_joints = action[:6]
        clipped_target = np.clip(
            target_joints,
            current - self._max_joint_delta_rad,
            current + self._max_joint_delta_rad,
        )
        gripper = float(np.clip(action[6], 0.0, 1.0))

        self._robot._ensure_control_mode()
        self._robot.robot.set_motion_mode("j")
        self._robot.robot.move_j(clipped_target.tolist())
        self._robot._set_gripper_fraction(gripper)
        self._gripper_fraction = gripper

    def close(self) -> None:
        self._robot.close()


def preprocess_image(image: np.ndarray, image_size: int) -> np.ndarray:
    image = image_tools.resize_with_pad(image, image_size, image_size)
    return image_tools.convert_to_uint8(image)


def draw_preview(
    global_raw: np.ndarray,
    wrist_raw: np.ndarray,
    global_model: np.ndarray,
    wrist_model: np.ndarray,
) -> None:
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


def build_observation(
    cameras: CameraRig, robot: PiperRobotClient, prompt: str, image_size: int, *, show_preview: bool
) -> dict:
    global_raw = cameras.get_global_image()
    wrist_raw = cameras.get_wrist_image()
    global_model = preprocess_image(global_raw, image_size)
    wrist_model = preprocess_image(wrist_raw, image_size)

    if show_preview:
        draw_preview(global_raw, wrist_raw, global_model, wrist_model)

    return {
        "observation/image": global_model,
        "observation/wrist_image": wrist_model,
        "observation/state": robot.get_state(),
        "prompt": prompt,
    }


def main(args: Args) -> None:
    client = websocket_client_policy.WebsocketClientPolicy(args.server_host, args.server_port)
    print("Connected to policy server:", client.get_server_metadata())

    cameras = CameraRig(args)
    robot = PiperRobotClient(args)

    action_chunk = None
    action_index = 0
    dt = 1.0 / args.control_hz

    try:
        for step in range(args.max_steps):
            start = time.perf_counter()

            if action_chunk is None or action_index >= min(args.open_loop_horizon, len(action_chunk)):
                obs = build_observation(
                    cameras,
                    robot,
                    args.prompt,
                    args.image_size,
                    show_preview=args.show_preview,
                )
                action_chunk = np.asarray(client.infer(obs)["actions"], dtype=np.float32)
                action_index = 0
                print(f"step={step} fetched action chunk shape={action_chunk.shape}")

            action = action_chunk[action_index]
            action_index += 1
            robot.send_action(action)

            elapsed = time.perf_counter() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    finally:
        if args.show_preview:
            import cv2

            cv2.destroyAllWindows()
        cameras.close()
        robot.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
