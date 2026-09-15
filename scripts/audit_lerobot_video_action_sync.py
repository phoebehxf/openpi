#!/usr/bin/env python3
"""Audit LeRobot v2.1 video/action synchronization and render annotated episodes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


def parse_range(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        bounds = part.strip().split("-")
        if len(bounds) == 1:
            result.append(int(bounds[0]))
        elif len(bounds) == 2:
            start, stop = map(int, bounds)
            result.extend(range(min(start, stop), max(start, stop) + 1))
        else:
            raise argparse.ArgumentTypeError(f"Invalid episode range: {part}")
    return sorted(set(result))


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames, fps


def smooth(values: np.ndarray, width: int = 7) -> np.ndarray:
    if width <= 1:
        return values
    return np.convolve(values, np.ones(width) / width, mode="same")


def visual_motion(frames: list[np.ndarray]) -> np.ndarray:
    small = [cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (96, 72)) for frame in frames]
    motion = np.zeros(len(small), dtype=np.float32)
    for index in range(1, len(small)):
        motion[index] = np.mean(cv2.absdiff(small[index], small[index - 1]))
    return smooth(motion)


def normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    std = values.std()
    return (values - values.mean()) / std if std > 1e-8 else np.zeros_like(values)


def best_lag(reference: np.ndarray, candidate: np.ndarray, max_lag: int) -> tuple[int, float]:
    reference, candidate = normalized(reference), normalized(candidate)
    best = (0, -1.0)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x, y = reference[-lag:], candidate[:lag]
        elif lag > 0:
            x, y = reference[:-lag], candidate[lag:]
        else:
            x, y = reference, candidate
        if len(x) < 20 or x.std() < 1e-8 or y.std() < 1e-8:
            continue
        corr = float(np.corrcoef(x, y)[0, 1])
        if corr > best[1]:
            best = (lag, corr)
    return best


def action_lead(state: np.ndarray, action: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Return frames by which an absolute action best leads measured state, plus RMSE."""
    best_lag_value, best_mse = 0, float("inf")
    for lead in range(-max_lag, max_lag + 1):
        if lead > 0:
            observed, commanded = state[lead:], action[:-lead]
        elif lead < 0:
            observed, commanded = state[:lead], action[-lead:]
        else:
            observed, commanded = state, action
        if len(observed) < 20:
            continue
        mse = float(np.mean(np.square(observed - commanded)))
        if mse < best_mse:
            best_lag_value, best_mse = lead, mse
    return best_lag_value, best_mse**0.5


def add_label(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    output = frame.copy()
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (output.shape[1], 26 * len(lines) + 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, output, 0.38, 0, output)
    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (10, 24 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def render_episode(
    destination: Path,
    rgb: list[np.ndarray],
    wrist: list[np.ndarray],
    state: np.ndarray,
    action: np.ndarray,
    fps: float,
) -> None:
    count = min(len(rgb), len(wrist), len(state), len(action))
    height = 360
    width = int(rgb[0].shape[1] * height / rgb[0].shape[0])
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {destination}")
    for index in range(count):
        left = cv2.resize(rgb[index], (width, height))
        right = cv2.resize(wrist[index], (width, height))
        state_text = " ".join(f"{value:+.2f}" for value in state[index, :6])
        action_text = " ".join(f"{value:+.2f}" for value in action[index, :6])
        lines = [
            f"frame={index:04d}  t={index / fps:6.2f}s",
            f"state q:  {state_text}",
            f"action q: {action_text}",
            f"gripper state/action: {state[index, 6]:.3f} / {action[index, 6]:.3f}",
        ]
        writer.write(np.hstack([add_label(left, lines), add_label(right, lines)]))
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episodes", type=parse_range, required=True, help="Example: 27-42 or 1,5,8-10")
    parser.add_argument("--output-dir", type=Path, default=Path("debug/dataset_sync_audit"))
    parser.add_argument("--max-lag", type=int, default=60)
    parser.add_argument("--render", action="store_true", help="Render annotated side-by-side videos.")
    args = parser.parse_args()

    info = json.loads((args.dataset / "meta/info.json").read_text())
    manifest_path = args.dataset / "meta/merge_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    if info.get("codebase_version") != "v2.1":
        raise ValueError("This auditor currently expects a LeRobot v2.1 dataset")
    video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    if len(video_keys) < 2:
        raise ValueError(f"Expected two video streams, found {video_keys}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for episode in args.episodes:
        fmt = {"episode_index": episode, "episode_chunk": episode // info["chunks_size"]}
        table = pq.read_table(args.dataset / info["data_path"].format(**fmt))
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        videos = []
        fps_values = []
        for key in video_keys[:2]:
            path = args.dataset / info["video_path"].format(**fmt, video_key=key)
            frames, fps = read_video(path)
            videos.append(frames)
            fps_values.append(fps)
        count = min(len(state), *(len(frames) for frames in videos))
        joint_motion = np.zeros(count, dtype=np.float32)
        joint_motion[1:] = np.linalg.norm(np.diff(state[:count, :6], axis=0), axis=1)
        motion_a = visual_motion(videos[0][:count])
        motion_b = visual_motion(videos[1][:count])
        lag_a, corr_a = best_lag(joint_motion, motion_a, args.max_lag)
        lag_b, corr_b = best_lag(joint_motion, motion_b, args.max_lag)
        action_lead_frames, action_state_rmse = action_lead(
            state[:count, :6], action[:count, :6], args.max_lag
        )
        record = {
            "episode": episode,
            "source": manifest[episode]["source"] if manifest is not None else "",
            "source_episode": manifest[episode]["source_episode_index"] if manifest is not None else episode,
            "data_frames": len(state),
            "video0_frames": len(videos[0]),
            "video1_frames": len(videos[1]),
            "joint_motion_mean": float(joint_motion.mean()),
            "video0_motion_mean": float(motion_a.mean()),
            "video1_motion_mean": float(motion_b.mean()),
            "video0_best_lag_frames": lag_a,
            "video0_best_corr": corr_a,
            "video1_best_lag_frames": lag_b,
            "video1_best_corr": corr_b,
            "action_lead_frames": action_lead_frames,
            "action_state_rmse": action_state_rmse,
        }
        records.append(record)
        print(record, flush=True)
        if args.render:
            render_episode(
                args.output_dir / f"episode_{episode:06d}_annotated.mp4",
                videos[0],
                videos[1],
                state,
                action,
                fps_values[0] or info["fps"],
            )

    with (args.output_dir / "sync_report.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {args.output_dir / 'sync_report.csv'}")


if __name__ == "__main__":
    main()
