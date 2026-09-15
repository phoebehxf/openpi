#!/usr/bin/env python3
"""Split each Piper pouring episode into three prompt-conditioned phases.

The source LeRobot v2.1 dataset is never modified.  Boundaries are inferred
from the final gripper close/open pair and the joint6 tilt trajectory.  The
output contains three new episodes per source episode, with parquet/video
indices and dataset statistics rebuilt from scratch.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from clean_lerobot_episodes import _array_stats
from clean_lerobot_episodes import _run_ffmpeg_trim
from clean_lerobot_episodes import _video_frame_count
from clean_lerobot_episodes import _video_stats
from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.utils import serialize_dict
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_SOURCE = "local/piper-pour-water-cleaned-merged-v1"
DEFAULT_OUTPUT = "local/piper-pour-water-three-phase-v1"
PHASES = (
    ("grasp_and_move", "Grasp the bottle and move it next to the red cup"),
    ("pour", "Tilt the bottle to pour water into the red cup"),
    ("return_and_place", "Return the bottle upright, place it down, and release it"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_SOURCE)
    parser.add_argument("--output-repo-id", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--lerobot-home",
        type=Path,
        default=Path(os.environ.get("HF_LEROBOT_HOME", Path(__file__).resolve().parents[1] / "local_datasets")),
    )
    parser.add_argument(
        "--tilt-onset-fraction",
        type=float,
        default=0.20,
        help="Fraction of episode joint6 tilt amplitude that starts the pour phase.",
    )
    parser.add_argument(
        "--peak-fraction",
        type=float,
        default=0.90,
        help="Return phase starts after the final sample above this fraction of peak tilt.",
    )
    parser.add_argument("--sustain-frames", type=int, default=5)
    parser.add_argument(
        "--overlap-frames",
        type=int,
        default=10,
        help="Context shared on both sides of each phase boundary (10 equals the action horizon).",
    )
    parser.add_argument("--min-phase-frames", type=int, default=30)
    parser.add_argument(
        "--visualize-dir",
        type=Path,
        help="Write boundary contact sheets here (works with --dry-run).",
    )
    parser.add_argument(
        "--visualize-episodes",
        default="0,10,20,30,40,49",
        help="Comma-separated source episodes to visualize, or 'all'.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values))


def _replace(table: pa.Table, key: str, values: np.ndarray) -> pa.Table:
    index = table.schema.get_field_index(key)
    if index < 0:
        raise KeyError(f"Missing parquet column: {key}")
    return table.set_column(index, key, pa.array(values, type=table.schema.field(index).type))


def _first_sustained(mask: np.ndarray, count: int) -> int | None:
    if len(mask) < count:
        return None
    hits = np.convolve(mask.astype(np.int16), np.ones(count, dtype=np.int16), mode="valid")
    found = np.flatnonzero(hits == count)
    return int(found[0]) if len(found) else None


def infer_boundaries(
    actions: np.ndarray,
    *,
    tilt_onset_fraction: float,
    peak_fraction: float,
    sustain_frames: int,
    overlap_frames: int,
    min_phase_frames: int,
) -> dict[str, Any]:
    """Infer three half-open ranges and diagnostic landmark frames."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise ValueError(f"Expected [frames, >=7] actions, got {actions.shape}")
    if not 0 < tilt_onset_fraction < peak_fraction <= 1:
        raise ValueError("Require 0 < tilt-onset-fraction < peak-fraction <= 1")

    closed = actions[:, 6] < 0.5
    closes = np.flatnonzero(closed[1:] & ~closed[:-1]) + 1
    opens = np.flatnonzero(~closed[1:] & closed[:-1]) + 1
    if not len(closes) or not len(opens):
        raise ValueError("No gripper close/open transitions")
    release = int(opens[-1])
    valid_closes = closes[closes < release]
    if not len(valid_closes):
        raise ValueError("No close transition before final release")
    grasp = int(valid_closes[-1])

    joint6 = actions[:, 5]
    reference_start = max(0, grasp - 30)
    upright = float(np.median(joint6[reference_start : grasp + 1]))
    tilt = np.abs(joint6 - upright)
    # Smooth command noise without shifting the landmarks materially.
    kernel = np.ones(sustain_frames, dtype=np.float32) / sustain_frames
    smooth = np.convolve(tilt, kernel, mode="same")
    interval = smooth[grasp:release]
    amplitude = float(interval.max())
    if amplitude < 0.35:
        raise ValueError(f"joint6 tilt amplitude is too small: {amplitude:.3f} rad")

    onset_offset = _first_sustained(interval >= tilt_onset_fraction * amplitude, sustain_frames)
    if onset_offset is None:
        raise ValueError("No sustained tilt onset")
    tilt_onset = grasp + onset_offset
    high = np.flatnonzero(interval >= peak_fraction * amplitude)
    if not len(high):
        raise ValueError("No peak-tilt samples")
    # The final high-tilt sample includes any deliberate pouring hold; the next
    # frame is where the return-to-upright phase begins.
    return_start = grasp + int(high[-1]) + 1

    n = len(actions)
    ranges = [
        (0, min(n, tilt_onset + overlap_frames)),
        (max(0, tilt_onset - overlap_frames), min(n, return_start + overlap_frames)),
        (max(0, return_start - overlap_frames), n),
    ]
    if any(stop - start < min_phase_frames for start, stop in ranges):
        raise ValueError(f"A phase is shorter than {min_phase_frames} frames: {ranges}")
    return {
        "grasp_frame": grasp,
        "release_frame": release,
        "upright_joint6": upright,
        "tilt_amplitude": amplitude,
        "tilt_onset_frame": tilt_onset,
        "return_start_frame": return_start,
        "ranges": ranges,
    }


def _rebuild_table(
    table: pa.Table,
    start: int,
    stop: int,
    *,
    episode_index: int,
    global_start: int,
    task_index: int,
    fps: int,
) -> pa.Table:
    result = table.slice(start, stop - start)
    n = len(result)
    result = _replace(result, "timestamp", np.arange(n, dtype=np.float32) / fps)
    result = _replace(result, "frame_index", np.arange(n, dtype=np.int64))
    result = _replace(result, "episode_index", np.full(n, episode_index, dtype=np.int64))
    result = _replace(result, "index", np.arange(global_start, global_start + n, dtype=np.int64))
    result = _replace(result, "task_index", np.full(n, task_index, dtype=np.int64))
    return result.replace_schema_metadata(None)


def _parse_visualize_episodes(value: str, available: set[int]) -> set[int]:
    if value.strip().lower() == "all":
        return available
    try:
        selected = {int(part.strip()) for part in value.split(",") if part.strip()}
    except ValueError as error:
        raise ValueError("--visualize-episodes must be comma-separated integers or 'all'") from error
    unknown = selected - available
    if unknown:
        raise ValueError(f"Unknown visualization episode indices: {sorted(unknown)}")
    return selected


def _read_video_frames(path: Path, indices: list[int]) -> list[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Cannot decode frame {index} from {path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


def _visualize_plan(
    source: Path,
    info: dict[str, Any],
    plan: dict[str, Any],
    actions: np.ndarray,
    output_dir: Path,
) -> Path:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    episode = int(plan["source_episode_index"])
    fmt = {"episode_index": episode, "episode_chunk": episode // int(info.get("chunks_size", 1000))}
    landmarks = [
        int(plan["grasp_frame"]),
        int(plan["tilt_onset_frame"]),
        int(plan["return_start_frame"]),
        int(plan["release_frame"]),
    ]
    labels = ["grasp", "tilt onset", "return start", "release"]
    video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    if len(video_keys) != 2:
        raise ValueError(f"Visualization expects two video streams, got {video_keys}")
    camera_frames = {
        key: _read_video_frames(source / info["video_path"].format(**fmt, video_key=key), landmarks)
        for key in video_keys
    }

    figure = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = figure.add_gridspec(3, 4, height_ratios=(1, 1, 0.9))
    for row, key in enumerate(video_keys):
        camera_name = key.rsplit(".", 1)[-1]
        for column, (frame, label, index) in enumerate(zip(camera_frames[key], labels, landmarks, strict=True)):
            axis = figure.add_subplot(grid[row, column])
            axis.imshow(frame)
            axis.set_title(f"{camera_name}: {label}\nframe {index}")
            axis.axis("off")

    axis = figure.add_subplot(grid[2, :])
    n = len(actions)
    tilt_onset, return_start = landmarks[1], landmarks[2]
    colors = ("#cfe8ff", "#ffe3a3", "#d9f2d9")
    spans = ((0, tilt_onset), (tilt_onset, return_start), (return_start, n))
    for (start, stop), (phase_name, _), color in zip(spans, PHASES, colors, strict=True):
        axis.axvspan(start, stop, color=color, alpha=0.65, label=phase_name)
    axis.plot(actions[:, 5], color="#315a9a", linewidth=1.8, label="joint6 action")
    axis.set_xlabel("source frame")
    axis.set_ylabel("joint6 [rad]", color="#315a9a")
    axis.tick_params(axis="y", labelcolor="#315a9a")
    gripper_axis = axis.twinx()
    gripper_axis.step(np.arange(n), actions[:, 6], where="post", color="#b83232", linewidth=1.3, label="gripper")
    gripper_axis.axhline(0.5, color="#b83232", linestyle=":", linewidth=0.8)
    gripper_axis.set_ylabel("gripper action (open=1, closed=0)", color="#b83232")
    gripper_axis.set_ylim(-0.08, 1.08)
    gripper_axis.tick_params(axis="y", labelcolor="#b83232")
    for index, label in zip(landmarks, labels, strict=True):
        axis.axvline(index, color="black", linestyle="--", linewidth=0.8)
        axis.text(index, axis.get_ylim()[1], f" {label}", rotation=90, va="top", fontsize=8)
    handles, legend_labels = axis.get_legend_handles_labels()
    axis.legend(handles, legend_labels, loc="lower center", ncol=4, fontsize=8)
    figure.suptitle(f"Pour phase split verification — source episode {episode:03d}", fontsize=15)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"episode_{episode:06d}_phase_boundaries.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def split_dataset(args: argparse.Namespace) -> Path | None:
    source = (args.lerobot_home / args.repo_id).resolve()
    output = (args.lerobot_home / args.output_repo_id).resolve()
    if source == output:
        raise ValueError("Source and output datasets must differ")
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    info = json.loads((source / "meta/info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        raise ValueError("Only LeRobot v2.1 datasets are supported")
    fps = int(info["fps"])
    chunks_size = int(info.get("chunks_size", 1000))
    video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    source_episodes = sorted(_rows(source / "meta/episodes.jsonl"), key=lambda row: row["episode_index"])
    available_indices = {int(row["episode_index"]) for row in source_episodes}
    visualize_indices = (
        _parse_visualize_episodes(args.visualize_episodes, available_indices)
        if args.visualize_dir is not None
        else set()
    )

    plans: list[dict[str, Any]] = []
    for row in source_episodes:
        source_index = int(row["episode_index"])
        fmt = {"episode_index": source_index, "episode_chunk": source_index // chunks_size}
        parquet = source / info["data_path"].format(**fmt)
        table = pq.read_table(parquet, columns=["action"])
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        boundary = infer_boundaries(
            actions,
            tilt_onset_fraction=args.tilt_onset_fraction,
            peak_fraction=args.peak_fraction,
            sustain_frames=args.sustain_frames,
            overlap_frames=args.overlap_frames,
            min_phase_frames=args.min_phase_frames,
        )
        plans.append({"source_episode_index": source_index, **boundary})
        ranges = " ".join(f"{PHASES[i][0]}=[{start},{stop})" for i, (start, stop) in enumerate(boundary["ranges"]))
        print(
            f"episode {source_index:03d}: grasp={boundary['grasp_frame']} "
            f"tilt={boundary['tilt_onset_frame']} return={boundary['return_start_frame']} "
            f"release={boundary['release_frame']} {ranges}",
            flush=True,
        )
        if source_index in visualize_indices:
            preview_path = _visualize_plan(source, info, plans[-1], actions, args.visualize_dir.resolve())
            print(f"  visualization: {preview_path}", flush=True)

    print(f"Planned {len(plans)} source episodes -> {len(plans) * len(PHASES)} phase episodes")
    if args.dry_run:
        return None

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    episodes: list[dict[str, Any]] = []
    episode_stats: list[dict[str, Any]] = []
    all_stats: list[dict[str, dict[str, np.ndarray]]] = []
    manifest: list[dict[str, Any]] = []
    global_index = 0
    try:
        for plan in plans:
            source_index = plan["source_episode_index"]
            source_fmt = {"episode_index": source_index, "episode_chunk": source_index // chunks_size}
            source_table = pq.read_table(source / info["data_path"].format(**source_fmt))
            for task_index, ((phase_name, prompt), (start, stop)) in enumerate(
                zip(PHASES, plan["ranges"], strict=True)
            ):
                new_index = len(episodes)
                new_table = _rebuild_table(
                    source_table,
                    start,
                    stop,
                    episode_index=new_index,
                    global_start=global_index,
                    task_index=task_index,
                    fps=fps,
                )
                new_fmt = {"episode_index": new_index, "episode_chunk": new_index // chunks_size}
                data_path = temporary / info["data_path"].format(**new_fmt)
                data_path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(new_table, data_path)

                video_paths: dict[str, Path] = {}
                for key in video_keys:
                    src_video = source / info["video_path"].format(**source_fmt, video_key=key)
                    dst_video = temporary / info["video_path"].format(**new_fmt, video_key=key)
                    _run_ffmpeg_trim(src_video, dst_video, start=start, stop=stop)
                    if _video_frame_count(dst_video) != len(new_table):
                        raise RuntimeError(f"Video length mismatch: {dst_video}")
                    video_paths[key] = dst_video

                stats = {name: _array_stats(np.asarray(new_table[name].to_pylist())) for name in new_table.column_names}
                for key, path in video_paths.items():
                    stats[key] = _video_stats(path)
                episodes.append({"episode_index": new_index, "tasks": [prompt], "length": len(new_table)})
                episode_stats.append({"episode_index": new_index, "stats": serialize_dict(stats)})
                all_stats.append(stats)
                manifest.append(
                    {
                        "episode_index": new_index,
                        "phase": phase_name,
                        "task": prompt,
                        "source_episode_index": source_index,
                        "source_start_frame": start,
                        "source_stop_frame": stop,
                        **{key: value for key, value in plan.items() if key not in {"ranges", "source_episode_index"}},
                    }
                )
                global_index += len(new_table)
                print(
                    f"Wrote episode {new_index:03d}: source={source_index:03d} "
                    f"phase={phase_name} frames={len(new_table)}",
                    flush=True,
                )

        output_info = copy.deepcopy(info)
        output_info.update(
            total_episodes=len(episodes),
            total_frames=global_index,
            total_tasks=len(PHASES),
            total_chunks=(len(episodes) + chunks_size - 1) // chunks_size,
            total_videos=len(episodes) * len(video_keys),
            splits={"train": f"0:{len(episodes)}"},
        )
        _write_json(temporary / "meta/info.json", output_info)
        _write_json(temporary / "meta/stats.json", serialize_dict(aggregate_stats(all_stats)))
        _write_jsonl(temporary / "meta/episodes.jsonl", episodes)
        _write_jsonl(temporary / "meta/episodes_stats.jsonl", episode_stats)
        _write_jsonl(
            temporary / "meta/tasks.jsonl",
            [{"task_index": index, "task": prompt} for index, (_, prompt) in enumerate(PHASES)],
        )
        _write_json(temporary / "meta/phase_split_manifest.json", manifest)
        temporary.rename(output)
    except Exception:
        print(f"Build failed; staging directory retained for diagnosis: {temporary}", flush=True)
        raise

    print(f"Complete: {output}; episodes={len(episodes)}, frames={global_index}")
    return output


if __name__ == "__main__":
    split_dataset(parse_args())
