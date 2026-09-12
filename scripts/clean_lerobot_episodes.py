#!/usr/bin/env python3
"""Trim ambiguous idle prefixes/suffixes from a LeRobot v2.1 dataset.

The source dataset is never modified. Parquet rows and every video stream are
cut to identical frame boundaries, episodes and global indices are rebuilt,
and v2.1 per-episode/global statistics are recomputed.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import cv2
from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import serialize_dict
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

VIDEO_FEATURE_TYPES = {"video"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Source repo id under HF_LEROBOT_HOME.")
    parser.add_argument("--output-repo-id", help="Default: SOURCE-cleaned-v1.")
    parser.add_argument(
        "--lerobot-home",
        type=Path,
        default=Path(os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser(),
    )
    parser.add_argument("--motion-threshold-rad", type=float, default=0.02)
    parser.add_argument("--gripper-threshold", type=float, default=0.10)
    parser.add_argument("--sustain-frames", type=int, default=3)
    parser.add_argument("--terminal-reference-frames", type=int, default=10)
    parser.add_argument("--terminal-postroll-frames", type=int, default=10)
    parser.add_argument("--min-episode-frames", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _first_sustained(mask: np.ndarray, length: int) -> int | None:
    if length <= 0:
        raise ValueError("sustain length must be positive")
    if mask.size < length:
        return None
    hits = np.convolve(mask.astype(np.int16), np.ones(length, dtype=np.int16), mode="valid")
    indices = np.flatnonzero(hits == length)
    return int(indices[0]) if indices.size else None


def choose_trim_bounds(
    actions: np.ndarray,
    *,
    motion_threshold_rad: float,
    gripper_threshold: float,
    sustain_frames: int,
    terminal_reference_frames: int,
    terminal_postroll_frames: int,
    min_episode_frames: int,
) -> tuple[int, int]:
    """Return half-open frame bounds that remove only home/terminal plateaus."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise ValueError(f"Expected actions shaped [frames, >=7], got {actions.shape}")
    if len(actions) < min_episode_frames:
        raise ValueError(f"Episode has only {len(actions)} frames (< {min_episode_frames}).")

    reference_count = min(10, len(actions))
    initial = np.median(actions[:reference_count], axis=0)
    departed = np.max(np.abs(actions[:, :6] - initial[:6]), axis=1) > motion_threshold_rad
    departed |= np.abs(actions[:, 6] - initial[6]) > gripper_threshold
    start = _first_sustained(departed, sustain_frames)
    if start is None:
        raise ValueError("No sustained departure from the initial pose was found.")

    terminal_count = min(terminal_reference_frames, len(actions) - start)
    terminal = np.median(actions[-terminal_count:], axis=0)
    outside_terminal = np.max(np.abs(actions[:, :6] - terminal[:6]), axis=1) > motion_threshold_rad
    outside_terminal |= np.abs(actions[:, 6] - terminal[6]) > gripper_threshold
    outside_after_start = np.flatnonzero(outside_terminal[start:])
    if outside_after_start.size:
        last_outside = start + int(outside_after_start[-1])
        stop = min(len(actions), last_outside + 1 + terminal_postroll_frames)
    else:
        stop = len(actions)

    if stop - start < min_episode_frames:
        raise ValueError(f"Trimmed episode would have only {stop - start} frames.")
    return start, stop


def _replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise KeyError(f"Required parquet column is missing: {name}")
    return table.set_column(index, name, pa.array(values, type=table.schema.field(index).type))


def rebuild_episode_table(
    table: pa.Table, start: int, stop: int, *, episode_index: int, global_start: int, fps: int
) -> pa.Table:
    result = table.slice(start, stop - start)
    length = len(result)
    result = _replace_column(result, "timestamp", np.arange(length, dtype=np.float32) / fps)
    result = _replace_column(result, "frame_index", np.arange(length, dtype=np.int64))
    result = _replace_column(result, "episode_index", np.full(length, episode_index, dtype=np.int64))
    result = _replace_column(result, "index", np.arange(global_start, global_start + length, dtype=np.int64))
    return result.replace_schema_metadata(None)


def _run_ffmpeg_trim(src: Path, dst: Path, *, start: int, stop: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-vf",
            f"trim=start_frame={start}:end_frame={stop},setpts=PTS-STARTPTS",
            "-frames:v",
            str(stop - start),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(dst),
        ],
        check=True,
        timeout=300,
    )


def _video_frame_count(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return int(result.stdout.strip())


def _array_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values)
    keepdims = values.ndim == 1
    result = {
        "min": np.min(values, axis=0, keepdims=keepdims),
        "max": np.max(values, axis=0, keepdims=keepdims),
        "mean": np.mean(values, axis=0, keepdims=keepdims),
        "std": np.std(values, axis=0, keepdims=keepdims),
        "count": np.array([len(values)]),
    }
    for label, quantile in (("q01", 0.01), ("q10", 0.10), ("q50", 0.50), ("q90", 0.90), ("q99", 0.99)):
        result[label] = np.quantile(values, quantile, axis=0, keepdims=keepdims)
    return result


def _video_stats(path: Path, *, max_samples: int = 100) -> dict[str, np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot decode video for statistics: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = set(np.linspace(0, frame_count - 1, min(max_samples, frame_count)).round().astype(int))
    samples: list[np.ndarray] = []
    index = 0
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        if index in wanted:
            samples.append(bgr[..., ::-1].transpose(2, 0, 1))
        index += 1
    capture.release()
    if not samples:
        raise RuntimeError(f"No frames decoded from {path}")
    values = np.stack(samples).astype(np.float32) / 255.0
    return {
        "min": values.min(axis=(0, 2, 3), keepdims=True).squeeze(0),
        "max": values.max(axis=(0, 2, 3), keepdims=True).squeeze(0),
        "mean": values.mean(axis=(0, 2, 3), keepdims=True).squeeze(0),
        "std": values.std(axis=(0, 2, 3), keepdims=True).squeeze(0),
        "count": np.array([len(values)]),
    }


def _episode_stats(table: pa.Table, video_paths: dict[str, Path]) -> dict[str, dict[str, np.ndarray]]:
    stats: dict[str, dict[str, np.ndarray]] = {}
    for name in table.column_names:
        column = table[name]
        values = np.asarray(column.to_pylist())
        stats[name] = _array_stats(values)
    for key, path in video_paths.items():
        stats[key] = _video_stats(path)
    return stats


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=4, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _source_episode_files(source: Path) -> list[Path]:
    paths = sorted((source / "data").glob("chunk-*/episode_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No v2.1 episode parquet files found under {source / 'data'}")
    return paths


def clean_dataset(args: argparse.Namespace) -> Path | None:
    output_repo_id = args.output_repo_id or f"{args.repo_id}-cleaned-v1"
    source = (args.lerobot_home / args.repo_id).resolve()
    output = (args.lerobot_home / output_repo_id).resolve()
    if source == output:
        raise ValueError("Source and output dataset paths must differ.")
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")

    info = json.loads((source / "meta" / "info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"Expected a v2.1 source dataset, got {info.get('codebase_version')!r}")
    fps = int(info["fps"])
    chunks_size = int(info.get("chunks_size", 1000))
    video_keys = [key for key, feature in info["features"].items() if feature["dtype"] in VIDEO_FEATURE_TYPES]
    source_episodes = {
        int(row["episode_index"]): row
        for row in (json.loads(line) for line in (source / "meta" / "episodes.jsonl").read_text().splitlines())
    }

    plans: list[dict[str, Any]] = []
    for source_index, path in enumerate(_source_episode_files(source)):
        table = pq.read_table(path)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        start, stop = choose_trim_bounds(
            actions,
            motion_threshold_rad=args.motion_threshold_rad,
            gripper_threshold=args.gripper_threshold,
            sustain_frames=args.sustain_frames,
            terminal_reference_frames=args.terminal_reference_frames,
            terminal_postroll_frames=args.terminal_postroll_frames,
            min_episode_frames=args.min_episode_frames,
        )
        plans.append(
            {
                "source_episode": source_index,
                "source_path": str(path),
                "source_frames": len(table),
                "start_frame": start,
                "stop_frame": stop,
                "output_frames": stop - start,
                "removed_prefix": start,
                "removed_suffix": len(table) - stop,
            }
        )

    total_before = sum(plan["source_frames"] for plan in plans)
    total_after = sum(plan["output_frames"] for plan in plans)
    print(f"source:   {source}")
    print(f"output:   {output}")
    print(f"episodes: {len(plans)}")
    print(f"frames:   {total_before} -> {total_after} (removed {total_before - total_after})")
    for plan in plans:
        print(
            f"episode {plan['source_episode']:03d}: {plan['source_frames']} -> {plan['output_frames']} "
            f"(prefix -{plan['removed_prefix']}, suffix -{plan['removed_suffix']})"
        )
    if args.dry_run:
        return None

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}; pass --overwrite to replace it.")
        shutil.rmtree(output)
    temporary = output.parent / f".{output.name}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    episode_rows: list[dict[str, Any]] = []
    episode_stats_rows: list[dict[str, Any]] = []
    all_stats: list[dict[str, dict[str, np.ndarray]]] = []
    global_index = 0
    try:
        (temporary / "meta").mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "meta" / "tasks.jsonl", temporary / "meta" / "tasks.jsonl")
        for new_index, plan in enumerate(plans):
            source_path = Path(plan["source_path"])
            source_table = pq.read_table(source_path)
            table = rebuild_episode_table(
                source_table,
                plan["start_frame"],
                plan["stop_frame"],
                episode_index=new_index,
                global_start=global_index,
                fps=fps,
            )
            chunk = new_index // chunks_size
            data_path = temporary / f"data/chunk-{chunk:03d}/episode_{new_index:06d}.parquet"
            data_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, data_path)

            output_videos: dict[str, Path] = {}
            for video_key in video_keys:
                source_video = (
                    source
                    / f"videos/chunk-{plan['source_episode'] // chunks_size:03d}/{video_key}/episode_{plan['source_episode']:06d}.mp4"
                )
                output_video = temporary / f"videos/chunk-{chunk:03d}/{video_key}/episode_{new_index:06d}.mp4"
                _run_ffmpeg_trim(
                    source_video,
                    output_video,
                    start=plan["start_frame"],
                    stop=plan["stop_frame"],
                )
                actual_frames = _video_frame_count(output_video)
                if actual_frames != len(table):
                    raise RuntimeError(
                        f"Video/table length mismatch for {output_video}: {actual_frames} != {len(table)}"
                    )
                output_videos[video_key] = output_video

            stats = _episode_stats(table, output_videos)
            all_stats.append(stats)
            source_episode = source_episodes.get(plan["source_episode"], {})
            episode_rows.append(
                {
                    "episode_index": new_index,
                    "tasks": source_episode.get("tasks", []),
                    "length": len(table),
                }
            )
            episode_stats_rows.append({"episode_index": new_index, "stats": serialize_dict(stats)})
            global_index += len(table)

        output_info = copy.deepcopy(info)
        output_info["total_episodes"] = len(plans)
        output_info["total_frames"] = global_index
        output_info["total_chunks"] = (len(plans) + chunks_size - 1) // chunks_size
        output_info["total_videos"] = len(plans) * len(video_keys)
        output_info["splits"] = {"train": f"0:{len(plans)}"}
        _write_json(temporary / "meta" / "info.json", output_info)
        _write_jsonl(temporary / "meta" / "episodes.jsonl", episode_rows)
        _write_jsonl(temporary / "meta" / "episodes_stats.jsonl", episode_stats_rows)
        _write_json(temporary / "meta" / "stats.json", serialize_dict(aggregate_stats(all_stats)))
        _write_json(
            temporary / "meta" / "cleaning_audit.json",
            {
                "source_repo_id": args.repo_id,
                "output_repo_id": output_repo_id,
                "fps": fps,
                "parameters": {
                    "motion_threshold_rad": args.motion_threshold_rad,
                    "gripper_threshold": args.gripper_threshold,
                    "sustain_frames": args.sustain_frames,
                    "terminal_reference_frames": args.terminal_reference_frames,
                    "terminal_postroll_frames": args.terminal_postroll_frames,
                    "min_episode_frames": args.min_episode_frames,
                },
                "total_frames_before": total_before,
                "total_frames_after": total_after,
                "episodes": plans,
            },
        )
        temporary.rename(output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    metadata = LeRobotDatasetMetadata(output_repo_id, root=output)
    if metadata.total_episodes != len(plans) or metadata.total_frames != total_after:
        raise RuntimeError("Post-write LeRobot metadata validation failed.")
    print(f"validated cleaned dataset: {output}")
    return output


def main() -> None:
    args = parse_args()
    if args.motion_threshold_rad <= 0 or args.gripper_threshold <= 0:
        raise SystemExit("Motion and gripper thresholds must be positive.")
    if args.sustain_frames <= 0 or args.terminal_reference_frames <= 0:
        raise SystemExit("Frame window sizes must be positive.")
    clean_dataset(args)


if __name__ == "__main__":
    main()
