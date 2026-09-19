#!/usr/bin/env python3
"""Split Piper cup-rack demonstrations into prompt-conditioned skill phases.

The source LeRobot v2.1 dataset is never modified. Grasp/release landmarks are
read from the gripper action. The less observable semantic boundaries (unhook
complete and rack alignment start) are estimated from joint-space distance and
can be corrected per episode with an overrides JSON file.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from clean_lerobot_episodes import _array_stats, _run_ffmpeg_trim, _video_frame_count, _video_stats
from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.utils import serialize_dict
import numpy as np
import pyarrow.parquet as pq

from split_pour_into_phases import _read_video_frames, _rebuild_table, _rows, _write_json, _write_jsonl


HANG_TASK = "Pick up the red cup from the table and hang it on the rack"
TAKE_TASK = "Take the red cup off the rack and place it on the table"

PHASES = {
    "hang": (
        ("grasp_and_move_to_rack", "Grasp the red cup and move it next to the rack"),
        ("align_and_hang", "Align the cup handle opening with the rack peg and hang the cup on it"),
    ),
    "take": (
        ("grasp_on_rack", "Grasp the red cup on the rack"),
        ("lift_diagonally_to_unhook", "Lift the red cup diagonally upward to unhook it from the rack"),
        ("place_on_table", "Move the red cup to the table, place it down, and release it"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="local/piper-cup-rack-multitask-cleaned-v2")
    parser.add_argument("--output-repo-id", default="local/piper-cup-rack-phases-v1")
    parser.add_argument(
        "--lerobot-home",
        type=Path,
        default=Path(os.environ.get("HF_LEROBOT_HOME", Path(__file__).resolve().parents[1] / "local_datasets")),
    )
    parser.add_argument("--off-hook-fraction", type=float, default=0.18)
    parser.add_argument("--alignment-fraction", type=float, default=0.35)
    parser.add_argument("--sustain-frames", type=int, default=5)
    parser.add_argument("--overlap-frames", type=int, default=10)
    parser.add_argument("--min-phase-frames", type=int, default=20)
    parser.add_argument(
        "--overrides",
        type=Path,
        help='JSON object keyed by episode index, e.g. {"3":{"align_start":170},"42":{"off_hook_end":205}}.',
    )
    parser.add_argument("--visualize-dir", type=Path)
    parser.add_argument("--visualize-episodes", default="all")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _first_sustained(mask: np.ndarray, count: int) -> int | None:
    if len(mask) < count:
        return None
    hits = np.convolve(mask.astype(np.int16), np.ones(count, dtype=np.int16), mode="valid")
    found = np.flatnonzero(hits == count)
    return int(found[0]) if len(found) else None


def _landmarks(actions: np.ndarray) -> tuple[int, int]:
    closed = actions[:, 6] < 0.5
    opens = np.flatnonzero(~closed[1:] & closed[:-1]) + 1
    release = int(opens[-1]) if len(opens) else len(actions) - 1
    closes = np.flatnonzero(closed[1:] & ~closed[:-1]) + 1
    valid = closes[closes < release]
    if len(valid):
        grasp = int(valid[-1])
    elif closed[0]:
        grasp = 0
    else:
        raise ValueError("No gripper close before release and episode does not start closed")
    if release <= grasp:
        raise ValueError(f"Invalid grasp/release order: {grasp}/{release}")
    return grasp, release


def infer_plan(
    actions: np.ndarray,
    task: str,
    *,
    off_hook_fraction: float,
    alignment_fraction: float,
    sustain_frames: int,
    overlap_frames: int,
    min_phase_frames: int,
    override: dict[str, int],
) -> dict[str, Any]:
    actions = np.asarray(actions, dtype=np.float32)
    grasp, release = _landmarks(actions)
    joints = actions[:, :6]
    n = len(actions)
    if task == TAKE_TASK:
        distance = np.linalg.norm(joints - joints[grasp], axis=1)
        peak = float(distance[grasp:release].max())
        hit = _first_sustained(distance[grasp:release] >= off_hook_fraction * peak, sustain_frames)
        if hit is None:
            raise ValueError("Could not infer off-hook completion")
        boundary = int(override.get("off_hook_end", grasp + hit))
        kind = "take"
        landmark_name = "off_hook_end"
        raw_ranges = ((0, grasp), (grasp, boundary), (boundary, n))
    elif task == HANG_TASK:
        target_lo = max(grasp, release - 10)
        target = np.median(joints[target_lo:release], axis=0)
        distance = np.linalg.norm(joints - target, axis=1)
        peak = float(distance[grasp:release].max())
        hit = _first_sustained(distance[grasp:release] <= alignment_fraction * peak, sustain_frames)
        if hit is None:
            raise ValueError("Could not infer rack-alignment start")
        boundary = int(override.get("align_start", grasp + hit))
        kind = "hang"
        landmark_name = "align_start"
        # Some collected clips already start with a grasped cup at the rack and
        # contain only the precision alignment/hanging skill. If an override
        # explicitly marks frame 0, omit the nonexistent grasp-and-move phase
        # instead of fabricating a tiny episode from overlap frames.
        if boundary == 0:
            raw_ranges = ((0, n),)
            phase_start_index = 1
        else:
            raw_ranges = ((0, boundary), (boundary, n))
            phase_start_index = 0
    else:
        raise ValueError(f"Unexpected task: {task!r}")

    if not grasp <= boundary <= release:
        raise ValueError(f"{landmark_name}={boundary} must be in [{grasp}, {release}]")
    ranges = []
    for index, (start, stop) in enumerate(raw_ranges):
        ranges.append((max(0, start - (overlap_frames if index else 0)), min(n, stop + overlap_frames)))
    if any(stop - start < min_phase_frames for start, stop in ranges):
        raise ValueError(f"A phase is shorter than {min_phase_frames} frames: {ranges}")
    return {
        "kind": kind,
        "phase_start_index": phase_start_index if kind == "hang" else 0,
        "grasp_frame": grasp,
        "release_frame": release,
        landmark_name: boundary,
        "ranges": ranges,
    }


def _selected(value: str, available: set[int]) -> set[int]:
    if value.lower().strip() == "all":
        return available
    result = {int(item.strip()) for item in value.split(",") if item.strip()}
    if unknown := result - available:
        raise ValueError(f"Unknown episode indices: {sorted(unknown)}")
    return result


def visualize(source: Path, info: dict[str, Any], plan: dict[str, Any], actions: np.ndarray, output: Path) -> Path:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    episode = int(plan["source_episode_index"])
    boundary_key = "off_hook_end" if plan["kind"] == "take" else "align_start"
    landmarks = [int(plan["grasp_frame"]), int(plan[boundary_key]), int(plan["release_frame"])]
    labels = ["grasp", boundary_key, "release"]
    fmt = {"episode_index": episode, "episode_chunk": episode // int(info.get("chunks_size", 1000))}
    video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    camera_frames = {
        key: _read_video_frames(source / info["video_path"].format(**fmt, video_key=key), landmarks)
        for key in video_keys
    }
    figure = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = figure.add_gridspec(len(video_keys) + 1, 3, height_ratios=(*([1] * len(video_keys)), 0.9))
    for row, key in enumerate(video_keys):
        for column, (frame, label, index) in enumerate(zip(camera_frames[key], labels, landmarks, strict=True)):
            axis = figure.add_subplot(grid[row, column])
            axis.imshow(frame)
            axis.set_title(f"{key.rsplit('.', 1)[-1]}: {label}\nframe {index}")
            axis.axis("off")
    axis = figure.add_subplot(grid[len(video_keys), :])
    velocity = np.r_[0.0, np.linalg.norm(np.diff(actions[:, :6], axis=0), axis=1)]
    axis.plot(velocity, color="#315a9a", label="joint-space speed")
    axis.set_ylabel("joint speed")
    grip_axis = axis.twinx()
    grip_axis.step(np.arange(len(actions)), actions[:, 6], where="post", color="#b83232", label="gripper")
    grip_axis.set_ylim(-0.08, 1.08)
    for index, label in zip(landmarks, labels, strict=True):
        axis.axvline(index, color="black", linestyle="--", linewidth=0.9)
        axis.text(index, axis.get_ylim()[1], f" {label}", rotation=90, va="top", fontsize=8)
    axis.set_xlabel("source frame")
    figure.suptitle(f"Cup-rack phase split — source episode {episode:03d} ({plan['kind']})")
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"episode_{episode:06d}_phase_boundaries.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def main() -> None:
    args = parse_args()
    source = (args.lerobot_home / args.repo_id).resolve()
    output = (args.lerobot_home / args.output_repo_id).resolve()
    if source == output:
        raise ValueError("Source and output must differ")
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    info = json.loads((source / "meta/info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        raise ValueError("Only LeRobot v2.1 datasets are supported")
    overrides = json.loads(args.overrides.read_text()) if args.overrides else {}
    source_episodes = sorted(_rows(source / "meta/episodes.jsonl"), key=lambda row: row["episode_index"])
    available = {int(row["episode_index"]) for row in source_episodes}
    selected = _selected(args.visualize_episodes, available) if args.visualize_dir else set()
    chunks_size = int(info.get("chunks_size", 1000))
    plans = []
    for row in source_episodes:
        episode = int(row["episode_index"])
        task = row["tasks"][0]
        fmt = {"episode_index": episode, "episode_chunk": episode // chunks_size}
        table = pq.read_table(source / info["data_path"].format(**fmt), columns=["action"])
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        plan = infer_plan(
            actions,
            task,
            off_hook_fraction=args.off_hook_fraction,
            alignment_fraction=args.alignment_fraction,
            sustain_frames=args.sustain_frames,
            overlap_frames=args.overlap_frames,
            min_phase_frames=args.min_phase_frames,
            override=overrides.get(str(episode), {}),
        )
        plan.update(source_episode_index=episode, source_task=task)
        plans.append(plan)
        boundary_key = "off_hook_end" if plan["kind"] == "take" else "align_start"
        print(
            f"episode {episode:03d} {plan['kind']}: grasp={plan['grasp_frame']} "
            f"{boundary_key}={plan[boundary_key]} release={plan['release_frame']} ranges={plan['ranges']}",
            flush=True,
        )
        if episode in selected:
            path = visualize(source, info, plan, actions, args.visualize_dir.resolve())
            print(f"  visualization: {path}", flush=True)
    print(f"Planned {len(plans)} source episodes -> {sum(len(p['ranges']) for p in plans)} phase episodes")
    if args.dry_run:
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    fps = int(info["fps"])
    task_prompts = [prompt for kind in ("hang", "take") for _, prompt in PHASES[kind]]
    episodes, episode_stats, all_stats, manifest = [], [], [], []
    global_index = 0
    try:
        for plan in plans:
            source_index = int(plan["source_episode_index"])
            source_fmt = {"episode_index": source_index, "episode_chunk": source_index // chunks_size}
            source_table = pq.read_table(source / info["data_path"].format(**source_fmt))
            phase_defs = PHASES[plan["kind"]][int(plan["phase_start_index"]) :]
            for (phase_name, prompt), (start, stop) in zip(phase_defs, plan["ranges"], strict=True):
                new_index = len(episodes)
                task_index = task_prompts.index(prompt)
                table = _rebuild_table(
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
                pq.write_table(table, data_path)
                video_paths = {}
                for key in video_keys:
                    src = source / info["video_path"].format(**source_fmt, video_key=key)
                    dst = temporary / info["video_path"].format(**new_fmt, video_key=key)
                    _run_ffmpeg_trim(src, dst, start=start, stop=stop)
                    if _video_frame_count(dst) != len(table):
                        raise RuntimeError(f"Video length mismatch: {dst}")
                    video_paths[key] = dst
                stats = {name: _array_stats(np.asarray(table[name].to_pylist())) for name in table.column_names}
                for key, path in video_paths.items():
                    stats[key] = _video_stats(path)
                episodes.append({"episode_index": new_index, "tasks": [prompt], "length": len(table)})
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
                        **{key: value for key, value in plan.items() if key not in {"ranges"}},
                    }
                )
                global_index += len(table)
                print(f"Wrote episode {new_index:03d}: source={source_index:03d} phase={phase_name}", flush=True)
        output_info = copy.deepcopy(info)
        output_info.update(
            total_episodes=len(episodes),
            total_frames=global_index,
            total_tasks=len(task_prompts),
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
            [{"task_index": index, "task": prompt} for index, prompt in enumerate(task_prompts)],
        )
        _write_json(temporary / "meta/phase_split_manifest.json", manifest)
        temporary.rename(output)
    except Exception:
        print(f"Build failed; staging directory retained: {temporary}", flush=True)
        raise
    print(f"Complete: {output}; episodes={len(episodes)}, frames={global_index}, tasks={len(task_prompts)}")


if __name__ == "__main__":
    main()
