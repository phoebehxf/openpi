#!/usr/bin/env python3
"""Merge the two local Piper v2.1 datasets without trimming or re-encoding."""

import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile

from clean_lerobot_episodes import _array_stats
from clean_lerobot_episodes import _video_frame_count
from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.utils import serialize_dict
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
# REPO_ID = "local/piper-cup-rack-multitask-cleaned-v2"
REPO_ID = "local/piper-pour-water-cleaned-merged-v1"
# SOURCES = ("piper-hang-cup-on-rack-cleaned-v1", "piper-take-cup-from-rack-cleaned-v1", "piper-cup-rack-2-cleaned-v1")
SOURCES = ("piper-pour-water-cleaned-v1", "piper-pour-water-2-cleaned-v1")


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def replace(table, key, values):
    i = table.schema.get_field_index(key)
    if i < 0:
        raise ValueError(f"Missing column: {key}")
    return table.set_column(i, key, pa.array(values, type=table.schema.field(i).type))


def merge(sources, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    infos = [json.loads((s / "meta/info.json").read_text()) for s in sources]
    for info in infos:
        if info["codebase_version"] != "v2.1":
            raise ValueError("Only converted v2.1 sources are supported")
        for key in ("fps", "robot_type", "features"):
            if info[key] != infos[0][key]:
                raise ValueError(f"Incompatible source {key}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    (temporary / "meta").mkdir()
    episodes, stats_rows, stats_all, provenance, tasks = [], [], [], [], []
    frame_offset = 0
    video_keys = [k for k, v in infos[0]["features"].items() if v["dtype"] == "video"]
    chunks_size = 1000
    try:
        for source, info in zip(sources, infos, strict=True):
            mapping = {}
            for task in rows(source / "meta/tasks.jsonl"):
                if task["task"] not in tasks:
                    tasks.append(task["task"])
                mapping[task["task_index"]] = tasks.index(task["task"])
            source_stats = {r["episode_index"]: r["stats"] for r in rows(source / "meta/episodes_stats.jsonl")}
            source_eps = sorted(rows(source / "meta/episodes.jsonl"), key=lambda r: r["episode_index"])
            if len(source_eps) != info["total_episodes"]:
                raise ValueError("Episode count mismatch")
            source_frames = 0
            for episode in source_eps:
                old, new = episode["episode_index"], len(episodes)
                fmt = {"episode_index": old, "episode_chunk": old // info["chunks_size"]}
                table = pq.read_table(source / info["data_path"].format(**fmt))
                n = len(table)
                if n != episode["length"] or not np.all(np.asarray(table["episode_index"]) == old):
                    raise ValueError(f"Invalid episode {source}/{old}")
                if not np.array_equal(np.asarray(table["frame_index"]), np.arange(n)):
                    raise ValueError("Non-contiguous frame_index")
                if not np.allclose(np.asarray(table["timestamp"]), np.arange(n) / info["fps"], atol=1e-4):
                    raise ValueError("Unexpected timestamps")
                table = replace(table, "episode_index", [new] * n)
                table = replace(table, "index", range(frame_offset, frame_offset + n))
                table = replace(table, "task_index", [mapping[t] for t in table["task_index"].to_pylist()])
                table = table.replace_schema_metadata(None)
                data_path = temporary / f"data/chunk-{new // chunks_size:03d}/episode_{new:06d}.parquet"
                data_path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, data_path)
                stats = {
                    k: {name: np.asarray(value) for name, value in values.items()}
                    for k, values in source_stats[old].items()
                    if k in video_keys
                }
                for key in table.column_names:
                    stats[key] = _array_stats(np.asarray(table[key].to_pylist()))
                for key in video_keys:
                    src_video = source / info["video_path"].format(**fmt, video_key=key)
                    if _video_frame_count(src_video) != n:
                        raise ValueError(f"Video frame count mismatch: {src_video}")
                    dst = temporary / f"videos/chunk-{new // chunks_size:03d}/{key}/episode_{new:06d}.mp4"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_video, dst)
                used_tasks = sorted(set(table["task_index"].to_pylist()))
                episodes.append({"episode_index": new, "tasks": [tasks[t] for t in used_tasks], "length": n})
                stats_rows.append({"episode_index": new, "stats": serialize_dict(stats)})
                stats_all.append(stats)
                provenance.append({"episode_index": new, "source": str(source), "source_episode_index": old})
                frame_offset += n
                source_frames += n
                print(f"Merged episode {new}: {source.name}/{old}, {n} frames", flush=True)
            if source_frames != info["total_frames"]:
                raise ValueError("Source total_frames mismatch")
        info = copy.deepcopy(infos[0])
        info.update(
            total_episodes=len(episodes),
            total_frames=frame_offset,
            total_tasks=len(tasks),
            chunks_size=chunks_size,
            total_chunks=(len(episodes) + chunks_size - 1) // chunks_size,
            total_videos=len(episodes) * len(video_keys),
            splits={"train": f"0:{len(episodes)}"},
            data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
        write_json(temporary / "meta/info.json", info)
        write_json(temporary / "meta/stats.json", serialize_dict(aggregate_stats(stats_all)))
        for filename, records in (
            ("episodes", episodes),
            ("episodes_stats", stats_rows),
            ("tasks", [{"task_index": i, "task": t} for i, t in enumerate(tasks)]),
        ):
            (temporary / f"meta/{filename}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        write_json(temporary / "meta/merge_manifest.json", provenance)
        temporary.rename(output)
    except Exception:
        print(f"Merge incomplete; diagnostic staging directory retained: {temporary}", flush=True)
        raise
    print(f"Complete: {output}; episodes={len(episodes)}, frames={frame_offset}, tasks={tasks}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-home", type=Path, default=Path.home() / ".cache/huggingface/lerobot/phoebe777777")
    parser.add_argument("--output", type=Path, default=ROOT / "local_datasets" / REPO_ID)
    args = parser.parse_args()
    merge([args.source_home / name for name in SOURCES], args.output.resolve())
