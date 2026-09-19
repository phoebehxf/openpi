#!/usr/bin/env python3
"""Render LeRobot episodes side-by-side with source frame numbers overlaid."""

import argparse
import json
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episodes", required=True, help="Comma-separated episode indices")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve()
    info = json.loads((dataset / "meta/info.json").read_text())
    episodes = {int(item.strip()) for item in args.episodes.split(",") if item.strip()}
    known = {
        json.loads(line)["episode_index"]
        for line in (dataset / "meta/episodes.jsonl").read_text().splitlines()
        if line.strip()
    }
    if unknown := episodes - known:
        raise ValueError(f"Unknown episode indices: {sorted(unknown)}")
    video_keys = [key for key, value in info["features"].items() if value["dtype"] == "video"]
    if len(video_keys) != 2:
        raise ValueError(f"Expected exactly two video streams, got {video_keys}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for episode in sorted(episodes):
        fmt = {
            "episode_index": episode,
            "episode_chunk": episode // int(info.get("chunks_size", 1000)),
        }
        inputs = [dataset / info["video_path"].format(**fmt, video_key=key) for key in video_keys]
        output = args.output_dir / f"episode_{episode:06d}_frames.mp4"
        labels = [key.rsplit(".", 1)[-1] for key in video_keys]
        # showinfo is deliberately avoided: drawtext's n is the zero-based
        # decoded source-frame number and is burned into every output frame.
        filters = (
            f"[0:v]drawtext=text='{labels[0]}  frame %{{n}}':x=12:y=12:"
            "fontsize=28:fontcolor=white:box=1:boxcolor=black@0.65[v0];"
            f"[1:v]drawtext=text='{labels[1]}  frame %{{n}}':x=12:y=12:"
            "fontsize=28:fontcolor=white:box=1:boxcolor=black@0.65[v1];"
            "[v0][v1]hstack=inputs=2[out]"
        )
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(inputs[0]),
            "-i",
            str(inputs[1]),
            "-filter_complex",
            filters,
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-preset",
            "fast",
            str(output),
        ]
        subprocess.run(command, check=True)
        print(output)


if __name__ == "__main__":
    main()
