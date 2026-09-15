#!/usr/bin/env python3
"""Copy a LeRobot v2.1 dataset while omitting selected episodes.

The source dataset is never modified. Remaining episodes are renumbered from zero,
because LeRobot expects contiguous episode indices. A mapping is printed before the
copy starts and is also stored in ``meta/merge_manifest.json`` by the merge helper.
"""

import argparse
import json
from pathlib import Path

from merge_cup_datasets import merge


def read_episode_indices(source: Path) -> list[int]:
    episodes_path = source / "meta/episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"Missing {episodes_path}")
    return [
        json.loads(line)["episode_index"]
        for line in episodes_path.read_text().splitlines()
        if line.strip()
    ]


def parse_indices(value: str) -> set[int]:
    try:
        result = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated integers, e.g. 32,42,43") from exc
    if not result or min(result) < 0:
        raise argparse.ArgumentTypeError("Episode indices must be non-negative integers")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude", type=parse_indices, required=True)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Create the filtered copy. Without this flag, only print the planned mapping.",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if source == output:
        raise ValueError("--output must differ from --source; in-place deletion is not supported")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    old_indices = read_episode_indices(source)
    missing = sorted(args.exclude.difference(old_indices))
    if missing:
        raise ValueError(f"Requested episode indices do not exist: {missing}")

    kept = [old for old in old_indices if old not in args.exclude]
    print(f"Source: {source}")
    print(f"Output: {output}")
    print(f"Episodes to omit: {sorted(args.exclude)}")
    print(f"Result: {len(kept)} of {len(old_indices)} episodes")
    print("Renumbering near each omission (new <- old):")
    nearby = {
        old
        for excluded in args.exclude
        for old in range(max(0, excluded - 2), excluded + 4)
    }
    for new, old in enumerate(kept):
        if old in nearby:
            print(f"  {new} <- {old}")

    if not args.yes:
        print("Dry run only. Add --yes to create the output dataset.")
        return

    merge([source], output, exclude_episodes=args.exclude)


if __name__ == "__main__":
    main()
