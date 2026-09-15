#!/usr/bin/env python3
"""Compare model-space image statistics between training videos and policy captures."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from openpi_client import image_tools


def metrics(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    luminance = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
    return np.asarray(
        [
            luminance.mean(),
            luminance.std(),
            *np.percentile(luminance, [5, 25, 50, 75, 95]),
            hsv[..., 1].mean(),
        ],
        dtype=np.float64,
    )


def prepare(image: np.ndarray) -> np.ndarray:
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))


def training_metrics(root: Path, camera: str, samples_per_video: int) -> np.ndarray:
    rows = []
    for path in sorted((root / "videos").glob(f"chunk-*/{camera}/episode_*.mp4")):
        capture = cv2.VideoCapture(str(path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for index in np.linspace(0, max(frame_count - 1, 0), samples_per_video, dtype=int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok:
                rows.append(metrics(prepare(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
        capture.release()
    return np.asarray(rows)


def capture_metrics(root: Path, filename: str) -> np.ndarray:
    rows = []
    for path in sorted(root.glob(f"request_*/{filename}")):
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is not None:
            rows.append(metrics(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    return np.asarray(rows)


def report(name: str, values: np.ndarray) -> None:
    labels = ("mean", "contrast", "p05", "p25", "p50", "p75", "p95", "saturation")
    print(f"{name}: frames={len(values)}")
    for index, label in enumerate(labels):
        column = values[:, index]
        print(
            f"  {label:10s} median={np.median(column):6.1f} "
            f"p05={np.percentile(column, 5):6.1f} p95={np.percentile(column, 95):6.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--samples-per-video", type=int, default=8)
    args = parser.parse_args()

    camera_pairs = (
        ("external", "observation.images.rgb", "external_decoded.png"),
        ("wrist", "observation.images.wrist", "wrist_decoded.png"),
    )
    for label, training_camera, capture_filename in camera_pairs:
        print(f"\n[{label}]")
        report("training", training_metrics(args.training_root, training_camera, args.samples_per_video))
        for capture_root in args.capture:
            report(capture_root.name, capture_metrics(capture_root, capture_filename))


if __name__ == "__main__":
    main()
