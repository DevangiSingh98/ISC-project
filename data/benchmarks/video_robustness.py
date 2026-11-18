"""Evaluate video verification behavior after common editorial transforms."""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engines.crypto.hasher import compute_sha256, inspect_video_metadata
from engines.forensics.deepfake_video import VideoForensicsDetector


Frame = Tuple[float, object]


def _read_video(path: Path) -> Tuple[float, List[object]]:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frames: List[object] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"Unable to decode any frames from {path}")
    return fps, frames


def _write_video(path: Path, fps: float, frames: Iterable[object], size: Tuple[int, int]) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(fps, 1.0),
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create transformed video at {path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _transforms(fps: float, frames: List[object]) -> Dict[str, Tuple[float, List[object]]]:
    height, width = frames[0].shape[:2]
    resized = [
        cv2.resize(frame, (max(2, width // 2), max(2, height // 2)), interpolation=cv2.INTER_AREA)
        for frame in frames
    ]
    return {
        "reencoded": (fps, frames),
        "resized_50_percent": (fps, resized),
        "trimmed_first_half": (fps, frames[: max(1, len(frames) // 2)]),
        "every_other_frame": (max(1.0, fps / 2.0), frames[::2]),
    }


def run_experiment(video_path: Path) -> Dict[str, object]:
    fps, frames = _read_video(video_path)
    detector = VideoForensicsDetector(sample_fps=1.0)
    reports: List[Dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="newsproof-video-") as temp_dir:
        for name, (transform_fps, transform_frames) in _transforms(fps, frames).items():
            height, width = transform_frames[0].shape[:2]
            transformed_path = Path(temp_dir) / f"{name}.mp4"
            _write_video(transformed_path, transform_fps, transform_frames, (width, height))
            raw_bytes = transformed_path.read_bytes()
            forensic_result = detector.analyze(str(transformed_path))
            reports.append(
                {
                    "transformation": name,
                    "sha256": compute_sha256(raw_bytes),
                    "metadata": inspect_video_metadata(str(transformed_path)),
                    "deepfake_score": forensic_result["deepfake_score"],
                    "temporal_inconsistency_score": forensic_result["temporal_inconsistency_score"],
                    "mean_manipulation_score": forensic_result["mean_manipulation_score"],
                    "frames_analyzed": forensic_result["frames_analyzed"],
                }
            )

    return {
        "source": str(video_path),
        "detector": "VideoForensicsDetector heuristic v2.0-full",
        "results": reports,
        "notes": [
            "A changed SHA-256 is expected after editorial transformation.",
            "This experiment measures score stability; it does not establish deepfake detection accuracy.",
            "Accuracy requires labeled real and manipulated video datasets with held-out sources.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Source video to transform.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmarks/video_robustness_results.json"),
        help="JSON output path.",
    )
    args = parser.parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(f"Source video not found: {args.video}")
    report = run_experiment(args.video)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote video robustness results to {args.output}")


if __name__ == "__main__":
    main()
