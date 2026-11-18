"""Measure perceptual fingerprint stability under common media transformations.

This experiment is intentionally independent of the detector benchmark. It
reports how pHash and dHash change when an image is compressed, resized,
cropped, or rotated, and writes machine-readable results for presentation.
"""

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from PIL import Image

# Allow direct execution from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engines.crypto.hasher import (
    compute_dhash,
    compute_phash,
    compute_sha256,
    hamming_distance,
)


Transform = Tuple[str, Callable[[Image.Image], Image.Image]]


def _jpeg_transform(quality: int) -> Callable[[Image.Image], Image.Image]:
    def transform(image: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")

    return transform


def _resize_transform(scale: float) -> Callable[[Image.Image], Image.Image]:
    def transform(image: Image.Image) -> Image.Image:
        width = max(1, round(image.width * scale))
        height = max(1, round(image.height * scale))
        return image.resize((width, height), Image.Resampling.LANCZOS)

    return transform


def _center_crop_transform(fraction: float) -> Callable[[Image.Image], Image.Image]:
    def transform(image: Image.Image) -> Image.Image:
        crop_width = max(1, round(image.width * fraction))
        crop_height = max(1, round(image.height * fraction))
        left = (image.width - crop_width) // 2
        top = (image.height - crop_height) // 2
        return image.crop((left, top, left + crop_width, top + crop_height))

    return transform


def _rotate_transform(degrees: float) -> Callable[[Image.Image], Image.Image]:
    def transform(image: Image.Image) -> Image.Image:
        return image.rotate(degrees, resample=Image.Resampling.BICUBIC, expand=False)

    return transform


def default_transforms() -> List[Transform]:
    return [
        ("jpeg_quality_95", _jpeg_transform(95)),
        ("jpeg_quality_75", _jpeg_transform(75)),
        ("jpeg_quality_50", _jpeg_transform(50)),
        ("jpeg_quality_25", _jpeg_transform(25)),
        ("resize_75_percent", _resize_transform(0.75)),
        ("resize_50_percent", _resize_transform(0.50)),
        ("resize_25_percent", _resize_transform(0.25)),
        ("center_crop_90_percent", _center_crop_transform(0.90)),
        ("center_crop_75_percent", _center_crop_transform(0.75)),
        ("rotate_5_degrees", _rotate_transform(5.0)),
    ]


def measure_image(image: Image.Image) -> Dict[str, object]:
    rgb_image = image.convert("RGB")
    encoded = io.BytesIO()
    rgb_image.save(encoded, format="PNG")
    raw_bytes = encoded.getvalue()
    return {
        "sha256": compute_sha256(raw_bytes),
        "phash": compute_phash(rgb_image),
        "dhash": compute_dhash(rgb_image),
        "width": rgb_image.width,
        "height": rgb_image.height,
        "byte_size": len(raw_bytes),
    }


def run_experiment(image_path: Path) -> Dict[str, object]:
    with Image.open(image_path) as source:
        original = source.convert("RGB")

    baseline = measure_image(original)
    results: List[Dict[str, object]] = []
    for name, transform in default_transforms():
        transformed = transform(original)
        measured = measure_image(transformed)
        results.append(
            {
                "transformation": name,
                "sha256_matches": measured["sha256"] == baseline["sha256"],
                "phash_hamming_distance": hamming_distance(
                    str(baseline["phash"]), str(measured["phash"])
                ),
                "dhash_hamming_distance": hamming_distance(
                    str(baseline["dhash"]), str(measured["dhash"])
                ),
                "width": measured["width"],
                "height": measured["height"],
                "byte_size": measured["byte_size"],
            }
        )

    return {
        "source": str(image_path),
        "baseline": baseline,
        "results": results,
        "notes": [
            "SHA-256 is expected to change when the encoded bytes change.",
            "Perceptual hash distances must be calibrated against false-match data before deployment.",
            "A crop can remove identifying visual content; a low distance does not prove pixel integrity.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--image",
        type=Path,
        default=root / "data" / "test_samples" / "authentic_press.jpg",
        help="Source image to transform.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "benchmarks" / "fingerprint_robustness_results.json",
        help="JSON output path.",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(f"Source image not found: {args.image}")

    report = run_experiment(args.image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote fingerprint robustness results to {args.output}")


if __name__ == "__main__":
    main()
