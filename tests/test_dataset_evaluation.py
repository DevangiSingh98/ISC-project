import hashlib
import json

import numpy as np
import pytest

from data.benchmarks.dataset_evaluation import (
    load_manifest,
    pixel_f1,
    pixel_iou,
    validate_manifest,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_manifest_rejects_cross_split_content_leakage():
    digest = _digest("same-image")
    records = [
        {
            "asset_id": "train-1",
            "path": "train/image.jpg",
            "split": "train",
            "category": "authentic",
            "content_sha256": digest,
        },
        {
            "asset_id": "test-1",
            "path": "test/image.jpg",
            "split": "test",
            "category": "authentic",
            "content_sha256": digest,
        },
    ]

    errors = validate_manifest(records)

    assert any("content leakage" in error for error in errors)


def test_manifest_loads_json_and_accepts_clean_splits(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "asset_id": "train-1",
                    "path": "train/image.jpg",
                    "split": "train",
                    "category": "authentic",
                    "content_sha256": _digest("train"),
                }
            ]
        ),
        encoding="utf-8",
    )

    records = load_manifest(manifest_path)

    assert validate_manifest(records) == []


def test_pixel_metrics_cover_perfect_and_empty_masks():
    expected = np.array([[1, 0], [0, 0]], dtype=float)

    assert pixel_iou(expected, expected) == 1.0
    assert pixel_f1(expected, expected) == 1.0
    assert pixel_iou(np.zeros_like(expected), np.zeros_like(expected)) == 1.0


def test_pixel_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError):
        pixel_iou(np.zeros((2, 2)), np.zeros((3, 3)))
