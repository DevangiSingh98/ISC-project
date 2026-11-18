"""Validation helpers for reproducible image-forensics evaluation datasets."""

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set

import numpy as np


REQUIRED_MANIFEST_FIELDS = {
    "asset_id",
    "path",
    "split",
    "category",
    "content_sha256",
}
VALID_SPLITS = {"train", "validation", "test"}


def load_manifest(path: Path) -> List[Dict[str, str]]:
    """Load a CSV or JSON list of dataset records."""
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise ValueError("JSON manifest must contain a list of objects.")
        return [{str(key): str(value) for key, value in item.items()} for item in data]
    raise ValueError("Manifest must be a .csv or .json file.")


def validate_manifest(records: Sequence[Mapping[str, str]]) -> List[str]:
    """Return all manifest validation errors, including content leakage."""
    errors: List[str] = []
    seen_asset_ids: Set[str] = set()
    hashes_by_split: Dict[str, Set[str]] = {}

    for index, record in enumerate(records):
        missing = REQUIRED_MANIFEST_FIELDS.difference(record)
        if missing:
            errors.append(f"row {index}: missing fields {sorted(missing)}")
            continue

        asset_id = record["asset_id"]
        if asset_id in seen_asset_ids:
            errors.append(f"row {index}: duplicate asset_id '{asset_id}'")
        seen_asset_ids.add(asset_id)

        split = record["split"].lower()
        if split not in VALID_SPLITS:
            errors.append(f"row {index}: invalid split '{split}'")

        content_sha256 = record["content_sha256"].lower()
        if len(content_sha256) != 64:
            errors.append(f"row {index}: content_sha256 must be a 64-character hex digest")
        else:
            try:
                int(content_sha256, 16)
            except ValueError:
                errors.append(f"row {index}: content_sha256 is not hexadecimal")
            hashes_by_split.setdefault(split, set()).add(content_sha256)

    split_names = sorted(hashes_by_split)
    for left_index, left_split in enumerate(split_names):
        for right_split in split_names[left_index + 1 :]:
            overlap = hashes_by_split[left_split].intersection(hashes_by_split[right_split])
            if overlap:
                errors.append(
                    f"content leakage between {left_split} and {right_split}: "
                    f"{len(overlap)} duplicate content hash(es)"
                )
    return errors


def file_sha256(path: Path) -> str:
    """Compute a dataset file digest for manifest generation."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_iou(predicted: np.ndarray, expected: np.ndarray, threshold: float = 0.5) -> float:
    """Compute binary intersection-over-union for predicted and expected masks."""
    predicted_mask = np.asarray(predicted) >= threshold
    expected_mask = np.asarray(expected) >= threshold
    if predicted_mask.shape != expected_mask.shape:
        raise ValueError("Predicted and expected masks must have the same shape.")
    union = np.logical_or(predicted_mask, expected_mask).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(predicted_mask, expected_mask).sum() / union)


def pixel_f1(predicted: np.ndarray, expected: np.ndarray, threshold: float = 0.5) -> float:
    """Compute binary pixel F1 for predicted and expected masks."""
    predicted_mask = np.asarray(predicted) >= threshold
    expected_mask = np.asarray(expected) >= threshold
    if predicted_mask.shape != expected_mask.shape:
        raise ValueError("Predicted and expected masks must have the same shape.")
    true_positive = np.logical_and(predicted_mask, expected_mask).sum()
    predicted_positive = predicted_mask.sum()
    expected_positive = expected_mask.sum()
    if predicted_positive == 0 and expected_positive == 0:
        return 1.0
    if true_positive == 0:
        return 0.0
    precision = true_positive / predicted_positive
    recall = true_positive / expected_positive
    return float(2 * precision * recall / (precision + recall))
