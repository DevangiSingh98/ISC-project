"""Fine-tune VideoMAE on a source-separated Celeb-DF subset.

Protocol:
- Select exactly 100 real and 100 fake videos.
- Split by source identity, not by individual clip.
- Target an 80-real/20-real and 80-fake/20-fake video split.
- Extract 16 evenly spaced face-cropped frames per video.
- Fine-tune only the VideoMAE classifier head and final two encoder blocks.
- Report video-level metrics.
"""
import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, VideoMAEForVideoClassification


VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
FRAMES_PER_CLIP = 16
FRAME_SIZE = 224


def _source_key(path: Path) -> str:
    match = re.match(r"(id\d+)_", path.stem)
    return match.group(1) if match else re.sub(r"_\d+$", "", path.stem)


def _read_clip(path: Path) -> List[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: List[np.ndarray] = []
    try:
        while len(frames) < 300:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"Unable to decode video: {path}")
    indices = np.linspace(0, len(frames) - 1, FRAMES_PER_CLIP).round().astype(int)
    return [frames[index] for index in indices]


def _face_crop(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
    if len(faces):
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        margin = int(0.25 * max(w, h))
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(frame.shape[1], x + w + margin), min(frame.shape[0], y + h + margin)
        face = frame[y1:y2, x1:x2]
        if face.size:
            return cv2.resize(face, (FRAME_SIZE, FRAME_SIZE))
    return cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))


def _collect(root: Path, per_class: int, seed: int) -> List[Tuple[Path, int]]:
    real_folder = root / "Celeb-real"
    fake_folder = root / "Celeb-synthesis"
    for folder in (real_folder, fake_folder):
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing Celeb-DF folder: {folder}")

    real_records = [
        (path, 0)
        for path in sorted(real_folder.rglob("*"))
        if path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    fake_records = [
        (path, 1)
        for path in sorted(fake_folder.rglob("*"))
        if path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if len(real_records) < per_class or len(fake_records) < per_class:
        raise ValueError(
            f"Not enough videos: real={len(real_records)}, fake={len(fake_records)}, "
            f"required={per_class} per class"
        )

    rng = random.Random(seed)
    rng.shuffle(real_records)
    rng.shuffle(fake_records)
    return real_records[:per_class] + fake_records[:per_class]


def _split(records: List[Tuple[Path, int]], seed: int) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    grouped: Dict[str, Dict[int, List[Tuple[Path, int]]]] = {}
    for record in records:
        source = _source_key(record[0])
        grouped.setdefault(source, {0: [], 1: []})
        grouped[source][record[1]].append(record)

    sources = sorted(grouped)
    rng = random.Random(seed)
    best_training = None
    best_validation = None
    best_error = (10**9, 10**9)

    for _ in range(5000):
        shuffled = sources[:]
        rng.shuffle(shuffled)
        split_index = max(1, len(shuffled) // 5)
        validation_sources = set(shuffled[:split_index])

        training, validation = [], []
        for source in sources:
            target = validation if source in validation_sources else training
            target.extend(grouped[source][0])
            target.extend(grouped[source][1])

        val_real = sum(label == 0 for _, label in validation)
        val_fake = sum(label == 1 for _, label in validation)
        train_real = sum(label == 0 for _, label in training)
        train_fake = sum(label == 1 for _, label in training)
        error = (
            abs(val_real - val_fake),
            abs(val_real - 20) + abs(val_fake - 20),
        )
        if error < best_error:
            best_error = error
            best_training = training
            best_validation = validation
        if val_real == 20 and val_fake == 20:
            break

    if best_training is None or best_validation is None:
        raise ValueError("Unable to construct a source-disjoint split.")
    training, validation = best_training, best_validation

    train_real = sum(label == 0 for _, label in training)
    train_fake = sum(label == 1 for _, label in training)
    val_real = sum(label == 0 for _, label in validation)
    val_fake = sum(label == 1 for _, label in validation)
    if val_real != 20 or val_fake != 20:
        raise ValueError(
            f"Could not produce a 20-real/20-fake validation split; "
            f"best split was real={val_real}, fake={val_fake}"
        )
    return training, validation


class VideoDataset(Dataset):
    def __init__(self, records: List[Tuple[Path, int]], processor, augment: bool = False):
        self.records = records
        self.processor = processor
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path, label = self.records[index]
        clip = _read_clip(path)
        frames = [_face_crop(frame) for frame in clip]
        if self.augment:
            if random.random() < 0.5:
                frames = [cv2.flip(frame, 1) for frame in frames]
            if random.random() < 0.5:
                brightness = random.uniform(0.90, 1.10)
                frames = [np.clip(frame * brightness, 0, 255).astype(np.uint8) for frame in frames]
        encoded = self.processor(frames, return_tensors="pt")
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def _video_probability(model, processor, path: Path, device) -> float:
    frames = [_face_crop(frame) for frame in _read_clip(path)]
    inputs = processor(frames, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    model.eval()
    with torch.inference_mode():
        logits = model(**inputs).logits
        probability = float(torch.softmax(logits, dim=-1)[0, 1].item())
    return probability


def train(args: argparse.Namespace) -> None:
    records = _collect(args.dataset, args.per_class, args.seed)
    training, validation = _split(records, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(args.base_model)
    model = VideoMAEForVideoClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label={0: "real", 1: "fake"},
        label2id={"real": 0, "fake": 1},
        ignore_mismatched_sizes=True,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False

    encoder_layers = model.videomae.encoder.layer
    for layer in encoder_layers[-2:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    model.to(device)
    train_loader = DataLoader(
        VideoDataset(training, processor, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(pixel_values=pixel_values).logits
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
        print(f"epoch={epoch + 1} loss={running_loss / max(1, len(train_loader)):.4f}")

    scores, labels = [], []
    for path, label in validation:
        score = _video_probability(model, processor, path, device)
        scores.append(score)
        labels.append(label)

    if len(set(labels)) == 2:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        optimal_index = int(np.argmax(tpr - fpr))
        decision_threshold = float(thresholds[optimal_index])
    else:
        decision_threshold = 0.5
    predictions = [1 if score >= decision_threshold else 0 for score in scores]
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
        "decision_threshold": decision_threshold,
        "validation_scores": [
            {"label": int(label), "fake_probability": round(float(score), 6)}
            for label, score in zip(labels, scores)
        ],
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "train_real": sum(label == 0 for _, label in training),
        "train_fake": sum(label == 1 for _, label in training),
        "validation_real": sum(label == 0 for _, label in validation),
        "validation_fake": sum(label == 1 for _, label in validation),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output / "videomae-celebdf"
    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    report = {
        "architecture": "VideoMAE",
        "base_model": args.base_model,
        "protocol": "100-real/100-fake, source-disjoint 80/20 split",
        "seed": args.seed,
        "device": str(device),
        "decision_threshold": decision_threshold,
        "metrics": metrics,
    }
    (args.output / "videomae-celebdf.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f"Saved checkpoint to {checkpoint_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("."))
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("models/celebdf_videomae"))
    parser.add_argument("--base-model", default="MCG-NJU/videomae-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
