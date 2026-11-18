"""Train the spatial detector on Tiny-GenImage from Hugging Face.

The script uses the dataset's real/test split for validation, keeps generator
metadata for reporting, and writes a state dict that the API detector loads.
It intentionally does not download the full dataset unless requested.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engines.forensics.synthetic.detector import LightweightSpatialBackbone


DATASET_ID = "TheKernel01/Tiny-GenImage"
IMAGE_SIZE = 128


class ImageArrayDataset(Dataset):
    def __init__(self, examples: List[Tuple[np.ndarray, int]]) -> None:
        self.images = [image for image, _ in examples]
        self.labels = [label for _, label in examples]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index].astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        tensor = torch.from_numpy(image.transpose(2, 0, 1))
        return tensor, torch.tensor(float(self.labels[index]), dtype=torch.float32)


def _row_to_example(row: Dict[str, object]) -> Tuple[np.ndarray, int, str]:
    image = row["image"]
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    image_array = np.asarray(image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.uint8)
    return image_array, int(row["label"]), str(row.get("generator", "unknown"))


def collect_balanced_examples(
    split: str,
    per_class: int,
    seed: int,
) -> Tuple[List[Tuple[np.ndarray, int]], Dict[str, int]]:
    """Collect a deterministic balanced sample from a streaming HF split."""
    dataset = load_dataset(DATASET_ID, split=split, streaming=True)
    buckets: Dict[int, List[Tuple[np.ndarray, int]]] = {0: [], 1: []}
    generators: Dict[str, int] = {}
    for row in dataset:
        image, label, generator = _row_to_example(row)
        if label not in buckets or len(buckets[label]) >= per_class:
            continue
        buckets[label].append((image, label))
        generators[generator] = generators.get(generator, 0) + 1
        if all(len(bucket) >= per_class for bucket in buckets.values()):
            break
    if any(len(bucket) < per_class for bucket in buckets.values()):
        counts = {str(label): len(bucket) for label, bucket in buckets.items()}
        raise RuntimeError(f"Could not collect a balanced {split} sample: {counts}")
    examples = buckets[0] + buckets[1]
    random.Random(seed).shuffle(examples)
    return examples, generators


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    loss_fn = nn.BCELoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).squeeze(1)
            total_loss += float(loss_fn(predictions, labels).item()) * len(labels)
            correct += int(((predictions >= 0.5) == (labels >= 0.5)).sum().item())
            total += len(labels)
    if total == 0:
        raise RuntimeError("Validation loader produced no examples.")
    return total_loss / total, correct / total


def train(args: argparse.Namespace) -> Dict[str, object]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_examples, train_generators = collect_balanced_examples(
        "train", args.train_per_class, args.seed
    )
    validation_examples, validation_generators = collect_balanced_examples(
        args.validation_split, args.validation_per_class, args.seed
    )
    train_loader = DataLoader(
        ImageArrayDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        ImageArrayDataset(validation_examples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = LightweightSpatialBackbone().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_fn = nn.BCELoss()
    history: List[Dict[str, float]] = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(images).squeeze(1)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            total += len(labels)
        validation_loss, validation_accuracy = evaluate(model, validation_loader, device)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": total_loss / total,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"train_loss={total_loss / total:.4f} "
            f"val_loss={validation_loss:.4f} "
            f"val_accuracy={validation_accuracy:.4f}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.to("cpu").state_dict(),
            "image_size": IMAGE_SIZE,
            "dataset": DATASET_ID,
            "label_mapping": {"0": "real", "1": "fake"},
            "train_per_class": args.train_per_class,
            "validation_per_class": args.validation_per_class,
            "validation_split": args.validation_split,
            "history": history,
        },
        output_path,
    )
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "weights": str(output_path),
                "dataset": DATASET_ID,
                "train_generators": train_generators,
                "validation_generators": validation_generators,
                "history": history,
                "device": str(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"weights": str(output_path), "metadata": str(metadata_path), "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-per-class", type=int, default=4000)
    parser.add_argument("--validation-per-class", type=int, default=1000)
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output",
        default="models/tiny_genimage_spatial.pt",
        help="Path for trained weights.",
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
