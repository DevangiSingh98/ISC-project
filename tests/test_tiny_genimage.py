import numpy as np
import torch
from PIL import Image

from data.benchmarks.tiny_genimage import ImageArrayDataset
from engines.forensics.synthetic.detector import (
    LightweightSpatialBackbone,
    AIGeneratedImageDetector,
)


def test_training_dataset_normalizes_images_and_labels():
    dataset = ImageArrayDataset(
        [
            (np.zeros((128, 128, 3), dtype=np.uint8), 0),
            (np.full((128, 128, 3), 255, dtype=np.uint8), 1),
        ]
    )

    image, label = dataset[1]

    assert image.shape == (3, 128, 128)
    assert image.dtype == torch.float32
    assert label.item() == 1.0


def test_detector_loads_trained_state_dict(tmp_path):
    weights_path = tmp_path / "weights.pt"
    model = LightweightSpatialBackbone()
    torch.save({"state_dict": model.state_dict()}, weights_path)

    detector = AIGeneratedImageDetector(model_path=str(weights_path), device="cpu")
    result = detector.analyze(Image.new("RGB", (320, 240), color="blue"))

    assert detector.weights_loaded is True
    assert result["spatial_model_trained"] is True
    assert 0.0 <= result["synthetic_score"] <= 1.0
