import pytest
import numpy as np
from PIL import Image, ImageFilter
from engines.forensics.manipulation.localizer import ManipulationLocalizer
from engines.forensics.noise_compression.compression_forensics import CompressionForensicsAnalyzer

@pytest.fixture
def clean_image():
    # Natural-looking synthetic image with uniform noise
    np.random.seed(42)
    base = np.zeros((300, 400, 3), dtype=np.uint8)
    for c in range(3):
        base[:, :, c] = np.random.normal(128, 15, (300, 400)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(base)

@pytest.fixture
def spliced_image(clean_image):
    # Create copy and paste an inconsistent foreign patch with distinct noise/gradient
    img = clean_image.copy()
    np.random.seed(99)
    foreign_patch = np.zeros((100, 100, 3), dtype=np.uint8)
    for c in range(3):
        foreign_patch[:, :, c] = np.random.normal(200, 35, (100, 100)).clip(0, 255).astype(np.uint8)
    foreign_pil = Image.fromarray(foreign_patch)
    img.paste(foreign_pil, (150, 100))
    return img

def test_clean_image_low_manipulation_score(clean_image):
    localizer = ManipulationLocalizer()
    score, mask, heatmap_b64 = localizer.generate_heatmap(clean_image)
    
    assert score < 0.45
    assert mask.shape == (300, 400)
    assert heatmap_b64.startswith("data:image/png;base64,")

def test_spliced_image_elevated_heatmap(spliced_image):
    localizer = ManipulationLocalizer()
    score, mask, heatmap_b64 = localizer.generate_heatmap(spliced_image)

    # Global score must be significantly elevated on spliced media
    assert score > 0.60
    
    # Spliced region [100:200, 150:250] should have higher mean score than untouched background
    spliced_roi = mask[100:200, 150:250]
    background_roi = mask[0:80, 0:100]
    assert np.mean(spliced_roi) > np.mean(background_roi)

def test_compression_forensics_ela_and_double_jpeg(clean_image, spliced_image):
    analyzer = CompressionForensicsAnalyzer()
    
    clean_results = analyzer.analyze(clean_image)
    assert 0.0 <= clean_results["compression_anomaly_score"] <= 1.0
    assert 0.0 <= clean_results["prnu_score"] <= 1.0

    spliced_results = analyzer.analyze(spliced_image)
    assert 0.0 <= spliced_results["compression_anomaly_score"] <= 1.0
    # Inconsistent splice increases ELA block variance
    assert spliced_results["ela_score"] >= clean_results["ela_score"] * 0.9
