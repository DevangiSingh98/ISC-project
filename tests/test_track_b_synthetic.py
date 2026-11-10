import pytest
import numpy as np
from PIL import Image
from engines.forensics.synthetic.detector import AIGeneratedImageDetector, SpectralFFTBranch

@pytest.fixture
def natural_photo():
    # Natural photo approximation: 1/f spatial gradient
    np.random.seed(101)
    x = np.linspace(0, 10, 256)
    y = np.linspace(0, 10, 256)
    xx, yy = np.meshgrid(x, y)
    grad = np.sin(xx / 2.0) + np.cos(yy / 2.0)
    # Add natural soft noise
    noise = np.random.normal(0, 0.2, (256, 256))
    img_data = np.clip((grad + noise + 2.0) * 50, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([img_data, img_data, img_data]))

@pytest.fixture
def synthetic_diffusion_artifact():
    # Generative model approximation: periodic checkerboard high-frequency grid
    np.random.seed(202)
    base = np.zeros((256, 256), dtype=np.float32)
    # Inject periodic upsampling checkerboard pattern (characteristic of deconvolution/upsamplers)
    y, x = np.ogrid[:256, :256]
    checker = np.sin(x * np.pi / 2.0) * np.sin(y * np.pi / 2.0)
    combined = np.clip(128.0 + checker * 40.0 + np.random.normal(0, 10, (256, 256)), 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([combined, combined, combined]))

def test_spectral_fft_branch_natural_vs_synthetic(natural_photo, synthetic_diffusion_artifact):
    branch = SpectralFFTBranch()
    
    gray_natural = np.asarray(natural_photo.convert("L"))
    score_nat, meta_nat = branch.analyze_frequency_spectrum(gray_natural)
    
    gray_synth = np.asarray(synthetic_diffusion_artifact.convert("L"))
    score_synth, meta_synth = branch.analyze_frequency_spectrum(gray_synth)

    # Synthetic image must trigger higher spectral anomaly than natural 1/f decay
    assert score_synth > score_nat
    assert meta_synth["spike_ratio"] >= meta_nat["spike_ratio"]

def test_ai_detector_end_to_end(natural_photo, synthetic_diffusion_artifact):
    detector = AIGeneratedImageDetector()
    
    res_nat = detector.analyze(natural_photo, deep_scan=False)
    assert 0.0 <= res_nat["synthetic_score"] <= 1.0
    assert res_nat["deep_scan_used"] is False
    assert res_nat["latency_ms"] < 2500.0 # Sub-2.5 second image inference

    res_synth = detector.analyze(synthetic_diffusion_artifact, deep_scan=False)
    assert res_synth["synthetic_score"] > res_nat["synthetic_score"]
    assert "uncertainty_variance" in res_synth

def test_deep_scan_flag_and_latency(natural_photo):
    detector = AIGeneratedImageDetector()
    res = detector.analyze(natural_photo, deep_scan=True)
    assert res["deep_scan_used"] is True
    assert res["latency_ms"] >= 40.0 # Verified deep scan execution path

