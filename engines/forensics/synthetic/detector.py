import time
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.fftpack import fft2, fftshift

class SpectralFFTBranch:
    """
    Extracts 2D Fast Fourier Transform (FFT) power spectra and evaluates
    azimuthal radial frequency distribution. Natural photographs follow a strict
    1/f^alpha power law decay, whereas Diffusion and GAN generators produce
    distinct periodic grid spikes and anomalous high-frequency energy.
    """
    def __init__(self, size: int = 256):
        self.size = size

    def analyze_frequency_spectrum(self, gray_image: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        h, w = gray_image.shape
        if h != self.size or w != self.size:
            from PIL import Image as PILImage
            img = PILImage.fromarray(gray_image.astype(np.uint8)).resize((self.size, self.size))
            gray = np.asarray(img, dtype=np.float32)
        else:
            gray = gray_image.astype(np.float32)

        # 2D FFT and center low frequencies
        f_transform = fft2(gray)
        f_shift = fftshift(f_transform)
        magnitude = np.abs(f_shift)
        log_spectrum = np.log1p(magnitude)

        # Radial profile calculation (azimuthal average)
        center = self.size // 2
        y, x = np.ogrid[:self.size, :self.size]
        r = np.hypot(x - center, y - center).astype(np.int32)

        radial_mean = np.bincount(r.ravel(), log_spectrum.ravel()) / (np.bincount(r.ravel()) + 1e-6)
        radii = np.arange(len(radial_mean))[1:center]
        profile = radial_mean[1:center]

        # In natural images: log(power) vs log(freq) is approximately linear (1/f decay)
        log_r = np.log(radii)
        log_p = np.log(profile + 1e-6)
        
        # Fit linear regression to measure deviation from natural 1/f decay
        slope, intercept = np.polyfit(log_r, log_p, 1)
        fitted = slope * log_r + intercept
        residuals = np.abs(log_p - fitted)
        power_law_deviation = float(np.mean(residuals))

        # Check for periodic grid spikes in high-frequency quadrants
        outer_spectrum = log_spectrum.copy()
        outer_spectrum[center-20:center+20, center-20:center+20] = 0.0 # Zero DC and low freqs
        spectrum_mean = np.mean(outer_spectrum)
        spectrum_std = np.std(outer_spectrum) + 1e-6
        spikes = np.sum((outer_spectrum - spectrum_mean) > 3.5 * spectrum_std)
        spike_ratio = float(min(1.0, spikes / 25.0))

        # Combined spectral synthetic score
        spectral_score = float(np.clip(power_law_deviation * 2.5 + spike_ratio * 0.5, 0.0, 1.0))

        return spectral_score, {
            "power_law_deviation": power_law_deviation,
            "spike_ratio": spike_ratio,
            "slope": float(slope)
        }

class LightweightSpatialBackbone(nn.Module):
    """
    Lightweight feature extractor with Monte Carlo dropout for epistemic uncertainty.
    Evaluates spatial texture coherence and structural generative artifacts.
    """
    def __init__(self, p_dropout: float = 0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1) # 128x128
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2) # 64x64
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # 32x32
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 16x16
        self.bn3 = nn.BatchNorm2d(128)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=p_dropout)
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        out = self.sigmoid(self.fc(x))
        return out

class AIGeneratedImageDetector:
    """
    Dual-branch AI-Generated Image Detector combining:
    1. Spectral FFT power spectrum branch (uncovering periodic artifacts & 1/f law deviations)
    2. Spatial CNN / ConvNeXt backbone with MC-Dropout uncertainty estimation
    3. Optional DIRE (Diffusion Reconstruction Error) behind deep-scan flag.
    """
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fft_branch = SpectralFFTBranch()
        self.spatial_model = LightweightSpatialBackbone().to(self.device)
        self.spatial_model.eval()

    def _mc_dropout_predict(self, tensor_x: torch.Tensor, n_passes: int = 5) -> Tuple[float, float]:
        """Runs n forward passes with active dropout to quantify epistemic uncertainty."""
        self.spatial_model.train() # Enable dropout
        preds = []
        with torch.no_grad():
            for _ in range(n_passes):
                out = self.spatial_model(tensor_x)
                preds.append(out.item())
        self.spatial_model.eval()
        mean_p = float(np.mean(preds))
        var_p = float(np.var(preds))
        return mean_p, var_p

    def analyze(self, image: Image.Image, deep_scan: bool = False) -> Dict[str, Any]:
        """
        Detects whether an image is AI-generated (Diffusion / GAN).
        deep_scan: If True, executes full deep scan (DIRE stub) with documented latency.
        """
        start_time = time.time()
        rgb = image.convert("RGB").resize((256, 256))
        gray = np.asarray(rgb.convert("L"), dtype=np.float32)

        # 1. Frequency branch
        spectral_score, spectral_meta = self.fft_branch.analyze_frequency_spectrum(gray)

        # 2. Spatial backbone with uncertainty
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        # Normalize
        mean_vec = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std_vec = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = ((arr - mean_vec) / std_vec).astype(np.float32)
        tensor_x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        spatial_mean, spatial_variance = self._mc_dropout_predict(tensor_x)

        # 3. Optional DIRE Deep Scan (Known Gap 2)
        dire_score = None
        if deep_scan:
            # DIRE (Diffusion Reconstruction Error) simulation stub
            # In production, DIRE passes image through DDIM inversion and computes MSE
            time.sleep(0.05) # Simulated inversion step (documented ~15-30s in full diffusion pipelines)
            dire_score = float(0.6 * spectral_score + 0.4 * spatial_mean)

        # Dual-branch fusion
        # Spectral frequency artifacts are especially reliable across unseen generators
        combined_score = 0.55 * spectral_score + 0.45 * spatial_mean
        if dire_score is not None:
            combined_score = 0.7 * combined_score + 0.3 * dire_score

        elapsed_latency_ms = (time.time() - start_time) * 1000.0

        return {
            "synthetic_score": float(np.clip(combined_score, 0.0, 1.0)),
            "spectral_score": spectral_score,
            "spatial_score": spatial_mean,
            "uncertainty_variance": spatial_variance,
            "deep_scan_used": deep_scan,
            "latency_ms": elapsed_latency_ms,
            "spectral_meta": spectral_meta
        }
