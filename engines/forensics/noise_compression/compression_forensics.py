import io
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from scipy.fftpack import dct

class CompressionForensicsAnalyzer:
    """
    Analyzes compression artifacts, Error Level Analysis (ELA),
    double-JPEG compression inconsistencies, and auxiliary PRNU sensor noise.
    """
    def __init__(self, ela_resave_quality: int = 90):
        self.ela_resave_quality = ela_resave_quality

    def compute_ela(self, image: Image.Image, scale: float = 15.0) -> Tuple[float, np.ndarray]:
        """
        Computes Error Level Analysis (ELA).
        Returns:
            (anomaly_score, ela_intensity_map)
        """
        rgb_img = image.convert("RGB")
        
        # Resave image to memory buffer at fixed JPEG quality
        buffer = io.BytesIO()
        rgb_img.save(buffer, format="JPEG", quality=self.ela_resave_quality)
        buffer.seek(0)
        resaved = Image.open(buffer)

        # Difference between original and resaved
        diff = ImageChops.difference(rgb_img, resaved)
        
        # Convert to numpy array
        diff_arr = np.asarray(diff, dtype=np.float32)
        diff_intensity = np.mean(diff_arr, axis=2) # H x W

        # Check flat canvas / document screenshot
        gray_arr = np.asarray(image.convert("L"), dtype=np.float32)
        flat_mask = (gray_arr > 235) | (gray_arr < 18)
        is_doc = float(np.mean(flat_mask)) > 0.35

        # Compute local block variances (16x16 blocks)
        h, w = diff_intensity.shape
        block_size = 16
        block_variances = []
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                if is_doc and np.mean(flat_mask[i:i+block_size, j:j+block_size]) > 0.6:
                    continue # Exclude flat digital canvas blocks from variance calculations
                block = diff_intensity[i:i+block_size, j:j+block_size]
                block_variances.append(np.var(block))

        if len(block_variances) > 4:
            var_std = float(np.std(block_variances))
            var_mean = float(np.mean(block_variances)) + 1e-6
            # High variance across blocks indicates inconsistent compression histories (splicing)
            heterogeneity = min(1.0, var_std / (var_mean * 2.5))
        else:
            heterogeneity = 0.0

        scaled_ela = np.clip(diff_intensity * scale, 0, 255).astype(np.uint8)
        return float(heterogeneity), scaled_ela

    def compute_double_jpeg_artifacts(self, image: Image.Image) -> float:
        """
        Analyzes 2D DCT coefficient histograms for periodicity indicative of
        double-JPEG compression with different quantization tables.
        """
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        h, w = gray.shape
        
        # Process in 8x8 blocks (standard JPEG grid)
        dct_coeffs = []
        for i in range(0, h - 8 + 1, 8):
            for j in range(0, w - 8 + 1, 8):
                block = gray[i:i+8, j:j+8] - 128.0
                block_dct = dct(dct(block.T, norm='ortho').T, norm='ortho')
                # Collect low-to-mid AC coefficients
                dct_coeffs.append(block_dct[1, 2])
                dct_coeffs.append(block_dct[2, 1])

        if len(dct_coeffs) < 64:
            return 0.0

        coeffs = np.array(dct_coeffs)
        # Check for histogram periodicity (zeros/spikes)
        hist, bin_edges = np.histogram(coeffs, bins=60, range=(-30, 30))
        # Zero-crossing / valley-to-peak ratio
        peaks = np.diff(np.sign(np.diff(hist)))
        num_peaks = np.sum(peaks < 0)
        
        # A normal single-compressed image has a smooth Laplacian/Gaussian-shaped histogram;
        # double-compressed exhibits periodic ripples/comb patterns.
        comb_metric = min(1.0, max(0.0, (num_peaks - 4) / 16.0))
        return float(comb_metric)

    def compute_auxiliary_prnu_inconsistency(self, image: Image.Image) -> float:
        """
        Auxiliary PRNU estimation via high-pass residual cross-correlation.
        Per Known Gap 4: auxiliary signal only, never a standalone verdict driver.
        """
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        h, w = gray.shape
        if h < 64 or w < 64:
            return 0.0

        # Simple high-pass spatial filter residual
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        # Pad and compute residual
        padded = np.pad(gray, 1, mode='edge')
        residual = (
            padded[1:-1, 1:-1] * 4
            - padded[:-2, 1:-1] - padded[2:, 1:-1]
            - padded[1:-1, :-2] - padded[1:-1, 2:]
        )

        # Split into quadrants and test cross-quadrant noise correlation
        mid_h, mid_w = h // 2, w // 2
        q1 = residual[:mid_h, :mid_w].flatten()
        q2 = residual[:mid_h, mid_w:].flatten()
        q3 = residual[mid_h:, :mid_w].flatten()
        q4 = residual[mid_h:, mid_w:].flatten()

        min_len = min(len(q1), len(q2), len(q3), len(q4))
        q1, q2, q3, q4 = q1[:min_len], q2[:min_len], q3[:min_len], q4[:min_len]

        corr12 = np.corrcoef(q1, q2)[0, 1]
        corr34 = np.corrcoef(q3, q4)[0, 1]
        
        # If noise properties vary wildly across regions, inconsistency is high
        inconsistency = abs(corr12 - corr34)
        return float(min(1.0, max(0.0, inconsistency)))

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        ela_score, ela_map = self.compute_ela(image)
        double_jpeg_score = self.compute_double_jpeg_artifacts(image)
        prnu_inconsistency = self.compute_auxiliary_prnu_inconsistency(image)

        # Combined compression anomaly score
        compression_anomaly = 0.5 * ela_score + 0.35 * double_jpeg_score + 0.15 * prnu_inconsistency

        return {
            "compression_anomaly_score": float(np.clip(compression_anomaly, 0.0, 1.0)),
            "ela_score": ela_score,
            "double_jpeg_score": double_jpeg_score,
            "prnu_score": prnu_inconsistency
        }

