import io
import base64
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image

class ManipulationLocalizer:
    """
    Manipulation Localizer inspired by Cat-Net and TruFor architectures.
    Extracts high-pass noise residuals (SRM filters) combined with RGB domain
    gradients to detect spliced, inpainted, or copy-moved regions.
    Produces both a global manipulation score and a pixel-level HxW heatmap.
    """
    def __init__(self):
        # SRM 3-kernel high-pass filter bank
        self.srm_kernel_1 = np.array([
            [0,  0,  0,  0, 0],
            [0, -1,  2, -1, 0],
            [0,  2, -4,  2, 0],
            [0, -1,  2, -1, 0],
            [0,  0,  0,  0, 0]
        ], dtype=np.float32) / 4.0

        self.srm_kernel_2 = np.array([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8,-12,  8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=np.float32) / 12.0

        self.srm_kernel_3 = np.array([
            [0,  0, -1,  0, 0],
            [0,  0,  3,  0, 0],
            [-1, 3, -8,  3, -1],
            [0,  0,  3,  0, 0],
            [0,  0, -1,  0, 0]
        ], dtype=np.float32) / 8.0

    def extract_noise_residual(self, gray: np.ndarray) -> np.ndarray:
        """Convolves image with SRM high-pass filter bank to extract noise residuals."""
        r1 = cv2.filter2D(gray, -1, self.srm_kernel_1)
        r2 = cv2.filter2D(gray, -1, self.srm_kernel_2)
        r3 = cv2.filter2D(gray, -1, self.srm_kernel_3)
        combined = (np.abs(r1) + np.abs(r2) + np.abs(r3)) / 3.0
        return combined

    def generate_heatmap(self, image: Image.Image) -> Tuple[float, np.ndarray, str]:
        """
        Analyzes the image for localized splicing, copy-move, or inpainting.
        Returns:
            (global_manipulation_score, raw_mask_HxW, base64_rgba_heatmap)
        """
        rgb_img = np.asarray(image.convert("RGB"), dtype=np.uint8)
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h, w = gray.shape

        # 1. Noise residual domain
        noise_res = self.extract_noise_residual(gray)

        # 2. Local noise inconsistency: sliding window standard deviation
        kernel_size = 15
        mean = cv2.blur(noise_res, (kernel_size, kernel_size))
        sq_mean = cv2.blur(noise_res ** 2, (kernel_size, kernel_size))
        variance = np.maximum(sq_mean - mean ** 2, 0)
        local_std = np.sqrt(variance)

        # 3. Spatial edge inconsistency (detects boundary seams from pasting)
        edges = cv2.Canny((gray).astype(np.uint8), 50, 150).astype(np.float32) / 255.0
        edge_density = cv2.GaussianBlur(edges, (21, 21), 5)

        # Normalize local noise anomaly map
        global_median = np.median(local_std)
        global_mad = np.median(np.abs(local_std - global_median)) + 1e-5
        z_scores = np.abs(local_std - global_median) / global_mad

        # Combine noise anomaly and boundary gradient
        raw_anomaly = z_scores * 0.7 + edge_density * 3.0 * 0.3
        
        # Smooth and scale to [0, 1] probability map
        smoothed = cv2.GaussianBlur(raw_anomaly, (11, 11), 3)
        norm_map = 1.0 / (1.0 + np.exp(-(smoothed - 2.5))) # Sigmoid transition centered at z=2.5

        # Calculate global manipulation score based on top percentile of tampered pixels
        # (Splicing typically affects 5% to 30% of image area)
        k_percentile = 98.0
        top_anomaly_val = float(np.percentile(norm_map, k_percentile))
        mean_top_anomaly = float(np.mean(norm_map[norm_map >= top_anomaly_val]))
        
        # Fraction of pixels showing elevated tampering confidence (> 0.6)
        tamper_area_ratio = float(np.mean(norm_map > 0.6))
        
        global_score = min(1.0, max(0.0, 0.75 * mean_top_anomaly + 0.25 * min(1.0, tamper_area_ratio * 5.0)))

        # Create RGBA heatmap overlay for display
        # Jet colormap: Blue (clean) -> Cyan -> Yellow -> Red (tampered)
        heatmap_color = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Alpha channel: transparent where score is low, opaque red where score is high
        alpha = (np.clip((norm_map - 0.2) / 0.8, 0.0, 1.0) * 180).astype(np.uint8)
        rgba = np.dstack((heatmap_rgb, alpha))

        # Encode to Base64 PNG
        heatmap_pil = Image.fromarray(rgba, mode="RGBA")
        buffer = io.BytesIO()
        heatmap_pil.save(buffer, format="PNG")
        base64_heatmap = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"

        return global_score, norm_map, base64_heatmap
