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
        Includes document/webpage screenshot detection and high-contrast font glyph
        attenuation to prevent mistaking text layout for camera sensor tampering.
        Returns:
            (global_manipulation_score, raw_mask_HxW, base64_rgba_heatmap)
        """
        rgb_img = np.asarray(image.convert("RGB"), dtype=np.uint8)
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h, w = gray.shape

        # 1. Noise residual domain
        noise_res = self.extract_noise_residual(gray)

        # 2. Check for digital document / webpage screenshot canvas (near white > 235 or near black < 18)
        flat_canvas = (gray > 235) | (gray < 18)
        flat_ratio = float(np.mean(flat_canvas))
        is_doc_screenshot = flat_ratio > 0.35
        self.last_is_document_screenshot = is_doc_screenshot

        # 3. Compute image gradients for structural edge attenuation
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)

        if is_doc_screenshot:
            # Mask out flat canvas and high-contrast font glyph strokes
            kernel_font = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated_flat = cv2.dilate(flat_canvas.astype(np.uint8), kernel_font) > 0
            text_or_flat = flat_canvas | (dilated_flat & (grad_mag > 20))
            valid_pixels = ~text_or_flat
        else:
            text_or_flat = np.zeros((h, w), dtype=bool)
            valid_pixels = np.ones((h, w), dtype=bool)

        # 4. Local noise inconsistency: sliding window standard deviation
        kernel_size = 15
        mean = cv2.blur(noise_res, (kernel_size, kernel_size))
        sq_mean = cv2.blur(noise_res ** 2, (kernel_size, kernel_size))
        variance = np.maximum(sq_mean - mean ** 2, 0)
        local_std = np.sqrt(variance)

        if np.sum(valid_pixels) < 256:
            # Degraded or empty document
            empty_map = np.zeros((h, w), dtype=np.float32)
            return 0.15, empty_map, "data:image/png;base64,"

        # Compute robust statistics ONLY on valid photographic pixels
        valid_std = local_std[valid_pixels]
        global_median = float(np.median(valid_std))
        global_mad = float(np.median(np.abs(valid_std - global_median)))
        global_mad = max(global_mad, 0.8) # Sensor noise floor prevents division by near-zero!

        z_scores = np.zeros((h, w), dtype=np.float32)
        z_scores[valid_pixels] = np.abs(local_std[valid_pixels] - global_median) / global_mad

        # Combine noise anomaly and boundary gradient
        if is_doc_screenshot:
            raw_anomaly = z_scores
        else:
            edges = cv2.Canny(gray.astype(np.uint8), 50, 150).astype(np.float32) / 255.0
            edge_density = cv2.GaussianBlur(edges, (21, 21), 5)
            raw_anomaly = z_scores * 0.7 + edge_density * 3.0 * 0.3

        # Clustered anomaly detection (morphological opening eliminates isolated edge artifacts)
        high_z = (z_scores > 2.8).astype(np.uint8)
        kernel_cl = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        clusters = cv2.morphologyEx(high_z, cv2.MORPH_OPEN, kernel_cl)

        # Smooth and scale to [0, 1] probability map for heatmap visualization
        smoothed = cv2.GaussianBlur(z_scores, (11, 11), 3)
        norm_map = 1.0 / (1.0 + np.exp(-(smoothed - 2.8)))
        norm_map[text_or_flat] = 0.0 # Suppress flat canvas & typography in heatmap overlay

        # Calculate cluster ratio of valid photographic pixels
        total_valid = np.sum(valid_pixels)
        cluster_ratio = float(np.sum(clusters)) / float(max(total_valid, 1))

        # Manipulation score driven by contiguous tamper cluster area
        if cluster_ratio < 0.015:
            global_score = float(np.clip(cluster_ratio * 15.0, 0.0, 0.30))
        else:
            global_score = float(np.clip(0.52 + cluster_ratio * 4.5, 0.52, 0.98))

        if is_doc_screenshot and cluster_ratio < 0.04:
            global_score = min(global_score, 0.20)

        # Create RGBA heatmap overlay for display
        heatmap_color = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Alpha channel: transparent where score is low, opaque where score is high
        alpha = (np.clip((norm_map - 0.25) / 0.75, 0.0, 1.0) * 180).astype(np.uint8)
        alpha[text_or_flat] = 0 # Keep text and flat canvas 100% transparent
        rgba = np.dstack((heatmap_rgb, alpha))

        # Encode to Base64 PNG
        heatmap_pil = Image.fromarray(rgba, mode="RGBA")
        buffer = io.BytesIO()
        heatmap_pil.save(buffer, format="PNG")
        base64_heatmap = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"

        return global_score, norm_map, base64_heatmap

