import numpy as np
import cv2
from PIL import Image

def measure_chromatic_aberration(img_path):
    img = Image.open(img_path).convert('RGB')
    arr = np.asarray(img, dtype=np.float32)
    h, w, _ = arr.shape
    
    # Analyze corners/periphery where physical lens dispersion causes R and B channel shifts
    cx, cy = w // 2, h // 2
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.hypot(x - cx, y - cy)
    max_dist = np.hypot(cx, cy)
    # Outer 30% of the image
    outer_mask = dist_from_center > 0.65 * max_dist
    
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    
    # High-pass edge locations in Green channel
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    g_edges = np.sqrt(gx**2 + gy**2)
    
    # Significant edges in the outer periphery
    edge_mask = outer_mask & (g_edges > 15.0)
    if np.sum(edge_mask) < 50:
        return 0.0, 0.0
    
    # Normalized radial gradient vectors from center
    rx = (x - cx) / (dist_from_center + 1e-6)
    ry = (y - cy) / (dist_from_center + 1e-6)
    
    # Radial directional derivative for R, G, B
    rx_r = cv2.Sobel(r, cv2.CV_32F, 1, 0, ksize=3)
    ry_r = cv2.Sobel(r, cv2.CV_32F, 0, 1, ksize=3)
    rad_deriv_r = rx_r * rx + ry_r * ry
    
    rx_b = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
    ry_b = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
    rad_deriv_b = rx_b * rx + ry_b * ry
    
    rad_deriv_g = gx * rx + gy * ry
    
    # Lens chromatic dispersion: (R - B) radial gradient difference along edges
    dispersion = np.abs(rad_deriv_r - rad_deriv_b)[edge_mask]
    mean_dispersion = float(np.mean(dispersion))
    dispersion_ratio = float(mean_dispersion / (np.mean(np.abs(rad_deriv_g)[edge_mask]) + 1e-6))
    
    return mean_dispersion, dispersion_ratio

samples = [
    ("REAL UN Gaza", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357622_7e7ff603.jpg"),
    ("REAL UN Nadia", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937346098_9eed34f1.jpg"),
    ("REAL UN Aerial", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357646_225cc16f.jpg"),
    ("REAL UN Assembly", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357654_26db2ec3.jpg"),
    ("AI Earthshot (ChatGPT)", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794938867451_a73fa26d.jpg"),
    ("AI Railway (ChatGPT)", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794938867468_6ba81fb5.jpg"),
]

for name, p in samples:
    disp, ratio = measure_chromatic_aberration(p)
    print(f"{name:25s} | Dispersion: {disp:8.3f} | Ratio: {ratio:8.3f}")

