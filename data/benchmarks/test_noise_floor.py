import numpy as np
import cv2
from PIL import Image

def measure_sensor_noise_floor(path):
    img = Image.open(path).convert('L')
    gray = np.asarray(img, dtype=np.float32)
    h, w = gray.shape

    # SRM high-pass residual
    k = np.array([
        [-1,  2, -2,  2, -1],
        [ 2, -6,  8, -6,  2],
        [-2,  8,-12,  8, -2],
        [ 2, -6,  8, -6,  2],
        [-1,  2, -2,  2, -1]
    ], dtype=np.float32) / 12.0
    res = cv2.filter2D(gray, -1, k)
    
    # Structural gradient to find truly smooth background regions
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    
    # Smooth areas: gradient < 4.0
    smooth_mask = grad < 4.0
    smooth_fraction = float(np.mean(smooth_mask))
    
    if np.sum(smooth_mask) > 500:
        smooth_res = res[smooth_mask]
        noise_variance = float(np.var(smooth_res))
        noise_mean_abs = float(np.mean(np.abs(smooth_res)))
    else:
        noise_variance = 0.0
        noise_mean_abs = 0.0
        
    # High texture regions: gradient > 20.0
    tex_mask = grad > 20.0
    if np.sum(tex_mask) > 500:
        tex_res = res[tex_mask]
        tex_noise_var = float(np.var(tex_res))
    else:
        tex_noise_var = 0.0
        
    # Ratio of texture noise to smooth noise (dynamic range of residual)
    dynamic_ratio = tex_noise_var / (noise_variance + 1e-4)

    return {
        "smooth_frac": smooth_fraction,
        "smooth_var": noise_variance,
        "smooth_abs": noise_mean_abs,
        "tex_var": tex_noise_var,
        "dynamic_ratio": dynamic_ratio
    }

samples = [
    ("REAL UN Gaza", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357622_7e7ff603.jpg"),
    ("REAL UN Nadia", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937346098_9eed34f1.jpg"),
    ("REAL UN Aerial", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357646_225cc16f.jpg"),
    ("REAL UN Assembly", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357654_26db2ec3.jpg"),
    ("REAL Press", "data/test_samples/authentic_press.jpg"),
    ("AI Earthshot (ChatGPT)", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794938867451_a73fa26d.jpg"),
    ("AI Railway (ChatGPT)", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794938867468_6ba81fb5.jpg"),
    ("AI Synth Sample", "data/test_samples/ai_synthetic_diffusion.jpg")
]

print(f"{'IMAGE':25s} | {'SMOOTH FRAC':12s} | {'SMOOTH VAR':12s} | {'SMOOTH ABS':12s} | {'TEX VAR':10s} | {'RATIO':10s}")
print("-" * 90)
for name, p in samples:
    r = measure_sensor_noise_floor(p)
    print(f"{name:25s} | {r['smooth_frac']:12.4f} | {r['smooth_var']:12.4f} | {r['smooth_abs']:12.4f} | {r['tex_var']:10.2f} | {r['dynamic_ratio']:10.2f}")

