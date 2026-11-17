import numpy as np
import cv2
from PIL import Image

def analyze_forensic_features(img_path):
    img = Image.open(img_path).convert('RGB')
    arr = np.asarray(img, dtype=np.float32)
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    
    # 1. High-pass residual (Laplacian & SRM)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_var = np.var(lap)
    
    # Noise floor in smooth regions: find flat patches (low gradient) and measure residual
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    flat_mask = grad_mag < 3.0
    if np.sum(flat_mask) > 100:
        flat_noise_var = float(np.var(lap[flat_mask]))
    else:
        flat_noise_var = 0.0
        
    # 2. Cross-channel correlation (Bayer filter vs Diffusion)
    # In real cameras, R, G, B are interpolated from Bayer CFAs, creating strong high-frequency correlation.
    # In diffusion models, channels often have independent high-frequency noise or unphysical cross-channel variance.
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    res_r = cv2.Laplacian(r, cv2.CV_32F)
    res_g = cv2.Laplacian(g, cv2.CV_32F)
    res_b = cv2.Laplacian(b, cv2.CV_32F)
    corr_rg = float(np.corrcoef(res_r.flatten(), res_g.flatten())[0, 1])
    corr_gb = float(np.corrcoef(res_g.flatten(), res_b.flatten())[0, 1])
    corr_rb = float(np.corrcoef(res_r.flatten(), res_b.flatten())[0, 1])
    mean_cross_corr = (corr_rg + corr_gb + corr_rb) / 3.0
    
    # 3. Kurtosis and entropy of high-pass noise
    from scipy.stats import kurtosis, skew
    kurt = float(kurtosis(lap.flatten()))
    
    # 4. Saturation and color distribution
    hsv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2HSV)
    sat = hsv[:,:,1]
    val = hsv[:,:,2]
    mean_sat = float(np.mean(sat))
    
    # 5. Gradient histogram ratio (diffusion images often have distinct edge softness / halo transitions)
    # Ratio of fine gradients (2-8) to sharp gradients (>20)
    fine_edges = np.sum((grad_mag >= 2) & (grad_mag <= 8))
    sharp_edges = np.sum(grad_mag > 20)
    edge_ratio = float(fine_edges) / float(max(sharp_edges, 1))

    return {
        "lap_var": lap_var,
        "flat_noise_var": flat_noise_var,
        "cross_corr": mean_cross_corr,
        "kurtosis": kurt,
        "mean_sat": mean_sat,
        "edge_ratio": edge_ratio
    }

samples = [
    ("REAL UN Gaza", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357622_7e7ff603.jpg"),
    ("REAL UN Nadia", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937346098_9eed34f1.jpg"),
    ("REAL UN Aerial", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357646_225cc16f.jpg"),
    ("REAL UN Assembly", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357654_26db2ec3.jpg"),
    ("REAL Press Sample", "data/test_samples/authentic_press.jpg"),
    ("AI Earthshot (ChatGPT)", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794938867451_a73fa26d.jpg"),
    ("AI Railway (ChatGPT)", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794938867468_6ba81fb5.jpg"),
    ("AI Diffusion Sample", "data/test_samples/ai_synthetic_diffusion.jpg")
]

print(f"{'IMAGE':25s} | {'FLAT NOISE':10s} | {'CROSS CORR':10s} | {'KURTOSIS':10s} | {'EDGE RATIO':10s} | {'LAP VAR':10s}")
print("-" * 85)
for name, p in samples:
    r = analyze_forensic_features(p)
    print(f"{name:25s} | {r['flat_noise_var']:10.4f} | {r['cross_corr']:10.4f} | {r['kurtosis']:10.2f} | {r['edge_ratio']:10.4f} | {r['lap_var']:10.2f}")

