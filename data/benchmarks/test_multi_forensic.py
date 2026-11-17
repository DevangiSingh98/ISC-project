import numpy as np
import cv2
from PIL import Image
from scipy.fftpack import fft2, fftshift

def extract_multi_forensic_features(p):
    img = Image.open(p).convert('RGB')
    arr = np.asarray(img, dtype=np.float32)
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape

    # Feature 1: VAE periodic lattice peak in autocorrelation
    k_srm = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]], dtype=np.float32)/12.0
    res = cv2.filter2D(gray, -1, k_srm)
    norm_res = (res - np.mean(res)) / (np.std(res) + 1e-6)
    
    # Check max positive autocorrelation at lags [8, 16] (classic VAE downsampling strides)
    vae_lags = [8, 16]
    max_vae_corr = max([float(np.mean(norm_res[:, :-lag] * norm_res[:, lag:])) for lag in vae_lags] +
                       [float(np.mean(norm_res[:-lag, :] * norm_res[lag:, :])) for lag in vae_lags])
    
    # Feature 2: High-frequency cross-channel color correlation variance
    # In real cameras, demosaicing couples R, G, B tightly. In diffusion, channels exhibit subtle color shifts.
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    r_h = cv2.Laplacian(r, cv2.CV_32F)
    g_h = cv2.Laplacian(g, cv2.CV_32F)
    b_h = cv2.Laplacian(b, cv2.CV_32F)
    
    # Local high-frequency color disagreement:
    color_diff = np.abs(r_h - g_h) + np.abs(b_h - g_h)
    color_disagreement = float(np.mean(color_diff) / (np.mean(np.abs(g_h)) + 1e-6))
    
    # Feature 3: Blur / Gradient sharpness falloff (Diffusion smooth edge profile)
    # Neural decoders produce smooth sigmoid-like edge transitions without sensor MTF ringing
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    g_mag = np.sqrt(gx**2 + gy**2)
    edge_pixels = g_mag > 10.0
    if np.sum(edge_pixels) > 100:
        # Ratio of Laplacian magnitude to first derivative along edges
        lap_mag = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        edge_sharpness = float(np.mean(lap_mag[edge_pixels] / (g_mag[edge_pixels] + 1e-6)))
    else:
        edge_sharpness = 0.0

    # Feature 4: Texture local entropy / noise variation
    # Sliding window std dev of noise residual
    mean_sq = cv2.blur(res**2, (15, 15))
    sq_mean = cv2.blur(res, (15, 15))**2
    local_std = np.sqrt(np.maximum(mean_sq - sq_mean, 0))
    std_iqr = float(np.percentile(local_std, 75) - np.percentile(local_std, 25))
    
    # Feature 5: 2D FFT radial curvature
    f_shift = fftshift(fft2(cv2.resize(gray, (256, 256))))
    mag = np.log1p(np.abs(f_shift))
    center = 128
    y, x = np.ogrid[:256, :256]
    r_map = np.hypot(x - center, y - center).astype(np.int32)
    radial_mean = np.bincount(r_map.ravel(), mag.ravel()) / (np.bincount(r_map.ravel()) + 1e-6)
    r_sub = radial_mean[15:100]
    log_r = np.log(np.arange(len(r_sub)) + 15)
    log_p = np.log(r_sub + 1e-6)
    slope, intercept = np.polyfit(log_r, log_p, 1)
    p_dev = float(np.mean(np.abs(log_p - (slope * log_r + intercept))))

    return {
        "max_vae_corr": max_vae_corr,
        "color_disagree": color_disagreement,
        "edge_sharpness": edge_sharpness,
        "std_iqr": std_iqr,
        "p_dev": p_dev
    }

samples = [
    ("REAL Gaza", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357622_7e7ff603.jpg"),
    ("REAL Nadia", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937346098_9eed34f1.jpg"),
    ("REAL Aerial", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357646_225cc16f.jpg"),
    ("REAL Assembly", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357654_26db2ec3.jpg"),
    ("REAL Press", "data/test_samples/authentic_press.jpg"),
    ("AI Earthshot (ChatGPT)", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794938867451_a73fa26d.jpg"),
    ("AI Railway (ChatGPT)", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794938867468_6ba81fb5.jpg"),
    ("AI Synth Sample", "data/test_samples/ai_synthetic_diffusion.jpg")
]

print(f"{'IMAGE':25s} | {'VAE CORR':10s} | {'COLOR DISAGREE':15s} | {'EDGE SHARPNESS':15s} | {'STD IQR':10s} | {'P DEV':10s}")
print("-" * 95)
for name, p in samples:
    r = extract_multi_forensic_features(p)
    print(f"{name:25s} | {r['max_vae_corr']:10.4f} | {r['color_disagree']:15.4f} | {r['edge_sharpness']:15.4f} | {r['std_iqr']:10.4f} | {r['p_dev']:10.4f}")

