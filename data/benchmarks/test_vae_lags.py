import numpy as np
import cv2
from PIL import Image

def test_vae_lag_autocorrelation(path):
    img = Image.open(path).convert('L')
    gray = np.asarray(img, dtype=np.float32)
    h, w = gray.shape
    
    # High-pass filter (Laplacian or SRM)
    k = np.array([
        [-1,  2, -2,  2, -1],
        [ 2, -6,  8, -6,  2],
        [-2,  8,-12,  8, -2],
        [ 2, -6,  8, -6,  2],
        [-1,  2, -2,  2, -1]
    ], dtype=np.float32) / 12.0
    res = cv2.filter2D(gray, -1, k)
    
    # Normalize residual
    res = (res - np.mean(res)) / (np.std(res) + 1e-6)
    
    # Test autocorrelation at lags 1 to 16
    lags = [2, 4, 6, 8, 10, 12, 14, 16]
    corrs_x = {}
    corrs_y = {}
    for lag in lags:
        cx = np.mean(res[:, :-lag] * res[:, lag:])
        cy = np.mean(res[:-lag, :] * res[lag:, :])
        corrs_x[lag] = float(cx)
        corrs_y[lag] = float(cy)
        
    return corrs_x, corrs_y

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

for name, p in samples:
    cx, cy = test_vae_lag_autocorrelation(p)
    print(f"{name:25s} | lag4={cx[4]:.4f} | lag8={cx[8]:.4f} | lag16={cx[16]:.4f}")

