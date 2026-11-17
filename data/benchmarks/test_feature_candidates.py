import numpy as np
import cv2
import io
from PIL import Image, ImageChops
from scipy.fftpack import fft2, fftshift

def test_feature_candidates(p):
    img = Image.open(p).convert('RGB')
    arr = np.asarray(img, dtype=np.float32)
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape

    # 1. ELA
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=90)
    buf.seek(0)
    resaved = Image.open(buf)
    diff = np.mean(np.asarray(ImageChops.difference(img, resaved), dtype=np.float32), axis=2)
    ela_mean = float(np.mean(diff))

    # 2. Camera Noise Residual / Steganographic Features (Spatial Rich Models residuals)
    k = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]], dtype=np.float32)/12.0
    res = cv2.filter2D(gray, -1, k)
    
    # 16x16 block stats
    b = 16
    lum_blocks = []
    noise_blocks = []
    for i in range(0, h - b + 1, b):
        for j in range(0, w - b + 1, b):
            lum = np.mean(gray[i:i+b, j:j+b])
            n_var = np.var(res[i:i+b, j:j+b])
            lum_blocks.append(lum)
            noise_blocks.append(n_var)
    shot_noise_corr = float(np.corrcoef(lum_blocks, noise_blocks)[0, 1])

    # 3. High-frequency Color Demosaicing Residual
    r, g, b_ch = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    cfa_residual = g - 0.5 * (r + b_ch)
    cfa_high = cv2.filter2D(cfa_residual, -1, k)
    cfa_kurt = float(np.mean(cfa_high**4) / (np.mean(cfa_high**2)**2 + 1e-6))
    cfa_energy = float(np.mean(cfa_high**2))

    # 4. Saturation-to-Luminance gradient ratio
    hsv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[:,:,1]
    gx_s = cv2.Sobel(sat, cv2.CV_32F, 1, 0)
    gy_s = cv2.Sobel(sat, cv2.CV_32F, 0, 1)
    mag_s = np.sqrt(gx_s**2 + gy_s**2)
    
    gx_v = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy_v = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag_v = np.sqrt(gx_v**2 + gy_v**2)
    
    sat_edge_ratio = float(np.mean(mag_s) / (np.mean(mag_v) + 1e-6))

    # 5. High-frequency Spectral Residual Falloff
    f_shift = fftshift(fft2(cv2.resize(gray, (256, 256))))
    spec_mag = np.abs(f_shift)
    center = 128
    y, x = np.ogrid[:256, :256]
    r_dist = np.hypot(x - center, y - center)
    nyq_mid = np.mean(spec_mag[(r_dist >= 40) & (r_dist < 80)])
    nyq_high = np.mean(spec_mag[(r_dist >= 80) & (r_dist < 120)])
    nyq_ratio = float(nyq_high / (nyq_mid + 1e-6))

    return {
        "shot_noise_corr": shot_noise_corr,
        "cfa_kurt": cfa_kurt,
        "cfa_energy": cfa_energy,
        "sat_edge_ratio": sat_edge_ratio,
        "nyq_ratio": nyq_ratio
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

print(f"{'IMAGE':23s} | {'SHOT NOISE':10s} | {'CFA KURT':10s} | {'CFA ENERGY':10s} | {'SAT RATIO':10s} | {'NYQ RATIO':10s}")
print("-" * 85)
for name, p in samples:
    r = test_feature_candidates(p)
    print(f"{name:23s} | {r['shot_noise_corr']:10.4f} | {r['cfa_kurt']:10.2f} | {r['cfa_energy']:10.2f} | {r['sat_edge_ratio']:10.4f} | {r['nyq_ratio']:10.4f}")
