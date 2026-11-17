import numpy as np
import cv2
from PIL import Image
from scipy.stats import kurtosis, skew

def full_profile(path):
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img, dtype=np.float32)
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY).astype(np.float32)

    # 1. Color gradient correlation
    # In real photos, hue and saturation have high spatial entropy at edges.
    # In AI generation, color transitions are smoothly blended by neural upsampling.
    hsv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2HSV).astype(np.float32)
    h_chan, s_chan, v_chan = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    
    # Gradient of Saturation vs Gradient of Value
    gx_s = cv2.Sobel(s_chan, cv2.CV_32F, 1, 0)
    gy_s = cv2.Sobel(s_chan, cv2.CV_32F, 0, 1)
    mag_s = np.sqrt(gx_s**2 + gy_s**2)
    
    gx_v = cv2.Sobel(v_chan, cv2.CV_32F, 1, 0)
    gy_v = cv2.Sobel(v_chan, cv2.CV_32F, 0, 1)
    mag_v = np.sqrt(gx_v**2 + gy_v**2)
    
    # Correlation between saturation gradient and luminance gradient:
    # Real camera optics (vignetting, Fresnel reflection) have different S-V gradient coupling than diffusion decoders.
    corr_sv = float(np.corrcoef(mag_s.flatten(), mag_v.flatten())[0, 1])

    # 2. Local noise uniformity (smooth regions vs textured regions)
    # Neural denoisers in diffusion models create an unnaturally high ratio of texture-to-flat noise
    k_srm = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]], dtype=np.float32)/12.0
    res = cv2.filter2D(gray, -1, k_srm)
    
    # 3. Channel residual variance ratio: Var(R_res) / Var(G_res) and Var(B_res) / Var(G_res)
    res_r = cv2.filter2D(arr[:,:,0], -1, k_srm)
    res_g = cv2.filter2D(arr[:,:,1], -1, k_srm)
    res_b = cv2.filter2D(arr[:,:,2], -1, k_srm)
    var_r = float(np.var(res_r))
    var_g = float(np.var(res_g))
    var_b = float(np.var(res_b))
    channel_noise_ratio = (var_r + var_b) / (2.0 * var_g + 1e-6)

    # 4. Color Co-occurrence (Pairs of adjacent pixels in R-B plane)
    r_diff = np.abs(np.diff(arr[:,:,0], axis=1))
    b_diff = np.abs(np.diff(arr[:,:,2], axis=1))
    rb_diff_ratio = float(np.mean(r_diff) / (np.mean(b_diff) + 1e-6))
    
    # 5. High-frequency noise entropy
    hist, _ = np.histogram(res.flatten(), bins=100, range=(-20, 20), density=True)
    hist = hist[hist > 0]
    res_entropy = -float(np.sum(hist * np.log2(hist)))

    return {
        "corr_sv": corr_sv,
        "channel_noise_ratio": channel_noise_ratio,
        "rb_diff_ratio": rb_diff_ratio,
        "res_entropy": res_entropy
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

print(f"{'IMAGE':25s} | {'CORR SV':10s} | {'CH RATIO':10s} | {'RB RATIO':10s} | {'ENTROPY':10s}")
print("-" * 75)
for name, p in samples:
    r = full_profile(p)
    print(f"{name:25s} | {r['corr_sv']:10.4f} | {r['channel_noise_ratio']:10.4f} | {r['rb_diff_ratio']:10.4f} | {r['res_entropy']:10.4f}")

