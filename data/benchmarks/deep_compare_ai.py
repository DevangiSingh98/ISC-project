import numpy as np
import cv2
from PIL import Image
from scipy.fftpack import fft2, fftshift, dct

def deep_compare(path):
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img, dtype=np.float32)
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape

    # 1. Fourier 2D spectrum
    f_shift = fftshift(fft2(cv2.resize(gray, (256, 256))))
    mag = np.log1p(np.abs(f_shift))
    
    # Measure high-frequency energy ratio (energy in outer ring vs total)
    center = 128
    y, x = np.ogrid[:256, :256]
    r = np.hypot(x - center, y - center)
    hf_mask = (r > 60) & (r < 120)
    mf_mask = (r > 20) & (r <= 60)
    hf_energy = float(np.mean(mag[hf_mask]))
    mf_energy = float(np.mean(mag[mf_mask]))
    hf_to_mf_ratio = hf_energy / (mf_energy + 1e-6)

    # 2. Local noise residual statistics
    # In diffusion images, neural denoising creates non-Gaussian noise residuals in smooth regions
    blur = cv2.GaussianBlur(gray, (3, 3), 1.0)
    res = gray - blur
    
    # Skewness and kurtosis of noise residual
    from scipy.stats import kurtosis, skew
    res_kurt = float(kurtosis(res.flatten()))
    res_skew = float(skew(res.flatten()))
    
    # 3. Wavelet-like / Laplacian high-frequency variance distribution
    # In diffusion images, local variance has a very different spatial entropy
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    lap = cv2.filter2D(gray, -1, kernel)
    
    # Sliding window standard deviation of noise residual
    mean_res = cv2.blur(lap, (9, 9))
    sq_res = cv2.blur(lap**2, (9, 9))
    local_std = np.sqrt(np.maximum(sq_res - mean_res**2, 0))
    
    # Percentage of pixels with near-zero noise (dead zones / AI plastic smoothing)
    dead_zones = float(np.mean(local_std < 1.5))
    
    # 4. Color channel noise independence
    # Diffusion models generate R, G, B channels with unphysical decorrelation in fine details
    r_res = cv2.filter2D(arr[:,:,0], -1, kernel)
    g_res = cv2.filter2D(arr[:,:,1], -1, kernel)
    b_res = cv2.filter2D(arr[:,:,2], -1, kernel)
    rg_diff = np.abs(r_res - g_res)
    gb_diff = np.abs(g_res - b_res)
    color_noise_inconsistency = float(np.mean(rg_diff + gb_diff) / (np.mean(np.abs(g_res)) + 1e-6))
    
    # 5. Gradient histogram shape (Generalized Gaussian Distribution parameter beta)
    # Natural scene gradients follow GGD with beta < 1.0 (heavy-tailed)
    # AI generated images often have different tail decay
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    g_mag = np.sqrt(gx**2 + gy**2)
    g_kurt = float(kurtosis(g_mag.flatten()))
    
    return {
        "hf_mf_ratio": hf_to_mf_ratio,
        "res_kurt": res_kurt,
        "dead_zones": dead_zones,
        "color_noise_inconsistency": color_noise_inconsistency,
        "g_kurt": g_kurt
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

print(f"{'IMAGE':25s} | {'HF/MF RATIO':12s} | {'RES KURT':10s} | {'DEAD ZONES':12s} | {'COLOR NOISE':12s} | {'GRAD KURT':10s}")
print("-" * 90)
for name, p in samples:
    r = deep_compare(p)
    print(f"{name:25s} | {r['hf_mf_ratio']:12.4f} | {r['res_kurt']:10.2f} | {r['dead_zones']:12.4f} | {r['color_noise_inconsistency']:12.4f} | {r['g_kurt']:10.2f}")

