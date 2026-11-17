import numpy as np
import cv2
from PIL import Image

def compute_synthetic_ensemble(path):
    img = Image.open(path).convert('RGB')
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape

    # 1. Spectral FFT branch (Checkerboard / Periodic Spikes)
    from scipy.fftpack import fft2, fftshift
    f_shift = fftshift(fft2(cv2.resize(gray, (256, 256))))
    mag = np.log1p(np.abs(f_shift))
    center = 128
    mask = np.ones((256, 256), dtype=bool)
    mask[center-20:center+20, center-20:center+20] = False
    mask[:, center-3:center+4] = False
    mask[center-3:center+4, :] = False
    vals = mag[mask]
    z = (vals - np.mean(vals)) / (np.std(vals) + 1e-6)
    spikes = int(np.sum(z > 5.0))
    spike_score = min(1.0, spikes / 4.0)

    # 2. VAE Periodic Autocorrelation (Latent Diffusion 8x/16x decoder footprint)
    k_srm = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]], dtype=np.float32)/12.0
    res = cv2.filter2D(gray, -1, k_srm)
    norm_res = (res - np.mean(res)) / (np.std(res) + 1e-6)
    
    vae_lags = [8, 16]
    corrs = [float(np.mean(norm_res[:, :-lag] * norm_res[:, lag:])) for lag in vae_lags] + \
            [float(np.mean(norm_res[:-lag, :] * norm_res[lag:, :])) for lag in vae_lags]
    max_vae_corr = max(corrs)
    vae_score = float(np.clip((max_vae_corr - 0.035) / 0.05, 0.0, 1.0))

    # 3. Diffusion Residual Noise-to-Luminance Coupling
    b = 16
    lum_blocks = []
    noise_blocks = []
    for i in range(0, h - b + 1, b):
        for j in range(0, w - b + 1, b):
            lum_blocks.append(np.mean(gray[i:i+b, j:j+b]))
            noise_blocks.append(np.var(res[i:i+b, j:j+b]))
    shot_corr = float(np.corrcoef(lum_blocks, noise_blocks)[0, 1])
    diff_noise_score = float(np.clip((shot_corr - 0.22) / 0.12, 0.0, 1.0))

    # Combined synthetic detector score
    # Any strong physical signature of generative models raises the synthetic score
    combined_score = max(spike_score * 0.85, vae_score * 0.80, diff_noise_score * 0.78)
    
    return {
        "combined": combined_score,
        "spike_score": spike_score,
        "max_vae_corr": max_vae_corr,
        "vae_score": vae_score,
        "shot_corr": shot_corr,
        "diff_noise_score": diff_noise_score
    }

samples = [
    ("REAL Gaza", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357622_7e7ff603.jpg"),
    ("REAL Nadia", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937346098_9eed34f1.jpg"),
    ("REAL Aerial", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357646_225cc16f.jpg"),
    ("REAL Assembly", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357654_26db2ec3.jpg"),
    ("REAL Press Sample", "data/test_samples/authentic_press.jpg"),
    ("AI Earthshot (ChatGPT)", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794938867451_a73fa26d.jpg"),
    ("AI Railway (ChatGPT)", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794938867468_6ba81fb5.jpg"),
    ("AI Synth Sample", "data/test_samples/ai_synthetic_diffusion.jpg")
]

print(f"{'IMAGE':23s} | {'COMBINED':10s} | {'SPIKES':10s} | {'VAE CORR':10s} | {'VAE SCORE':10s} | {'SHOT CORR':10s} | {'DIFF SCORE':10s}")
print("-" * 105)
for name, p in samples:
    r = compute_synthetic_ensemble(p)
    print(f"{name:23s} | {r['combined']:10.3f} | {r['spike_score']:10.3f} | {r['max_vae_corr']:10.4f} | {r['vae_score']:10.3f} | {r['shot_corr']:10.4f} | {r['diff_noise_score']:10.3f}")

