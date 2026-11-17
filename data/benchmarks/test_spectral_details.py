import numpy as np
import cv2
from PIL import Image
from scipy.fftpack import fft2, fftshift, dct

def inspect_spectral_details(path):
    img = Image.open(path).convert('L').resize((256, 256))
    gray = np.asarray(img, dtype=np.float32)
    
    # 2D FFT
    f_shift = fftshift(fft2(gray))
    mag = np.log1p(np.abs(f_shift))
    
    # Radial profile
    center = 128
    y, x = np.ogrid[:256, :256]
    r = np.hypot(x - center, y - center).astype(np.int32)
    radial_mean = np.bincount(r.ravel(), mag.ravel()) / (np.bincount(r.ravel()) + 1e-6)
    
    # Radii 10 to 100
    radii = np.arange(len(radial_mean))[10:100]
    profile = radial_mean[10:100]
    log_r = np.log(radii)
    log_p = np.log(profile + 1e-6)
    
    # Linear fit
    slope, intercept = np.polyfit(log_r, log_p, 1)
    res = np.abs(log_p - (slope * log_r + intercept))
    
    # High-frequency curvature (second derivative of log_p w.r.t log_r)
    d2 = np.diff(log_p, 2)
    
    # Azimuthal anisotropy (variance across angles at fixed radius)
    # Natural photos have strong directional preference (horizontal/vertical scene structures)
    # Synthetic diffusion models often have unnaturally isotropic / circularly symmetric high frequencies
    angles = np.arctan2(y - center, x - center)
    hf_ring = (r >= 40) & (r <= 80)
    ring_angles = angles[hf_ring]
    ring_mags = mag[hf_ring]
    # Bin into 16 angular sectors
    angle_bins = np.linspace(-np.pi, np.pi, 17)
    sector_means = []
    for i in range(16):
        in_sector = (ring_angles >= angle_bins[i]) & (ring_angles < angle_bins[i+1])
        if np.sum(in_sector) > 10:
            sector_means.append(np.mean(ring_mags[in_sector]))
    angular_var = float(np.var(sector_means)) if len(sector_means) > 0 else 0.0
    
    # 2D DCT coefficient statistics (8x8 blocks)
    h, w = gray.shape
    dct_ac = []
    for i in range(0, h - 8 + 1, 8):
        for j in range(0, w - 8 + 1, 8):
            b = gray[i:i+8, j:j+8] - 128.0
            d = dct(dct(b.T, norm='ortho').T, norm='ortho')
            dct_ac.append(d[7, 7]) # Highest AC diagonal frequency
    ac_highest = np.array(dct_ac)
    hf_dct_kurt = float(np.mean(ac_highest**4) / (np.mean(ac_highest**2)**2 + 1e-6))
    hf_dct_std = float(np.std(ac_highest))
    
    return {
        "slope": float(slope),
        "res_mean": float(np.mean(res)),
        "angular_var": angular_var,
        "hf_dct_kurt": hf_dct_kurt,
        "hf_dct_std": hf_dct_std
    }

samples = [
    ("REAL Gaza", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357622_7e7ff603.jpg"),
    ("REAL Nadia", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937346098_9eed34f1.jpg"),
    ("REAL Aerial", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357646_225cc16f.jpg"),
    ("REAL Assembly", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357654_26db2ec3.jpg"),
    ("REAL Press", "data/test_samples/authentic_press.jpg"),
    ("AI Earthshot", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794938867451_a73fa26d.jpg"),
    ("AI Railway", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794938867468_6ba81fb5.jpg"),
    ("AI Synth Sample", "data/test_samples/ai_synthetic_diffusion.jpg")
]

print(f"{'NAME':18s} | {'SLOPE':8s} | {'RES MEAN':10s} | {'ANGULAR VAR':12s} | {'DCT KURT':10s} | {'DCT STD':10s}")
print("-" * 80)
for name, p in samples:
    r = inspect_spectral_details(p)
    print(f"{name:18s} | {r['slope']:8.3f} | {r['res_mean']:10.4f} | {r['angular_var']:12.4f} | {r['hf_dct_kurt']:10.2f} | {r['hf_dct_std']:10.3f}")

