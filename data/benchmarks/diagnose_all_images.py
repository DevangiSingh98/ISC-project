"""Diagnose all user-uploaded images through the full pipeline."""
import sys, os
import numpy as np
import cv2
sys.path.insert(0, r"c:\Users\singh\Desktop\ISC Project")
from PIL import Image
from engines.forensics.synthetic.detector import AIGeneratedImageDetector
from engines.forensics.manipulation.localizer import ManipulationLocalizer
from engines.forensics.noise_compression.compression_forensics import CompressionForensicsAnalyzer

synth = AIGeneratedImageDetector()
local = ManipulationLocalizer()
comp = CompressionForensicsAnalyzer()

images = {
    "REAL Gaza Refugee Camp": r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357622_7e7ff603.jpg",
    "REAL Nadia Bookstore":   r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937346098_9eed34f1.jpg",
    "REAL Aerial Highway":    r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357646_225cc16f.jpg",
    "REAL UN Assembly":       r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357654_26db2ec3.jpg",
    "AI ChatGPT Earthshot":   r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794938867451_a73fa26d.jpg",
    "AI ChatGPT Railway":     r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794938867468_6ba81fb5.jpg",
}

print("=" * 130)
print(f"{'Image':<30} {'manip':>8} {'synth':>8} {'spect':>8} {'comp':>8} {'diffuse':>8} {'clust_r':>8} {'max_cr':>8} {'vae_sc':>8} {'diff_ns':>8} {'shot_c':>8}")
print("=" * 130)

k_srm = np.array([
    [-1,  2, -2,  2, -1],
    [ 2, -6,  8, -6,  2],
    [-2,  8,-12,  8, -2],
    [ 2, -6,  8, -6,  2],
    [-1,  2, -2,  2, -1]
], dtype=np.float32) / 12.0

for label, path in images.items():
    if not os.path.exists(path):
        print(f"{label:<30} FILE NOT FOUND")
        continue
    img = Image.open(path)
    
    # Synthetic analysis
    s = synth.analyze(img)
    
    # Manipulation analysis
    manip_score, _, _ = local.generate_heatmap(img)
    is_diffuse = getattr(local, 'last_is_diffuse', 'N/A')
    max_comp_ratio = getattr(local, 'last_max_comp_ratio', 'N/A')
    cluster_ratio = getattr(local, 'last_cluster_ratio', 'N/A')
    
    # Compression analysis
    c = comp.analyze(img)
    comp_score = c.get("double_jpeg_score", 0)
    
    # Get VAE and diff noise internals
    orig_gray = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY).astype(np.float32)
    h_o, w_o = orig_gray.shape
    if max(h_o, w_o) > 1024:
        scale_fac = 1024.0 / max(h_o, w_o)
        orig_gray = cv2.resize(orig_gray, (int(w_o * scale_fac), int(h_o * scale_fac)))
    
    srm_res = cv2.filter2D(orig_gray, -1, k_srm)
    norm_res = (srm_res - np.mean(srm_res)) / (np.std(srm_res) + 1e-6)
    
    vae_lags = [8, 16]
    corrs = [float(np.mean(norm_res[:, :-lag] * norm_res[:, lag:])) for lag in vae_lags] + \
            [float(np.mean(norm_res[:-lag, :] * norm_res[lag:, :])) for lag in vae_lags]
    max_vae_corr = max(corrs)
    vae_score = float(np.clip((max_vae_corr - 0.035) / 0.05, 0.0, 1.0))
    
    b_sz = 16
    lum_blocks = []
    noise_blocks = []
    h_g, w_g = orig_gray.shape
    for bi in range(0, h_g - b_sz + 1, b_sz):
        for bj in range(0, w_g - b_sz + 1, b_sz):
            lum_blocks.append(np.mean(orig_gray[bi:bi+b_sz, bj:bj+b_sz]))
            noise_blocks.append(np.var(srm_res[bi:bi+b_sz, bj:bj+b_sz]))
    shot_corr = float(np.corrcoef(lum_blocks, noise_blocks)[0, 1]) if len(lum_blocks) > 10 else 0.0
    diff_noise_score = float(np.clip((shot_corr - 0.22) / 0.12, 0.0, 1.0))
    
    print(f"{label:<30} {manip_score:>8.3f} {s['synthetic_score']:>8.3f} {s['spectral_score']:>8.3f} {comp_score:>8.3f} {str(is_diffuse):>8} {str(cluster_ratio)[:8]:>8} {str(max_comp_ratio)[:8]:>8} {vae_score:>8.3f} {diff_noise_score:>8.3f} {shot_corr:>8.3f}")

print("\n" + "=" * 130)
print("\nFusion verdict simulation:")
print("-" * 100)
for label, path in images.items():
    if not os.path.exists(path):
        continue
    img = Image.open(path)
    s = synth.analyze(img)
    manip_score, _, _ = local.generate_heatmap(img)
    c = comp.analyze(img)
    
    synth_sc = s['synthetic_score']
    comp_sc = c.get("double_jpeg_score", 0)
    
    is_corroborated = (
        (manip_score >= 0.55 and comp_sc >= 0.38) or
        (manip_score >= 0.75)
    )
    
    if is_corroborated and (manip_score >= synth_sc - 0.10):
        verdict = "SUSPICIOUS / MANIPULATED"
    elif synth_sc >= 0.55:
        verdict = "AI-GENERATED"
    elif manip_score >= 0.35:
        verdict = "LIKELY AUTHENTIC (texture)"
    else:
        verdict = "LIKELY AUTHENTIC (clean)"
    
    print(f"  {label:<30} manip={manip_score:.3f} synth={synth_sc:.3f} comp={comp_sc:.3f} corr={is_corroborated} → {verdict}")
