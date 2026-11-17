import os
from PIL import Image
from engines.crypto.c2pa_validator import C2PAValidator
from engines.forensics.manipulation.localizer import ManipulationLocalizer
from engines.forensics.noise_compression.compression_forensics import CompressionForensicsAnalyzer
from engines.fusion.rules_engine import FusionRulesEngine
from apps.api.app.schemas.verdict import ForensicsDetails

# Test combining the new synthetic detector logic with full fusion
from test_synthetic_ensemble import compute_synthetic_ensemble

c2pa = C2PAValidator()
localizer = ManipulationLocalizer()
comp = CompressionForensicsAnalyzer()
fusion = FusionRulesEngine()

all_test_imgs = [
    ("REAL Gaza", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357622_7e7ff603.jpg"),
    ("REAL Nadia", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937346098_9eed34f1.jpg"),
    ("REAL Aerial", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357646_225cc16f.jpg"),
    ("REAL Assembly", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357654_26db2ec3.jpg"),
    ("REAL Press", "data/test_samples/authentic_press.jpg"),
    ("AI Earthshot (ChatGPT)", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794938867451_a73fa26d.jpg"),
    ("AI Railway (ChatGPT)", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794938867468_6ba81fb5.jpg"),
    ("AI Synth Sample", "data/test_samples/ai_synthetic_diffusion.jpg"),
    ("Spliced Sample", "data/test_samples/spliced_manipulated.jpg")
]

print(f"{'IMAGE':25s} | {'VERDICT':35s} | {'CONF':6s} | {'SYNTH':6s} | {'MANIP':6s}")
print("-" * 85)
for name, p in all_test_imgs:
    raw = open(p, "rb").read()
    ext = os.path.splitext(p)[1]
    prov = c2pa.validate_bytes(raw, ext)
    img = Image.open(p)
    res_loc = localizer.generate_heatmap(img)
    res_comp = comp.analyze(img)
    
    syn_res = compute_synthetic_ensemble(p)
    synth_score = syn_res["combined"]
    
    forensics = ForensicsDetails(
        manipulation_score=res_loc[0],
        synthetic_score=synth_score,
        compression_anomaly_score=res_comp['compression_anomaly_score'],
        prnu_score=res_comp['prnu_score'],
        uncertainty_variance=0.001,
        is_degraded=False,
        is_document_screenshot=getattr(localizer, 'last_is_document_screenshot', False)
    )
    verdict, conf, expl, signals = fusion.evaluate(prov, forensics)
    print(f"{name:25s} | {verdict.value:35s} | {conf:5.1f}% | {synth_score:6.3f} | {res_loc[0]:6.3f}")

