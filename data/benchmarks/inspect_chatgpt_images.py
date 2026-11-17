import os
from PIL import Image
from engines.crypto.c2pa_validator import C2PAValidator
from engines.forensics.synthetic.detector import AIGeneratedImageDetector
from engines.forensics.manipulation.localizer import ManipulationLocalizer
from engines.forensics.noise_compression.compression_forensics import CompressionForensicsAnalyzer
from engines.fusion.rules_engine import FusionRulesEngine
from apps.api.app.schemas.verdict import ForensicsDetails

c2pa = C2PAValidator()
detector = AIGeneratedImageDetector()
localizer = ManipulationLocalizer()
comp = CompressionForensicsAnalyzer()
fusion = FusionRulesEngine()

img1_path = r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794938867451_a73fa26d.jpg"
img2_path = r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794938867468_6ba81fb5.jpg"

for label, p in [("Earthshot AI", img1_path), ("Railway AI", img2_path)]:
    raw = open(p, "rb").read()
    ext = os.path.splitext(p)[1]
    prov = c2pa.validate_bytes(raw, ext)
    img = Image.open(p)
    res_loc = localizer.generate_heatmap(img)
    res_syn = detector.analyze(img)
    res_comp = comp.analyze(img)
    
    forensics = ForensicsDetails(
        manipulation_score=res_loc[0],
        synthetic_score=res_syn['synthetic_score'],
        compression_anomaly_score=res_comp['compression_anomaly_score'],
        prnu_score=res_comp['prnu_score'],
        uncertainty_variance=res_syn['uncertainty_variance'],
        is_degraded=False,
        is_document_screenshot=getattr(localizer, 'last_is_document_screenshot', False)
    )
    verdict, conf, expl, signals = fusion.evaluate(prov, forensics)
    
    print(f"=== {label} ===")
    print(f"  VERDICT: {verdict.value} ({conf:.1f}%)")
    print(f"  EXPLANATION: {expl}")
    print("  PROVENANCE:")
    print(f"    has_c2pa: {prov.has_c2pa}, is_valid: {prov.is_valid}, is_trusted: {prov.is_trusted}")
    print(f"    issuer: {prov.issuer}, declared_actions: {prov.declared_actions}")
    if prov.raw_manifest:
        print(f"    raw_manifest: {prov.raw_manifest}")
    print("  FORENSICS:")
    print(f"    synthetic: total={res_syn['synthetic_score']:.4f}, spectral={res_syn['spectral_score']:.4f}, p_dev={res_syn['spectral_meta']['power_law_deviation']:.4f}, spikes={res_syn['spectral_meta']['spike_ratio']:.4f}, spatial={res_syn['spatial_score']:.4f}")
    print(f"    manipulation: score={res_loc[0]:.4f}, cluster_ratio={localizer.last_cluster_ratio:.4f}")
    print(f"    compression: anomaly={res_comp['compression_anomaly_score']:.4f}, prnu={res_comp['prnu_score']:.4f}")
    print()

