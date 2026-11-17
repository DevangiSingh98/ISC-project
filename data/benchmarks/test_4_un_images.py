import os
from PIL import Image
from engines.forensics.synthetic.detector import AIGeneratedImageDetector
from engines.forensics.manipulation.localizer import ManipulationLocalizer
from engines.forensics.noise_compression.compression_forensics import CompressionForensicsAnalyzer
from engines.fusion.rules_engine import FusionRulesEngine
from apps.api.app.schemas.verdict import ProvenanceDetails, ForensicsDetails

detector = AIGeneratedImageDetector()
localizer = ManipulationLocalizer()
comp = CompressionForensicsAnalyzer()
fusion = FusionRulesEngine()

user_imgs = [
    r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937346098_9eed34f1.jpg",
    r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357622_7e7ff603.jpg",
    r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357646_225cc16f.jpg",
    r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357654_26db2ec3.jpg"
]

labels = ["Nadia Murad / Bookstore", "Gaza Refugee Camp", "Aerial Highway / Park", "UN General Assembly"]

for p, label in zip(user_imgs, labels):
    fname = os.path.basename(p)
    img = Image.open(p)
    res_loc = localizer.generate_heatmap(img)
    res_syn = detector.analyze(img)
    res_comp = comp.analyze(img)
    
    prov = ProvenanceDetails(has_c2pa=False, is_valid=False)
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
    print(f"=== {label} ({fname}) ===")
    print(f"  Verdict: {verdict.value} ({conf:.1f}%)")
    print(f"  synthetic: total={res_syn['synthetic_score']:.3f}, spectral={res_syn['spectral_score']:.3f}, p_dev={res_syn['spectral_meta']['power_law_deviation']:.3f}, spikes={res_syn['spectral_meta']['spike_ratio']:.3f}")
    print(f"  manipulation: score={res_loc[0]:.3f}, cluster_ratio={getattr(localizer, 'last_cluster_ratio', 0.0):.4f}")
    print(f"  compression: anomaly={res_comp['compression_anomaly_score']:.3f}, prnu={res_comp['prnu_score']:.3f}")
    print(f"  explanation: {expl}")
    print()

