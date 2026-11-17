import os
import numpy as np
from PIL import Image
from engines.forensics.manipulation.localizer import ManipulationLocalizer
from engines.forensics.noise_compression.compression_forensics import CompressionForensicsAnalyzer
from engines.forensics.synthetic.detector import AIGeneratedImageDetector

localizer = ManipulationLocalizer()
comp = CompressionForensicsAnalyzer()
detector = AIGeneratedImageDetector()

test_list = [
    ("Spliced Sample", "data/test_samples/spliced_manipulated.jpg"),
    ("Authentic Press", "data/test_samples/authentic_press.jpg"),
    ("Gaza UN", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357622_7e7ff603.jpg"),
    ("Nadia UN", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937346098_9eed34f1.jpg"),
    ("Aerial UN", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357646_225cc16f.jpg"),
    ("Assembly UN", r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357654_26db2ec3.jpg")
]

for name, p in test_list:
    img = Image.open(p)
    res_loc = localizer.generate_heatmap(img)
    res_comp = comp.analyze(img)
    res_syn = detector.analyze(img)
    print(f"{name:18s} | manip={res_loc[0]:.3f} (cl={localizer.last_cluster_ratio:.4f}) | ela={res_comp['ela_score']:.3f} | d_jpeg={res_comp['double_jpeg_score']:.3f} | prnu={res_comp['prnu_score']:.4f} | synth={res_syn['synthetic_score']:.3f}")

