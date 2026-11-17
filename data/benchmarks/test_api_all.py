"""Quick API test for all images."""
import requests, json, os

images = {
    "REAL Gaza":     r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357622_7e7ff603.jpg",
    "REAL Nadia":    r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937346098_9eed34f1.jpg",
    "REAL Aerial":   r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357646_225cc16f.jpg",
    "REAL UN Assy":  r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794937357654_26db2ec3.jpg",
    "AI Earthshot":  r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794938867451_a73fa26d.jpg",
    "AI Railway":    r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794938867468_6ba81fb5.jpg",
}

for label, path in images.items():
    if not os.path.exists(path):
        print(f"{label}: FILE NOT FOUND")
        continue
    with open(path, "rb") as f:
        resp = requests.post("http://127.0.0.1:8000/api/v1/verify", files={"file": (os.path.basename(path), f, "image/jpeg")})
    data = resp.json()
    verdict = data.get("verdict", "???")
    confidence = data.get("confidence", "???")
    manip = data.get("forensics", {}).get("manipulation_score", "???")
    synth = data.get("forensics", {}).get("synthetic_score", "???")
    print(f"  {label:<20} → {verdict:<40} conf={confidence}%  manip={manip}  synth={synth}")

