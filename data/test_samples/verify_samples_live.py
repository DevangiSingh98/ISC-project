import requests

samples = requests.get('http://127.0.0.1:8000/api/v1/samples').json()
for s in samples:
    fname = s['filename']
    img_bytes = requests.get('http://127.0.0.1:8000/api/v1/samples/' + fname).content
    files = {'file': (fname, img_bytes, 'image/jpeg')}
    res = requests.post('http://127.0.0.1:8000/api/v1/verify', files=files, data={'deep_scan': 'false'}).json()
    status = 'MATCH' if res['verdict'] == s['expected_verdict'] else 'DIFF'
    print(f"[{status}] {s['title']} -> {res['verdict']} (Confidence: {res['confidence_score']}%)")

