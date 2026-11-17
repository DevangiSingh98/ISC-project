import urllib.request, json, uuid

BASE = "http://127.0.0.1:8000"
path = r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794923367724_bd37ef0d.jpg"

with open(path, "rb") as f:
    file_data = f.read()

for i in range(10):
    boundary = uuid.uuid4().hex
    CRLF = b"\r\n"
    body = b""
    body += ("--" + boundary + "\r\n").encode()
    body += b"Content-Disposition: form-data; name=\"file\"; filename=\"test.jpg\"\r\n"
    body += b"Content-Type: image/jpeg\r\n\r\n"
    body += file_data + CRLF
    body += ("--" + boundary + "\r\n").encode()
    body += b"Content-Disposition: form-data; name=\"deep_scan\"\r\n\r\nfalse\r\n"
    body += ("--" + boundary + "--\r\n").encode()

    req = urllib.request.Request(
        BASE + "/api/v1/verify",
        data=body,
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary},
        method="POST"
    )
    res = json.loads(urllib.request.urlopen(req).read())
    print(f"Pass {i+1}: {res['verdict']} ({res['confidence_score']}%) manip={res['forensics']['manipulation_score']:.3f} synth={res['forensics']['synthetic_score']:.3f} var={res['forensics']['uncertainty_variance']:.6f} expl={res['explanation'][:60]}")
