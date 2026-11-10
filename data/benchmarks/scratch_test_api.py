"""Quick API smoke-test for the two user screenshots."""
import urllib.request, json, uuid

BASE = "http://127.0.0.1:8000"

def api_verify(path, label):
    boundary = uuid.uuid4().hex
    with open(path, "rb") as f:
        file_data = f.read()
    # Build multipart body manually
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
        method="POST",
    )
    res = json.loads(urllib.request.urlopen(req).read())
    verdict = res["verdict"]
    conf = res["confidence_score"]
    is_doc = res.get("forensics", {}).get("is_document_screenshot", False)
    manip = res.get("forensics", {}).get("manipulation_score", -1)
    print(f"{label}")
    print(f"  Verdict: {verdict}  ({conf}%)")
    print(f"  is_document_screenshot: {is_doc}  manipulation_score: {manip:.4f}")
    print(f"  Explanation: {res.get('explanation','')[:120]}")
    print()

# Health check
print("Health:", json.loads(urllib.request.urlopen(BASE + "/health").read())["status"])
print()

api_verify(
    r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794284686678_71030ded.jpg",
    "Zoomed article (TOI)"
)
api_verify(
    r"C:\Users\singh\.gemini\antigravity\brain\665a9fec-00da-48fe-b182-dbe26c3b7721\.user_uploaded\media_1794289421048_a795f5cf.jpg",
    "Full webpage screenshot (TOI)"
)

