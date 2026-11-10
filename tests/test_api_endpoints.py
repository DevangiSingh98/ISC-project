import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from apps.api.app.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def dummy_jpeg_bytes():
    buf = io.BytesIO()
    img = Image.new("RGB", (320, 320), color=(150, 200, 250))
    img.save(buf, format="JPEG")
    return buf.getvalue()

def test_api_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "merkle_root" in data

def test_api_verify_and_report_lifecycle(client, dummy_jpeg_bytes):
    # 1. POST /api/v1/verify
    files = {"file": ("press_photo.jpg", dummy_jpeg_bytes, "image/jpeg")}
    resp = client.post("/api/v1/verify", files=files, data={"deep_scan": "false"})
    assert resp.status_code == 200
    res = resp.json()
    
    assert "asset_id" in res
    assert "verdict" in res
    assert "confidence_score" in res
    assert "forensics" in res
    assert "ledger" in res
    assert res["deep_scan_used"] is False

    asset_id = res["asset_id"]

    # 2. GET /api/v1/result/{asset_id}
    res_get = client.get(f"/api/v1/result/{asset_id}")
    assert res_get.status_code == 200
    assert res_get.json()["asset_id"] == asset_id

    # 3. GET /api/v1/manifest/{asset_id}
    manifest_resp = client.get(f"/api/v1/manifest/{asset_id}")
    assert manifest_resp.status_code == 200
    assert "has_c2pa" in manifest_resp.json()

    # 4. GET /api/v1/report/{asset_id}/jsonld
    jsonld_resp = client.get(f"/api/v1/report/{asset_id}/jsonld")
    assert jsonld_resp.status_code == 200
    assert jsonld_resp.json()["@type"] == "MediaReview"

    # 5. GET /api/v1/report/{asset_id}/pdf
    pdf_resp = client.get(f"/api/v1/report/{asset_id}/pdf")
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"
    assert pdf_resp.content.startswith(b"%PDF-")

    # 6. GET /api/v1/ledger
    ledger_resp = client.get("/api/v1/ledger")
    assert ledger_resp.status_code == 200
    assert ledger_resp.json()["total_verdicts"] >= 1

    # 7. GET /api/v1/ledger/proof/0
    proof_resp = client.get("/api/v1/ledger/proof/0")
    assert proof_resp.status_code == 200
    assert "proof" in proof_resp.json()

def test_api_html_routes(client):
    # Dashboard route
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "NewsProof" in root_resp.text

    # Public verify route
    v_resp = client.get("/v/dummy-asset-123")
    assert v_resp.status_code == 200
    assert "NewsProof Public Trust Signal" in v_resp.text

