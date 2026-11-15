import os
import uuid
import io
import tempfile
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from PIL import Image

from apps.api.app.schemas.verdict import (
    VerdictType,
    VerificationResult,
    LedgerReference,
    ProvenanceDetails,
    ForensicsDetails,
    ContributingSignal
)
from apps.api.app.schemas.video_verdict import (
    VideoVerificationResult,
    VideoForensicsDetails,
    VideoProvenanceDetails,
    VideoMetadata,
    VideoLedgerReference,
    SuspiciousFrame,
)
from engines.crypto.hasher import compute_sha256, compute_phash, inspect_image_metadata, inspect_video_metadata
from engines.crypto.c2pa_validator import C2PAValidator
from engines.crypto.durable_registry import default_durable_registry
from engines.forensics.manipulation.localizer import ManipulationLocalizer
from engines.forensics.synthetic.detector import AIGeneratedImageDetector
from engines.forensics.noise_compression.compression_forensics import CompressionForensicsAnalyzer
from engines.forensics.deepfake_video import VideoForensicsDetector
from engines.fusion.rules_engine import FusionRulesEngine
from engines.fusion.merkle_ledger import default_transparency_log
from engines.fusion.report_generator import ReportGenerator


app = FastAPI(
    title="NewsProof API",
    version="1.0.0",
    description="Intelligent Image & Video Authenticity Verification for Online News"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Engine Singletons
c2pa_validator = C2PAValidator()
localizer = ManipulationLocalizer()
synthetic_detector = AIGeneratedImageDetector()
compression_analyzer = CompressionForensicsAnalyzer()
fusion_engine = FusionRulesEngine()
report_generator = ReportGenerator()
video_detector = VideoForensicsDetector(sample_fps=1.0)

# In-memory stores for verification results
results_store: dict[str, VerificationResult] = {}
video_results_store: dict[str, VideoVerificationResult] = {}

SAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "test_samples"))

AVAILABLE_SAMPLES = [
    {
        "filename": "c2pa_signed_authentic.jpg",
        "title": "C2PA Verified Photo",
        "expected_verdict": "Verified Authentic",
        "description": "Cryptographically signed by Reuters with valid X.509 certificate chain."
    },
    {
        "filename": "c2pa_edited_declared.jpg",
        "title": "Declared Editorial Edits",
        "expected_verdict": "Authentic - Edited (Declared)",
        "description": "Signed image disclosing approved crop and color adjustments in C2PA manifest."
    },
    {
        "filename": "authentic_press.jpg",
        "title": "Unsigned Authentic Photo",
        "expected_verdict": "Likely Authentic",
        "description": "Clean camera visual without C2PA manifest; clean noise and natural 1/f spectrum."
    },
    {
        "filename": "spliced_manipulated.jpg",
        "title": "Spliced / Forged Image",
        "expected_verdict": "Suspicious / Manipulated",
        "description": "Foreign image patch spliced in; localized SRM noise anomalies isolated in heatmap."
    },
    {
        "filename": "ai_synthetic_diffusion.jpg",
        "title": "AI-Generated Visual",
        "expected_verdict": "AI-Generated",
        "description": "High-frequency 2D FFT periodic grid artifacts typical of diffusion upsampling."
    },
    {
        "filename": "c2pa_discrepancy.jpg",
        "title": "Core Invariant Discrepancy",
        "expected_verdict": "Discrepancy: Requires Editorial Review",
        "description": "Valid C2PA signature coexisting with localized tampering; surfaces both proof and heatmap."
    },
    {
        "filename": "c2pa_broken_tampered.jpg",
        "title": "Broken Provenance",
        "expected_verdict": "Provenance Broken",
        "description": "Signed with C2PA credentials but image bytes tampered with post-signing."
    }
]

@app.get("/api/v1/samples")
def list_samples():
    return AVAILABLE_SAMPLES

@app.get("/api/v1/samples/{filename}")
def get_sample_file(filename: str):
    safe_name = os.path.basename(filename)
    path = os.path.join(SAMPLES_DIR, safe_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Sample not found")
    return FileResponse(path, media_type="image/jpeg")

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    dashboard_path = os.path.join(os.path.dirname(__file__), "..", "..", "web", "index.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>NewsProof API Running</h1>"

@app.get("/v/{asset_id}", response_class=HTMLResponse)
def serve_public_verify(asset_id: str):
    verify_path = os.path.join(os.path.dirname(__file__), "..", "..", "web", "public_verify.html")
    if os.path.exists(verify_path):
        with open(verify_path, "r", encoding="utf-8") as f:
            return f.read()
    return f"<h1>NewsProof Verification Record: {asset_id}</h1>"

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "NewsProof API",
        "version": "1.0.0",
        "merkle_root": default_transparency_log.get_root()
    }

@app.post("/api/v1/verify", response_model=VerificationResult)
async def verify_image(
    file: UploadFile = File(...),
    deep_scan: bool = Form(False)
):
    try:
        raw_bytes = await file.read()
        if len(raw_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Ingestion: Hashing & Metadata
        sha256_hash = compute_sha256(raw_bytes)
        try:
            image = Image.open(io.BytesIO(raw_bytes))
            image.load()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid or unsupported image format.")

        phash_str = compute_phash(image)
        meta = inspect_image_metadata(image, raw_bytes)
        is_degraded = meta["is_degraded"]

        # Track A: Cryptographic Provenance (C2PA)
        ext = os.path.splitext(file.filename or "image.jpg")[1] or ".jpg"
        provenance = c2pa_validator.validate_bytes(raw_bytes, extension=ext)

        # Durable Credentials check: If no C2PA embedded, lookup in pHash registry
        if not provenance.has_c2pa:
            recovered = default_durable_registry.lookup_by_phash(phash_str)
            if recovered:
                provenance.has_c2pa = True
                provenance.is_valid = True
                provenance.is_trusted = True
                provenance.issuer = recovered["matched_record"].get("publisher", "Recovered Registered Credential")
                provenance.raw_manifest = recovered["recovered_manifest"]
                provenance.cert_chain_status = f"RECOVERED_VIA_PHASH_REGISTRY (Hamming dist={recovered['distance']})"

        # Track B: AI Forensics
        manip_score, raw_mask, heatmap_b64 = localizer.generate_heatmap(image)
        is_doc = getattr(localizer, "last_is_document_screenshot", False)
        synth_results = synthetic_detector.analyze(image, deep_scan=deep_scan)
        compression_results = compression_analyzer.analyze(image)

        forensics = ForensicsDetails(
            manipulation_score=manip_score,
            synthetic_score=synth_results["synthetic_score"],
            compression_anomaly_score=compression_results["compression_anomaly_score"],
            prnu_score=compression_results["prnu_score"],
            uncertainty_variance=synth_results["uncertainty_variance"],
            is_degraded=is_degraded,
            is_document_screenshot=is_doc,
            heatmap_base64=heatmap_b64,
            dimensions=[meta["width"], meta["height"]]
        )

        # Decision & Fusion
        verdict, confidence, explanation, signals = fusion_engine.evaluate(provenance, forensics)

        # Append to Merkle Transparency Log
        ledger_entry = default_transparency_log.record_verdict(
            asset_sha256=sha256_hash,
            verdict=verdict.value,
            confidence_score=confidence
        )

        ledger_ref = LedgerReference(
            leaf_hash=ledger_entry["leaf_hash"],
            merkle_root=ledger_entry["merkle_root"],
            tree_index=ledger_entry["index"],
            timestamp=ledger_entry["timestamp"],
            model_versions_hash=ledger_entry["model_versions_hash"]
        )

        asset_id = str(uuid.uuid4())
        result = VerificationResult(
            asset_id=asset_id,
            sha256=sha256_hash,
            phash=phash_str,
            verdict=verdict,
            confidence_score=confidence,
            explanation=explanation,
            top_signals=signals,
            provenance=provenance,
            forensics=forensics,
            ledger=ledger_ref,
            deep_scan_used=deep_scan
        )

        results_store[asset_id] = result
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal verification error: {str(e)}")

@app.get("/api/v1/result/{asset_id}", response_model=VerificationResult)
def get_verification_result(asset_id: str):
    if asset_id not in results_store:
        raise HTTPException(status_code=404, detail="Verification result not found.")
    return results_store[asset_id]

@app.get("/api/v1/manifest/{asset_id}")
def get_manifest(asset_id: str):
    if asset_id not in results_store:
        raise HTTPException(status_code=404, detail="Asset not found.")
    res = results_store[asset_id]
    return {
        "asset_id": asset_id,
        "has_c2pa": res.provenance.has_c2pa,
        "is_valid": res.provenance.is_valid,
        "is_trusted": res.provenance.is_trusted,
        "issuer": res.provenance.issuer,
        "declared_actions": res.provenance.declared_actions,
        "edit_history": res.provenance.edit_history,
        "raw_manifest": res.provenance.raw_manifest,
        "untrusted_warning": res.provenance.untrusted_warning
    }

@app.get("/api/v1/report/{asset_id}/jsonld")
def get_report_jsonld(asset_id: str):
    if asset_id not in results_store:
        raise HTTPException(status_code=404, detail="Asset not found.")
    res = results_store[asset_id]
    return report_generator.generate_json_ld(res)

@app.get("/api/v1/report/{asset_id}/pdf")
def get_report_pdf(asset_id: str):
    if asset_id not in results_store:
        raise HTTPException(status_code=404, detail="Asset not found.")
    res = results_store[asset_id]
    pdf_bytes = report_generator.generate_pdf(res)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=newsproof_certificate_{asset_id}.pdf"}
    )

@app.get("/api/v1/ledger")
def get_ledger_status():
    return {
        "merkle_root": default_transparency_log.get_root(),
        "total_verdicts": len(default_transparency_log.entries),
        "recent_entries": default_transparency_log.entries[-20:]
    }

@app.get("/api/v1/ledger/proof/{index}")
def get_inclusion_proof(index: int):
    try:
        leaf_hash, proof = default_transparency_log.get_inclusion_proof(index)
        return {
            "index": index,
            "leaf_hash": leaf_hash,
            "merkle_root": default_transparency_log.get_root(),
            "proof": proof
        }
    except IndexError:
        raise HTTPException(status_code=404, detail="Index out of range in ledger.")


# ---------------------------------------------------------------------------
# Video Verification Endpoints
# ---------------------------------------------------------------------------

VIDEO_ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm"}
VIDEO_MAX_BYTES = 200 * 1024 * 1024  # 200 MB


@app.post("/api/v1/verify/video", response_model=VideoVerificationResult)
async def verify_video(
    file: UploadFile = File(...),
):
    """
    Accepts an MP4, MOV, or WebM video file and runs the full two-track
    forensic pipeline:
      Track A – C2PA cryptographic provenance (if manifest present)
      Track B – Frame-level AI forensics (reuses image engines on frames)
               + Temporal optical flow analysis
               + Haar face detection for deepfake focus
    Returns a VideoVerificationResult with verdict, confidence, signals,
    per-frame suspicious thumbnails, and a Merkle ledger anchor.
    """
    try:
        # --- File validation ---
        raw_bytes = await file.read()
        if len(raw_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded video file is empty.")
        if len(raw_bytes) > VIDEO_MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Video file exceeds maximum allowed size ({VIDEO_MAX_BYTES // (1024*1024)} MB)."
            )

        filename = file.filename or "video.mp4"
        ext = os.path.splitext(filename)[1].lower()
        if ext not in VIDEO_ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported video format '{ext}'. Accepted: MP4, MOV, WebM."
            )

        # --- Write to temp file (cv2.VideoCapture requires a file path) ---
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        try:
            # --- SHA-256 hash ---
            sha256_hash = compute_sha256(raw_bytes)

            # --- Container metadata ---
            raw_meta = inspect_video_metadata(tmp_path)
            video_meta = VideoMetadata(
                duration_sec=raw_meta["duration_sec"],
                fps=raw_meta["fps"],
                frame_count=raw_meta["frame_count"],
                width=raw_meta["width"],
                height=raw_meta["height"],
                codec=raw_meta["codec"],
                byte_size=len(raw_bytes),
            )

            # Basic sanity: video must have at least 1 frame and non-zero dimensions
            if video_meta.frame_count == 0 or video_meta.width == 0 or video_meta.height == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Video file appears to be corrupt or unreadable (zero frames or dimensions)."
                )

            # --- Track A: C2PA Provenance ---
            provenance = c2pa_validator.validate_video_bytes(raw_bytes, extension=ext)

            # --- Track B: Video Forensics ---
            raw_forensics = video_detector.analyze(tmp_path)

        finally:
            # Always clean up the temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # --- Build VideoForensicsDetails ---
        suspicious_frames = [
            SuspiciousFrame(
                frame_index=sf["frame_index"],
                timestamp_sec=sf["timestamp_sec"],
                manipulation_score=sf["manipulation_score"],
                synthetic_score=sf["synthetic_score"],
                face_detected=sf["face_detected"],
                heatmap_base64=sf.get("heatmap_base64"),
                thumbnail_base64=sf.get("thumbnail_base64"),
            )
            for sf in raw_forensics["suspicious_frames"]
        ]

        forensics = VideoForensicsDetails(
            mean_manipulation_score=raw_forensics["mean_manipulation_score"],
            mean_synthetic_score=raw_forensics["mean_synthetic_score"],
            deepfake_score=raw_forensics["deepfake_score"],
            temporal_inconsistency_score=raw_forensics["temporal_inconsistency_score"],
            face_detected_in_video=raw_forensics["face_detected_in_video"],
            frame_count_total=raw_forensics["frame_count_total"],
            frames_analyzed=raw_forensics["frames_analyzed"],
            sample_fps=raw_forensics["sample_fps"],
            suspicious_frames=suspicious_frames,
            worst_frame_heatmap_base64=raw_forensics.get("worst_frame_heatmap_base64"),
            worst_frame_thumbnail_base64=raw_forensics.get("worst_frame_thumbnail_base64"),
            worst_frame_timestamp_sec=raw_forensics.get("worst_frame_timestamp_sec"),
        )

        # --- Decision & Fusion ---
        verdict, confidence, explanation, signals = fusion_engine.evaluate_video(
            provenance, forensics
        )

        # --- Merkle Transparency Log ---
        ledger_entry = default_transparency_log.record_verdict(
            asset_sha256=sha256_hash,
            verdict=verdict.value,
            confidence_score=confidence,
        )
        ledger_ref = VideoLedgerReference(
            leaf_hash=ledger_entry["leaf_hash"],
            merkle_root=ledger_entry["merkle_root"],
            tree_index=ledger_entry["index"],
            timestamp=ledger_entry["timestamp"],
            model_versions_hash=ledger_entry["model_versions_hash"],
        )

        asset_id = str(uuid.uuid4())
        result = VideoVerificationResult(
            asset_id=asset_id,
            sha256=sha256_hash,
            media_type="video",
            verdict=verdict,
            confidence_score=confidence,
            explanation=explanation,
            top_signals=signals,
            metadata=video_meta,
            provenance=provenance,
            forensics=forensics,
            ledger=ledger_ref,
        )

        video_results_store[asset_id] = result
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal video verification error: {str(e)}")


@app.get("/api/v1/result/video/{asset_id}", response_model=VideoVerificationResult)
def get_video_result(asset_id: str):
    """Retrieves a previously computed video verification result by asset ID."""
    if asset_id not in video_results_store:
        raise HTTPException(status_code=404, detail="Video verification result not found.")
    return video_results_store[asset_id]
