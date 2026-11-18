"""
Tests for the video verification pipeline.

These tests generate synthetic MP4 video data in-memory using
cv2.VideoWriter so there is no dependency on external video files.
They cover:
  - VideoForensicsDetector.analyze()  (Track B engine)
  - /api/v1/verify/video endpoint     (full pipeline via TestClient)
  - FusionRulesEngine.evaluate_video() (fusion logic)
  - Existing image tests are unaffected (separate test file)
"""
import io
import os
import struct
import tempfile
import numpy as np
import pytest
import cv2
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_video(
    path: str,
    num_frames: int = 10,
    fps: float = 5.0,
    width: int = 320,
    height: int = 240,
    noisy: bool = False,
) -> None:
    """
    Writes a minimal valid MP4 (via cv2.VideoWriter with mp4v codec) to path.
    If noisy=True, each frame has random noise to trigger a higher manipulation
    score; otherwise frames are solid colour gradients.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    rng = np.random.default_rng(42)
    for i in range(num_frames):
        if noisy:
            frame = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
        else:
            # Solid blue-ish gradient frame — deterministic
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = 80 + i * 2       # B channel
            frame[:, :, 1] = 120               # G channel
            frame[:, :, 2] = 60 + i * 3       # R channel
        out.write(frame)
    out.release()


# ---------------------------------------------------------------------------
# VideoForensicsDetector unit tests
# ---------------------------------------------------------------------------

class TestVideoForensicsDetector:

    def test_capped_sampling_covers_entire_video(self):
        from engines.forensics.deepfake_video import (
            MAX_FRAMES_TO_ANALYSE,
            VideoForensicsDetector,
        )

        indices = VideoForensicsDetector(sample_fps=1.0)._select_frame_indices(
            total_frames=10_000,
            source_fps=25.0,
        )

        assert len(indices) == MAX_FRAMES_TO_ANALYSE
        assert indices[0] == 0
        assert indices[-1] == 9_999
        assert indices[MAX_FRAMES_TO_ANALYSE // 2] > 4_000

    def test_analyze_returns_all_required_keys(self, tmp_path):
        """Engine must return all keys needed to build VideoForensicsDetails."""
        from engines.forensics.deepfake_video import VideoForensicsDetector
        path = str(tmp_path / "clean.mp4")
        _make_synthetic_video(path, num_frames=6, fps=3.0)

        detector = VideoForensicsDetector(sample_fps=1.0)
        result = detector.analyze(path)

        required_keys = [
            "mean_manipulation_score",
            "mean_synthetic_score",
            "deepfake_score",
            "temporal_inconsistency_score",
            "face_detected_in_video",
            "frame_count_total",
            "frames_analyzed",
            "sample_fps",
            "suspicious_frames",
            "worst_frame_heatmap_base64",
            "worst_frame_thumbnail_base64",
            "worst_frame_timestamp_sec",
            "latency_ms",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_scores_are_in_valid_range(self, tmp_path):
        """All probability scores must be in [0, 1]."""
        from engines.forensics.deepfake_video import VideoForensicsDetector
        path = str(tmp_path / "clean2.mp4")
        _make_synthetic_video(path, num_frames=5, fps=5.0)

        detector = VideoForensicsDetector(sample_fps=1.0)
        result = detector.analyze(path)

        for key in ["mean_manipulation_score", "mean_synthetic_score",
                    "deepfake_score", "temporal_inconsistency_score"]:
            val = result[key]
            assert 0.0 <= val <= 1.0, f"{key} = {val} out of [0,1]"

    def test_frames_analyzed_is_nonzero_for_valid_video(self, tmp_path):
        """Engine should analyse at least 1 frame for a valid video."""
        from engines.forensics.deepfake_video import VideoForensicsDetector
        path = str(tmp_path / "frames.mp4")
        _make_synthetic_video(path, num_frames=10, fps=5.0)

        detector = VideoForensicsDetector(sample_fps=1.0)
        result = detector.analyze(path)

        assert result["frames_analyzed"] > 0

    def test_suspicious_frames_is_a_list(self, tmp_path):
        """Suspicious frames must always be a list."""
        from engines.forensics.deepfake_video import VideoForensicsDetector
        path = str(tmp_path / "sus.mp4")
        _make_synthetic_video(path, num_frames=5, fps=5.0)

        detector = VideoForensicsDetector(sample_fps=1.0)
        result = detector.analyze(path)

        assert isinstance(result["suspicious_frames"], list)
        for frame in result["suspicious_frames"]:
            assert 0.0 <= frame["temporal_inconsistency_score"] <= 1.0

    def test_thumbnail_is_base64_jpeg(self, tmp_path):
        """Worst-frame thumbnail, if present, must be a valid JPEG data URI."""
        from engines.forensics.deepfake_video import VideoForensicsDetector
        path = str(tmp_path / "thumb.mp4")
        _make_synthetic_video(path, num_frames=6, fps=3.0)

        detector = VideoForensicsDetector(sample_fps=1.0)
        result = detector.analyze(path)

        thumb = result["worst_frame_thumbnail_base64"]
        if thumb is not None:
            assert thumb.startswith("data:image/jpeg;base64,"), \
                f"Thumbnail prefix unexpected: {thumb[:40]}"

    def test_empty_result_on_nonexistent_file(self):
        """Engine must return a safe zero-score result for unreadable paths."""
        from engines.forensics.deepfake_video import VideoForensicsDetector
        detector = VideoForensicsDetector(sample_fps=1.0)
        result = detector.analyze("/nonexistent/path/video.mp4")

        # Should not raise; scores should be 0
        assert result["frames_analyzed"] == 0
        assert result["deepfake_score"] == 0.0


# ---------------------------------------------------------------------------
# VideoMetadata extractor unit tests
# ---------------------------------------------------------------------------

class TestVideoMetadataExtractor:

    def test_inspect_video_metadata_returns_correct_shape(self, tmp_path):
        from engines.crypto.hasher import inspect_video_metadata
        path = str(tmp_path / "meta.mp4")
        _make_synthetic_video(path, num_frames=10, fps=5.0, width=320, height=240)

        meta = inspect_video_metadata(path)

        assert meta["width"] == 320
        assert meta["height"] == 240
        assert meta["fps"] == pytest.approx(5.0, abs=0.5)
        assert meta["frame_count"] >= 10
        assert meta["duration_sec"] > 0.0
        assert isinstance(meta["codec"], str)


# ---------------------------------------------------------------------------
# Fusion engine evaluate_video() unit tests
# ---------------------------------------------------------------------------

class TestFusionEngineVideo:

    def _make_forensics(self, deepfake=0.1, temporal=0.1, manip=0.1, synth=0.1, face=False):
        from apps.api.app.schemas.video_verdict import VideoForensicsDetails
        return VideoForensicsDetails(
            mean_manipulation_score=manip,
            mean_synthetic_score=synth,
            deepfake_score=deepfake,
            temporal_inconsistency_score=temporal,
            face_detected_in_video=face,
            frame_count_total=30,
            frames_analyzed=10,
            sample_fps=1.0,
        )

    def _make_provenance(self, has_c2pa=False, is_valid=False, is_trusted=False,
                         hash_match=None, issuer=None, declared=None):
        from apps.api.app.schemas.video_verdict import VideoProvenanceDetails
        return VideoProvenanceDetails(
            has_c2pa=has_c2pa,
            is_valid=is_valid,
            is_trusted=is_trusted,
            hash_match=hash_match,
            issuer=issuer,
            declared_actions=declared or [],
        )

    def test_clean_video_returns_likely_authentic(self):
        from engines.fusion.rules_engine import FusionRulesEngine
        from apps.api.app.schemas.verdict import VerdictType
        engine = FusionRulesEngine()
        prov = self._make_provenance()
        foren = self._make_forensics(deepfake=0.05, temporal=0.05, manip=0.05)
        verdict, conf, _, _ = engine.evaluate_video(prov, foren)
        assert verdict == VerdictType.LIKELY_AUTHENTIC
        assert conf >= 50.0

    def test_deepfake_video_returns_suspicious(self):
        from engines.fusion.rules_engine import FusionRulesEngine
        from apps.api.app.schemas.verdict import VerdictType
        engine = FusionRulesEngine()
        prov = self._make_provenance()
        foren = self._make_forensics(deepfake=0.85, temporal=0.70, face=True)
        verdict, conf, _, _ = engine.evaluate_video(prov, foren)
        assert verdict == VerdictType.SUSPICIOUS_MANIPULATED
        assert conf >= 50.0

    def test_valid_c2pa_no_anomaly_returns_verified(self):
        from engines.fusion.rules_engine import FusionRulesEngine
        from apps.api.app.schemas.verdict import VerdictType
        engine = FusionRulesEngine()
        prov = self._make_provenance(has_c2pa=True, is_valid=True, is_trusted=True,
                                     hash_match=True, issuer="Reuters")
        foren = self._make_forensics()
        verdict, conf, _, _ = engine.evaluate_video(prov, foren)
        assert verdict == VerdictType.VERIFIED_AUTHENTIC
        assert conf >= 95.0

    def test_valid_but_untrusted_c2pa_is_not_verified(self):
        from engines.fusion.rules_engine import FusionRulesEngine
        from apps.api.app.schemas.verdict import VerdictType
        engine = FusionRulesEngine()
        prov = self._make_provenance(
            has_c2pa=True,
            is_valid=True,
            is_trusted=False,
            hash_match=True,
            issuer="Unknown Publisher",
        )
        foren = self._make_forensics()

        verdict, confidence, explanation, signals = engine.evaluate_video(prov, foren)

        assert verdict == VerdictType.UNVERIFIABLE
        assert confidence == 40.0
        assert "not trusted" in explanation
        assert any(signal.name == "Untrusted Video Cryptographic Issuer" for signal in signals)

    def test_broken_c2pa_returns_provenance_broken(self):
        from engines.fusion.rules_engine import FusionRulesEngine
        from apps.api.app.schemas.verdict import VerdictType
        engine = FusionRulesEngine()
        prov = self._make_provenance(has_c2pa=True, is_valid=False, hash_match=False)
        foren = self._make_forensics()
        verdict, conf, _, _ = engine.evaluate_video(prov, foren)
        assert verdict == VerdictType.PROVENANCE_BROKEN

    def test_video_model_inference_failure_returns_unverifiable(self):
        from engines.fusion.rules_engine import FusionRulesEngine
        from apps.api.app.schemas.verdict import VerdictType
        engine = FusionRulesEngine()
        prov = self._make_provenance()
        foren = self._make_forensics(deepfake=0.99)
        foren.video_model_status = "inference_failed"

        verdict, confidence, explanation, signals = engine.evaluate_video(prov, foren)

        assert verdict == VerdictType.UNVERIFIABLE
        assert confidence == 40.0
        assert "could not be completed reliably" in explanation
        assert any(signal.name == "Video Model Analysis Failure" for signal in signals)

    def test_temporal_inconsistency_alone_triggers_suspicious(self):
        from engines.fusion.rules_engine import FusionRulesEngine
        from apps.api.app.schemas.verdict import VerdictType
        engine = FusionRulesEngine()
        prov = self._make_provenance()
        foren = self._make_forensics(deepfake=0.2, temporal=0.80, manip=0.1)
        verdict, _, _, _ = engine.evaluate_video(prov, foren)
        assert verdict == VerdictType.SUSPICIOUS_MANIPULATED


# ---------------------------------------------------------------------------
# API endpoint integration tests
# ---------------------------------------------------------------------------

class TestVideoAPIEndpoint:

    @pytest.fixture(autouse=True)
    def client(self):
        from apps.api.app.main import app
        self.client = TestClient(app)

    def _make_video_bytes(self, num_frames=6, fps=3.0) -> bytes:
        """Generate a minimal synthetic MP4 and return its bytes."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            _make_synthetic_video(tmp_path, num_frames=num_frames, fps=fps)
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_verify_video_returns_200(self):
        video_bytes = self._make_video_bytes()
        resp = self.client.post(
            "/api/v1/verify/video",
            files={"file": ("test_video.mp4", video_bytes, "video/mp4")},
        )
        assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"

    def test_verify_video_rejects_unsupported_extension(self):
        resp = self.client.post(
            "/api/v1/verify/video",
            files={"file": ("test_video.avi", b"not-a-video", "video/x-msvideo")},
        )
        assert resp.status_code == 400
        assert "Unsupported video format" in resp.json()["detail"]

    def test_verify_video_rejects_oversized_upload(self, monkeypatch):
        import apps.api.app.main as main_module

        monkeypatch.setattr(main_module, "VIDEO_MAX_BYTES", 4)
        resp = self.client.post(
            "/api/v1/verify/video",
            files={"file": ("test_video.mp4", b"12345", "video/mp4")},
        )
        assert resp.status_code == 413
        assert "maximum allowed size" in resp.json()["detail"]

    def test_verify_video_response_has_required_fields(self):
        video_bytes = self._make_video_bytes()
        resp = self.client.post(
            "/api/v1/verify/video",
            files={"file": ("test_video.mp4", video_bytes, "video/mp4")},
        )
        data = resp.json()
        assert "asset_id" in data
        assert "verdict" in data
        assert "confidence_score" in data
        assert "explanation" in data
        assert "metadata" in data
        assert "forensics" in data
        assert "provenance" in data
        assert "ledger" in data
        assert data["media_type"] == "video"

    def test_verify_video_forensics_fields_in_range(self):
        video_bytes = self._make_video_bytes()
        resp = self.client.post(
            "/api/v1/verify/video",
            files={"file": ("test_video.mp4", video_bytes, "video/mp4")},
        )
        data = resp.json()
        f = data["forensics"]
        for key in ["deepfake_score", "temporal_inconsistency_score",
                    "mean_manipulation_score", "mean_synthetic_score"]:
            val = f[key]
            assert 0.0 <= val <= 1.0, f"{key} = {val} out of [0,1]"

    def test_verify_video_result_retrievable_by_id(self):
        video_bytes = self._make_video_bytes()
        resp = self.client.post(
            "/api/v1/verify/video",
            files={"file": ("test_video.mp4", video_bytes, "video/mp4")},
        )
        data = resp.json()
        asset_id = data["asset_id"]

        get_resp = self.client.get(f"/api/v1/result/video/{asset_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["asset_id"] == asset_id

    def test_verify_video_rejects_image_file(self):
        """Uploading an image to the video endpoint should return 400."""
        import PIL.Image
        buf = io.BytesIO()
        PIL.Image.new("RGB", (100, 100), color=(255, 0, 0)).save(buf, format="JPEG")
        buf.seek(0)
        resp = self.client.post(
            "/api/v1/verify/video",
            files={"file": ("photo.jpg", buf.getvalue(), "image/jpeg")},
        )
        assert resp.status_code == 400

    def test_verify_video_rejects_empty_file(self):
        resp = self.client.post(
            "/api/v1/verify/video",
            files={"file": ("empty.mp4", b"", "video/mp4")},
        )
        assert resp.status_code == 400

    def test_existing_image_endpoint_still_works(self):
        """Confirm that the existing /api/v1/verify image endpoint is unbroken."""
        import PIL.Image
        buf = io.BytesIO()
        img = PIL.Image.new("RGB", (300, 300), color=(100, 150, 200))
        img.save(buf, format="JPEG")
        buf.seek(0)
        resp = self.client.post(
            "/api/v1/verify",
            data={"deep_scan": "false"},
            files={"file": ("photo.jpg", buf.getvalue(), "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "verdict" in data
        assert "forensics" in data
        assert "provenance" in data
