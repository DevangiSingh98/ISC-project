"""
Video Verification Schemas
Separate from image schemas to preserve backward-compatibility with
the existing image pipeline. These models describe the structure of a
full video authenticity report including per-frame forensic details.
"""
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# Re-use the same verdict taxonomy and contributing signal model from the
# image pipeline so the frontend can share verdict-coloring logic.
from apps.api.app.schemas.verdict import VerdictType, ContributingSignal


class SuspiciousFrame(BaseModel):
    """Describes a single video frame flagged as potentially manipulated."""
    frame_index: int
    timestamp_sec: float
    manipulation_score: float = Field(..., ge=0.0, le=1.0)
    synthetic_score: float = Field(..., ge=0.0, le=1.0)
    temporal_inconsistency_score: float = Field(0.0, ge=0.0, le=1.0)
    face_detected: bool = False
    heatmap_base64: Optional[str] = None   # RGBA PNG heatmap overlay
    thumbnail_base64: Optional[str] = None  # small JPEG thumbnail for the strip


class VideoMetadata(BaseModel):
    """Container-level metadata extracted from the video file."""
    duration_sec: float
    fps: float
    frame_count: int
    width: int
    height: int
    codec: Optional[str] = None
    byte_size: int


class VideoForensicsDetails(BaseModel):
    """
    Aggregated forensic analysis across all sampled video frames.
    Includes both per-frame image-domain scores (reusing image engines)
    and video-specific temporal signals.
    """
    # Overall frame-level scores (mean of analysed frames)
    mean_manipulation_score: float = Field(..., ge=0.0, le=1.0)
    mean_synthetic_score: float = Field(..., ge=0.0, le=1.0)

    # Video-specific temporal signals
    deepfake_score: float = Field(..., ge=0.0, le=1.0)
    video_model_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    video_model_status: str = "heuristic_fallback"
    video_model_kind: Optional[str] = None
    video_model_threshold: float = Field(0.5, ge=0.0, le=1.0)
    temporal_inconsistency_score: float = Field(..., ge=0.0, le=1.0)
    face_detected_in_video: bool = False

    # Frame sampling statistics
    frame_count_total: int
    frames_analyzed: int
    sample_fps: float  # frames per second actually analysed

    # Flagged frames (above anomaly threshold)
    suspicious_frames: List[SuspiciousFrame] = []

    # Representative heatmap from the most suspicious frame (if any)
    worst_frame_heatmap_base64: Optional[str] = None
    worst_frame_thumbnail_base64: Optional[str] = None
    worst_frame_timestamp_sec: Optional[float] = None


class VideoProvenanceDetails(BaseModel):
    """C2PA / cryptographic provenance details for a video asset."""
    has_c2pa: bool = False
    is_valid: bool = False
    is_trusted: bool = False
    issuer: Optional[str] = None
    cert_chain_status: Optional[str] = None
    hash_match: Optional[bool] = None
    declared_actions: List[str] = []
    edit_history: List[Dict[str, Any]] = []
    raw_manifest: Optional[Dict[str, Any]] = None
    untrusted_warning: Optional[str] = None
    container_format: Optional[str] = None  # e.g. "MP4", "MOV", "WebM"


class VideoLedgerReference(BaseModel):
    """Merkle transparency ledger anchor for a video verification event."""
    leaf_hash: str
    merkle_root: str
    tree_index: int
    timestamp: str
    model_versions_hash: str


class VideoVerificationResult(BaseModel):
    """
    Top-level response model for a video authenticity verification request.
    Analogous to VerificationResult for images, but includes video-specific
    temporal forensics and per-frame suspicious frame data.
    """
    asset_id: str
    sha256: str                    # SHA-256 of the raw video bytes
    media_type: str = "video"      # always "video" for this result type

    verdict: VerdictType
    confidence_score: float = Field(..., ge=0.0, le=100.0)
    explanation: str
    top_signals: List[ContributingSignal] = []

    metadata: VideoMetadata
    provenance: VideoProvenanceDetails
    forensics: VideoForensicsDetails
    ledger: Optional[VideoLedgerReference] = None
