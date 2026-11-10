from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class VerdictType(str, Enum):
    VERIFIED_AUTHENTIC = "Verified Authentic"
    AUTHENTIC_EDITED_DECLARED = "Authentic - Edited (Declared)"
    LIKELY_AUTHENTIC = "Likely Authentic"
    SUSPICIOUS_MANIPULATED = "Suspicious / Manipulated"
    AI_GENERATED = "AI-Generated"
    PROVENANCE_BROKEN = "Provenance Broken"
    UNVERIFIABLE = "Unverifiable"
    DISCREPANCY_REQUIRES_REVIEW = "Discrepancy: Requires Editorial Review"

class ContributingSignal(BaseModel):
    name: str
    weight: float = Field(..., ge=0.0, le=1.0)
    score: float = Field(..., ge=0.0, le=1.0)
    description: str

class ProvenanceDetails(BaseModel):
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

class ForensicsDetails(BaseModel):
    manipulation_score: float = Field(..., ge=0.0, le=1.0)
    synthetic_score: float = Field(..., ge=0.0, le=1.0)
    compression_anomaly_score: float = Field(..., ge=0.0, le=1.0)
    prnu_score: Optional[float] = None
    uncertainty_variance: float = 0.0
    is_degraded: bool = False
    is_document_screenshot: bool = False
    heatmap_base64: Optional[str] = None
    dimensions: Optional[List[int]] = None

class LedgerReference(BaseModel):
    leaf_hash: str
    merkle_root: str
    tree_index: int
    timestamp: str
    model_versions_hash: str
    signature: Optional[str] = None

class VerificationResult(BaseModel):
    asset_id: str
    sha256: str
    phash: str
    verdict: VerdictType
    confidence_score: float = Field(..., ge=0.0, le=100.0)
    explanation: str
    top_signals: List[ContributingSignal] = []
    provenance: ProvenanceDetails
    forensics: ForensicsDetails
    ledger: Optional[LedgerReference] = None
    deep_scan_used: bool = False

