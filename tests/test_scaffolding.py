import os
from apps.api.app.schemas.verdict import VerdictType, VerificationResult, ProvenanceDetails, ForensicsDetails

def test_verdict_types():
    assert len(VerdictType) == 8
    assert VerdictType.DISCREPANCY_REQUIRES_REVIEW.value == "Discrepancy: Requires Editorial Review"
    assert VerdictType.VERIFIED_AUTHENTIC.value == "Verified Authentic"
    assert VerdictType.UNVERIFIABLE.value == "Unverifiable"

def test_schema_serialization():
    result = VerificationResult(
        asset_id="test_001",
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        phash="8f8f8f8f8f8f8f8f",
        verdict=VerdictType.LIKELY_AUTHENTIC,
        confidence_score=94.5,
        explanation="Forensics indicate clean pixel patterns with no evidence of synthetic or manual manipulation.",
        provenance=ProvenanceDetails(has_c2pa=False),
        forensics=ForensicsDetails(
            manipulation_score=0.03,
            synthetic_score=0.04,
            compression_anomaly_score=0.02,
            uncertainty_variance=0.001,
            is_degraded=False
        )
    )
    assert result.verdict == VerdictType.LIKELY_AUTHENTIC
    assert result.confidence_score == 94.5

