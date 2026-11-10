import pytest
from apps.api.app.schemas.verdict import (
    VerdictType,
    VerificationResult,
    ProvenanceDetails,
    ForensicsDetails,
    LedgerReference
)
from engines.fusion.merkle_ledger import MerkleTransparencyLog
from engines.crypto.durable_registry import DurableCredentialRegistry
from engines.fusion.report_generator import ReportGenerator

def test_merkle_transparency_log_append_and_proof():
    ledger = MerkleTransparencyLog()
    
    # Record multiple verdicts
    entry1 = ledger.record_verdict(
        asset_sha256="aaa111",
        verdict=VerdictType.VERIFIED_AUTHENTIC.value,
        confidence_score=99.0,
        timestamp="2026-11-10T00:00:00Z"
    )
    entry2 = ledger.record_verdict(
        asset_sha256="bbb222",
        verdict=VerdictType.SUSPICIOUS_MANIPULATED.value,
        confidence_score=92.5,
        timestamp="2026-11-10T00:01:00Z"
    )
    entry3 = ledger.record_verdict(
        asset_sha256="ccc333",
        verdict=VerdictType.AI_GENERATED.value,
        confidence_score=96.0,
        timestamp="2026-11-10T00:02:00Z"
    )

    root = ledger.get_root()
    assert len(root) == 64

    # Verify inclusion proof for entry 1
    leaf_hash, proof = ledger.get_inclusion_proof(1)
    is_valid = MerkleTransparencyLog.verify_proof(leaf_hash, proof, root)
    assert is_valid is True

    # Tampered proof must fail
    fake_leaf = "00" * 32
    assert MerkleTransparencyLog.verify_proof(fake_leaf, proof, root) is False

def test_durable_credential_registry():
    registry = DurableCredentialRegistry(max_distance=6)
    
    original_phash = "ffff0000aaaa5555"
    registry.register_manifest(
        sha256="hash123",
        phash=original_phash,
        manifest_data={"title": "Reuters Exclusive", "claim_generator": "Reuters CMS"},
        publisher="Reuters"
    )
    assert registry.count() == 1

    # Exact match query
    match_exact = registry.lookup_by_phash(original_phash)
    assert match_exact is not None
    assert match_exact["distance"] == 0
    assert match_exact["recovered_manifest"]["title"] == "Reuters Exclusive"

    # Query with 2-bit Hamming distance (simulating social media recompression)
    # Change 1 hex char: 5 (0101) -> 4 (0100) is 1-bit difference
    slightly_altered_phash = "ffff0000aaaa5554"
    match_near = registry.lookup_by_phash(slightly_altered_phash)
    assert match_near is not None
    assert match_near["distance"] == 1
    assert match_near["recovered_manifest"]["title"] == "Reuters Exclusive"

    # Distant query (completely different image)
    distant_phash = "0000ffff5555aaaa"
    assert registry.lookup_by_phash(distant_phash) is None

def test_report_generator_json_ld_and_pdf():
    generator = ReportGenerator()
    
    sample_result = VerificationResult(
        asset_id="asset_test_999",
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        phash="8f8f8f8f8f8f8f8f",
        verdict=VerdictType.DISCREPANCY_REQUIRES_REVIEW,
        confidence_score=78.5,
        explanation="Valid signature detected alongside pixel anomalies.",
        provenance=ProvenanceDetails(has_c2pa=True, is_valid=True, issuer="Associated Press"),
        forensics=ForensicsDetails(
            manipulation_score=0.85,
            synthetic_score=0.10,
            compression_anomaly_score=0.60
        ),
        ledger=LedgerReference(
            leaf_hash="leaf12345",
            merkle_root="root67890",
            tree_index=0,
            timestamp="2026-11-10T00:00:00Z",
            model_versions_hash="v1-audit"
        )
    )

    # 1. JSON-LD check
    json_ld = generator.generate_json_ld(sample_result)
    assert json_ld["@context"] == "https://schema.org"
    assert json_ld["@type"] == "MediaReview"
    assert json_ld["reviewRating"]["alternateName"] == "Discrepancy: Requires Editorial Review"

    # 2. PDF generation check
    pdf_bytes = generator.generate_pdf(sample_result)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF-")

