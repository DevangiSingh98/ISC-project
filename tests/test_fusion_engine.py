import pytest
import numpy as np
from apps.api.app.schemas.verdict import (
    VerdictType,
    ProvenanceDetails,
    ForensicsDetails
)
from engines.fusion.rules_engine import FusionRulesEngine
from engines.forensics.calibrator import ProbabilityCalibrator, compute_ece

@pytest.fixture
def engine():
    return FusionRulesEngine()

def test_verdict_row_1_verified_authentic(engine):
    prov = ProvenanceDetails(
        has_c2pa=True,
        is_valid=True,
        is_trusted=True,
        hash_match=True,
        issuer="Reuters Verified Desk"
    )
    forensics = ForensicsDetails(
        manipulation_score=0.10,
        synthetic_score=0.08,
        compression_anomaly_score=0.05,
        uncertainty_variance=0.002,
        is_degraded=False
    )
    verdict, conf, expl, signals = engine.evaluate(prov, forensics)
    assert verdict == VerdictType.VERIFIED_AUTHENTIC
    assert conf >= 95.0
    assert "Cryptographically verified authentic" in expl

def test_verdict_row_2_authentic_edited_declared(engine):
    prov = ProvenanceDetails(
        has_c2pa=True,
        is_valid=True,
        is_trusted=True,
        hash_match=True,
        issuer="Associated Press",
        declared_actions=["c2pa.cropped", "c2pa.color_adjusted"]
    )
    forensics = ForensicsDetails(
        manipulation_score=0.20,
        synthetic_score=0.10,
        compression_anomaly_score=0.15,
        uncertainty_variance=0.003,
        is_degraded=False
    )
    verdict, conf, expl, signals = engine.evaluate(prov, forensics)
    assert verdict == VerdictType.AUTHENTIC_EDITED_DECLARED
    assert "disclosed editorial modifications" in expl

def test_verdict_row_3_likely_authentic(engine):
    prov = ProvenanceDetails(has_c2pa=False)
    forensics = ForensicsDetails(
        manipulation_score=0.12,
        synthetic_score=0.15,
        compression_anomaly_score=0.10,
        uncertainty_variance=0.004,
        is_degraded=False
    )
    verdict, conf, expl, signals = engine.evaluate(prov, forensics)
    assert verdict == VerdictType.LIKELY_AUTHENTIC
    assert conf >= 80.0

def test_verdict_row_4_suspicious_manipulated(engine):
    prov = ProvenanceDetails(has_c2pa=False)
    forensics = ForensicsDetails(
        manipulation_score=0.82,
        synthetic_score=0.20,
        compression_anomaly_score=0.75,
        uncertainty_variance=0.005,
        is_degraded=False
    )
    verdict, conf, expl, signals = engine.evaluate(prov, forensics)
    assert verdict == VerdictType.SUSPICIOUS_MANIPULATED
    assert conf >= 80.0
    assert any("Noise Residual" in s.name for s in signals)

def test_verdict_row_5_ai_generated(engine):
    prov = ProvenanceDetails(has_c2pa=False)
    forensics = ForensicsDetails(
        manipulation_score=0.15,
        synthetic_score=0.91,
        compression_anomaly_score=0.10,
        uncertainty_variance=0.003,
        is_degraded=False
    )
    verdict, conf, expl, signals = engine.evaluate(prov, forensics)
    assert verdict == VerdictType.AI_GENERATED
    assert conf >= 85.0
    assert any("Spectral" in s.name for s in signals)

def test_verdict_row_6_provenance_broken(engine):
    # Manifest present, but hash mismatch (file altered post-signing)
    prov = ProvenanceDetails(
        has_c2pa=True,
        is_valid=False,
        hash_match=False,
        issuer="BBC News"
    )
    forensics = ForensicsDetails(
        manipulation_score=0.05,
        synthetic_score=0.05,
        compression_anomaly_score=0.05,
        uncertainty_variance=0.001,
        is_degraded=False
    )
    verdict, conf, expl, signals = engine.evaluate(prov, forensics)
    assert verdict == VerdictType.PROVENANCE_BROKEN
    assert conf >= 95.0

def test_verdict_row_7_unverifiable_degraded_and_high_variance(engine):
    # 7a. Degraded image (resolution < 256px)
    prov = ProvenanceDetails(has_c2pa=False)
    forensics_degraded = ForensicsDetails(
        manipulation_score=0.10,
        synthetic_score=0.10,
        compression_anomaly_score=0.10,
        uncertainty_variance=0.001,
        is_degraded=True
    )
    v_deg, _, expl_deg, _ = engine.evaluate(prov, forensics_degraded)
    assert v_deg == VerdictType.UNVERIFIABLE
    assert "low-resolution or heavily degraded" in expl_deg

    # 7b. High epistemic variance
    forensics_variance = ForensicsDetails(
        manipulation_score=0.50,
        synthetic_score=0.50,
        compression_anomaly_score=0.30,
        uncertainty_variance=0.12, # exceeds threshold
        is_degraded=False
    )
    v_var, _, expl_var, _ = engine.evaluate(prov, forensics_variance)
    assert v_var == VerdictType.UNVERIFIABLE
    assert "high prediction variance" in expl_var

def test_verdict_row_8_core_invariant_discrepancy(engine):
    """
    CORE INVARIANT:
    A valid C2PA signature is never silently overridden by an isolated AI score.
    If a valid signature coexists with an elevated forensic anomaly, the engine
    returns 'Discrepancy: Requires Editorial Review'.
    """
    prov = ProvenanceDetails(
        has_c2pa=True,
        is_valid=True,
        is_trusted=True,
        hash_match=True,
        issuer="AFP Photo Desk",
        declared_actions=[]
    )
    forensics = ForensicsDetails(
        manipulation_score=0.88, # High tampering detected
        synthetic_score=0.15,
        compression_anomaly_score=0.70,
        uncertainty_variance=0.002,
        is_degraded=False
    )
    verdict, conf, expl, signals = engine.evaluate(prov, forensics)
    assert verdict == VerdictType.DISCREPANCY_REQUIRES_REVIEW
    assert "Discrepancy detected" in expl
    assert any("Cryptographic Signature" in s.name for s in signals)
    assert any("Pixel Forensics Anomaly" in s.name for s in signals)

def test_calibration_ece_and_isotonic():
    np.random.seed(42)
    # Simulated uncalibrated overconfident probabilities
    true_labels = np.random.binomial(1, 0.7, 500)
    raw_probs = np.clip(true_labels * 0.85 + np.random.normal(0, 0.2, 500), 0.05, 0.95)

    raw_ece = compute_ece(raw_probs, true_labels)
    assert raw_ece >= 0.0

    calibrator = ProbabilityCalibrator()
    calibrator.fit(raw_probs, true_labels)

    # In-distribution ECE should be well calibrated (< 0.05)
    assert calibrator.in_distribution_ece is not None
    assert calibrator.in_distribution_ece < 0.05

    # Out of distribution / held out evaluation
    unseen_probs = np.random.uniform(0.3, 0.9, 100)
    unseen_labels = np.random.binomial(1, 0.5, 100)
    held_out_ece = calibrator.evaluate_held_out(unseen_probs, unseen_labels)
    assert held_out_ece >= 0.0
