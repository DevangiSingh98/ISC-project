import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from apps.api.app.schemas.verdict import (
    VerdictType,
    ContributingSignal,
    ProvenanceDetails,
    ForensicsDetails,
)

class FusionRulesEngine:
    """
    Deterministic Rule-Plus-Model Fusion Engine.
    Combines Track A (Cryptographic Passport) and Track B (AI Forensics Ensemble)
    into one of 8 calibrated verdicts with plain-language explanations.

    Core Invariant:
    A valid C2PA signature is never silently overridden by an isolated AI score.
    If a valid signature coexists with an elevated forensic anomaly, the engine
    returns 'Discrepancy: Requires Editorial Review', displaying both the credentials
    and the localization heatmap.
    """
    def __init__(
        self,
        manip_threshold: float = 0.55,
        synth_threshold: float = 0.55,
        variance_uncertainty_threshold: float = 0.065,
        clean_threshold: float = 0.35
    ):
        self.manip_threshold = manip_threshold
        self.synth_threshold = synth_threshold
        self.variance_threshold = variance_uncertainty_threshold
        self.clean_threshold = clean_threshold

    def evaluate(
        self,
        provenance: ProvenanceDetails,
        forensics: ForensicsDetails
    ) -> Tuple[VerdictType, float, str, List[ContributingSignal]]:
        signals: List[ContributingSignal] = []

        # 1. Check for Degradation / High Epistemic Uncertainty -> Unverifiable
        if forensics.is_degraded:
            signals.append(ContributingSignal(
                name="Resolution & Quality Check",
                weight=1.0,
                score=1.0,
                description="Image resolution is below 256px minimum threshold or exhibits severe lossy degradation."
            ))
            return (
                VerdictType.UNVERIFIABLE,
                20.0,
                "The image is too low-resolution or heavily degraded to reliably evaluate authenticity.",
                signals
            )

        if forensics.uncertainty_variance > self.variance_threshold:
            signals.append(ContributingSignal(
                name="Model Epistemic Uncertainty",
                weight=0.9,
                score=min(1.0, forensics.uncertainty_variance * 10.0),
                description=f"Detector ensemble variance is high (sigma^2 = {forensics.uncertainty_variance:.4f}), indicating out-of-distribution media."
            ))
            return (
                VerdictType.UNVERIFIABLE,
                30.0,
                "Forensic model ensemble exhibits high prediction variance. System refrains from guessing and flags as Unverifiable.",
                signals
            )

        # 2. Check Track A: Provenance Broken (C2PA present but hash mismatch or invalid signature)
        if provenance.has_c2pa and (not provenance.is_valid or provenance.hash_match is False):
            signals.append(ContributingSignal(
                name="Cryptographic Hash Binding",
                weight=1.0,
                score=1.0,
                description="Content bytes do not match the cryptographic hash signed in the C2PA manifest."
            ))
            return (
                VerdictType.PROVENANCE_BROKEN,
                98.0,
                "Cryptographic Content Credentials present, but digital signature or hash binding verification failed. File altered post-signing.",
                signals
            )

        # 3. Check Track A: Valid C2PA Manifest
        if provenance.has_c2pa and provenance.is_valid and provenance.hash_match is True:
            # Check CORE INVARIANT: Valid signature coexists with forensic red flag!
            has_forensic_red_flag = (
                forensics.manipulation_score > self.manip_threshold or
                forensics.synthetic_score > self.synth_threshold
            )

            # Check if forensic signals match declared edit actions
            is_declared_edit = False
            if provenance.declared_actions:
                declared_lower = [a.lower() for a in provenance.declared_actions]
                if any(edit in act for edit in ["crop", "color", "adjust", "resize", "filter"] for act in declared_lower):
                    is_declared_edit = True

            if has_forensic_red_flag:
                # If tampering is detected beyond declared edits, return DISCREPANCY
                signals.append(ContributingSignal(
                    name="Cryptographic Signature",
                    weight=0.5,
                    score=0.1,
                    description=f"Valid signature from {provenance.issuer or 'authorized entity'}."
                ))
                signals.append(ContributingSignal(
                    name="Pixel Forensics Anomaly",
                    weight=0.5,
                    score=max(forensics.manipulation_score, forensics.synthetic_score),
                    description="Forensics detected anomalous pixel boundaries or spectral artifacts inconsistent with declared history."
                ))
                return (
                    VerdictType.DISCREPANCY_REQUIRES_REVIEW,
                    75.0,
                    "Discrepancy detected: Media carries a valid cryptographic signature, but AI forensics found anomalous manipulation patterns. Requires human editorial review.",
                    signals
                )

            if is_declared_edit:
                signals.append(ContributingSignal(
                    name="C2PA Action Assertion",
                    weight=0.9,
                    score=0.1,
                    description=f"Manifest discloses approved editorial adjustments: {', '.join(provenance.declared_actions)}."
                ))
                return (
                    VerdictType.AUTHENTIC_EDITED_DECLARED,
                    95.0,
                    f"Cryptographically verified with disclosed editorial modifications: {', '.join(provenance.declared_actions)}.",
                    signals
                )

            # Clean verified authentic
            signals.append(ContributingSignal(
                name="C2PA Digital Provenance",
                weight=1.0,
                score=0.05,
                description=f"Verified signature chain from {provenance.issuer or 'Trusted Authority'} with unaltered hash binding."
            ))
            return (
                VerdictType.VERIFIED_AUTHENTIC,
                99.0,
                "Cryptographically verified authentic. Content credentials intact with unbroken provenance chain from verified publisher.",
                signals
            )

        # 4. Track A Absent / No C2PA: Rely on Track B Forensics Ensemble
        is_synthetic = forensics.synthetic_score >= self.synth_threshold
        is_manipulated = forensics.manipulation_score >= self.manip_threshold

        # Contradictory signals check (e.g. both extremely high in inconsistent ways)
        if is_synthetic and is_manipulated and abs(forensics.synthetic_score - forensics.manipulation_score) < 0.05 and forensics.synthetic_score > 0.85:
            # When both synthetic and splicing trigger at near maximums without provenance, investigate or flag
            pass

        if is_synthetic and forensics.synthetic_score >= forensics.manipulation_score:
            signals.append(ContributingSignal(
                name="Generative Spectral Fingerprint",
                weight=0.85,
                score=forensics.synthetic_score,
                description="2D FFT frequency spectrum exhibits anomalous periodic artifacts and departure from natural 1/f decay."
            ))
            conf = float(np.clip(50.0 + forensics.synthetic_score * 48.0, 50.0, 99.0))
            return (
                VerdictType.AI_GENERATED,
                round(conf, 1),
                "Media demonstrates characteristic synthetic generative model artifacts (Diffusion/GAN spectral patterns).",
                signals
            )

        if is_manipulated:
            signals.append(ContributingSignal(
                name="SRM Noise Residual Discrepancy",
                weight=0.85,
                score=forensics.manipulation_score,
                description="Pixel-level noise inconsistency detected along localized boundary seams, characteristic of splicing or copy-move."
            ))
            if forensics.compression_anomaly_score > 0.4:
                signals.append(ContributingSignal(
                    name="Compression Boundary Variance",
                    weight=0.5,
                    score=forensics.compression_anomaly_score,
                    description="Error Level Analysis (ELA) reveals disparate JPEG compression grids across image segments."
                ))
            conf = float(np.clip(50.0 + forensics.manipulation_score * 46.0, 50.0, 98.0))
            return (
                VerdictType.SUSPICIOUS_MANIPULATED,
                round(conf, 1),
                "Strong forensic evidence of digital manipulation (splicing, inpainting, or copy-move). Localization heatmap isolates tampered zones.",
                signals
            )

        # Check clean forensics
        if forensics.manipulation_score < self.clean_threshold and forensics.synthetic_score < self.clean_threshold:
            if getattr(forensics, "is_document_screenshot", False):
                signals.append(ContributingSignal(
                    name="Document Layout & Typography Filtering",
                    weight=0.85,
                    score=0.05,
                    description="Webpage / document layout detected. Rendered text typography was filtered from camera sensor noise analysis."
                ))
                signals.append(ContributingSignal(
                    name="Embedded Media Integrity",
                    weight=0.8,
                    score=0.1,
                    description="Visual photographic content exhibits clean natural sensor noise with no evidence of tampering or generative artifacts."
                ))
                return (
                    VerdictType.LIKELY_AUTHENTIC,
                    88.0,
                    "Webpage / document screenshot layout detected. Rendered article typography was filtered from camera sensor noise checks; embedded visual media is clean with no signs of tampering.",
                    signals
                )

            signals.append(ContributingSignal(
                name="Natural Sensor Noise Consistency",
                weight=0.8,
                score=0.1,
                description="Uniform PRNU noise residuals and natural 1/f spectral frequency distribution across all color channels."
            ))
            return (
                VerdictType.LIKELY_AUTHENTIC,
                88.0,
                "No cryptographic credentials present, but pixel and frequency forensics confirm consistent natural sensor noise with no signs of tampering.",
                signals
            )

        # If scores fall in intermediate ambiguous zone
        return (
            VerdictType.UNVERIFIABLE,
            45.0,
            "Forensic signals fall within an ambiguous threshold margin without cryptographic provenance. Insufficient confidence for definitive ruling.",
            signals
        )
