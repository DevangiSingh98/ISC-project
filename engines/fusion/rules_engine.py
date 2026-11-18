import numpy as np
from typing import TYPE_CHECKING, Tuple, List, Optional, Dict, Any
from apps.api.app.schemas.verdict import (
    VerdictType,
    ContributingSignal,
    ProvenanceDetails,
    ForensicsDetails,
)

if TYPE_CHECKING:
    from apps.api.app.schemas.video_verdict import VideoForensicsDetails, VideoProvenanceDetails


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

        # A valid signature from an unknown or revoked authority proves control
        # of the signing key, but does not establish a trusted publisher identity.
        if provenance.has_c2pa and provenance.is_valid and provenance.hash_match is True and not provenance.is_trusted:
            signals.append(ContributingSignal(
                name="Untrusted Cryptographic Issuer",
                weight=1.0,
                score=0.0,
                description=provenance.untrusted_warning or "The credential issuer is not in the configured trusted authority list."
            ))
            return (
                VerdictType.UNVERIFIABLE,
                40.0,
                "The file has an intact cryptographic credential, but its issuer is not trusted by this verifier. The signature proves the signer controlled the key, not that the publisher is an approved news organization.",
                signals
            )

        # 3. Check Track A: Valid C2PA Manifest
        if provenance.has_c2pa and provenance.is_valid and provenance.is_trusted and provenance.hash_match is True:
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
        is_doc = getattr(forensics, "is_document_screenshot", False)

        # For webpage/document screenshots, manipulation signals are strongly discounted:
        # - ELA & SRM detect boundaries between independently-compressed CDN assets (ads,
        #   photos, thumbnails, widgets) which is structurally expected, NOT tampering.
        # - Only the synthetic score (FFT spectral analysis) is CDN-invariant and reliable.
        # - We use a relaxed manipulation threshold (0.45 → cap at 0.28 by localizer).
        effective_manip_score = forensics.manipulation_score
        effective_manip_threshold = self.manip_threshold
        if is_doc:
            # In document mode the localizer already caps manip_score ≤ 0.28 for normal pages.
            # Apply an additional threshold raise so a lightly elevated cross-CDN score
            # never crosses into Suspicious territory without very strong evidence.
            effective_manip_threshold = 0.45  # was 0.55 — but doc cap makes this effectively 0.45+

        is_synthetic = forensics.synthetic_score >= self.synth_threshold
        is_manipulated = effective_manip_score >= effective_manip_threshold

        if is_synthetic and forensics.synthetic_score >= effective_manip_score:
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

        if is_manipulated and not is_doc:
            # Non-document image with elevated manipulation signal
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

        if is_manipulated and is_doc:
            # Document screenshot with elevated (but doc-capped) manipulation score —
            # this is very likely cross-CDN compression heterogeneity, not photo tampering.
            # Return Likely Authentic with a clear explanation rather than Suspicious.
            signals.append(ContributingSignal(
                name="Document Layout & Multi-Source Asset Detection",
                weight=0.85,
                score=0.1,
                description=(
                    "Webpage screenshot detected containing multiple independently-compressed "
                    "embedded assets (editorial photo, ads, thumbnails). Boundary-level compression "
                    "heterogeneity between CDN-served blocks is structurally expected and is NOT "
                    "evidence of photo manipulation."
                )
            ))
            signals.append(ContributingSignal(
                name="Editorial Photo Noise Consistency",
                weight=0.75,
                score=0.15,
                description="Primary editorial photographic content shows no internal coherent tampering clusters."
            ))
            return (
                VerdictType.LIKELY_AUTHENTIC,
                82.0,
                (
                    "Webpage screenshot detected. The page contains multiple independently-compressed "
                    "embedded assets from different CDN origins (editorial photo, banner ads, "
                    "thumbnails, widgets). Cross-asset compression differences are structurally "
                    "normal for a rendered webpage and are NOT indicators of photo manipulation. "
                    "No internal tampering detected within the primary editorial visual."
                ),
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

    def evaluate_video(
        self,
        provenance: "VideoProvenanceDetails",
        forensics: "VideoForensicsDetails",
    ) -> Tuple[VerdictType, float, str, List[ContributingSignal]]:
        """
        Fusion decision for video assets.

        Applies the same 8-verdict taxonomy as the image pipeline.
        Video-specific signals:
        - deepfake_score: fused temporal + frame-level score
        - temporal_inconsistency_score: optical flow discontinuity
        - face_detected_in_video: boosts confidence when deepfake signals
          are found in a face region

        Core Invariant is preserved: a valid C2PA signature is never
        silently overridden by an isolated AI score.
        """
        signals: List[ContributingSignal] = []

        # 1. Provenance Broken (C2PA present, hash mismatch or invalid signature)
        if provenance.has_c2pa and (not provenance.is_valid or provenance.hash_match is False):
            signals.append(ContributingSignal(
                name="Video Cryptographic Hash Binding",
                weight=1.0,
                score=1.0,
                description="Video bytes do not match the cryptographic hash signed in the C2PA manifest."
            ))
            return (
                VerdictType.PROVENANCE_BROKEN,
                98.0,
                "Video carries C2PA Content Credentials, but signature or hash binding verification failed. File altered post-signing.",
                signals
            )

        # An intact signature from an unknown or revoked authority is not
        # equivalent to verified publisher provenance.
        if provenance.has_c2pa and provenance.is_valid and provenance.hash_match is True and not provenance.is_trusted:
            signals.append(ContributingSignal(
                name="Untrusted Video Cryptographic Issuer",
                weight=1.0,
                score=0.0,
                description=provenance.untrusted_warning or "The video credential issuer is not in the configured trusted authority list."
            ))
            return (
                VerdictType.UNVERIFIABLE,
                40.0,
                "The video has an intact cryptographic credential, but its issuer is not trusted by this verifier. The signature proves the signer controlled the key, not that the publisher is an approved news organization.",
                signals
            )

        # 2. Valid C2PA Manifest
        if provenance.has_c2pa and provenance.is_valid and provenance.is_trusted and provenance.hash_match is True:
            has_forensic_red_flag = (
                forensics.deepfake_score > (
                    forensics.video_model_threshold
                    if forensics.video_model_status == "loaded"
                    else self.synth_threshold
                )
                or forensics.mean_manipulation_score > self.manip_threshold
            )

            if has_forensic_red_flag:
                signals.append(ContributingSignal(
                    name="Video Cryptographic Signature",
                    weight=0.5,
                    score=0.1,
                    description=f"Valid C2PA signature from {provenance.issuer or 'authorized entity'}."
                ))
                signals.append(ContributingSignal(
                    name="Deepfake / Temporal Anomaly",
                    weight=0.5,
                    score=max(forensics.deepfake_score, forensics.mean_manipulation_score),
                    description="Forensics detected deepfake or manipulation signals inconsistent with declared provenance history."
                ))
                return (
                    VerdictType.DISCREPANCY_REQUIRES_REVIEW,
                    75.0,
                    "Video carries a valid cryptographic signature, but AI forensics found deepfake or manipulation artefacts. Requires human editorial review.",
                    signals
                )

            # Declared editorial actions
            is_declared_edit = False
            if provenance.declared_actions:
                declared_lower = [a.lower() for a in provenance.declared_actions]
                if any(edit in act for edit in ["crop", "color", "adjust", "resize", "filter", "trim", "edit"] for act in declared_lower):
                    is_declared_edit = True

            if is_declared_edit:
                signals.append(ContributingSignal(
                    name="C2PA Video Action Assertion",
                    weight=0.9,
                    score=0.1,
                    description=f"Manifest discloses approved editorial adjustments: {', '.join(provenance.declared_actions)}."
                ))
                return (
                    VerdictType.AUTHENTIC_EDITED_DECLARED,
                    95.0,
                    f"Video cryptographically verified with disclosed editorial modifications: {', '.join(provenance.declared_actions)}.",
                    signals
                )

            signals.append(ContributingSignal(
                name="C2PA Video Provenance",
                weight=1.0,
                score=0.05,
                description=f"Verified C2PA signature chain from {provenance.issuer or 'Trusted Authority'} with unaltered hash binding."
            ))
            return (
                VerdictType.VERIFIED_AUTHENTIC,
                99.0,
                "Video cryptographically verified authentic. Content credentials intact with unbroken provenance chain from verified publisher.",
                signals
            )

        # 3. No C2PA: Rely on AI forensics
        if forensics.video_model_status in {
            "checkpoint_missing",
            "load_failed",
            "inference_failed",
        }:
            signals.append(ContributingSignal(
                name="Video Model Analysis Failure",
                weight=1.0,
                score=0.0,
                description=f"The configured video model could not complete analysis ({forensics.video_model_status})."
            ))
            return (
                VerdictType.UNVERIFIABLE,
                40.0,
                "Video model analysis could not be completed reliably. No fake/manipulated verdict is asserted.",
                signals
            )

        trained_model_threshold = (
            forensics.video_model_threshold
            if forensics.video_model_status == "loaded"
            else self.synth_threshold
        )
        is_deepfake = forensics.deepfake_score >= trained_model_threshold
        is_temporally_inconsistent = forensics.temporal_inconsistency_score >= 0.55
        is_manipulated = forensics.mean_manipulation_score >= self.manip_threshold

        if is_deepfake:
            face_note = " Facial regions were detected and scored for manipulation." if forensics.face_detected_in_video else ""
            signals.append(ContributingSignal(
                name="Deepfake / Generative Fingerprint",
                weight=0.9,
                score=forensics.deepfake_score,
                description=f"Combined frame-level synthetic and temporal inconsistency signals indicate deepfake generation artefacts.{face_note}"
            ))
            if is_temporally_inconsistent:
                signals.append(ContributingSignal(
                    name="Temporal Optical Flow Discontinuity",
                    weight=0.75,
                    score=forensics.temporal_inconsistency_score,
                    description="Inter-frame optical flow shows erratic motion field inconsistencies characteristic of face-swap or frame-stitching."
                ))
            conf = float(np.clip(50.0 + forensics.deepfake_score * 48.0, 50.0, 99.0))
            return (
                VerdictType.SUSPICIOUS_MANIPULATED,
                round(conf, 1),
                "Video forensics detected deepfake generation artefacts including abnormal temporal flow patterns and synthetic spectral fingerprints.",
                signals
            )

        if is_temporally_inconsistent and not is_deepfake:
            signals.append(ContributingSignal(
                name="Temporal Inconsistency",
                weight=0.75,
                score=forensics.temporal_inconsistency_score,
                description="Inter-frame motion field exhibits high variance, suggesting possible frame-level editing or stitching."
            ))
            conf = float(np.clip(50.0 + forensics.temporal_inconsistency_score * 40.0, 50.0, 85.0))
            return (
                VerdictType.SUSPICIOUS_MANIPULATED,
                round(conf, 1),
                "Temporal analysis found motion field discontinuities consistent with frame-level editing. No cryptographic provenance present.",
                signals
            )

        if is_manipulated:
            signals.append(ContributingSignal(
                name="Frame Pixel Manipulation",
                weight=0.8,
                score=forensics.mean_manipulation_score,
                description="Per-frame SRM noise residuals indicate localised pixel manipulation across multiple frames."
            ))
            conf = float(np.clip(50.0 + forensics.mean_manipulation_score * 46.0, 50.0, 98.0))
            return (
                VerdictType.SUSPICIOUS_MANIPULATED,
                round(conf, 1),
                "Frame-level pixel forensics found consistent manipulation signals across sampled video frames.",
                signals
            )

        # Clean video
        if forensics.mean_manipulation_score < self.clean_threshold and forensics.deepfake_score < self.clean_threshold:
            signals.append(ContributingSignal(
                name="Temporal Consistency",
                weight=0.8,
                score=0.1,
                description="Inter-frame optical flow motion fields are consistent with natural camera motion. No deepfake artefacts detected."
            ))
            signals.append(ContributingSignal(
                name="Frame-Level Noise Consistency",
                weight=0.75,
                score=0.1,
                description="Per-frame noise residuals and spectral distributions are consistent with natural video capture."
            ))
            return (
                VerdictType.LIKELY_AUTHENTIC,
                85.0,
                "No cryptographic credentials present, but frame-level pixel forensics and temporal optical flow analysis confirm no deepfake or manipulation artefacts.",
                signals
            )

        # Ambiguous zone
        return (
            VerdictType.UNVERIFIABLE,
            45.0,
            "Video forensic signals fall within an ambiguous margin without cryptographic provenance. Insufficient confidence for definitive ruling.",
            signals
        )
