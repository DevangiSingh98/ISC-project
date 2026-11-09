import io
import json
from typing import Dict, Any, Optional
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from apps.api.app.schemas.verdict import VerificationResult

class ReportGenerator:
    """
    Generates machine-readable JSON-LD and exportable PDF authenticity audit certificates.
    """
    def __init__(self, public_verify_base_url: str = "https://verify.newsproof.org"):
        self.public_verify_base_url = public_verify_base_url

    def generate_json_ld(self, result: VerificationResult) -> Dict[str, Any]:
        """Produces Schema.org MediaReview compliant JSON-LD representation."""
        return {
            "@context": "https://schema.org",
            "@type": "MediaReview",
            "url": f"{self.public_verify_base_url}/v/{result.asset_id}",
            "originalMediaContext": "News publication visual",
            "itemReviewed": {
                "@type": "ImageObject",
                "contentUrl": f"{self.public_verify_base_url}/media/{result.asset_id}",
                "sha256": result.sha256,
                "perceptualHash": result.phash
            },
            "reviewRating": {
                "@type": "Rating",
                "ratingValue": result.confidence_score,
                "bestRating": 100,
                "worstRating": 0,
                "alternateName": result.verdict.value
            },
            "reviewBody": result.explanation,
            "provenance": {
                "c2paVerified": result.provenance.is_valid,
                "issuer": result.provenance.issuer,
                "declaredActions": result.provenance.declared_actions
            },
            "transparencyLedger": {
                "merkleRoot": result.ledger.merkle_root if result.ledger else None,
                "leafHash": result.ledger.leaf_hash if result.ledger else None,
                "timestamp": result.ledger.timestamp if result.ledger else None
            }
        }

    def generate_pdf(self, result: VerificationResult) -> bytes:
        """Generates an audit-ready PDF verification report."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=6
        )
        subtitle_style = ParagraphStyle(
            "DocSub",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.HexColor("#64748B"),
            spaceAfter=12
        )
        section_style = ParagraphStyle(
            "SectionHead",
            parent=styles["Heading2"],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#1E293B"),
            spaceBefore=10,
            spaceAfter=4
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#334155")
        )

        elements = []

        # Header
        elements.append(Paragraph("NEWSPROOF AUTHENTICITY CERTIFICATE", title_style))
        elements.append(Paragraph(f"Verification ID: {result.asset_id} &bull; SHA-256: {result.sha256[:16]}...", subtitle_style))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#CBD5E1"), spaceAfter=14))

        # Verdict Card Table
        verdict_color = "#10B981" # Green
        if "Suspicious" in result.verdict.value or "Broken" in result.verdict.value:
            verdict_color = "#EF4444" # Red
        elif "AI-Generated" in result.verdict.value:
            verdict_color = "#6366F1" # Indigo
        elif "Discrepancy" in result.verdict.value:
            verdict_color = "#F59E0B" # Amber

        card_data = [
            [
                Paragraph(f"<b>FINAL VERDICT</b><br/><font size='16' color='{verdict_color}'><b>{result.verdict.value}</b></font>", body_style),
                Paragraph(f"<b>CONFIDENCE</b><br/><font size='16'><b>{result.confidence_score:.1f}%</b></font>", body_style)
            ]
        ]
        card_table = Table(card_data, colWidths=[360, 180])
        card_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
            ("PADDING", (0, 0), (-1, -1), 10),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        elements.append(card_table)
        elements.append(Spacer(1, 10))

        # Plain language explanation
        elements.append(Paragraph("<b>Summary & Evidence Rationale:</b>", section_style))
        elements.append(Paragraph(result.explanation, body_style))
        elements.append(Spacer(1, 10))

        # Two-Track Breakdown Table
        prov_status = "Valid & Intact" if result.provenance.is_valid else ("None Found" if not result.provenance.has_c2pa else "Invalid / Broken")
        details_data = [
            ["EVIDENCE STREAM", "RESULT & SIGNALS"],
            ["Track A: C2PA Provenance", f"Status: {prov_status}\nIssuer: {result.provenance.issuer or 'N/A'}\nDeclared: {', '.join(result.provenance.declared_actions) or 'None'}"],
            ["Track B: Pixel Manipulation", f"Tamper Score: {result.forensics.manipulation_score:.3f}\nNoise Residual: Evaluated via SRM Filter Bank"],
            ["Track B: Generative AI", f"Synthetic Score: {result.forensics.synthetic_score:.3f}\nSpectral Artifacts: 2D FFT Azimuthal Profile"],
            ["Track B: Compression", f"Anomaly Score: {result.forensics.compression_anomaly_score:.3f}\nELA Variance: Analyzed"],
        ]
        breakdown_table = Table(details_data, colWidths=[180, 360])
        breakdown_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(breakdown_table)
        elements.append(Spacer(1, 12))

        # Ledger Reference
        if result.ledger:
            elements.append(Paragraph("<b>Tamper-Evident Transparency Ledger Anchor:</b>", section_style))
            ledger_text = f"Merkle Root: {result.ledger.merkle_root}<br/>Leaf Hash: {result.ledger.leaf_hash}<br/>Timestamp: {result.ledger.timestamp}"
            elements.append(Paragraph(ledger_text, body_style))

        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()
