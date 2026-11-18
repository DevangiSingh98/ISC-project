import io

from PIL import Image

from apps.api.app.schemas.verdict import ForensicsDetails, ProvenanceDetails, VerdictType
from engines.crypto.c2pa_validator import C2PAValidator
from engines.crypto.signer import C2PASigner
from engines.fusion.rules_engine import FusionRulesEngine


def _write_image(path, color):
    Image.new("RGB", (320, 240), color=color).save(path, format="JPEG")


def test_signed_fake_does_not_prove_truth(tmp_path):
    fake_path = tmp_path / "synthetic_claim.jpg"
    signed_path = tmp_path / "signed_synthetic_claim.jpg"
    _write_image(fake_path, (240, 20, 20))

    C2PASigner().sign_asset(
        input_image_path=str(fake_path),
        output_image_path=str(signed_path),
        title="Fabricated Scene",
        publisher="Authorized Publisher",
    )
    initial_provenance = C2PAValidator().validate_file(str(signed_path))
    assert initial_provenance.issuer
    # Treat the signer as authorized for this test so it isolates the
    # limitation that publisher authorization does not establish truthfulness.
    provenance = initial_provenance.model_copy(
        update={
            "is_trusted": True,
            "untrusted_warning": None,
            "cert_chain_status": "VALID_AND_TRUSTED",
        }
    )
    forensics = ForensicsDetails(
        manipulation_score=0.05,
        synthetic_score=0.05,
        compression_anomaly_score=0.05,
        uncertainty_variance=0.0,
    )

    verdict, _, explanation, _ = FusionRulesEngine().evaluate(provenance, forensics)

    assert provenance.is_valid is True
    assert verdict == VerdictType.VERIFIED_AUTHENTIC
    assert "cryptographically verified" in explanation.lower()


def test_signed_file_with_modified_bytes_is_rejected(tmp_path):
    source_path = tmp_path / "source.jpg"
    signed_path = tmp_path / "signed.jpg"
    tampered_path = tmp_path / "tampered.jpg"
    _write_image(source_path, (20, 140, 240))

    C2PASigner().sign_asset(str(source_path), str(signed_path))
    tampered = bytearray(signed_path.read_bytes())
    tampered[-80:] = b"\x00" * 80
    tampered_path.write_bytes(tampered)

    result = C2PAValidator().validate_file(str(tampered_path))

    assert result.has_c2pa is True
    assert result.is_valid is False
    assert result.hash_match is False
