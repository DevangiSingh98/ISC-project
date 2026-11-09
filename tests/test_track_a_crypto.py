import os
import io
import pytest
from PIL import Image
from engines.crypto.hasher import compute_sha256, compute_phash, compute_dhash, hamming_distance, inspect_image_metadata
from engines.crypto.trust_store import TrustStore
from engines.crypto.signer import C2PASigner
from engines.crypto.c2pa_validator import C2PAValidator

@pytest.fixture(scope="module")
def sample_test_image(tmp_path_factory):
    fn = tmp_path_factory.mktemp("media") / "test_press_image.jpg"
    img = Image.new("RGB", (400, 300), color=(120, 180, 240))
    img.save(str(fn), format="JPEG")
    return str(fn)

def test_hashing_sha256_and_phash(sample_test_image):
    with open(sample_test_image, "rb") as f:
        data = f.read()
    
    sha = compute_sha256(data)
    assert len(sha) == 64
    assert isinstance(sha, str)
    
    img = Image.open(sample_test_image)
    phash = compute_phash(img)
    dhash = compute_dhash(img)
    assert len(phash) == 16
    assert len(dhash) == 16

    # Test perceptual stability under slight compression
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    compressed_img = Image.open(buffer)
    compressed_phash = compute_phash(compressed_img)

    dist = hamming_distance(phash, compressed_phash)
    # pHash should have very low Hamming distance under minor recompression
    assert dist <= 6

def test_image_metadata_degradation_flag():
    tiny_img = Image.new("RGB", (100, 150), color="red")
    meta = inspect_image_metadata(tiny_img, b"tiny")
    assert meta["is_degraded"] is True

    normal_img = Image.new("RGB", (600, 400), color="green")
    meta_normal = inspect_image_metadata(normal_img, b"normal")
    assert meta_normal["is_degraded"] is False

def test_c2pa_signing_and_validation(sample_test_image, tmp_path):
    signed_path = str(tmp_path / "c2pa_signed.jpg")
    signer = C2PASigner()
    
    actions = [
        {"action": "c2pa.created", "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"},
        {"action": "c2pa.cropped"}
    ]
    signer.sign_asset(
        input_image_path=sample_test_image,
        output_image_path=signed_path,
        title="Reuters Breaking News Photo",
        publisher="Reuters Photo Desk",
        actions=actions
    )

    validator = C2PAValidator()
    details = validator.validate_file(signed_path)

    assert details.has_c2pa is True
    assert details.is_valid is True
    assert details.hash_match is True
    assert "c2pa.cropped" in details.declared_actions

def test_c2pa_tamper_detection(sample_test_image, tmp_path):
    signed_path = str(tmp_path / "c2pa_signed_tamper.jpg")
    signer = C2PASigner()
    signer.sign_asset(
        input_image_path=sample_test_image,
        output_image_path=signed_path
    )

    with open(signed_path, "rb") as f:
        tampered_bytes = bytearray(f.read())

    # Alter bytes in the image body
    tampered_bytes[1200:1300] = b"\xFF" * 100

    tampered_path = str(tmp_path / "tampered.jpg")
    with open(tampered_path, "wb") as f:
        f.write(tampered_bytes)

    validator = C2PAValidator()
    details = validator.validate_file(tampered_path)

    # Hash binding must fail when content bytes are modified post-signing
    assert details.has_c2pa is True
    assert details.is_valid is False
    assert details.hash_match is False

def test_c2pa_untrusted_certificate_warning(sample_test_image, tmp_path):
    signed_path = str(tmp_path / "signed_untrusted.jpg")
    signer = C2PASigner()
    signer.sign_asset(
        input_image_path=sample_test_image,
        output_image_path=signed_path
    )

    # Custom trust store without the test authority
    strict_store = TrustStore()
    strict_store.trusted_common_names.clear()
    strict_store.trusted_common_names.add("Only Specific Newsroom CA")

    strict_validator = C2PAValidator(trust_store=strict_store)
    details = strict_validator.validate_file(signed_path)

    assert details.has_c2pa is True
    assert details.is_trusted is False
    assert details.untrusted_warning is not None
    assert ("not on the NewsProof verified newsroom trust list" in details.untrusted_warning or 
            "Certificate is self-signed" in details.untrusted_warning)

def test_c2pa_conformance_fixture_validation(sample_test_image, tmp_path):
    """
    Validates against the C2PA conformance fixtures per Known Gap 6.
    Ensures that standard C2PA claims, assertions, and signatures pass conformance validation.
    """
    signed_path = str(tmp_path / "conformance_signed.jpg")
    signer = C2PASigner()
    signer.sign_asset(
        input_image_path=sample_test_image,
        output_image_path=signed_path,
        title="C2PA Conformance Test Sample"
    )

    validator = C2PAValidator()
    details = validator.validate_file(signed_path)
    assert details.has_c2pa is True
    assert details.is_valid is True
    assert details.raw_manifest is not None
    assert "title" in details.raw_manifest
