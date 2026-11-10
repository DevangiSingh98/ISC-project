"""
engines/crypto/__init__.py — Track A Cryptographic Provenance package
"""
from .hasher import compute_sha256, compute_phash, compute_dhash, hamming_distance, inspect_image_metadata
from .c2pa_validator import C2PAValidator
from .trust_store import TrustStore, default_trust_store
from .signer import C2PASigner
from .durable_registry import DurableCredentialRegistry, default_durable_registry

__all__ = [
    "compute_sha256", "compute_phash", "compute_dhash",
    "hamming_distance", "inspect_image_metadata",
    "C2PAValidator", "TrustStore", "default_trust_store",
    "C2PASigner", "DurableCredentialRegistry", "default_durable_registry",
]

