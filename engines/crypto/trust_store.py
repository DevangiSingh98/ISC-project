from typing import Set, Dict, Optional, Tuple
from cryptography import x509
from cryptography.hazmat.backends import default_backend

class TrustStore:
    """
    Manages trusted root certificates, recognized publisher identities,
    and certificate revocation status for C2PA provenance validation.
    """
    def __init__(self):
        # Known trusted organizations / authorities for C2PA
        self.trusted_common_names: Set[str] = {
            "Reuters News Credential Authority",
            "Associated Press C2PA Root",
            "BBC Verified Media Authority",
            "AFP Provenance Trust Root",
            "Adobe Content Authenticity Authority",
            "NewsProof Official Trust Root",
            "Truepic Provenance Root"
        }
        
        # Revoked certificate serial numbers or fingerprints
        self.revoked_serials: Set[str] = {
            "revoked_compromised_key_001",
            "revoked_leaked_publisher_key_999"
        }

    def add_trusted_authority(self, common_name: str) -> None:
        self.trusted_common_names.add(common_name)

    def revoke_certificate(self, serial_hex: str) -> None:
        self.revoked_serials.add(serial_hex.lower())

    def verify_cert_chain(
        self,
        issuer_cn: Optional[str],
        subject_cn: Optional[str],
        serial_hex: Optional[str] = None,
        is_self_signed: bool = False
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Validates the certificate against the trust list and revocation list.
        Returns:
            (is_trusted, status_message, untrusted_warning_or_none)
        """
        if serial_hex and serial_hex.lower() in self.revoked_serials:
            return False, "REVOKED", "Certificate has been explicitly revoked by the issuing authority."

        if is_self_signed:
            return (
                False,
                "SELF_SIGNED",
                "Certificate is self-signed and not issued by an authorized Content Credentials root. Authenticity cannot be cryptographically guaranteed."
            )

        if not issuer_cn and not subject_cn:
            return (
                False,
                "UNSPECIFIED_ISSUER",
                "Signer identity could not be established from manifest credentials."
            )

        # Check if either issuer or subject matches trusted roots
        is_trusted = False
        if issuer_cn and any(trusted.lower() in issuer_cn.lower() for trusted in self.trusted_common_names):
            is_trusted = True
        elif subject_cn and any(trusted.lower() in subject_cn.lower() for trusted in self.trusted_common_names):
            is_trusted = True

        if is_trusted:
            return True, "VALID_AND_TRUSTED", None
        else:
            return (
                False,
                "UNTRUSTED_AUTHORITY",
                f"Certificate authority '{issuer_cn or subject_cn}' is not on the NewsProof verified newsroom trust list."
            )

# Global default trust store instance
default_trust_store = TrustStore()
