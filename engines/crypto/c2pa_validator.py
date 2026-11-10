import json
import os
import tempfile
from typing import Dict, Any, Optional, List
import c2pa
from apps.api.app.schemas.verdict import ProvenanceDetails
from .trust_store import default_trust_store, TrustStore

class C2PAValidator:
    """
    Validates C2PA manifests, verifies cryptographic hash bindings,
    and checks certificate chains against trusted newsroom authorities.
    """
    def __init__(self, trust_store: Optional[TrustStore] = None):
        self.trust_store = trust_store or default_trust_store

    def validate_file(self, file_path: str) -> ProvenanceDetails:
        """Parses and validates C2PA Content Credentials from a media file path."""
        try:
            with c2pa.Reader(file_path) as reader:
                manifest_json_str = reader.json()
        except c2pa.C2paError:
            # Asset has no C2PA manifest or unrecognized container
            return ProvenanceDetails(has_c2pa=False, is_valid=False)
        except Exception:
            return ProvenanceDetails(has_c2pa=False, is_valid=False)

        if not manifest_json_str or manifest_json_str.strip() == "{}":
            return ProvenanceDetails(has_c2pa=False, is_valid=False)

        try:
            store_data = json.loads(manifest_json_str)
        except Exception:
            return ProvenanceDetails(has_c2pa=True, is_valid=False)

        active_id = store_data.get("active_manifest")
        if not active_id:
            return ProvenanceDetails(has_c2pa=False, is_valid=False)

        manifest = store_data.get("manifests", {}).get(active_id, {})
        val_status = store_data.get("validation_status", [])

        # Check for hash mismatch or tampering
        hash_match = True
        signature_broken = False
        untrusted_found = False

        for status_item in val_status:
            code = status_item.get("code", "")
            if "mismatch" in code.lower() or "hash" in code.lower():
                hash_match = False
            if "signature" in code.lower() and "untrusted" not in code.lower():
                signature_broken = True
            if "untrusted" in code.lower():
                untrusted_found = True

        sig_info = manifest.get("signature_info", {})
        issuer = sig_info.get("issuer")
        common_name = sig_info.get("common_name")
        serial = sig_info.get("cert_serial_number")

        # Trust store check
        is_trusted, trust_status, untrusted_warning = self.trust_store.verify_cert_chain(
            issuer_cn=issuer,
            subject_cn=common_name,
            serial_hex=serial,
            is_self_signed=untrusted_found
        )

        # Extract declared actions
        declared_actions: List[str] = []
        assertions = manifest.get("assertions", [])
        for assertion in assertions:
            label = assertion.get("label", "")
            if "actions" in label:
                action_data = assertion.get("data", {}).get("actions", [])
                for act in action_data:
                    action_name = act.get("action")
                    if action_name:
                        declared_actions.append(action_name)

        # Extract edit history / ingredients
        edit_history: List[Dict[str, Any]] = []
        ingredients = manifest.get("ingredients", [])
        for ing in ingredients:
            edit_history.append({
                "title": ing.get("title"),
                "relationship": ing.get("relationship"),
                "format": ing.get("format")
            })

        is_valid = hash_match and not signature_broken

        return ProvenanceDetails(
            has_c2pa=True,
            is_valid=is_valid,
            is_trusted=is_trusted,
            issuer=common_name or issuer,
            cert_chain_status=trust_status,
            hash_match=hash_match,
            declared_actions=declared_actions,
            edit_history=edit_history,
            raw_manifest=manifest,
            untrusted_warning=untrusted_warning
        )

    def validate_bytes(self, data: bytes, extension: str = ".jpg") -> ProvenanceDetails:
        """Validates C2PA Content Credentials from raw bytes via a temporary file."""
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            return self.validate_file(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

