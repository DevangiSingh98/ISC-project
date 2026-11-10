import os
import json
from typing import Dict, Any, Optional, List
import c2pa

# Default paths for signing cert and private key
DEFAULT_CERT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "test_samples", "certs")
DEFAULT_CERT_PATH = os.path.join(DEFAULT_CERT_DIR, "news_signer.pub")
DEFAULT_KEY_PATH = os.path.join(DEFAULT_CERT_DIR, "news_signer.pem")

class C2PASigner:
    """
    Publisher signing service: issues C2PA Content Credentials
    bound to media assets using authorized publisher keys.
    """
    def __init__(self, cert_path: Optional[str] = None, key_path: Optional[str] = None, alg: str = "es256"):
        self.cert_path = cert_path or DEFAULT_CERT_PATH
        self.key_path = key_path or DEFAULT_KEY_PATH
        self.alg = alg

    def sign_asset(
        self,
        input_image_path: str,
        output_image_path: str,
        title: str = "Verified News Media",
        publisher: str = "NewsProof Verified Publisher",
        actions: Optional[List[Dict[str, Any]]] = None,
        digital_source_type: str = "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"
    ) -> str:
        """
        Signs the asset at input_image_path and writes signed asset to output_image_path.
        Returns the active manifest JSON string.
        """
        if not os.path.exists(self.cert_path) or not os.path.exists(self.key_path):
            raise FileNotFoundError(f"Signer certificate or key not found at {self.cert_path}")

        with open(self.cert_path, "rb") as f:
            cert_bytes = f.read()
        with open(self.key_path, "rb") as f:
            key_bytes = f.read()

        signer_info = c2pa.C2paSignerInfo(self.alg.encode("utf-8"), cert_bytes, key_bytes, None)
        signer = c2pa.Signer.from_info(signer_info)

        if not actions:
            actions = [
                {
                    "action": "c2pa.created",
                    "digitalSourceType": digital_source_type
                }
            ]

        manifest_def = {
            "claim_generator": f"NewsProof/1.0 ({publisher})",
            "title": title,
            "assertions": [
                {
                    "label": "c2pa.actions",
                    "data": {
                        "actions": actions
                    }
                }
            ]
        }

        builder = c2pa.Builder.from_json(json.dumps(manifest_def))
        builder.sign_file(input_image_path, output_image_path, signer)

        with c2pa.Reader(output_image_path) as reader:
            return reader.json()

