from typing import Dict, Any, Optional, List
from .hasher import hamming_distance

class DurableCredentialRegistry:
    """
    Durable Credentials Registry (FR-11).
    When social networks or re-uploaders strip embedded C2PA metadata,
    this registry allows recovering the signed manifest via perceptual hash (pHash)
    similarity lookup (Hamming distance threshold <= 6).
    """
    def __init__(self, max_distance: int = 6):
        self.max_distance = max_distance
        # Storage: list of registered records
        self.records: List[Dict[str, Any]] = []

    def register_manifest(
        self,
        sha256: str,
        phash: str,
        manifest_data: Dict[str, Any],
        publisher: str = "Verified News Publisher"
    ) -> None:
        """Stores asset manifest and perceptual hash in registry."""
        self.records.append({
            "sha256": sha256,
            "phash": phash,
            "manifest_data": manifest_data,
            "publisher": publisher
        })

    def lookup_by_phash(self, query_phash: str) -> Optional[Dict[str, Any]]:
        """
        Searches for registered asset whose perceptual hash matches query_phash
        within the Hamming distance tolerance.
        """
        best_match = None
        min_dist = float("inf")

        for record in self.records:
            dist = hamming_distance(query_phash, record["phash"])
            if dist <= self.max_distance and dist < min_dist:
                min_dist = dist
                best_match = {
                    "matched_record": record,
                    "distance": dist,
                    "recovered_manifest": record["manifest_data"]
                }

        return best_match

    def count(self) -> int:
        return len(self.records)

default_durable_registry = DurableCredentialRegistry()
