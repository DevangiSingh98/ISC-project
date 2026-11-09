import hashlib
import json
import time
from typing import List, Dict, Any, Optional, Tuple

def sha256_str(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

def combine_hashes(h1: str, h2: str) -> str:
    return hashlib.sha256((h1 + h2).encode("utf-8")).hexdigest()

class MerkleTransparencyLog:
    """
    Append-only Merkle-tree verification transparency ledger.
    Every verification verdict is recorded with its asset SHA-256,
    timestamp, verdict type, and model version hash.
    Provides verifiable inclusion proofs.
    """
    def __init__(self, model_versions_hash: str = "v1-trufor-fft-srm-c2pa"):
        self.model_versions_hash = model_versions_hash
        self.entries: List[Dict[str, Any]] = []
        self.leaf_hashes: List[str] = []

    def record_verdict(
        self,
        asset_sha256: str,
        verdict: str,
        confidence_score: float,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """Appends a new verification record and recomputes the Merkle root."""
        ts = timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        leaf_payload = f"{asset_sha256}:{verdict}:{confidence_score:.2f}:{self.model_versions_hash}:{ts}"
        leaf_hash = sha256_str(leaf_payload)

        entry_index = len(self.entries)
        entry = {
            "index": entry_index,
            "asset_sha256": asset_sha256,
            "verdict": verdict,
            "confidence_score": confidence_score,
            "timestamp": ts,
            "model_versions_hash": self.model_versions_hash,
            "leaf_hash": leaf_hash
        }
        self.entries.append(entry)
        self.leaf_hashes.append(leaf_hash)

        merkle_root = self.get_root()
        entry["merkle_root"] = merkle_root
        return entry

    def get_root(self) -> str:
        """Computes current Merkle root of all appended leaf hashes."""
        if not self.leaf_hashes:
            return sha256_str("empty_ledger")
        
        current_level = self.leaf_hashes.copy()
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                h1 = current_level[i]
                if i + 1 < len(current_level):
                    h2 = current_level[i + 1]
                else:
                    h2 = h1 # Duplicate odd node
                next_level.append(combine_hashes(h1, h2))
            current_level = next_level

        return current_level[0]

    def get_inclusion_proof(self, index: int) -> Tuple[str, List[Dict[str, str]]]:
        """
        Generates inclusion audit path for a leaf at given index.
        Returns:
            (leaf_hash, audit_path_pairs)
        """
        if index < 0 or index >= len(self.leaf_hashes):
            raise IndexError("Leaf index out of bounds")

        leaf_hash = self.leaf_hashes[index]
        proof: List[Dict[str, str]] = []
        
        current_level = self.leaf_hashes.copy()
        idx = index

        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                h1 = current_level[i]
                h2 = current_level[i + 1] if i + 1 < len(current_level) else h1
                next_level.append(combine_hashes(h1, h2))

            is_right_sibling = (idx % 2 == 1)
            sibling_idx = idx - 1 if is_right_sibling else idx + 1
            if sibling_idx < len(current_level):
                sibling_hash = current_level[sibling_idx]
            else:
                sibling_hash = current_level[idx]

            proof.append({
                "direction": "left" if is_right_sibling else "right",
                "hash": sibling_hash
            })

            idx = idx // 2
            current_level = next_level

        return leaf_hash, proof

    @staticmethod
    def verify_proof(leaf_hash: str, proof: List[Dict[str, str]], expected_root: str) -> bool:
        """Verifies an inclusion proof against expected Merkle root."""
        current_hash = leaf_hash
        for step in proof:
            sibling = step["hash"]
            if step["direction"] == "left":
                current_hash = combine_hashes(sibling, current_hash)
            else:
                current_hash = combine_hashes(current_hash, sibling)
        return current_hash == expected_root

# Default singleton ledger instance
default_transparency_log = MerkleTransparencyLog()
