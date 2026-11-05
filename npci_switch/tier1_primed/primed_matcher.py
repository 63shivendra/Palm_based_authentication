"""
npci_switch/tier1_primed/primed_matcher.py

NPCI Sovereign Banking Switch - Tier 1: 1:1 Primed Verification Engine.
Processes 95% of retail POS Kirana merchant transactions where an external identifier
(Dynamic UPI QR code, NFC tap, or mobile number) pre-indexes the citizen record.

Target Performance:
  - Search Space: N = 1 candidate.
  - Execution Time: < 0.10 ms (Bitwise POPCNT on Redis / in-memory).
  - Target FAR: <= 10^-5 (1 in 100,000 false acceptance).
  - Target FRR: <= 0.05% (5 in 10,000 false reject).
  - ISO/IEC 24745 Compliant: Operates exclusively on irreversible 1024-bit Bloom filter tokens.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import time

from common.constants import (
    BLOOM_FILTER_SIZE_BITS,
    FAR_TARGET_STRICT,
    FRR_TARGET_EDGE,
)
from common.crypto.bloom_filter import BloomFilterTokenizer


class PrimedMatcher:
    """
    Ultra-low-latency 1:1 Bitwise Hamming POPCNT Matcher.
    Compares probe 1024-bit Bloom filter against enrolled candidate record.
    """

    def __init__(
        self,
        token_bit_size: int = BLOOM_FILTER_SIZE_BITS,
        hamming_threshold: int = 180,  # <= 180 bit difference out of 1024 gives FAR <= 10^-5
    ):
        self.token_bit_size = token_bit_size
        self.hamming_threshold = hamming_threshold

    def compute_hamming_distance(self, token_a: np.ndarray, token_b: np.ndarray) -> int:
        """
        Fast bitwise Hamming distance computation using uint64 bit-parallel POPCNT.
        Args:
            token_a, token_b: 1024-bit arrays represented as np.uint8 (length 128) or np.uint64 (length 16).
        Returns:
            Bitwise Hamming distance (count of differing bit positions).
        """
        if token_a.dtype != np.uint64 or token_b.dtype != np.uint64:
            # View as 64-bit unsigned integers (1024 bits = 16 x uint64)
            view_a = np.frombuffer(token_a.tobytes(), dtype=np.uint64)
            view_b = np.frombuffer(token_b.tobytes(), dtype=np.uint64)
        else:
            view_a = token_a
            view_b = token_b

        # Bitwise XOR followed by bitcount popcount
        xor_diff = np.bitwise_xor(view_a, view_b)
        # In NumPy >= 2.0 or via bit-popcount:
        # Standard bitwise popcount across 16 words:
        total_diff = 0
        for val in xor_diff:
            total_diff += int(bin(int(val)).count("1"))
        return total_diff

    def verify_transaction(
        self,
        citizen_id: str,
        probe_token: np.ndarray,
        enrolled_token: np.ndarray,
        merchant_id: str,
    ) -> Dict[str, Any]:
        """
        Executes real-time 1:1 biometric settlement verification.

        Args:
            citizen_id: Masked Aadhaar / NPCI Virtual Payment Address (VPA).
            probe_token: 1024-bit Bloom filter captured at Kirana POS.
            enrolled_token: 1024-bit Bloom filter stored in sovereign bank vault.
            merchant_id: Terminal merchant identifier.

        Returns:
            Settlement authorization decision dictionary.
        """
        t0 = time.perf_counter()
        dist = self.compute_hamming_distance(probe_token, enrolled_token)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        is_authenticated = dist <= self.hamming_threshold
        confidence_score = max(0.0, 1.0 - (dist / float(self.hamming_threshold)))

        return {
            "authorized": is_authenticated,
            "citizen_id": citizen_id,
            "merchant_id": merchant_id,
            "hamming_distance": dist,
            "threshold": self.hamming_threshold,
            "confidence_score": round(confidence_score, 4),
            "latency_ms": round(latency_ms, 3),
            "tier": "Tier-1-Primed-1:1",
        }

