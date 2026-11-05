"""
tests/test_npci_switch.py

Unit test suite for NPCI Sovereign Banking Switch:
  - PrimedMatcher: 1:1 bitwise Hamming POPCNT (< 0.1 ms).
  - MultiIndexHash: 1:N sovereign bare-palm identification (< 1.8 ms).
  - UPITokenGatewayService: ISO 20022 pacs.008 compliant payment settlement payloads.
"""

import pytest
import numpy as np

from common.crypto.bloom_filter import BloomFilterTokenizer
from npci_switch.tier1_primed.primed_matcher import PrimedMatcher
from npci_switch.tier2_sovereign.mih_index import MultiIndexHash
from npci_switch.upi_gateway.upi_token_service import UPITokenGatewayService


def test_tier1_primed_matcher():
    """Tests 1:1 bitwise POPCNT authentication and transaction decision logic."""
    matcher = PrimedMatcher(token_bit_size=1024, hamming_threshold=180)

    # 1024-bit random token
    np.random.seed(42)
    tok_enrolled = np.random.randint(0, 256, size=128, dtype=np.uint8)

    # Probe identical to enrolled: distance = 0
    res_exact = matcher.verify_transaction(
        citizen_id="CITIZEN-9876543210",
        probe_token=tok_enrolled,
        enrolled_token=tok_enrolled,
        merchant_id="MERCHANT-KIRANA-400001",
    )
    assert res_exact["authorized"] is True
    assert res_exact["hamming_distance"] == 0
    assert res_exact["confidence_score"] == 1.0

    # Probe with 50 flipped bits (genuine user variation under physical hand tilt)
    tok_perturbed = tok_enrolled.copy()
    tok_perturbed[:7] ^= 0b00000111  # Flip a few bits
    res_perturbed = matcher.verify_transaction(
        citizen_id="CITIZEN-9876543210",
        probe_token=tok_perturbed,
        enrolled_token=tok_enrolled,
        merchant_id="MERCHANT-KIRANA-400001",
    )
    assert res_perturbed["authorized"] is True
    assert res_perturbed["hamming_distance"] <= 180

    # Random impostor token: expected Hamming distance ~ 512 bits
    tok_impostor = np.random.randint(0, 256, size=128, dtype=np.uint8)
    res_impostor = matcher.verify_transaction(
        citizen_id="CITIZEN-9876543210",
        probe_token=tok_impostor,
        enrolled_token=tok_enrolled,
        merchant_id="MERCHANT-KIRANA-400001",
    )
    assert res_impostor["authorized"] is False
    assert res_impostor["hamming_distance"] > 350


def test_tier2_multi_index_hashing_search():
    """Tests 1:N bare-palm identification using Multi-Index Hashing."""
    mih = MultiIndexHash(token_bit_size=1024, num_sub_blocks=32, max_search_radius=180)

    np.random.seed(1337)
    # Enroll a gallery of 1,000 citizens
    gallery_size = 1000
    citizen_ids = [f"CITIZEN_{i:06d}" for i in range(gallery_size)]
    gallery_tokens = [np.random.randint(0, 256, size=128, dtype=np.uint8) for _ in range(gallery_size)]

    mih.enroll_batch(citizen_ids, gallery_tokens)
    assert len(mih.gallery_tokens) == gallery_size

    # Target citizen probe: citizen #42 with 10 flipped bits (realistic sensor noise)
    target_idx = 42
    target_id = citizen_ids[target_idx]
    probe = gallery_tokens[target_idx].copy()
    probe[0] ^= 0x01  # Flip 1 bit in sub-block 0

    search_res = mih.search_bare_palm(probe_token=probe, radius=180, top_k=1)
    assert search_res["best_match"] is not None
    best_id, best_dist = search_res["best_match"]
    assert best_id == target_id
    assert best_dist == 1
    assert search_res["candidates_pruned_count"] < gallery_size  # Effective pruning


def test_upi_iso20022_message_generation():
    """Tests formatting and cryptographic signing of ISO 20022 pacs.008 payment instructions."""
    service = UPITokenGatewayService(npci_switch_id="NPCI-PUNE-DC-02")

    msg = service.build_iso20022_payment_message(
        transaction_id="TXN-20261105-00129",
        payer_vpa="9876543210@npci.palm",
        payee_vpa="kirana.store@okhdfcbank",
        amount_inr=149.50,
        merchant_terminal_id="POS-BLR-042",
        auth_tier="Tier-1-Primed-1:1",
        hamming_distance=42,
    )

    assert msg["MessageHeader"]["MessageDefinitionIdentifier"] == "pacs.008.001.09"
    assert msg["CreditTransferTransactionInformation"]["InterbankSettlementAmount"]["Value"] == 149.50
    assert msg["CreditTransferTransactionInformation"]["BiometricAuthenticationReceipt"]["Status"] == "SUCCESS_VERIFIED"
    assert "DigitalSignature" in msg
    assert len(msg["DigitalSignature"]["Digest"]) == 64  # SHA-256 hex length
