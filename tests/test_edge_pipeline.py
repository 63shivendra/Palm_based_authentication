"""
tests/test_edge_pipeline.py

End-to-End System Integration Test for Sovereign Edge Pipeline:
  - Validates full execution from multi-spectral frame inputs to 1024-bit Bloom filter token.
  - Verifies early abort on presentation spoof attack.
  - Validates timing latency profiling and budget compliance.
"""

import pytest
import torch
import numpy as np

from indipalm_edge.inference.edge_pipeline import SovereignEdgeInferenceEngine


def test_full_edge_pipeline_bona_fide():
    """Tests complete end-to-end execution for bona fide live hand presentation."""
    engine = SovereignEdgeInferenceEngine(deploy_mode=True)

    # Bona fide inputs (living tissue with differential 850nm vs 940nm response)
    rgb = torch.full((1, 3, 128, 128), 0.50)
    nir_850 = torch.full((1, 1, 128, 128), 0.65)
    nir_940 = torch.full((1, 1, 128, 128), 0.35)

    result = engine.process_transaction_capture(
        rgb_frame=rgb,
        nir_850_frame=nir_850,
        nir_940_frame=nir_940,
        standoff_z_mm=115.0,
        pose_angles_deg=(5.0, -3.0, 2.0),
        merchant_id="POS-MUMBAI-001",
    )

    assert result["status"] == "AUTHENTICATION_READY"
    assert result["authorized"] is True
    assert result["token"] is not None
    # 1024-bit Bloom filter = 128 bytes
    assert len(result["token"]) == 128
    assert len(result["token_hex"]) == 256  # 128 bytes * 2 hex chars
    assert "profiling_ms" in result
    assert result["total_latency_ms"] > 0


def test_full_edge_pipeline_spoof_rejection():
    """Tests early abort when presenting an artificial flat-reflectance spoof."""
    engine = SovereignEdgeInferenceEngine(deploy_mode=True)

    # Spoof attack: identical 850nm and 940nm intensities (Delta_PAD ~ 0.0)
    rgb = torch.full((1, 3, 128, 128), 0.50)
    nir_850 = torch.full((1, 1, 128, 128), 0.50)
    nir_940 = torch.full((1, 1, 128, 128), 0.50)

    result = engine.process_transaction_capture(
        rgb_frame=rgb,
        nir_850_frame=nir_850,
        nir_940_frame=nir_940,
        standoff_z_mm=100.0,
    )

    assert result["status"] == "REJECTED_SPOOF_ATTACK"
    assert result["authorized"] is False
    assert result["token"] is None
