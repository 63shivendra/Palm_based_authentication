"""
tests/test_rep_msfnet.py

Unit test suite for Rep-MSFNet:
  - RepConv2d multi-branch execution.
  - Algebraic structural reparameterization: verifies that the fused single 3x3 Conv
    produces identical outputs to the multi-branch block within FP32 numerical tolerance (< 1e-4).
  - HaarDWT2d 2D wavelet sub-band decomposition [LL, LH, HL, HH].
  - Dual-stream feature extraction and unit hypersphere embedding projection.
"""

import pytest
import torch
import torch.nn as nn

from indipalm_edge.models.rep_msfnet import (
    RepConv2d,
    HaarDWT2d,
    SpatialFrequencyBlock,
    RepMSFNetStream,
    RepMSFNet,
)


def test_repconv2d_reparameterization():
    """
    Verifies that RepConv2d outputs before and after structural reparameterization
    match with high numerical precision (max absolute difference < 1e-4).
    """
    torch.manual_seed(42)
    in_c, out_c = 16, 16
    x = torch.randn(2, in_c, 32, 32)

    rep_block = RepConv2d(in_channels=in_c, out_channels=out_c, stride=1)
    rep_block.eval()  # Must be in eval mode for deterministic BatchNorm behavior

    with torch.no_grad():
        out_multi_branch = rep_block(x)

    # Reparameterize into single Conv3x3
    rep_block.reparameterize()
    assert rep_block.deploy is True
    assert hasattr(rep_block, "rbr_reparam")
    assert not hasattr(rep_block, "rbr_dense")

    with torch.no_grad():
        out_fused = rep_block(x)

    max_diff = (out_multi_branch - out_fused).abs().max().item()
    assert max_diff < 1e-4, f"Fused Conv mismatch too large: {max_diff}"


def test_haar_dwt2d_decomposition():
    """Verifies Haar 2D DWT subband dimensions and energy conservation properties."""
    x = torch.randn(2, 4, 32, 32)
    dwt = HaarDWT2d(in_channels=4)

    ll, lh, hl, hh = dwt(x)
    assert ll.shape == (2, 4, 16, 16)
    assert lh.shape == (2, 4, 16, 16)
    assert hl.shape == (2, 4, 16, 16)
    assert hh.shape == (2, 4, 16, 16)

    # Constant uniform image should concentrate all energy in LL with zero in LH, HL, HH
    constant_img = torch.ones(1, 1, 16, 16)
    dwt_1c = HaarDWT2d(in_channels=1)
    ll_c, lh_c, hl_c, hh_c = dwt_1c(constant_img)
    assert torch.allclose(ll_c, torch.ones_like(ll_c) * 2.0, atol=1e-5)
    assert torch.allclose(lh_c, torch.zeros_like(lh_c), atol=1e-5)
    assert torch.allclose(hl_c, torch.zeros_like(hl_c), atol=1e-5)
    assert torch.allclose(hh_c, torch.zeros_like(hh_c), atol=1e-5)


def test_rep_msfnet_full_pipeline():
    """Tests the full dual-stream RepMSFNet network and embedding projection."""
    model = RepMSFNet(base_channels=16, embedding_dim=512)
    model.eval()

    rgb = torch.randn(2, 3, 128, 128)
    nir = torch.randn(2, 1, 128, 128)

    rgb_f2, rgb_f3, nir_f2, nir_f3 = model.forward_features(rgb, nir)
    assert rgb_f2.shape == (2, 32, 32, 32)
    assert rgb_f3.shape == (2, 64, 16, 16)
    assert nir_f2.shape == (2, 32, 32, 32)
    assert nir_f3.shape == (2, 64, 16, 16)

    # Embedding projection
    emb = model.project_embedding(rgb_f3)
    assert emb.shape == (2, 512)
    # L2 normalized on unit sphere S^511
    norms = torch.norm(emb, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

