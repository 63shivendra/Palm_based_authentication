"""
tests/test_diffusion_unet.py

Unit test suite for IndiDiff-Palm generative diffusion architecture:
  - CrossSpectralBidirectionalAttentionBridge (CS-BAB) cross-conditioning.
  - PhysicsConditioningEncoder multi-modal encoding.
  - DualStreamDiffusionUNet noise prediction for coupled RGB + NIR streams.
"""

import pytest
import torch

from indidiff_palm.models.cross_attention_bridge import CrossSpectralBidirectionalAttentionBridge
from indidiff_palm.models.conditioning_encoder import PhysicsConditioningEncoder
from indidiff_palm.models.dual_stream_unet import DualStreamDiffusionUNet


def test_cross_spectral_bridge():
    """Verifies that the CS-BAB bridge performs bidirectional attention exchange."""
    bridge = CrossSpectralBidirectionalAttentionBridge(channels=32, num_heads=4)
    f_rgb = torch.randn(2, 32, 16, 16)
    f_nir = torch.randn(2, 32, 16, 16)

    out_rgb, out_nir = bridge(f_rgb, f_nir)
    assert out_rgb.shape == f_rgb.shape
    assert out_nir.shape == f_nir.shape


def test_conditioning_encoder():
    """Verifies multi-modal physics condition projection."""
    cond_enc = PhysicsConditioningEncoder(cond_channels=32, id_dim=512)

    depth = torch.rand(2, 1, 32, 32)
    mehndi = torch.rand(2, 1, 32, 32)
    melanin = torch.rand(2, 1)
    z_id = torch.randn(2, 512)

    spatial_cond, global_cond = cond_enc(depth, mehndi, melanin, z_id)
    assert spatial_cond.shape == (2, 32, 32, 32)
    assert global_cond.shape == (2, 32)


def test_dual_stream_unet_forward():
    """Verifies end-to-end forward pass predicting joint RGB + NIR noise."""
    unet = DualStreamDiffusionUNet(base_channels=16, time_emb_dim=64, cond_channels=16)

    b, h, w = 2, 32, 32
    x_rgb = torch.randn(b, 3, h, w)
    x_nir = torch.randn(b, 1, h, w)
    t = torch.tensor([10, 250], dtype=torch.long)

    depth = torch.rand(b, 1, h, w)
    mehndi = torch.rand(b, 1, h, w)
    melanin = torch.rand(b, 1)
    z_id = torch.randn(b, 512)

    eps_rgb, eps_nir = unet(x_rgb, x_nir, t, depth, mehndi, melanin, z_id)

    assert eps_rgb.shape == (b, 3, h, w)
    assert eps_nir.shape == (b, 1, h, w)
    assert not torch.isnan(eps_rgb).any()
    assert not torch.isnan(eps_nir).any()

