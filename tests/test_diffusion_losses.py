"""
tests/test_diffusion_losses.py

Unit test suite for IndiDiff-Palm generative training losses:
  - WaveletDirectionalLoss: 2D Haar DWT directional high-frequency subbands [LH, HL, HH].
  - MurrayHemodynamicLoss: Cubic radius conservation (r_p^3 = r_d1^3 + r_d2^3) and vascular continuity.
"""

import pytest
import torch

from indidiff_palm.losses.wavelet_loss import WaveletDirectionalLoss
from indidiff_palm.losses.murray_loss import MurrayHemodynamicLoss


def test_wavelet_directional_loss():
    """Wavelet directional loss should be zero for identical inputs and positive for dissimilar images."""
    loss_fn = WaveletDirectionalLoss(in_channels=1, levels=2)

    pred = torch.randn(2, 1, 64, 64)
    target = pred.clone()

    zero_loss = loss_fn(pred, target)
    assert zero_loss.item() < 1e-5

    # Blurry image should have high high-frequency wavelet loss
    blurred = torch.nn.functional.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
    blur_loss = loss_fn(blurred, pred)
    assert blur_loss.item() > 0.05


def test_murray_hemodynamic_loss():
    """Murray loss should be zero for identical vascular profiles and non-negative for perturbed maps."""
    murray_fn = MurrayHemodynamicLoss()

    nir_pred = torch.full((1, 1, 64, 64), 0.5)
    nir_targ = nir_pred.clone()

    identical_loss = murray_fn(nir_pred, nir_targ)
    assert identical_loss.item() < 1e-4

    # Adding artificial fragmented noise increases hemodynamic continuity and profile loss
    noisy_nir = nir_pred + torch.randn_like(nir_pred) * 0.2
    noisy_loss = murray_fn(noisy_nir, nir_targ)
    assert noisy_loss.item() > 1e-3

