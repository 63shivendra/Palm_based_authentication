"""
tests/test_active_pad.py

Unit test suite for Active Multi-Spectral Stroboscopic PAD (ISO/IEC 30107-3):
  - Palmar Core Differential Index (Delta_PAD) calculation.
  - Rejection of synthetic presentation attacks (flat spectral response Delta_PAD ~ 0.0).
  - Demographic melanin calibration across Fitzpatrick skin types IV through VI.
  - Screen glare / specular highlight detection.
"""

import pytest
import torch

from indipalm_edge.models.active_pad import ActiveStroboscopicPAD


def test_bona_fide_living_tissue_acceptance():
    """Bona fide living dermis with differential Hb/HbO2 absorption should be accepted as live."""
    pad = ActiveStroboscopicPAD()

    # Living tissue: 850nm backscatter is significantly higher than 940nm
    # (e.g. I_850 = 0.65, I_940 = 0.35 -> Delta_PAD = (0.65 - 0.35)/(0.65 + 0.35) = 0.30)
    img_850 = torch.full((1, 1, 128, 128), 0.65)
    img_940 = torch.full((1, 1, 128, 128), 0.35)
    rgb = torch.full((1, 3, 128, 128), 0.50)  # Diffuse skin tone

    result = pad(img_850, img_940, rgb)
    assert bool(result["is_live"][0].item()) is True
    assert 0.18 <= float(result["delta_pad"][0].item()) <= 0.42
    assert float(result["liveness_score"][0].item()) > 0.70


def test_presentation_attack_rejection_silicone():
    """Silicone / 2D photographic attacks with flat spectral reflectance should be rejected."""
    pad = ActiveStroboscopicPAD()

    # Synthetic spoof: identical reflectance under 850nm and 940nm (Delta_PAD ~ 0.0)
    img_850 = torch.full((1, 1, 128, 128), 0.50)
    img_940 = torch.full((1, 1, 128, 128), 0.50)
    rgb = torch.full((1, 3, 128, 128), 0.50)

    result = pad(img_850, img_940, rgb)
    assert bool(result["is_live"][0].item()) is False
    assert abs(float(result["delta_pad"][0].item())) < 0.05


def test_screen_glare_attack_rejection():
    """Phone / tablet screens showing a palm image produce specular glares and saturated pixels."""
    pad = ActiveStroboscopicPAD()

    img_850 = torch.full((1, 1, 128, 128), 0.65)
    img_940 = torch.full((1, 1, 128, 128), 0.35)

    # RGB image with glass reflection / saturated hot spot
    rgb = torch.full((1, 3, 128, 128), 0.50)
    rgb[:, :, 50:70, 50:70] = 0.99  # Saturated specular reflection

    result = pad(img_850, img_940, rgb)
    assert float(result["specular_score"][0].item()) > 0.60
    assert bool(result["is_live"][0].item()) is False


def test_demographic_calibration_fitzpatrick():
    """
    Verifies that melanin calibration dynamically modulates lower bounds
    so that deeply pigmented skin (Fitzpatrick VI) is not falsely rejected.
    """
    pad = ActiveStroboscopicPAD()

    # Fitzpatrick IV (fairer brown, higher luminance L*)
    rgb_fitz4 = torch.full((1, 3, 128, 128), 0.60)
    tau_min4, tau_max4 = pad.calibrate_demographic_thresholds(rgb_fitz4)

    # Fitzpatrick VI (deep rich melanin, lower luminance L*)
    rgb_fitz6 = torch.full((1, 3, 128, 128), 0.20)
    tau_min6, tau_max6 = pad.calibrate_demographic_thresholds(rgb_fitz6)

    # Fitzpatrick VI should have a more accommodating lower threshold to prevent demographic false rejections
    assert float(tau_min6.item()) < float(tau_min4.item())

