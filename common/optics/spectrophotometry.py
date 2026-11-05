"""
Spectrophotometric Skin Optics & Presentation Attack Detection.

Implements biophysical equations governing light transmission, melanin absorption,
hemoglobin deoxygenation, and active differential PAD ratios complying with ISO/IEC 30107-3.

Copyright (c) 2026 Sovereign Biometrics Architecture Team.
Licensed under the Apache License, Version 2.0.
"""

import numpy as np
from typing import Tuple
from ..constants import (
    WAVELENGTH_NIR_PRIMARY,
    WAVELENGTH_NIR_SECONDARY,
    WAVELENGTH_GREEN_REF,
    MELANIN_EXTINCTION_BASE,
    MELANIN_EXPONENT,
    DELTA_PAD_LIVING_MIN,
    DELTA_PAD_LIVING_MAX,
    DELTA_PAD_SPOOF_MAX,
    ALPHA_HEMOGLOBIN_WATER_WEIGHT,
)


def compute_epidermal_melanin_absorption(wavelength_nm: float, melanin_fraction: float) -> float:
    """
    Computes epidermal absorption coefficient mu_a_epi(lambda) in cm^-1.

    Mathematical Formulation (Jacques, 2013):
        mu_a_epi(lambda) = C_mel * 6.6e11 * (lambda)^{-3.48} + (1 - C_mel) * mu_a_baseline(lambda)
    """
    c_mel = np.clip(melanin_fraction, 0.01, 0.45)
    mu_a_mel = MELANIN_EXTINCTION_BASE * (wavelength_nm ** MELANIN_EXPONENT)
    mu_a_baseline = 0.244 + 85.3 * np.exp(-(wavelength_nm - 154.0) / 66.2)
    return float(c_mel * mu_a_mel + (1.0 - c_mel) * mu_a_baseline)


def compute_differential_pad_ratio(
    intensity_850: np.ndarray,
    intensity_940: np.ndarray,
    alpha: float = ALPHA_HEMOGLOBIN_WATER_WEIGHT,
) -> float:
    """
    Computes the Palmar Core Differential Index Delta_PAD for active liveness verification.

    Mathematical Formulation:
        Delta_PAD = ( I_850 - alpha * I_940 ) / ( I_850 + alpha * I_940 )

    Biophysical Grounding:
        At 850 nm: Deoxygenated hemoglobin exhibits strong absorption peak.
        At 940 nm: Subcutaneous water molecules exhibit strong harmonic vibrational absorption.
        Living vascular tissue: Delta_PAD in [0.18, 0.42].
        Silicone/latex molds, paper prints, OLED screens: Delta_PAD in [-0.03, +0.03].
    """
    mean_850 = np.mean(intensity_850)
    mean_940 = np.mean(intensity_940)

    numerator = mean_850 - alpha * mean_940
    denominator = mean_850 + alpha * mean_940 + 1e-12

    return float(numerator / denominator)


def verify_liveness_iso30107(delta_pad: float) -> Tuple[bool, str]:
    """
    Evaluates Presentation Attack Detection compliance against ISO/IEC 30107-3.
    """
    if DELTA_PAD_LIVING_MIN <= delta_pad <= DELTA_PAD_LIVING_MAX:
        return True, "GENUINE_LIVING_DERMIS"
    if abs(delta_pad) <= DELTA_PAD_SPOOF_MAX:
        return False, "PRESENTATION_ATTACK_SPOOF_FLAT_SPECTRUM"
    if delta_pad < DELTA_PAD_LIVING_MIN:
        return False, "PRESENTATION_ATTACK_ABNORMAL_WATER_RATIO"
    return False, "PRESENTATION_ATTACK_OPTICAL_SATURATION"


def compute_melanin_factor_cielab(l_star: float, a_star: float, b_star: float) -> float:
    """
    Computes Normalized Difference Skin Index (NDSI) melanin factor xi_mel
    from CIE L*a*b* color coordinates to prevent demographic rejection across
    Fitzpatrick skin types IV through VI.

    xi_mel = 1.0 - (L* / 100.0) + 0.2 * (100.0 / (a* - b* + 1e-6))
    """
    l_norm = np.clip(l_star / 100.0, 0.0, 1.0)
    diff = max(1.0, abs(a_star - b_star))
    xi_mel = 1.0 - l_norm + 0.2 * (100.0 / diff)
    return float(np.clip(xi_mel, 0.0, 3.0))


def lawsone_dye_transmission(wavelength_nm: float, oxidation_age_days: float = 2.0) -> float:
    """
    Simulates optical transmittance of Lawsone (2-hydroxy-1,4-naphthoquinone) dye
    across the palmar stratum corneum.

    At lambda = 850 nm: Transmittance is IDENTICALLY 1.0 (Zero NIR absorption).
    At lambda = 525 nm: Transmittance drops to 0.05 - 0.20 (Dense visible absorption).
    """
    if wavelength_nm >= 800.0:
        return 1.0  # Transparent in Near-Infrared

    # In visible band (400 - 700 nm)
    decay = np.clip(oxidation_age_days / 7.0, 0.1, 1.0)
    transmittance = 1.0 / (1.0 + np.exp(-(wavelength_nm - 580.0) / 30.0))
    return float(np.clip(transmittance * (1.0 - 0.8 * decay), 0.02, 1.0))



def simulate_physical_tissue_reflectance(
    wavelength_nm: float,
    is_living: bool = True,
    water_fraction: float = 0.70,
    melanin_fraction: float = 0.35,
    blood_fraction: float = 0.05,
) -> float:
    """
    Simulates diffuse reflectance R_d(lambda) of palmar tissue or presentation spoof
    from first-principles radiative transfer and spectrophotometry.

    Physical Constants:
        At 850 nm: mu_a(water) = 0.043 cm^-1, mu_a(deoxy-Hb) = 0.080 cm^-1
        At 940 nm: mu_a(water) = 0.290 cm^-1 (Harmonic vibrational absorption peak)
    """
    # Dermal reduced scattering (mm^-1 converted to cm^-1)
    mu_s_prime = (1.1e3 * (wavelength_nm ** -1.0)) * 10.0

    if not is_living:
        # Inorganic silicone / paper / display spoof: flat scattering, zero water/Hb resonance
        mu_a_spoof = 0.15 + 0.005 * ((wavelength_nm - 850.0) / 100.0)
        return float(mu_s_prime / (mu_s_prime + mu_a_spoof))

    # Living tissue: water + hemoglobin + baseline
    if abs(wavelength_nm - 850.0) <= 20.0:
        mu_a_water = 0.043
        mu_a_blood = 0.080
    elif abs(wavelength_nm - 940.0) <= 20.0:
        mu_a_water = 0.290  # 6.7x higher water absorption!
        mu_a_blood = 0.075
    else:
        mu_a_water = 0.050
        mu_a_blood = 0.070

    # Total absorption: mu_a = V_water * mu_a_water + V_blood * mu_a_blood + baseline
    mu_a_total = (
        water_fraction * mu_a_water
        + blood_fraction * mu_a_blood
        + compute_epidermal_melanin_absorption(wavelength_nm, melanin_fraction) * 0.01
    )

    # Standard diffuse reflectance formula: R_d = mu_s' / (mu_s' + mu_a)
    r_diffuse = mu_s_prime / (mu_s_prime + mu_a_total)
    return float(np.clip(r_diffuse, 0.05, 0.95))


def compute_sensor_detected_signal(
    wavelength_nm: float,
    is_living: bool = True,
    melanin_fraction: float = 0.35,
    exposure_us: float = 40.0,
    optical_power_w: float = 5.0,
) -> float:
    """
    Computes physical digital signal (0-255 DN) recorded by Sony IMX296 CMOS sensor
    under 40 us pulsed VCSEL illumination.

    Incorporates:
    1. Diffuse tissue reflectance R_d(lambda).
    2. Wavelength-dependent CMOS Silicon Quantum Efficiency:
       eta_QE(850 nm) approx 0.28
       eta_QE(940 nm) approx 0.15
    """
    r_d = simulate_physical_tissue_reflectance(
        wavelength_nm=wavelength_nm,
        is_living=is_living,
        melanin_fraction=melanin_fraction,
    )

    # Silicon CMOS quantum efficiency curve
    if wavelength_nm <= 860.0:
        eta_qe = 0.28
    else:
        eta_qe = 0.15

    # Optical photon energy h * nu
    h_planck = 6.626e-34
    c_light = 3.0e8
    photon_energy = (h_planck * c_light) / (wavelength_nm * 1e-9)

    # Incident photons during 40 us
    optical_energy_j = optical_power_w * (exposure_us * 1e-6)
    photons_emitted = optical_energy_j / photon_energy

    # Detected electrons: Q = photons * r_d * eta_qe * geometric_factor
    geometric_factor = 2.5e-13  # Aperture f/1.8 collection solid angle
    electrons = photons_emitted * r_d * eta_qe * geometric_factor

    # Conversion to 8-bit digital number (gain = 255 / 10500 e- full-well)
    digital_output = np.clip(electrons * (255.0 / 10500.0), 5.0, 250.0)
    return float(digital_output)
