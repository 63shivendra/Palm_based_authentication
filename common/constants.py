"""
IndiPalm-SovereignNet: Physical, Optical, and Cryptographic System Constants.

Copyright (c) 2026 Sovereign Biometrics Architecture Team.
Licensed under the Apache License, Version 2.0.
"""

from dataclasses import dataclass
from typing import Tuple

# ==============================================================================
# Optical & Radiative Transfer Constants
# ==============================================================================
WAVELENGTH_NIR_PRIMARY: float = 850.0       # Primary vein imaging band (nm)
WAVELENGTH_NIR_SECONDARY: float = 940.0     # Differential PAD water-absorption band (nm)
WAVELENGTH_GREEN_REF: float = 525.0         # Visible melanin reference band (nm)
WAVELENGTH_RED_REF: float = 630.0           # Visible Lawsone dye reference band (nm)

OPTICAL_FILTER_CWL: float = 850.0           # Central Wavelength (nm)
OPTICAL_FILTER_FWHM: float = 20.0           # Full Width at Half Maximum (nm) [840 nm - 860 nm]
OPTICAL_FILTER_OD_OUT_OF_BAND: float = 4.0  # Optical Density >= 4 outside band (10^-4 transmittance)

VCSEL_PULSE_DURATION_US: float = 40.0       # Global shutter gate & VCSEL strobe duration (microseconds)
VCSEL_PEAK_OPTICAL_POWER_W: float = 5.0     # Peak optical power during 40 us strobe (Watts)
SOLAR_REJECTION_RATIO: float = 0.982        # 98.2% ambient solar photon rejection

# Refractive Indices of Human Skin Layers
REFRACTIVE_INDEX_AIR: float = 1.000
REFRACTIVE_INDEX_STRATUM_CORNEUM: float = 1.530
REFRACTIVE_INDEX_EPIDERMIS: float = 1.430
REFRACTIVE_INDEX_DERMIS: float = 1.400
REFRACTIVE_INDEX_LIPID_FILM: float = 1.480

# Spectrophotometric Constants for Epidermal Melanin (Jacques, 2013)
MELANIN_EXTINCTION_BASE: float = 6.6e11     # Base absorption coefficient factor
MELANIN_EXPONENT: float = -3.48             # Power-law wavelength exponent

# Reduced Scattering Coefficient in Dermis (Meglinski & Matcher, 2002)
DERMAL_SCATTERING_BASE: float = 1.1e3       # Base scattering factor (mm^-1)
DERMAL_SCATTERING_EXPONENT: float = -1.0    # Rayleigh-Mie power exponent
DERMAL_ANISOTROPY_G: float = 0.88           # Forward scattering anisotropy factor

# ==============================================================================
# Biomechanical & Ergonomic Constants
# ==============================================================================
STANDOFF_MIN_MM: float = 80.0               # Minimum ergonomic hover distance (mm)
STANDOFF_MAX_MM: float = 150.0              # Maximum ergonomic hover distance (mm)
MAX_TILT_ROLL_DEG: float = 35.0             # Maximum pitch / roll presentation angle (degrees)
MAX_CUPPING_ANGLE_DEG: float = 30.0         # Maximum hand cupping curvature (degrees)
STEREO_BASELINE_MM: float = 18.0            # Horizontal optical center baseline (mm)

# ==============================================================================
# Presentation Attack Detection (ISO/IEC 30107-3)
# ==============================================================================
DELTA_PAD_LIVING_MIN: float = 0.18          # Lower bound for living dermis differential ratio
DELTA_PAD_LIVING_MAX: float = 0.42          # Upper bound for living dermis differential ratio
DELTA_PAD_SPOOF_MAX: float = 0.05           # Maximum ratio for silicone/paper/display spoofs
ALPHA_HEMOGLOBIN_WATER_WEIGHT: float = 1.05 # Photometric balancing coefficient alpha

# ==============================================================================
# Cancellable Biometrics (ISO/IEC 24745)
# ==============================================================================
FEATURE_EMBEDDING_DIM: int = 512            # Continuous hypersphere embedding dimension (z in S^511)
IOM_NUM_HASHES_K: int = 128                 # Number of ordinal rank projections (t_1 ... t_K)
IOM_SUBSET_SIZE_M: int = 16                 # Subset projection cardinality |S_k|
BLOOM_FILTER_LENGTH_BITS: int = 1024        # Locality-Sensitive Bloom Filter token length (bits)
MAX_PERMISSIBLE_D_SYS: float = 0.011        # Maximum system unlinkability metric (ISO/IEC 24745)
STOLEN_TOKEN_COSINE_CEILING: float = 0.28   # Theoretical recovery cosine ceiling under LP attacks

# ==============================================================================
# Central Banking Switch Scalability (NPCI / UPI)
# ==============================================================================
TIER1_HAMMING_THRESHOLD: int = 185          # Bitwise Hamming distance verification threshold (out of 1024)
TIER2_MIH_NUM_SUBBLOCKS: int = 16           # MIH partitioned sub-blocks (m = 16 blocks of 64 bits)
TIER2_MIH_SUBBLOCK_BITS: int = 64           # Bits per sub-block (1024 / 16 = 64 bits)
BANKING_FRAUD_FAR_CEILING: float = 1e-5     # 1 in 100,000 False Acceptance Rate threshold
CUSTOMER_EXPERIENCE_FRR_CEILING: float = 0.0008 # 0.08% False Rejection Rate threshold

# ==============================================================================
# NXP i.MX8M Plus Edge Hardware Execution Budgets
# ==============================================================================
EDGE_LATENCY_BUDGET_MS: float = 36.0        # Strict hardware execution budget (ms)
EDGE_MEASURED_TARGET_MS: float = 30.1       # Profiling target latency on VeriSilicon VIP8000 NPU (ms)
EDGE_POWER_BUDGET_WATTS: float = 3.5        # Maximum thermal power dissipation (Watts)
ROI_IMAGE_SIZE: Tuple[int, int] = (128, 128)# Standardized palm ROI resolution (pixels)

# ==============================================================================
# Canonical System Constants & Aliases
# ==============================================================================
PALM_ROI_SIZE: Tuple[int, int] = (128, 128)
EMBEDDING_DIM: int = FEATURE_EMBEDDING_DIM
BLOOM_FILTER_SIZE_BITS: int = BLOOM_FILTER_LENGTH_BITS
IOM_NUM_HASH_FUNCTIONS: int = IOM_NUM_HASHES_K
EDGE_MAX_TOTAL_LATENCY_MS: float = EDGE_LATENCY_BUDGET_MS

PAD_DIFFERENTIAL_MIN: float = DELTA_PAD_LIVING_MIN
PAD_DIFFERENTIAL_MAX: float = DELTA_PAD_LIVING_MAX
PAD_MAX_ACER: float = 0.0010                # Target ACER < 0.10%

NIR_WAVELENGTH_PRIMARY_NM: float = WAVELENGTH_NIR_PRIMARY
NIR_WAVELENGTH_SECONDARY_NM: float = WAVELENGTH_NIR_SECONDARY
FAR_TARGET_STRICT: float = BANKING_FRAUD_FAR_CEILING
FRR_TARGET_EDGE: float = CUSTOMER_EXPERIENCE_FRR_CEILING

FITZPATRICK_SKIN_TYPES = ["Type-IV", "Type-V", "Type-VI"]
MURRAY_LAW_EXPONENT: float = 3.0
HEMODYNAMIC_PRESSURE_DROP_EXPONENT: float = 4.0


