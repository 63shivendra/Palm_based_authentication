"""IndiDiff-Palm loss functions module."""
from indidiff_palm.losses.wavelet_loss import WaveletDirectionalLoss
from indidiff_palm.losses.murray_loss import MurrayHemodynamicLoss

__all__ = ["WaveletDirectionalLoss", "MurrayHemodynamicLoss"]
