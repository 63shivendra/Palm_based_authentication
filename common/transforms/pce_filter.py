"""
common/transforms/pce_filter.py

PyTorch and NumPy implementations of the Modified Finite Radon Transform (MFRAT)
and Palm Crease Energy (PCE) extractor from PCE-Palm (AAAI 2024) and Diff-Palm (CVPR 2025).

Extracts directional crease skeletons independent of:
  - Melanin / skin pigmentation
  - Cultural Mehndi (henna) dyes
  - Sensor exposure and ambient lighting variations
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mfrat_filter_bank(ksize: int = 31, angle_step: int = 30) -> np.ndarray:
    """Constructs multi-angle line detection filters across 0° to 180°."""
    half = ksize // 2
    angles = [i * angle_step for i in range(180 // angle_step)]
    filters = []
    
    # 1D Gaussian weighting
    sigma = 0.01
    t = np.linspace(-sigma * 2, sigma * 2, ksize)
    gauss = np.exp(-t**2 / (2 * sigma**2))
    mask_row = gauss[None, :]
    mask_col = gauss[:, None]

    for angle in angles:
        filt = np.zeros((ksize, ksize), dtype=np.float32)
        rad = math.radians(angle if angle <= 90 else angle - 180)
        
        if -45 < (angle if angle <= 90 else angle - 180) < 45:
            tan_theta = math.tan(rad)
            for dx in range(-half, half + 1):
                dy = round(dx * tan_theta)
                px, py = half + dx, half + dy
                if 0 <= py < ksize and 0 <= px < ksize:
                    filt[py, px] = 1.0
        else:
            cot_theta = math.cos(rad) / math.sin(rad)
            for dy in range(-half, half + 1):
                dx = round(dy * cot_theta)
                px, py = half + dx, half + dy
                if 0 <= py < ksize and 0 <= px < ksize:
                    filt[py, px] = 1.0
                    
        if angle <= 45 or angle >= 135:
            filt = filt * mask_row
        else:
            filt = filt * mask_col
            
        filt_sum = np.sum(np.abs(filt))
        if filt_sum > 0:
            filt = filt / filt_sum
            
        filters.append(filt)

    # Invert so valleys (creases) produce maximum positive response
    filter_bank = np.stack(filters, axis=0) * -1.0
    return filter_bank


class PalmCreaseEnergyFilter(nn.Module):
    """
    Differentiable PyTorch module computing Palm Crease Energy (PCE)
    via multi-directional MFRAT filters and local bandpass subtraction.
    """
    def __init__(self, ksize: int = 31, angle_step: int = 30):
        super().__init__()
        self.ksize = ksize
        bank = build_mfrat_filter_bank(ksize, angle_step)  # (N_angles, K, K)
        num_angles = bank.shape[0]
        # Reshape to (N_angles, 1, K, K) for conv2d
        bank_t = torch.from_numpy(bank).unsqueeze(1).float()
        self.register_buffer("filter_bank", bank_t)
        
        # Gaussian smoothing kernel for bandpass detail isolation
        half = ksize // 2
        coords = torch.arange(ksize).float() - half
        g1d = torch.exp(-coords**2 / (2 * 4.0**2))
        g2d = g1d[:, None] * g1d[None, :]
        g2d = g2d / g2d.sum()
        self.register_buffer("gauss_kernel", g2d.unsqueeze(0).unsqueeze(0).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, 1, H, W) in range [-1, 1] or [0, 1]
        Output: (B, 1, H, W) normalized crease energy map in [0, 1]
        """
        # 1. Bandpass filter: high-pass details = x - lowpass(x)
        pad = self.ksize // 2
        x_low = F.conv2d(x, self.gauss_kernel, padding=pad)
        details = x - x_low
        
        # 2. Multi-angle MFRAT directional filtering
        responses = F.conv2d(details, self.filter_bank, padding=pad) # (B, N_angles, H, W)
        
        # 3. Maximum directional projection (crease energy)
        pce_map, _ = torch.max(responses, dim=1, keepdim=True) # (B, 1, H, W)
        
        # 4. Normalize per sample to zero-mean unit-variance
        pce_norm = F.relu(pce_map) # keep only valley crease depressions
        pce_max = torch.amax(pce_norm, dim=(-2, -1), keepdim=True) + 1e-7
        pce_norm = pce_norm / pce_max
        return pce_norm

