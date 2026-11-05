"""
indidiff_palm/losses/wavelet_loss.py

Wavelet-Domain Directional Phase Loss for IndiDiff-Palm Generative Multi-Spectral Diffusion.
Forces the generative diffusion engine to accurately capture high-frequency friction ridges,
flexion creases, and deep vascular bifurcations without blurriness or spectral collapse.

Formulation:
  Multi-level 2D Haar Discrete Wavelet Transform decomposing images into directional subbands:
    - LH: Horizontal micro-ridges (high horizontal frequencies)
    - HL: Vertical principal flexion creases (high vertical frequencies)
    - HH: Diagonal ridge junctions and vascular bifurcations
  Loss combines multiscale subband L1 norm with directional spatial gradient phase alignment.
"""

from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from indipalm_edge.models.rep_msfnet import HaarDWT2d


class WaveletDirectionalLoss(nn.Module):
    """
    Multi-Scale Directional Wavelet Loss for multi-spectral image synthesis.
    Computes sub-band consistency across high-frequency components [LH, HL, HH].
    """

    def __init__(
        self,
        in_channels: int = 1,
        levels: int = 2,
        lambda_phase: float = 0.5,
        attenuate_ll: float = 0.1,
    ):
        super().__init__()
        self.levels = levels
        self.lambda_phase = lambda_phase
        self.attenuate_ll = attenuate_ll

        # Hierarchical DWT decomposition stages
        self.dwt_stages = nn.ModuleList([HaarDWT2d(in_channels) for _ in range(levels)])

    def _spatial_gradient(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes spatial gradients along x and y dimensions."""
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates multi-scale directional wavelet loss between predicted and target images.

        Args:
            pred: (B, C, H, W) synthesized image.
            target: (B, C, H, W) reference or ground-truth procedural image.

        Returns:
            Scalar loss tensor.
        """
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        curr_pred = pred
        curr_target = target

        for level, dwt in enumerate(self.dwt_stages):
            # Compute Haar DWT subbands
            ll_pred, lh_pred, hl_pred, hh_pred = dwt(curr_pred)
            ll_targ, lh_targ, hl_targ, hh_targ = dwt(curr_target)

            # High-frequency directional subbands
            hf_pred = [lh_pred, hl_pred, hh_pred]
            hf_targ = [lh_targ, hl_targ, hh_targ]

            # 1. Low-frequency attenuation (prevents global skin tone differences from dominating)
            total_loss += self.attenuate_ll * F.l1_loss(ll_pred, ll_targ)

            # 2. High-frequency directional subband L1 reconstruction
            for p_sub, t_sub in zip(hf_pred, hf_targ):
                # Subband magnitude matching
                subband_l1 = F.l1_loss(p_sub, t_sub)
                total_loss += subband_l1

                # Directional phase gradient matching
                p_dx, p_dy = self._spatial_gradient(p_sub)
                t_dx, t_dy = self._spatial_gradient(t_sub)
                phase_loss = F.l1_loss(p_dx, t_dx) + F.l1_loss(p_dy, t_dy)
                total_loss += self.lambda_phase * phase_loss

            # Propagate low-frequency approximation to next scale
            curr_pred = ll_pred
            curr_target = ll_targ

        return total_loss
