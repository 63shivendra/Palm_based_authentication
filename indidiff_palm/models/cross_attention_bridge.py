"""
indidiff_palm/models/cross_attention_bridge.py

Cross-Spectral Bidirectional Attention Bridge (CS-BAB) for IndiDiff-Palm.
Enforces biophysical spatial registration between visible surface friction creases (RGB)
and sub-surface venous vascular networks (NIR) during generative diffusion denoising steps.

Formulation:
  Given intermediate latent representations F_rgb and F_nir:
    Q_rgb = F_rgb * W_q_rgb,  K_nir = F_nir * W_k_nir,  V_nir = F_nir * W_v_nir
    Q_nir = F_nir * W_q_nir,  K_rgb = F_rgb * W_k_rgb,  V_rgb = F_rgb * W_v_rgb

    Attn_rgb = Softmax(Q_rgb * K_nir^T / sqrt(d)) * V_nir
    Attn_nir = Softmax(Q_nir * K_rgb^T / sqrt(d)) * V_rgb

  F_rgb_out = F_rgb + gamma_rgb * Attn_rgb
  F_nir_out = F_nir + gamma_nir * Attn_nir
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossSpectralBidirectionalAttentionBridge(nn.Module):
    """
    Bidirectional Cross-Attention Bridge connecting RGB and NIR latent streams.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        # RGB -> NIR projections
        self.to_q_rgb = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.to_k_nir = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.to_v_nir = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.proj_rgb = nn.Conv2d(channels, channels, kernel_size=1)

        # NIR -> RGB projections
        self.to_q_nir = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.to_k_rgb = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.to_v_rgb = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.proj_nir = nn.Conv2d(channels, channels, kernel_size=1)

        # Learnable gating residual parameters (initialized to small value for stable warmup)
        self.gamma_rgb = nn.Parameter(torch.zeros(1))
        self.gamma_nir = nn.Parameter(torch.zeros(1))

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
        b, c, _, _ = q.shape
        # Reshape to (B, heads, N, head_dim) where N = H * W
        q_flat = q.view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        k_flat = k.view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        v_flat = v.view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)

        # Scaled dot-product attention
        scale = 1.0 / (self.head_dim ** 0.5)
        sim = torch.matmul(q_flat, k_flat.transpose(-2, -1)) * scale
        attn = F.softmax(sim, dim=-1)

        out = torch.matmul(attn, v_flat)  # (B, heads, N, head_dim)
        out = out.permute(0, 1, 3, 2).contiguous().view(b, c, h, w)
        return out

    def forward(
        self,
        f_rgb: torch.Tensor,
        f_nir: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f_rgb: (B, C, H, W) visible epidermal feature map.
            f_nir: (B, C, H, W) subcutaneous vascular feature map.
        Returns:
            (f_rgb_out, f_nir_out): Aligned and cross-conditioned feature maps.
        """
        _, _, h, w = f_rgb.shape

        # 1. RGB attends to NIR (learns underlying vascular topology under skin)
        q_rgb = self.to_q_rgb(f_rgb)
        k_nir = self.to_k_nir(f_nir)
        v_nir = self.to_v_nir(f_nir)
        attn_rgb = self.proj_rgb(self._attention(q_rgb, k_nir, v_nir, h, w))
        f_rgb_out = f_rgb + self.gamma_rgb * attn_rgb

        # 2. NIR attends to RGB (learns palmar fascia constraint along principal flexion creases)
        q_nir = self.to_q_nir(f_nir)
        k_rgb = self.to_k_rgb(f_rgb)
        v_rgb = self.to_v_rgb(f_rgb)
        attn_nir = self.proj_nir(self._attention(q_nir, k_rgb, v_rgb, h, w))
        f_nir_out = f_nir + self.gamma_nir * attn_nir

        return f_rgb_out, f_nir_out
