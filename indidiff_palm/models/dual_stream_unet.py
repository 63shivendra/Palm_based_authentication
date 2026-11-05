"""
indidiff_palm/models/dual_stream_unet.py

Coupled Dual-Stream Denoising U-Net for IndiDiff-Palm.
Synthesizes registered, high-fidelity RGB palm surface textures and
sub-surface NIR vascular networks under biophysical conditioning.

Features:
  - Parallel visible (RGB, 3-ch) and vascular (NIR, 1-ch) streams.
  - Sinusoidal timestep diffusion embedding.
  - Cross-Spectral Bidirectional Attention Bridges (CS-BAB) at multi-resolution scales.
  - Conditioned on 3D depth, Lawsone Mehndi dye masks, and melanin indices.
"""

from typing import Tuple, Dict, Any, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from indidiff_palm.models.cross_attention_bridge import CrossSpectralBidirectionalAttentionBridge
from indidiff_palm.models.conditioning_encoder import PhysicsConditioningEncoder


class SinusoidalTimeEmbedding(nn.Module):
    """Generates sinusoidal positional embeddings for diffusion timestep t."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb_scale)
        args = t.unsqueeze(1).float() * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    """Residual convolutional block with timestep and global condition injection."""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        # Add timestep context
        t_proj = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t_proj
        h = self.act(self.conv2(h))
        return h + self.skip(x)


class DualStreamDiffusionUNet(nn.Module):
    """
    Coupled RGB + NIR Denoising Diffusion U-Net.
    """

    def __init__(
        self,
        base_channels: int = 32,
        time_emb_dim: int = 128,
        cond_channels: int = 64,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.time_emb_dim = time_emb_dim

        # Timestep encoder
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Physics conditioning encoder
        self.cond_encoder = PhysicsConditioningEncoder(cond_channels=cond_channels)

        # RGB Input & NIR Input projections (spatial cond concatenated)
        # RGB input: 3 channels + cond_channels
        # NIR input: 1 channel + cond_channels
        self.in_conv_rgb = nn.Conv2d(3 + cond_channels, base_channels, kernel_size=3, padding=1)
        self.in_conv_nir = nn.Conv2d(1 + cond_channels, base_channels, kernel_size=3, padding=1)

        # Level 1 (Resolution: H x W)
        self.res1_rgb = ResBlock(base_channels, base_channels, time_emb_dim)
        self.res1_nir = ResBlock(base_channels, base_channels, time_emb_dim)
        self.bridge1 = CrossSpectralBidirectionalAttentionBridge(base_channels, num_heads=2)

        # Downsample Level 1 -> Level 2 (Resolution: H/2 x W/2)
        self.down_rgb = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.down_nir = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)

        # Level 2 (Resolution: H/2 x W/2)
        self.res2_rgb = ResBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        self.res2_nir = ResBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        self.bridge2 = CrossSpectralBidirectionalAttentionBridge(base_channels * 2, num_heads=4)

        # Bottleneck (Resolution: H/2 x W/2)
        self.mid_rgb = ResBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        self.mid_nir = ResBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        self.bridge_mid = CrossSpectralBidirectionalAttentionBridge(base_channels * 2, num_heads=4)

        # Upsample Level 2 -> Level 1 (Resolution: H x W)
        self.up_rgb = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
        )
        self.up_nir = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
        )

        # Decoder Level 1 with skip connection (base_channels + base_channels)
        self.dec_rgb = ResBlock(base_channels * 2, base_channels, time_emb_dim)
        self.dec_nir = ResBlock(base_channels * 2, base_channels, time_emb_dim)
        self.bridge_dec = CrossSpectralBidirectionalAttentionBridge(base_channels, num_heads=2)

        # Output Noise Prediction Heads
        self.out_head_rgb = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
        )
        self.out_head_nir = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x_rgb_t: torch.Tensor,
        x_nir_t: torch.Tensor,
        t: torch.Tensor,
        depth_map: torch.Tensor,
        mehndi_mask: torch.Tensor,
        melanin_index: torch.Tensor,
        identity_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts diffusion noise epsilon_t for coupled RGB and NIR streams.

        Args:
            x_rgb_t: (B, 3, H, W) noisy visible palm tensor at timestep t.
            x_nir_t: (B, 1, H, W) noisy NIR vein tensor at timestep t.
            t: (B,) diffusion integer timesteps in [0, T].
            depth_map: (B, 1, H, W) 3D hand mesh standoff and cupping map.
            mehndi_mask: (B, 1, H, W) Lawsone Mehndi dye staining map.
            melanin_index: (B, 1) melanin absorption fraction.
            identity_embedding: (B, D) identity latent code.

        Returns:
            (eps_pred_rgb, eps_pred_nir): Predicted Gaussian noise components.
        """
        # 1. Compute timestep embedding and condition tensors
        t_emb = self.time_embed(t)
        spatial_cond, global_cond = self.cond_encoder(
            depth_map, mehndi_mask, melanin_index, identity_embedding
        )
        # Combine global condition into timestep embedding
        t_emb = t_emb + F.pad(global_cond, (0, self.time_emb_dim - global_cond.shape[-1]))

        # Match spatial conditioning resolution if needed
        if spatial_cond.shape[-2:] != x_rgb_t.shape[-2:]:
            spatial_cond = F.interpolate(
                spatial_cond, size=x_rgb_t.shape[-2:], mode="bilinear", align_corners=False
            )

        # 2. Input stage (concatenate spatial physics conditions)
        h_rgb = self.in_conv_rgb(torch.cat([x_rgb_t, spatial_cond], dim=1))
        h_nir = self.in_conv_nir(torch.cat([x_nir_t, spatial_cond], dim=1))

        # 3. Level 1 Encoder + CS-BAB 1
        h_rgb = self.res1_rgb(h_rgb, t_emb)
        h_nir = self.res1_nir(h_nir, t_emb)
        h_rgb, h_nir = self.bridge1(h_rgb, h_nir)
        skip_rgb = h_rgb
        skip_nir = h_nir

        # 4. Downsampling Level 1 -> Level 2
        h_rgb = self.down_rgb(h_rgb)
        h_nir = self.down_nir(h_nir)

        # 5. Level 2 Encoder + CS-BAB 2
        h_rgb = self.res2_rgb(h_rgb, t_emb)
        h_nir = self.res2_nir(h_nir, t_emb)
        h_rgb, h_nir = self.bridge2(h_rgb, h_nir)

        # 6. Bottleneck Mid + CS-BAB Mid
        h_rgb = self.mid_rgb(h_rgb, t_emb)
        h_nir = self.mid_nir(h_nir, t_emb)
        h_rgb, h_nir = self.bridge_mid(h_rgb, h_nir)

        # 7. Upsampling Level 2 -> Level 1
        h_rgb = self.up_rgb(h_rgb)
        h_nir = self.up_nir(h_nir)

        # 8. Level 1 Decoder + Skip Connection + CS-BAB Dec
        h_rgb = torch.cat([h_rgb, skip_rgb], dim=1)
        h_nir = torch.cat([h_nir, skip_nir], dim=1)
        h_rgb = self.dec_rgb(h_rgb, t_emb)
        h_nir = self.dec_nir(h_nir, t_emb)
        h_rgb, h_nir = self.bridge_dec(h_rgb, h_nir)

        # 9. Output noise predictions
        eps_rgb = self.out_head_rgb(h_rgb)
        eps_nir = self.out_head_nir(h_nir)

        return eps_rgb, eps_nir
