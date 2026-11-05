"""
indidiff_palm/models/conditioning_encoder.py

Physics-Grounded Multi-Modal Conditioning Encoder for IndiDiff-Palm.
Encodes biophysical parameters into multi-scale spatial and global conditioning vectors
for guiding cross-spectral diffusion image synthesis:
  1. 3D Hand Mesh Standoff & Cupping Depth Map Z_mesh (1 channel).
  2. Lawsone Mehndi Staining Mask M_dye (1 channel, 2-hydroxy-1,4-naphthoquinone).
  3. Melanin Absorption Index xi_mel (CIE L*a*b* scalar, Fitzpatrick IV-VI).
  4. Sovereign Biometric Identity Embedding z_id (D-dimensional vector on S^(D-1)).
"""

from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.constants import EMBEDDING_DIM


class PhysicsConditioningEncoder(nn.Module):
    """
    Encodes spatial maps (Depth, Mehndi) and global parameters (Melanin, Identity)
    into aligned conditioning representations for diffusion U-Net blocks.
    """

    def __init__(
        self,
        cond_channels: int = 64,
        id_dim: int = EMBEDDING_DIM,
    ):
        super().__init__()
        self.cond_channels = cond_channels

        # Spatial condition encoder: processes concatenated (Z_mesh, M_dye) -> (B, 2, H, W)
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(2, cond_channels // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(cond_channels // 2, cond_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        # Global parameters encoder: processes (xi_mel [1], z_id [D]) -> (B, 1 + D)
        self.global_mlp = nn.Sequential(
            nn.Linear(1 + id_dim, cond_channels * 2),
            nn.SiLU(),
            nn.Linear(cond_channels * 2, cond_channels),
        )

    def forward(
        self,
        depth_map: torch.Tensor,
        mehndi_mask: torch.Tensor,
        melanin_index: torch.Tensor,
        identity_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            depth_map: (B, 1, H, W) continuous depth Z values in [0, 1].
            mehndi_mask: (B, 1, H, W) binary/continuous dye concentration in [0, 1].
            melanin_index: (B, 1) scalar in [0, 1].
            identity_embedding: (B, D) L2-normalized vector on S^(D-1).

        Returns:
            spatial_cond: (B, C, H, W) spatial conditioning tensor.
            global_cond: (B, C) global context vector.
        """
        # Spatial processing
        spatial_in = torch.cat([depth_map, mehndi_mask], dim=1)  # (B, 2, H, W)
        spatial_cond = self.spatial_stem(spatial_in)  # (B, C, H, W)

        # Global context processing
        global_in = torch.cat([melanin_index, identity_embedding], dim=1)  # (B, 1 + D)
        global_cond = self.global_mlp(global_in)  # (B, C)

        return spatial_cond, global_cond
