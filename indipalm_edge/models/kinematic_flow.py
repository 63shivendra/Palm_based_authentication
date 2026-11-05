"""
indipalm_edge/models/kinematic_flow.py

32x32 Biomechanical Kinematic Flow Field Normalizer.

Executes fine non-rigid unwarping directly within on-chip NPU SRAM,
constrained by musculoskeletal elasticity and negative Jacobian penalties
to resolve thenar crease shearing under up to 30° hand cupping.

Supports both PyTorch tensors and NumPy arrays for seamless execution
across edge hardware inference pipelines and analytical test suites.

Copyright (c) 2026 Sovereign Biometrics Architecture Team.
Licensed under the Apache License, Version 2.0.
"""

from typing import Tuple, Dict, Any, Union, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiomechanicalKinematicFlowField(nn.Module):
    """
    In-network feature-space Spatial Transformer that warps intermediate
    feature representations inside on-chip NPU SRAM.
    """

    def __init__(
        self,
        grid_size: Union[Tuple[int, int], int] = (32, 32),
        channels: int = 64,
        in_channels: int = 64,
        elasticity_lambda: float = 0.15,
        jacobian_penalty_weight: float = 100.0,
    ):
        super().__init__()
        if isinstance(grid_size, int):
            self.grid_h, self.grid_w = grid_size, grid_size
        else:
            self.grid_h, self.grid_w = grid_size

        self.channels = in_channels if in_channels is not None else channels
        self.elasticity_lambda = elasticity_lambda
        self.jacobian_penalty_weight = jacobian_penalty_weight

        # PyTorch flow regressor: lightweight depthwise-separable conv predicting (B, 2, H, W)
        self.flow_regressor = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
            nn.Tanh(),  # Bounded in [-1, 1]
        )

        # Precompute base identity coordinate grid in [-1, 1] for numpy
        y = np.linspace(-1.0, 1.0, self.grid_h, dtype=np.float32)
        x = np.linspace(-1.0, 1.0, self.grid_w, dtype=np.float32)
        self.grid_x, self.grid_y = np.meshgrid(x, y)

    def forward(
        self, feature_tensor: Union[torch.Tensor, np.ndarray], flow_delta: Optional[torch.Tensor] = None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], np.ndarray]:
        """
        Forward pass for PyTorch execution.
        Args:
            feature_tensor: (B, C, H, W) tensor.
            flow_delta: Optional precomputed (B, 2, H, W) flow.
        Returns:
            (unwarped_features, flow_field)
        """
        if isinstance(feature_tensor, np.ndarray):
            return self.warp_feature_tensor(feature_tensor, flow_delta)

        b, c, h, w = feature_tensor.shape
        if flow_delta is None:
            # Predict optical flow field, scaled by small factor for stability
            flow_delta = self.flow_regressor(feature_tensor) * 0.15

        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=feature_tensor.device, dtype=feature_tensor.dtype),
            torch.linspace(-1.0, 1.0, w, device=feature_tensor.device, dtype=feature_tensor.dtype),
            indexing="ij",
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)  # (B, H, W, 2)
        sampling_grid = base_grid + flow_delta.permute(0, 2, 3, 1)

        unwarped = F.grid_sample(feature_tensor, sampling_grid, align_corners=True, mode="bilinear")
        return unwarped, flow_delta

    def warp_feature_tensor(
        self,
        feature_tensor: np.ndarray,
        flow_delta: np.ndarray,
    ) -> np.ndarray:
        """
        Bilinearly samples feature_tensor using normalized flow displacement (NumPy execution).
        """
        h, w, c = feature_tensor.shape
        assert h == self.grid_h and w == self.grid_w

        sample_x = self.grid_x + flow_delta[:, :, 0]
        sample_y = self.grid_y + flow_delta[:, :, 1]

        pixel_x = ((sample_x + 1.0) * 0.5) * (w - 1)
        pixel_y = ((sample_y + 1.0) * 0.5) * (h - 1)

        x0 = np.clip(np.floor(pixel_x).astype(np.int32), 0, w - 2)
        x1 = x0 + 1
        y0 = np.clip(np.floor(pixel_y).astype(np.int32), 0, h - 2)
        y1 = y0 + 1

        wx1 = pixel_x - x0
        wx0 = 1.0 - wx1
        wy1 = pixel_y - y0
        wy0 = 1.0 - wy1

        wx0 = wx0[:, :, np.newaxis]
        wx1 = wx1[:, :, np.newaxis]
        wy0 = wy0[:, :, np.newaxis]
        wy1 = wy1[:, :, np.newaxis]

        top_left = feature_tensor[y0, x0, :]
        top_right = feature_tensor[y0, x1, :]
        bottom_left = feature_tensor[y1, x0, :]
        bottom_right = feature_tensor[y1, x1, :]

        unwarped = (
            wy0 * (wx0 * top_left + wx1 * top_right)
            + wy1 * (wx0 * bottom_left + wx1 * bottom_right)
        )
        return unwarped

    def compute_biomechanical_loss(
        self, flow_delta: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, Dict[str, float]]:
        """
        Computes musculoskeletal regularization loss and Jacobian folding penalty.
        """
        if isinstance(flow_delta, torch.Tensor):
            # PyTorch implementation
            du_dx = flow_delta[:, 0, :, 1:] - flow_delta[:, 0, :, :-1]
            du_dy = flow_delta[:, 0, 1:, :] - flow_delta[:, 0, :-1, :]
            dv_dx = flow_delta[:, 1, :, 1:] - flow_delta[:, 1, :, :-1]
            dv_dy = flow_delta[:, 1, 1:, :] - flow_delta[:, 1, :-1, :]

            elasticity_loss = (
                torch.mean(du_dx ** 2) + torch.mean(du_dy ** 2) +
                torch.mean(dv_dx ** 2) + torch.mean(dv_dy ** 2)
            )

            # Jacobian determinant
            j00 = 1.0 + du_dx[:, 1:, :]
            j01 = du_dy[:, :, 1:]
            j10 = dv_dx[:, 1:, :]
            j11 = 1.0 + dv_dy[:, :, 1:]
            det_j = j00 * j11 - j01 * j10
            folding_penalty = torch.mean(F.relu(-det_j) ** 2) * self.jacobian_penalty_weight
            return elasticity_loss * self.elasticity_lambda + folding_penalty

        # NumPy implementation
        du_dx = np.gradient(flow_delta[:, :, 0], axis=1)
        du_dy = np.gradient(flow_delta[:, :, 0], axis=0)
        dv_dx = np.gradient(flow_delta[:, :, 1], axis=1)
        dv_dy = np.gradient(flow_delta[:, :, 1], axis=0)

        j00 = 1.0 + du_dx
        j01 = du_dy
        j10 = dv_dx
        j11 = 1.0 + dv_dy

        elasticity_loss = float(np.mean(du_dx ** 2 + du_dy ** 2 + dv_dx ** 2 + dv_dy ** 2))
        det_j = j00 * j11 - j01 * j10
        folding_violation = np.maximum(0.0, -det_j)
        folding_penalty = float(np.mean(folding_violation ** 2) * self.jacobian_penalty_weight)
        total_loss = elasticity_loss * self.elasticity_lambda + folding_penalty

        return {
            "total_biomech_loss": total_loss,
            "elasticity_loss": elasticity_loss,
            "folding_penalty": folding_penalty,
            "min_det_jacobian": float(np.min(det_j)),
        }
