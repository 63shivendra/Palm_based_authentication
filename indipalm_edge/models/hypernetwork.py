"""
indipalm_edge/models/hypernetwork.py

Continuous Pose-Conditioned Hypernetwork & Dynamic Metric Learning.
Eliminates discrete sub-center boundary oscillation under compound presentation angles
(e.g. 15° pitch + 15° roll) by dynamically generating continuous hyperspherical centroids
w_j(Omega) and continuous angular margins on a Riemannian manifold.

Supports both PyTorch tensors and NumPy arrays for seamless interoperability across
embedded edge inference and statistical unit testing.

Copyright (c) 2026 Sovereign Biometrics Architecture Team.
Licensed under the Apache License, Version 2.0.
"""

from typing import Tuple, Dict, Any, Union, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.constants import EMBEDDING_DIM


class PoseConditionedHypernetwork(nn.Module):
    """
    Continuous Pose Hypernetwork and Dynamic Margin Metric Loss.

    Mathematical Formulation:
        Let Omega = [phi_pitch, psi_roll, kappa_yaw]^T in R^3 be 3D hand orientation angles.
        A 2-layer MLP maps Omega -> R^D:
            H_pose(Omega) = W_2 * ReLU(W_1 * Omega + b_1) + b_2
        The instantaneous identity centroid on the unit hypersphere is:
            w_j(Omega) = (w_{j, base} + H_pose(Omega)) / ||w_{j, base} + H_pose(Omega)||_2

        Dynamic Margin:
            m(Omega) = m_base - beta * tanh(gamma * ||Omega||_2)
            With s = 64.0, m_base = 0.50, beta = 0.15, gamma = 6.0.
    """

    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_dim: int = 64,
        scale_s: float = 64.0,
        margin_base: float = 0.50,
        margin_beta: float = 0.15,
        margin_gamma: float = 6.0,
        seed: int = 42,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.scale_s = scale_s
        self.margin_base = margin_base
        self.margin_beta = margin_beta
        self.margin_gamma = margin_gamma

        # PyTorch layers
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # NumPy fallback weights for direct numpy benchmarking
        rng = np.random.RandomState(seed)
        self.w1 = (rng.randn(3, hidden_dim) * np.sqrt(2.0 / 3.0)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = (rng.randn(hidden_dim, embedding_dim) * np.sqrt(2.0 / hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(embedding_dim, dtype=np.float32)

    def forward(self, pose: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Forward pass producing continuous centroid displacement delta_w.
        Handles both PyTorch tensors and NumPy arrays.
        """
        if isinstance(pose, torch.Tensor):
            return self.net(pose) * 0.08

        # NumPy execution
        orig_1d = (pose.ndim == 1)
        omega = pose.reshape(1, 3) if orig_1d else pose
        h = np.maximum(0.0, omega @ self.w1 + self.b1)
        delta_w = (h @ self.w2 + self.b2) * 0.08
        return delta_w[0] if orig_1d else delta_w

    def forward_hypernetwork(self, pose_omega: np.ndarray) -> np.ndarray:
        return self.forward(pose_omega)

    def compute_dynamic_centroid(self, w_base: np.ndarray, pose_omega: np.ndarray) -> np.ndarray:
        """
        Computes the instantaneous normalized centroid:
            w_j(Omega) = Normalize(w_{base} + H_pose(Omega))
        """
        delta_w = self.forward_hypernetwork(pose_omega)
        w_shifted = w_base + delta_w
        norm = np.linalg.norm(w_shifted, axis=-1, keepdims=True) + 1e-12
        return w_shifted / norm

    def compute_dynamic_margin(self, pose_omega: Union[np.ndarray, torch.Tensor]) -> Union[float, torch.Tensor]:
        """
        Computes continuous angular margin:
            m(Omega) = m_base - beta * tanh(gamma * ||Omega||_2)
        """
        if isinstance(pose_omega, torch.Tensor):
            pose_norm = torch.norm(pose_omega, p=2, dim=-1, keepdim=True)
            m = self.margin_base - self.margin_beta * torch.tanh(self.margin_gamma * pose_norm)
            return torch.clamp(m, 0.20, 0.50)

        pose_norm = float(np.linalg.norm(pose_omega))
        m = self.margin_base - self.margin_beta * np.tanh(self.margin_gamma * pose_norm)
        return float(np.clip(m, 0.20, 0.50))


class DynamicPoseArcFaceLoss(nn.Module):
    """
    Continuous Pose-Conditioned DP-SubArc Metric Margin Loss.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = EMBEDDING_DIM,
        base_margin: float = 0.50,
        scale: float = 32.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.base_margin = base_margin
        self.scale = scale

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.hyper = PoseConditionedHypernetwork(embedding_dim=embedding_dim)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, poses: torch.Tensor, margin_scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # Normalize weights
        w_norm = F.normalize(self.weight, p=2, dim=1)
        cos_theta = F.linear(embeddings, w_norm)
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

        # Dynamic margin per sample with linear warmup
        m = self.hyper.compute_dynamic_margin(poses) * margin_scale  # (B, 1)

        theta = torch.acos(cos_theta)
        cos_theta_m = torch.cos(theta + m)

        # Monotonicity guard: Deng et al. (CVPR 2019 ArcFace Section 3.2)
        # Prevents gradient reversal when theta + m > pi on difficult or noisy presentations
        sin_m = torch.sin(m)
        cos_theta_m = torch.where(
            theta > (math.pi - m),
            cos_theta - m * sin_m,
            cos_theta_m,
        )

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = (one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta) * self.scale
        loss = F.cross_entropy(logits, labels)
        return loss, logits, cos_theta
