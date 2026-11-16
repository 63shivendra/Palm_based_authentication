"""
Unit Tests for Edge AI Models.

Verifies:
1. 32x32 Biomechanical Kinematic Flow Field feature warping and Jacobian elasticity.
2. Dense Spatial Cross-Attention Feature Fusion (MCA-FF) with Lawsone dye gating.
3. Continuous Pose-Conditioned Hypernetwork and continuous margin metric.
"""

import numpy as np
import pytest
from indipalm_edge.models.kinematic_flow import BiomechanicalKinematicFlowField
from indipalm_edge.models.mca_ff_fusion import DenseSpatialCrossAttentionFusion
from indipalm_edge.models.hypernetwork import PoseConditionedHypernetwork


def test_kinematic_flow_field_warping():
    flow_engine = BiomechanicalKinematicFlowField(grid_size=(32, 32), channels=64)

    # Synthetic feature tensor of shape (32, 32, 64)
    rng = np.random.RandomState(42)
    feat = rng.randn(32, 32, 64).astype(np.float32)

    # 1. Zero flow: identity transform
    zero_flow = np.zeros((32, 32, 2), dtype=np.float32)
    unwarped_identity = flow_engine.warp_feature_tensor(feat, zero_flow)
    assert np.allclose(feat, unwarped_identity, atol=1e-3)

    # 2. Smooth physical deformation flow
    y, x = np.mgrid[0:32, 0:32]
    # Small gentle translation field
    smooth_flow = np.zeros((32, 32, 2), dtype=np.float32)
    smooth_flow[:, :, 0] = 0.05 * np.sin(x / 5.0)
    smooth_flow[:, :, 1] = 0.05 * np.cos(y / 5.0)

    unwarped_smooth = flow_engine.warp_feature_tensor(feat, smooth_flow)
    assert unwarped_smooth.shape == (32, 32, 64)

    # Compute biomechanical loss
    loss_dict = flow_engine.compute_biomechanical_loss(smooth_flow)
    assert loss_dict["folding_penalty"] == 0.0  # Smooth flow has det(J) > 0 everywhere
    assert loss_dict["min_det_jacobian"] > 0.8  # Far from non-physical folding!


def test_mca_ff_dense_fusion():
    fusion_engine = DenseSpatialCrossAttentionFusion(channels=64, d_k=32, seed=101)

    rng = np.random.RandomState(202)
    f_rgb = rng.randn(16, 16, 64).astype(np.float32)
    f_nir = rng.randn(16, 16, 64).astype(np.float32)

    # Cross-attention forward pass with zero hardcoded masks or heuristics
    res = fusion_engine.forward(f_rgb, f_nir)
    assert res["fused_features"].shape == (16, 16, 64)
    assert res["spatial_gate_m"].shape == (16, 16)
    # The spatial gating confidence map must be bounded strictly in [0, 1]
    assert np.all(res["spatial_gate_m"] >= 0.0) and np.all(res["spatial_gate_m"] <= 1.0)
    # Both channels must contribute to the dynamic output
    assert 0.05 < np.mean(res["spatial_gate_m"]) < 0.95


def test_continuous_pose_hypernetwork():
    hyper_engine = PoseConditionedHypernetwork(embedding_dim=512, hidden_dim=64, seed=303)

    w_base = np.random.randn(512).astype(np.float32)
    w_base /= np.linalg.norm(w_base)

    pose_zero = np.zeros(3, dtype=np.float32)
    pose_compound = np.array([np.radians(15.0), np.radians(15.0), 0.0], dtype=np.float32)

    w_zero = hyper_engine.compute_dynamic_centroid(w_base, pose_zero)
    w_compound = hyper_engine.compute_dynamic_centroid(w_base, pose_compound)

    # Unit norms
    assert np.isclose(np.linalg.norm(w_zero), 1.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(w_compound), 1.0, atol=1e-5)

    # Continuous deformation: cosine between w_zero and w_compound should be high but distinct (> 0.85)
    cos_shift = np.dot(w_zero, w_compound)
    assert 0.85 < cos_shift < 1.0, f"Expected smooth continuous shift, got {cos_shift}"

    # Dynamic margin: high pose angle relaxes margin smoothly
    m_zero = hyper_engine.compute_dynamic_margin(pose_zero)
    m_compound = hyper_engine.compute_dynamic_margin(pose_compound)
    assert m_zero > m_compound  # Compound tilt reduces margin to absorb intra-class variance
    assert m_compound >= 0.20

