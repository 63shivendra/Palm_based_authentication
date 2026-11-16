"""
tests/test_training_pipelines.py

Unit test suite verifying execution of both training pipelines:
  - IndiDiff-Palm Coupled Diffusion training loop step.
  - IndiPalm-Edge Biometric metric margin learning step and reparameterization.
"""

import pytest
import torch
import torch.nn.functional as F

from indidiff_palm.pipelines.train_diffusion import (
    ProceduralPalmDataset,
    DiffusionNoiseScheduler,
    DualStreamDiffusionUNet,
)
from indipalm_edge.training.train_edge_model import (
    MultiSpectralIdentityDataset,
    EdgeBiometricModel,
)
from indipalm_edge.models.hypernetwork import DynamicPoseArcFaceLoss


def test_diffusion_training_step():
    """Verifies a single forward-backward gradient step of the diffusion training pipeline."""
    device = torch.device("cpu")
    dataset = ProceduralPalmDataset(num_samples=2, seed=99)
    sample = dataset[0]

    rgb_0 = sample["rgb"].unsqueeze(0)
    nir_0 = sample["nir"].unsqueeze(0)
    depth = sample["depth"].unsqueeze(0)
    mehndi = sample["mehndi"].unsqueeze(0)
    melanin = sample["melanin"].unsqueeze(0)
    z_id = sample["identity_embedding"].unsqueeze(0)

    model = DualStreamDiffusionUNet(base_channels=16, time_emb_dim=64, cond_channels=16)
    scheduler = DiffusionNoiseScheduler(timesteps=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    t = torch.tensor([50], dtype=torch.long)
    eps_rgb = torch.randn_like(rgb_0)
    eps_nir = torch.randn_like(nir_0)

    rgb_t = scheduler.add_noise(rgb_0, eps_rgb, t)
    nir_t = scheduler.add_noise(nir_0, eps_nir, t)

    pred_eps_rgb, pred_eps_nir = model(rgb_t, nir_t, t, depth, mehndi, melanin, z_id)
    loss = F.mse_loss(pred_eps_rgb, eps_rgb) + F.mse_loss(pred_eps_nir, eps_nir)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss)
    assert loss.item() > 0


def test_edge_model_training_step():
    """Verifies a single forward-backward step of the edge recognition model with dynamic pose loss."""
    device = torch.device("cpu")
    num_classes = 5
    model = EdgeBiometricModel(num_classes=num_classes, embedding_dim=128)
    criterion = DynamicPoseArcFaceLoss(
        num_classes=num_classes, embedding_dim=128, base_margin=0.5, scale=16.0
    )
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()), lr=1e-3
    )

    rgb = torch.randn(2, 3, 128, 128)
    nir = torch.randn(2, 1, 128, 128)
    pose = torch.zeros(2, 3)
    labels = torch.tensor([0, 1], dtype=torch.long)

    emb, flow_nir, mask = model(rgb, nir, pose)
    l_metric, logits = criterion(emb, labels, pose)
    l_biomech = model.kinematic_flow.compute_biomechanical_loss(flow_nir)
    loss = l_metric + 0.05 * l_biomech

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss)
    assert logits.shape == (2, num_classes)

    # Test structural reparameterization
    model.backbone.reparameterize()
    assert model.backbone.rgb_stream.stem[0].deploy is True

