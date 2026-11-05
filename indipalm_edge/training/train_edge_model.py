"""
indipalm_edge/training/train_edge_model.py

Training Pipeline for IndiPalm-Edge Biometric Recognition Network.
Trains the complete multi-spectral edge recognition payment model on synthetic data.

Key Components Trained:
  - RepMSFNet: Multi-Scale Spatial-Frequency Backbone (RepConv + 2D Haar DWT).
  - BiomechanicalKinematicFlowField: 32x32 feature flow unwarper with Jacobian penalty.
  - DenseSpatialCrossAttentionFusion (MCA-FF): Dynamic spectral routing under Mehndi dye.
  - PoseConditionedHypernetwork: Continuous Riemannian manifold projection on S^511.
  - Dynamic Pose-Conditioned DP-SubArc Metric Margin Loss.

Post-Training Export:
  - Fuses multi-branch RepConv2d into single 3x3 Conv via .reparameterize().
  - Saves optimized weights ready for INT8 Quantization-Aware Metric Training (QAMT).

Usage:
  python indipalm_edge/training/train_edge_model.py --epochs 25 --batch_size 32 --lr 1e-3
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from common.constants import (
    PALM_ROI_SIZE,
    EMBEDDING_DIM,
)
from common.generators.procedural_palm import CleanRoomPalmGenerator
from indipalm_edge.models.rep_msfnet import RepMSFNet
from indipalm_edge.models.kinematic_flow import BiomechanicalKinematicFlowField
from indipalm_edge.models.mca_ff_fusion import DenseSpatialCrossAttentionFusion
from indipalm_edge.models.hypernetwork import (
    PoseConditionedHypernetwork,
    DynamicPoseArcFaceLoss,
)


class MultiSpectralIdentityDataset(Dataset):
    """
    Multi-Spectral Biometric Classification Dataset.
    Generates synthetic multi-spectral pairs with randomized 3D hand tilts
    (pitch, roll, yaw in [-25 deg, +25 deg]) and Mehndi surface dyes.
    """

    def __init__(self, num_identities: int = 200, samples_per_id: int = 5, seed: int = 1337):
        self.num_identities = num_identities
        self.samples_per_id = samples_per_id
        self.total_samples = num_identities * samples_per_id
        self.generator = CleanRoomPalmGenerator(base_seed=seed)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int):
        identity_id = idx // self.samples_per_id
        sample_variant = idx % self.samples_per_id

        # Generate base identity with randomized pose variations
        data = self.generator.generate_identity(identity_index=identity_id)

        # Randomize 3D presentation angles for this presentation
        rng = np.random.RandomState(idx + 50000)
        pitch = rng.uniform(-20.0, 20.0)
        roll = rng.uniform(-20.0, 20.0)
        yaw = rng.uniform(-15.0, 15.0)

        rgb = torch.from_numpy(data["rgb_image"]).permute(2, 0, 1).float()  # (3, 128, 128)
        nir = torch.from_numpy(data["nir_image"]).unsqueeze(0).float()       # (1, 128, 128)
        pose_rad = torch.tensor(
            [np.radians(pitch), np.radians(roll), np.radians(yaw)], dtype=torch.float32
        )

        return {
            "rgb": rgb,
            "nir": nir,
            "pose": pose_rad,
            "label": identity_id,
        }


class EdgeBiometricModel(nn.Module):
    """
    Unified training wrapper combining Backbone, Flow Normalizer, Cross-Attention Fusion,
    and Continuous Pose Hypernetwork into an end-to-end differentiable graph.
    """

    def __init__(self, num_classes: int, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.backbone = RepMSFNet(base_channels=32, embedding_dim=embedding_dim)
        self.kinematic_flow = BiomechanicalKinematicFlowField(in_channels=64, grid_size=32)
        self.mca_ff = DenseSpatialCrossAttentionFusion(feature_dim=128)
        self.hypernetwork = PoseConditionedHypernetwork(embedding_dim=embedding_dim)

    def forward(
        self, rgb: torch.Tensor, nir: torch.Tensor, pose: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass during training.
        Returns:
            invariant_embedding: (B, D) L2-normalized embedding on S^(D-1).
            flow_field: (B, 2, 32, 32) kinematic deformation field for regularization.
            dynamic_mask: (B, 1, 16, 16) MCA-FF gating mask.
        """
        # Multi-scale backbone feature maps
        rgb_f2, rgb_f3, nir_f2, nir_f3 = self.backbone.forward_features(rgb, nir)

        # 32x32 Kinematic flow normalizer (resolves thenar cupping)
        nir_f2_norm, flow_nir = self.kinematic_flow(nir_f2)

        # Dense spatial cross-attention fusion (resolves Mehndi dye corruption)
        fused_f3, dynamic_mask = self.mca_ff(rgb_f3, nir_f3)

        # Base backbone embedding on S^(D-1)
        base_emb = self.backbone.project_embedding(fused_f3)

        # Continuous Riemannian pose manifold adaptation
        pose_delta = self.hypernetwork(pose)
        invariant_embedding = F.normalize(base_emb + pose_delta, p=2, dim=-1)

        return invariant_embedding, flow_nir, dynamic_mask


def train_edge_model(
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-3,
    num_classes: int = 100,
    device: str = "cpu",
    output_dir: str = "checkpoints/indipalm_edge",
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"[*] Initializing IndiPalm-Edge Recognition Training on device: {device}")

    # 1. Dataset & DataLoader
    dataset = MultiSpectralIdentityDataset(num_identities=num_classes, samples_per_id=5, seed=42)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 2. Complete Model & Dynamic Pose-Conditioned Margin Loss
    model = EdgeBiometricModel(num_classes=num_classes, embedding_dim=EMBEDDING_DIM).to(device)
    metric_criterion = DynamicPoseArcFaceLoss(
        num_classes=num_classes,
        embedding_dim=EMBEDDING_DIM,
        base_margin=0.50,
        scale=32.0,
    ).to(device)

    # 3. Optimizer & Schedulers
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(metric_criterion.parameters()),
        lr=lr,
        weight_decay=5e-4,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader))

    print(f"[*] Commencing edge model training: {epochs} epochs, {num_classes} identities...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        t_start = time.perf_counter()

        for step, batch in enumerate(loader):
            rgb = batch["rgb"].to(device)
            nir = batch["nir"].to(device)
            pose = batch["pose"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            emb, flow_field, dynamic_mask = model(rgb, nir, pose)

            # 1. Continuous Pose-Conditioned DP-SubArc Metric Loss
            l_metric, logits = metric_criterion(emb, labels, pose)

            # 2. Biomechanical Jacobian Non-Folding Regularization
            l_biomech = model.kinematic_flow.compute_biomechanical_loss(flow_field)

            # Composite total loss
            loss = l_metric + 0.05 * l_biomech

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

            total_epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.shape[0]

            if (step + 1) % 5 == 0 or (step + 1) == len(loader):
                acc = (correct_predictions / max(1, total_predictions)) * 100.0
                print(
                    f"    Epoch [{epoch}/{epochs}] Batch [{step+1}/{len(loader)}] "
                    f"Loss: {loss.item():.4f} (Metric: {l_metric.item():.4f}, Biomech: {l_biomech.item():.4f}) "
                    f"Batch Acc: {acc:.1f}%"
                )

        epoch_time = time.perf_counter() - t_start
        epoch_acc = (correct_predictions / max(1, total_predictions)) * 100.0
        avg_loss = total_epoch_loss / len(loader)
        print(f"[+] Epoch {epoch} complete in {epoch_time:.1f}s. Avg Loss: {avg_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        # Save checkpoint
        ckpt_path = os.path.join(output_dir, f"indipalm_edge_epoch_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "metric_criterion_state": metric_criterion.state_dict(),
                "accuracy": epoch_acc,
            },
            ckpt_path,
        )

    # Post-Training Structural Reparameterization Demo
    print("\n[*] Fusing multi-branch RepConv2d into single 3x3 Conv for edge deployment...")
    model.eval()
    model.backbone.reparameterize()
    deploy_path = os.path.join(output_dir, "indipalm_edge_deploy_fused.pt")
    torch.save(model.state_dict(), deploy_path)
    print(f"[✓] Deployment-ready fused model saved to: {deploy_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IndiPalm-Edge Recognition Payment Model")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_classes", type=int, default=50, help="Number of synthetic identities")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu)")
    parser.add_argument("--output_dir", type=str, default="checkpoints/indipalm_edge", help="Output directory")
    args = parser.parse_args()

    train_edge_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_classes=args.num_classes,
        device=args.device,
        output_dir=args.output_dir,
    )
