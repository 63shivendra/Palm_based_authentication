import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indipalm_edge.training.train_edge_model import MultiSpectralIdentityDataset, EdgeBiometricModel
from indipalm_edge.models.hypernetwork import DynamicPoseArcFaceLoss
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = MultiSpectralIdentityDataset(num_identities=20, samples_per_id=10, seed=42)
loader = DataLoader(ds, batch_size=16, shuffle=True)

model = EdgeBiometricModel(num_classes=20).to(device)
criterion = DynamicPoseArcFaceLoss(num_classes=20, base_margin=0.35, scale=32.0).to(device)
optimizer = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=1e-3)

print("Testing 5 training steps...")
for step, batch in enumerate(loader):
    if step >= 5:
        break
    rgb = batch["rgb"].to(device)
    nir = batch["nir"].to(device)
    pose = batch["pose"].to(device)
    labels = batch["label"].to(device)
    
    optimizer.zero_grad()
    emb, flow, mask = model(rgb, nir, pose)
    loss, logits, cos_theta = criterion(emb, labels, pose, margin_scale=0.2)
    loss.backward()
    
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    acc = (cos_theta.argmax(dim=1) == labels).float().mean() * 100.0
    print(f"Step {step+1}: Loss={loss.item():.4f}, Acc={acc.item():.1f}%, GradNorm={grad_norm.item():.4f}, EmbNorm={emb.norm(dim=1).mean().item():.3f}")

