import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from indipalm_edge.training.train_edge_model import EdgeBiometricModel, MultiSpectralIdentityDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EdgeBiometricModel(num_classes=200).to(device)
ckpt = torch.load("checkpoints/indipalm_edge/indipalm_edge_best_generalizable.pt", map_location=device)
model.load_state_dict(ckpt["model_state"], strict=False)
model.eval()

# Synthetic input
ds = MultiSpectralIdentityDataset(num_identities=5, samples_per_id=2, seed=42)
s0_rgb = ds[0]["rgb"].unsqueeze(0).to(device)
s0_nir = ds[0]["nir"].unsqueeze(0).to(device)

s2_rgb = ds[2]["rgb"].unsqueeze(0).to(device) # diff id
s2_nir = ds[2]["nir"].unsqueeze(0).to(device)

# Real input
pats = glob.glob('data/PalmMatchDB_ROI/**/*.png', recursive=True)
r0_img = np.array(Image.open(pats[0]).convert("RGB").resize((128, 128))) / 255.0
r1_img = np.array(Image.open(pats[100]).convert("RGB").resize((128, 128))) / 255.0
r0_rgb = torch.from_numpy(r0_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
r0_nir = r0_rgb.mean(dim=1, keepdim=True)
r1_rgb = torch.from_numpy(r1_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
r1_nir = r1_rgb.mean(dim=1, keepdim=True)

with torch.no_grad():
    # 1. Stem & Backbone
    s0_rf2, s0_rf3, s0_nf2, s0_nf3 = model.backbone.forward_features(s0_rgb, s0_nir)
    s2_rf2, s2_rf3, s2_nf2, s2_nf3 = model.backbone.forward_features(s2_rgb, s2_nir)
    
    r0_rf2, r0_rf3, r0_nf2, r0_nf3 = model.backbone.forward_features(r0_rgb, r0_nir)
    r1_rf2, r1_rf3, r1_nf2, r1_nf3 = model.backbone.forward_features(r1_rgb, r1_nir)
    
    print("--- BACKBONE STAGE 3 FEATURES (16x16, 128 channels) ---")
    print(f"Synth s0_rf3: min={s0_rf3.min():.3f}, max={s0_rf3.max():.3f}, mean={s0_rf3.mean():.3f}, std={s0_rf3.std():.3f}")
    print(f"Synth s2_rf3: min={s2_rf3.min():.3f}, max={s2_rf3.max():.3f}, mean={s2_rf3.mean():.3f}, std={s2_rf3.std():.3f}")
    print(f"Real r0_rf3:  min={r0_rf3.min():.3f}, max={r0_rf3.max():.3f}, mean={r0_rf3.mean():.3f}, std={r0_rf3.std():.3f}")
    print(f"Real r1_rf3:  min={r1_rf3.min():.3f}, max={r1_rf3.max():.3f}, mean={r1_rf3.mean():.3f}, std={r1_rf3.std():.3f}")
    
    cos_rf3_synth = F.cosine_similarity(s0_rf3.flatten(1), s2_rf3.flatten(1)).item()
    cos_rf3_real  = F.cosine_similarity(r0_rf3.flatten(1), r1_rf3.flatten(1)).item()
    print(f"Cosine between s0 and s2 (diff synth ids) at rf3: {cos_rf3_synth:.4f}")
    print(f"Cosine between r0 and r1 (diff real palms) at rf3: {cos_rf3_real:.4f}")

    # 2. Cross attention fusion
    s0_fused, s0_mask = model.mca_ff(s0_rf3, s0_nf3)
    s2_fused, s2_mask = model.mca_ff(s2_rf3, s2_nf3)
    r0_fused, r0_mask = model.mca_ff(r0_rf3, r0_nf3)
    r1_fused, r1_mask = model.mca_ff(r1_rf3, r1_nf3)
    
    print("\n--- MCA-FF FUSION MASK & FEATURES ---")
    print(f"Synth mask s0: min={s0_mask.min():.3f}, max={s0_mask.max():.3f}, mean={s0_mask.mean():.3f}")
    print(f"Real mask r0:  min={r0_mask.min():.3f}, max={r0_mask.max():.3f}, mean={r0_mask.mean():.3f}")
    cos_fused_synth = F.cosine_similarity(s0_fused.flatten(1), s2_fused.flatten(1)).item()
    cos_fused_real  = F.cosine_similarity(r0_fused.flatten(1), r1_fused.flatten(1)).item()
    print(f"Cosine between s0 and s2 at fused_f3: {cos_fused_synth:.4f}")
    print(f"Cosine between r0 and r1 at fused_f3: {cos_fused_real:.4f}")

    # 3. Head projection
    s0_proj = model.backbone.project_embedding(s0_fused)
    s2_proj = model.backbone.project_embedding(s2_fused)
    r0_proj = model.backbone.project_embedding(r0_fused)
    r1_proj = model.backbone.project_embedding(r1_fused)
    print(f"\nFinal projected cosine s0 vs s2 (synth): {torch.dot(s0_proj[0], s2_proj[0]).item():.4f}")
    print(f"Final projected cosine r0 vs r1 (real):  {torch.dot(r0_proj[0], r1_proj[0]).item():.4f}")

