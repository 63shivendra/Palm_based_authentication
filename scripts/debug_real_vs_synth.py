import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import numpy as np
from PIL import Image
import torch
from indipalm_edge.training.train_edge_model import EdgeBiometricModel, MultiSpectralIdentityDataset

pats = glob.glob('data/PalmMatchDB_ROI/**/*.png', recursive=True)
print('Found ROI pngs:', len(pats))

# Check real image stats
im_real = np.array(Image.open(pats[0]).convert("RGB").resize((128, 128))) / 255.0
im_real2 = np.array(Image.open(pats[100]).convert("RGB").resize((128, 128))) / 255.0

print(f"Real sample 0: shape={im_real.shape}, min={im_real.min():.3f}, max={im_real.max():.3f}, mean={im_real.mean():.3f}, std={im_real.std():.3f}")
print(f"Real sample 100: shape={im_real2.shape}, min={im_real2.min():.3f}, max={im_real2.max():.3f}, mean={im_real2.mean():.3f}, std={im_real2.std():.3f}")

# Check synthetic image stats
ds = MultiSpectralIdentityDataset(num_identities=5, samples_per_id=2, seed=42)
item0 = ds[0]
synth_rgb = item0["rgb"].permute(1, 2, 0).numpy()
synth_nir = item0["nir"][0].numpy()
print(f"Synthetic RGB: shape={synth_rgb.shape}, min={synth_rgb.min():.3f}, max={synth_rgb.max():.3f}, mean={synth_rgb.mean():.3f}, std={synth_rgb.std():.3f}")
print(f"Synthetic NIR: shape={synth_nir.shape}, min={synth_nir.min():.3f}, max={synth_nir.max():.3f}, mean={synth_nir.mean():.3f}, std={synth_nir.std():.3f}")

# Now test model predictions on real vs synthetic
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EdgeBiometricModel(num_classes=200).to(device)
ckpt = torch.load("checkpoints/indipalm_edge/indipalm_edge_best_generalizable.pt", map_location=device)
model.load_state_dict(ckpt["model_state"], strict=False)
model.eval()

# Synthetic embedding test
s0 = item0["rgb"].unsqueeze(0).to(device)
s0_nir = item0["nir"].unsqueeze(0).to(device)
item1 = ds[1] # same ID, different sample
s1 = item1["rgb"].unsqueeze(0).to(device)
s1_nir = item1["nir"].unsqueeze(0).to(device)
item2 = ds[2] # different ID
s2 = item2["rgb"].unsqueeze(0).to(device)
s2_nir = item2["nir"].unsqueeze(0).to(device)
pose0 = torch.zeros((1, 3), device=device)

with torch.no_grad():
    e_s0 = model.extract_embedding(s0, s0_nir, pose0).cpu().numpy()[0]
    e_s1 = model.extract_embedding(s1, s1_nir, pose0).cpu().numpy()[0]
    e_s2 = model.extract_embedding(s2, s2_nir, pose0).cpu().numpy()[0]

print("\n--- SYNTHETIC EMBEDDING TESTS ---")
print(f"Genuine synthetic similarity (same ID):  {np.dot(e_s0, e_s1):.4f}")
print(f"Impostor synthetic similarity (diff ID): {np.dot(e_s0, e_s2):.4f}")

# Real embedding test
r0 = torch.from_numpy(im_real).permute(2, 0, 1).unsqueeze(0).float().to(device)
r0_nir = r0.mean(dim=1, keepdim=True)
r1 = torch.from_numpy(im_real2).permute(2, 0, 1).unsqueeze(0).float().to(device)
r1_nir = r1.mean(dim=1, keepdim=True)

with torch.no_grad():
    e_r0 = model.extract_embedding(r0, r0_nir, pose0).cpu().numpy()[0]
    e_r1 = model.extract_embedding(r1, r1_nir, pose0).cpu().numpy()[0]

print("\n--- REAL EMBEDDING TESTS ---")
print(f"Real pair similarity: {np.dot(e_r0, e_r1):.4f}")
print(f"Norm of e_r0: {np.linalg.norm(e_r0):.4f}")
print(f"Std of e_r0 elements: {np.std(e_r0):.6f}")
print(f"First 10 dims of e_r0:\n{e_r0[:10]}")
print(f"First 10 dims of e_r1:\n{e_r1[:10]}")
print(f"Difference (e_r0 - e_r1) norm: {np.linalg.norm(e_r0 - e_r1):.6f}")
