import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
import torch
from common.transforms.pce_filter import PalmCreaseEnergyFilter
from indipalm_edge.training.train_edge_model import MultiSpectralIdentityDataset

pce = PalmCreaseEnergyFilter(ksize=31, angle_step=30)

# Real image
pats = glob.glob('data/PalmMatchDB_ROI/**/*.png', recursive=True)
im_real = np.array(Image.open(pats[0]).convert("L").resize((128, 128))) / 255.0
im_real2 = np.array(Image.open(pats[1]).convert("L").resize((128, 128))) # same palm, different image
im_real_other = np.array(Image.open(pats[500]).convert("L").resize((128, 128))) # different palm

r0_t = torch.from_numpy(im_real).unsqueeze(0).unsqueeze(0).float()
r1_t = torch.from_numpy(im_real2 / 255.0).unsqueeze(0).unsqueeze(0).float()
ro_t = torch.from_numpy(im_real_other / 255.0).unsqueeze(0).unsqueeze(0).float()

pce_r0 = pce(r0_t)[0, 0].numpy()
pce_r1 = pce(r1_t)[0, 0].numpy()
pce_ro = pce(ro_t)[0, 0].numpy()

# Synthetic image
ds = MultiSpectralIdentityDataset(num_identities=5, samples_per_id=2, seed=42)
s0 = ds[0]["rgb"].mean(dim=0, keepdim=True).unsqueeze(0)
s1 = ds[1]["rgb"].mean(dim=0, keepdim=True).unsqueeze(0) # same id
s2 = ds[2]["rgb"].mean(dim=0, keepdim=True).unsqueeze(0) # diff id

pce_s0 = pce(s0)[0, 0].numpy()
pce_s1 = pce(s1)[0, 0].numpy()
pce_s2 = pce(s2)[0, 0].numpy()

print("PCE ON REAL IMAGES:")
print("  Real r0 PCE min/max/mean/std:", pce_r0.min(), pce_r0.max(), pce_r0.mean(), pce_r0.std())
print("  Cosine between same palm (r0 vs r1):", np.dot(pce_r0.flatten(), pce_r1.flatten()) / (np.linalg.norm(pce_r0.flatten()) * np.linalg.norm(pce_r1.flatten())))
print("  Cosine between diff palm (r0 vs ro):", np.dot(pce_r0.flatten(), pce_ro.flatten()) / (np.linalg.norm(pce_r0.flatten()) * np.linalg.norm(pce_ro.flatten())))

print("\nPCE ON SYNTHETIC IMAGES:")
print("  Synth s0 PCE min/max/mean/std:", pce_s0.min(), pce_s0.max(), pce_s0.mean(), pce_s0.std())
print("  Cosine between same synth ID (s0 vs s1):", np.dot(pce_s0.flatten(), pce_s1.flatten()) / (np.linalg.norm(pce_s0.flatten()) * np.linalg.norm(pce_s1.flatten())))
print("  Cosine between diff synth ID (s0 vs s2):", np.dot(pce_s0.flatten(), pce_s2.flatten()) / (np.linalg.norm(pce_s0.flatten()) * np.linalg.norm(pce_s2.flatten())))

