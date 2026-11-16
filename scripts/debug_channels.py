import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
import torch

pats = glob.glob('data/PalmMatchDB_ROI/**/*.png', recursive=True)
im_real = np.array(Image.open(pats[0]).convert("RGB").resize((128, 128))) / 255.0

from indipalm_edge.training.train_edge_model import MultiSpectralIdentityDataset
ds = MultiSpectralIdentityDataset(num_identities=5, samples_per_id=2, seed=42)
item0 = ds[0]
synth_rgb = item0["rgb"].permute(1, 2, 0).numpy()

print("SYNTHETIC RGB:")
for c in range(3):
    print(f"  Channel {c}: min={synth_rgb[:, :, c].min():.3f}, max={synth_rgb[:, :, c].max():.3f}, mean={synth_rgb[:, :, c].mean():.3f}, std={synth_rgb[:, :, c].std():.3f}")

print("\nREAL RGB:")
for c in range(3):
    print(f"  Channel {c}: min={im_real[:, :, c].min():.3f}, max={im_real[:, :, c].max():.3f}, mean={im_real[:, :, c].mean():.3f}, std={im_real[:, :, c].std():.3f}")

print("\nImage corner and center values:")
print(f"Synth top-left corner: {synth_rgb[0, 0, :]}")
print(f"Real top-left corner:  {im_real[0, 0, :]}")
print(f"Synth center:          {synth_rgb[64, 64, :]}")
print(f"Real center:           {im_real[64, 64, :]}")

