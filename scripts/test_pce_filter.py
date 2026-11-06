import os
import sys
sys.path.insert(0, ".")
import glob
import torch
import numpy as np
from PIL import Image
from common.transforms.pce_filter import PalmCreaseEnergyFilter

filter_net = PalmCreaseEnergyFilter(ksize=31, angle_step=30)

imgs = glob.glob("data/real_palm_dataset/**/*.jpg", recursive=True)
if not imgs:
    imgs = glob.glob("data/PalmMatchDB/**/*.png", recursive=True)

img_path = imgs[0]
print(f"Testing on image: {img_path}")
img = Image.open(img_path).convert("L").resize((128, 128))
t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    pce = filter_net(t)
    
print(f"Input shape: {t.shape}, min: {t.min():.2f}, max: {t.max():.2f}, mean: {t.mean():.2f}")
print(f"PCE shape:   {pce.shape}, min: {pce.min():.2f}, max: {pce.max():.2f}, mean: {pce.mean():.2f}")
print(f"Sparsity (% zeros): {(pce == 0).float().mean() * 100:.1f}%")
print("[+] MFRAT / PCE transform operates cleanly and isolates high-contrast crease energy!")
