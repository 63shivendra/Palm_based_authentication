import os
import sys
sys.path.insert(0, ".")
import time
import torch
from common.transforms.pce_filter import PalmCreaseEnergyFilter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing PCE Filter on {device}...")

filter_net = PalmCreaseEnergyFilter(ksize=31, angle_step=30).to(device)
batch = torch.randn(64, 1, 128, 128, device=device)

# Warmup
_ = filter_net(batch)

start = time.time()
for _ in range(10):
    out = filter_net(batch)
torch.cuda.synchronize()
elapsed = time.time() - start

print(f"Processed 640 images in {elapsed:.3f}s ({640/elapsed:.1f} imgs/sec).")
print(f"Output shape: {out.shape}, min: {out.min():.2f}, max: {out.max():.2f}")

