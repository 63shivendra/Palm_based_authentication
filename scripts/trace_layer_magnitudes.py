import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import numpy as np
from PIL import Image
import torch

from indipalm_edge.training.train_edge_model import EdgeBiometricModel, MultiSpectralIdentityDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EdgeBiometricModel(num_classes=200).to(device)
ckpt = torch.load("checkpoints/indipalm_edge/indipalm_edge_best_generalizable.pt", map_location=device)
model.load_state_dict(ckpt["model_state"], strict=False)
model.eval()

ds = MultiSpectralIdentityDataset(num_identities=5, samples_per_id=2, seed=42)
s0_rgb = ds[0]["rgb"].unsqueeze(0).to(device)
s0_nir = ds[0]["nir"].unsqueeze(0).to(device)

pats = glob.glob('data/PalmMatchDB_ROI/**/*.png', recursive=True)
r0_img = np.array(Image.open(pats[0]).convert("RGB").resize((128, 128))) / 255.0
r0_rgb = torch.from_numpy(r0_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
r0_nir = r0_rgb.mean(dim=1, keepdim=True)

stream = model.backbone.rgb_stream

print("=== TRACING RGB STREAM LAYER BY LAYER ===")
print("Input RGB: synth min/max = {:.3f}/{:.3f}, real min/max = {:.3f}/{:.3f}".format(
    s0_rgb.min(), s0_rgb.max(), r0_rgb.min(), r0_rgb.max()
))

# 1. Stem
x_s = stream.stem(s0_rgb)
x_r = stream.stem(r0_rgb)
print("After Stem (32x128x128): synth std={:.3f}, real std={:.3f}, synth max={:.3f}, real max={:.3f}".format(
    x_s.std(), x_r.std(), x_s.max(), x_r.max()
))

# 2. Stage 1: SpatialFrequencyBlock
# Let's inspect inside Stage 1:
# DWT vs Spatial branch
sf_block = stream.stage1
print("\n--- Inside Stage 1 ---")
# DWT branch
dwt_s = sf_block.dwt(x_s)
dwt_r = sf_block.dwt(x_r)
print("DWT output: synth std={:.3f}, real std={:.3f}, synth max={:.3f}, real max={:.3f}".format(
    dwt_s.std(), dwt_r.std(), dwt_s.max(), dwt_r.max()
))

dwt_conv_s = sf_block.freq_conv(dwt_s)
dwt_conv_r = sf_block.freq_conv(dwt_r)
print("Freq Conv output: synth std={:.3f}, real std={:.3f}, synth max={:.3f}, real max={:.3f}".format(
    dwt_conv_s.std(), dwt_conv_r.std(), dwt_conv_s.max(), dwt_conv_r.max()
))

spat_s = sf_block.spatial_branch(x_s)
spat_r = sf_block.spatial_branch(x_r)
print("Spatial branch output: synth std={:.3f}, real std={:.3f}, synth max={:.3f}, real max={:.3f}".format(
    spat_s.std(), spat_r.std(), spat_s.max(), spat_r.max()
))

x_s = sf_block(x_s)
x_r = sf_block(x_r)
print("Stage 1 combined output: synth std={:.3f}, real std={:.3f}".format(x_s.std(), x_r.std()))

# 3. Stage 2
x_s = stream.stage2(x_s)
x_r = stream.stage2(x_r)
print("Stage 2 output: synth std={:.3f}, real std={:.3f}".format(x_s.std(), x_r.std()))

# 4. Stage 3
x_s = stream.stage3(x_s)
x_r = stream.stage3(x_r)
print("Stage 3 output: synth std={:.3f}, real std={:.3f}".format(x_s.std(), x_r.std()))

