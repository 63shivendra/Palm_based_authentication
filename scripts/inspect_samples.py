import os
import glob
from PIL import Image
import numpy as np

print("=== Real Palm Dataset Samples ===")
real_files = glob.glob("data/real_palm_dataset/**/*.jpg", recursive=True)
for p in real_files[:5]:
    img = Image.open(p)
    arr = np.array(img)
    print(f"{os.path.basename(p)}: shape={arr.shape}, mode={img.mode}, min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")

print("\n=== AxonData Contactless Smartphone Samples ===")
axon_files = glob.glob("data/axondata_contactless/**/*.jpg", recursive=True)
for p in axon_files[:5]:
    img = Image.open(p)
    arr = np.array(img)
    print(f"{os.path.basename(p)}: shape={arr.shape}, mode={img.mode}, min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")

