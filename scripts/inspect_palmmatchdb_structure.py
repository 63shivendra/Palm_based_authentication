import os
import glob
from collections import defaultdict
from PIL import Image
import numpy as np

base_dir = "data/PalmMatchDB"
all_images = []
for root, dirs, files in os.walk(base_dir):
    for f in files:
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            all_images.append(os.path.join(root, f))

print(f"Total valid image files found: {len(all_images):,}")

# Inspect first 10 paths to understand subjects, categories, and hands
subjects = defaultdict(list)
for p in all_images:
    # Normalize separators
    rel = os.path.relpath(p, base_dir).replace("\\", "/")
    parts = rel.split("/")
    # e.g., ['PalmMatchDB', 'MALE PALMPRINTS', 'm103 age 60', 'm103 12m', 'ei_1611579869040-removebg-preview-5.png']
    # Subject identifier is typically parts[2] e.g. "m103 age 60"
    if len(parts) >= 3:
        sub_folder = parts[2] if parts[0] == "PalmMatchDB" else parts[1]
        subjects[sub_folder].append(p)

print(f"Total parsed subjects: {len(subjects)}")
print(f"Sample subjects:")
for s in list(subjects.keys())[:10]:
    print(f"  Subject: '{s}' -> {len(subjects[s])} images")

# Check image size and format
if all_images:
    sample_img = Image.open(all_images[0])
    arr = np.array(sample_img)
    print(f"\nSample Image: {all_images[0]}")
    print(f"Format: {sample_img.format}, Size: {sample_img.size}, Mode: {sample_img.mode}")
    print(f"Array shape: {arr.shape}, Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean():.2f}")

