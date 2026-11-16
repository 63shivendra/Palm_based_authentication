import os
import glob
import cv2
import numpy as np

def inspect_dataset(name, path, exts=['.png', '.jpg', '.bmp']):
    print(f"=== Inspecting {name} ({path}) ===")
    files = []
    for root, _, fs in os.walk(path):
        for f in fs:
            if any(f.lower().endswith(e) for e in exts):
                files.append(os.path.join(root, f))
    print(f"Total images found: {len(files)}")
    if not files:
        return
    
    shapes = []
    channels = []
    sample_stats = []
    for f in files[:20]:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is not None:
            shapes.append(img.shape[:2])
            channels.append(img.shape[2] if len(img.shape) > 2 else 1)
            sample_stats.append((f, img.shape, img.dtype, np.min(img), np.max(img), np.mean(img)))
            
    print(f"Sample shapes: {set(shapes[:10])}")
    print(f"Channels: {set(channels[:10])}")
    print("First 3 samples:")
    for s in sample_stats[:3]:
        print(f"  {s[0]}: shape={s[1]}, dtype={s[2]}, min={s[3]}, max={s[4]}, mean={s[5]:.2f}")

inspect_dataset("PalmMatchDB", "data/PalmMatchDB")
inspect_dataset("real_palm_dataset", "data/real_palm_dataset")
inspect_dataset("axondata_contactless", "data/axondata_contactless")

