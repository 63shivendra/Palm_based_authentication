import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from collections import defaultdict
from common.transforms.pce_filter import PalmCreaseEnergyFilter
from benchmarks.iso_iec_evaluator import ISO19795AccuracyEvaluator

pce_filter = PalmCreaseEnergyFilter(ksize=31, angle_step=30)

files = glob.glob("data/PalmMatchDB_ROI/**/*.png", recursive=True)
print(f"Total files in PalmMatchDB_ROI: {len(files)}")

classes = defaultdict(list)
for f in files:
    parts = f.replace("\\", "/").split("/")
    try:
        sub = [p for p in parts if ("F" in p or "M" in p) and "(" in p][0]
        hand = [p for p in parts if "left" in p.lower() or "right" in p.lower()][0].lower()
        class_id = f"{sub}_{hand}"
        classes[class_id].append(f)
    except Exception:
        continue

print(f"Total classes: {len(classes)}")

# Select first 50 palms for fast benchmark test
selected_classes = list(classes.keys())[:50]
print(f"Testing on {len(selected_classes)} palms...")

features = {}
for cid in selected_classes:
    for f in classes[cid][:10]: # up to 10 presentations per palm
        img = np.array(Image.open(f).convert("L").resize((128, 128))) / 255.0
        t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            feat = pce_filter(t)[0, 0].numpy()
            # Downsample to 32x32 for fast cosine matching
            feat_32 = Image.fromarray((feat * 255).astype(np.uint8)).resize((32, 32), Image.BILINEAR)
            v = np.array(feat_32, dtype=np.float32).flatten()
            norm = np.linalg.norm(v)
            if norm > 1e-6:
                v = v / norm
            features[f] = (cid, v)

# Compute genuine and impostor scores
genuine_sims = []
impostor_sims = []
items = list(features.values())

for i in range(len(items)):
    for j in range(i + 1, min(i + 200, len(items))):
        c_i, v_i = items[i]
        c_j, v_j = items[j]
        cos = float(np.dot(v_i, v_j))
        if c_i == c_j:
            genuine_sims.append(cos)
        else:
            impostor_sims.append(cos)

gen_arr = np.array(genuine_sims)
imp_arr = np.array(impostor_sims)

print(f"\nResults on 50 Real Palms:")
print(f"  Genuine Pairs: {len(gen_arr)}, Mean: {gen_arr.mean():.4f}, Std: {gen_arr.std():.4f}")
print(f"  Impostor Pairs: {len(imp_arr)}, Mean: {imp_arr.mean():.4f}, Std: {imp_arr.std():.4f}")
print(f"  Separation Margin: {gen_arr.mean() - imp_arr.mean():.4f}")

gen_dist = 1.0 - gen_arr
imp_dist = 1.0 - imp_arr
evaluator = ISO19795AccuracyEvaluator(gen_dist, imp_dist)
eer = evaluator.compute_eer(num_thresholds=1000)
print(f"  Equal Error Rate (EER): {eer['eer']*100:.2f}%")
print(f"  Optimal Threshold: {1.0 - eer['eer_threshold']:.4f}")

