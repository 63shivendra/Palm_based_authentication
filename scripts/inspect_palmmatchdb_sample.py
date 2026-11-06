import os
import glob
import cv2
import numpy as np

files = glob.glob("data/PalmMatchDB/**/*.png", recursive=True)
print(f"Total files: {len(files)}")
sample = files[0]
print(f"Sample: {sample}")

img = cv2.imread(sample, cv2.IMREAD_UNCHANGED)
print(f"Shape: {img.shape}")
# BGR + Alpha
b, g, r, a = cv2.split(img)
print(f"Alpha min: {np.min(a)}, max: {np.max(a)}, non-zero pixels: {np.count_nonzero(a)} / {a.size} ({np.count_nonzero(a)/a.size*100:.1f}%)")

# Find the bounding box of non-zero alpha
ys, xs = np.where(a > 50)
if len(ys) > 0:
    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()
    print(f"Hand bounding box: x=[{xmin}, {xmax}], y=[{ymin}, {ymax}], w={xmax-xmin}, h={ymax-ymin}")
    
    # Palm center is typically around the lower half of the hand (above wrist, below fingers)
    # The hand extends from wrist (bottom) to fingertips (top).
    # Palm region in a vertical hand is roughly y in [ymin + 0.45*h, ymin + 0.85*h], x in [xmin + 0.15*w, xmin + 0.85*w]
    print(f"Hand aspect ratio (h/w): {(ymax-ymin)/(xmax-xmin):.2f}")

