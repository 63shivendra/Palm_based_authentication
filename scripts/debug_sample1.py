import cv2
import numpy as np

files = ["data/PalmMatchDB/PalmMatchDB/FEMALE PALMPRINTS/female Age(15-24)/F101(15-24)/F12MB/left/ei_1608938994605-removebg-preview.png"]
import glob
all_f = glob.glob("data/PalmMatchDB/**/*.png", recursive=True)
sample1 = all_f[100]
print("File:", sample1)
img = cv2.imread(sample1, cv2.IMREAD_UNCHANGED)
print("Shape:", img.shape)
a = img[:, :, 3]
print("Alpha min:", a.min(), "max:", a.max(), "unique:", np.unique(a)[:10])
mask = (a > 100).astype(np.uint8)
dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
print("Dist min:", dist.min(), "max:", dist.max())
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist)
print("min_val:", min_val, "max_val:", max_val, "loc:", max_loc)

