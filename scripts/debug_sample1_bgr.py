import cv2
import numpy as np

f = "data/PalmMatchDB/PalmMatchDB/FEMALE PALMPRINTS/female Age(15-24)/F104(15-24)/left/ei_1608974287314-removebg-preview (12).png"
img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
bgr = img[:, :, :3]
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
print("Gray min:", gray.min(), "max:", gray.max(), "mean:", gray.mean())
# Background is usually white or very light (>240) or dark (<15)
print("Pixels > 240:", np.count_nonzero(gray > 240) / gray.size)
print("Pixels < 15:", np.count_nonzero(gray < 15) / gray.size)

