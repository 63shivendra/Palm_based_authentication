import os
import glob
import cv2
import numpy as np

def extract_palm_roi_edt(img_path, target_size=(128, 128)):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    
    # Check if alpha channel exists
    if img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        gray = cv2.cvtColor(cv2.merge([b, g, r]), cv2.COLOR_BGR2GRAY)
        mask = (a > 50).astype(np.uint8)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
    if np.count_nonzero(mask) < 500:
        return cv2.resize(gray, target_size)
        
    # Distance transform
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # In a hand mask, fingers are thin, palm is thick.
    # The maximum distance to boundary in the whole hand is inside the palm!
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist)
    cx, cy = max_loc
    radius = int(max_val)
    
    # Radius of palm inscribed circle gives the exact scale!
    # Crop a square region of side = 2.2 * radius centered at (cx, cy)
    crop_size = int(radius * 2.2)
    half = crop_size // 2
    
    h, w = gray.shape
    x1 = max(0, cx - half)
    x2 = min(w, cx + half)
    y1 = max(0, cy - half)
    y2 = min(h, cy + half)
    
    cropped = gray[y1:y2, x1:x2]
    
    # Resize to target 128x128
    roi = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    return roi, (cx, cy, radius, (x1, y1, x2, y2))

files = glob.glob("data/PalmMatchDB/**/*.png", recursive=True)
for i in range(5):
    res = extract_palm_roi_edt(files[i * 100])
    if res:
        roi, stats = res
        print(f"Sample {i}: center=({stats[0]}, {stats[1]}), radius={stats[2]}, crop={stats[3]}, roi_mean={np.mean(roi):.2f}, nonzeros={np.count_nonzero(roi)}")

