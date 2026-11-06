"""
scripts/crop_palmmatchdb_roi.py

Batch extracts canonical 128x128 palmar disc ROIs for all 10,455 PalmMatchDB images
using Euclidean Distance Transform (EDT) Maximum Inscribed Circle and contour orientation.
Saves pre-processed ROIs to data/PalmMatchDB_ROI/ for blazingly fast training and evaluation.
"""

import os
import glob
import math
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import cv2
import numpy as np

def extract_canonical_palm_roi(img_path, target_size=(128, 128)):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
            
        if len(img.shape) == 3 and img.shape[2] == 4:
            a = img[:, :, 3]
            mask = (a > 50).astype(np.uint8) * 255
            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        else:
            gray = img
            _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
            
        # If image is already a cropped ROI (high density of non-zero skin)
        if np.count_nonzero(mask) > 0.85 * mask.size:
            return cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
            
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return cv2.resize(gray, target_size)
            
        hand_cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(hand_cnt) < 1000:
            return cv2.resize(gray, target_size)
            
        # Distance transform for Maximum Inscribed Circle (palm center & radius)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        min_v, max_v, min_l, max_l = cv2.minMaxLoc(dist)
        cx, cy = max_l
        radius = max_v
        
        if radius < 15 or radius > 1e10:
            return cv2.resize(gray, target_size)
            
        # Principal axis orientation via second-order spatial moments
        moments = cv2.moments(hand_cnt)
        if moments['m00'] > 0:
            mu20 = moments['mu20'] / moments['m00']
            mu02 = moments['mu02'] / moments['m00']
            mu11 = moments['mu11'] / moments['m00']
            angle_rad = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
            rot_angle = math.degrees(angle_rad) - 90.0
            while rot_angle > 45.0:
                rot_angle -= 90.0
            while rot_angle < -45.0:
                rot_angle += 90.0
        else:
            rot_angle = 0.0
            
        # Rotate around palm center
        M = cv2.getRotationMatrix2D((float(cx), float(cy)), rot_angle, 1.0)
        h, w = gray.shape
        rotated_gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Crop square of size 2.2 * radius around (cx, cy)
        side = int(radius * 2.2)
        x1 = int(cx - side // 2)
        x2 = int(cx + side // 2)
        y1 = int(cy - side // 2)
        y2 = int(cy + side // 2)
        
        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - w)
        pad_b = max(0, y2 - h)
        
        if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
            padded = cv2.copyMakeBorder(rotated_gray, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT)
            crop = padded[y1 + pad_t : y2 + pad_t, x1 + pad_l : x2 + pad_l]
        else:
            crop = rotated_gray[y1:y2, x1:x2]
            
        return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
    except Exception as e:
        return None

def process_single(args):
    src, dst = args
    if os.path.exists(dst):
        return True
    roi = extract_canonical_palm_roi(src)
    if roi is not None:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cv2.imwrite(dst, roi)
        return True
    return False

def main():
    src_base = "data/PalmMatchDB"
    dst_base = "data/PalmMatchDB_ROI"
    
    pngs = glob.glob(f"{src_base}/**/*.png", recursive=True)
    jpgs = glob.glob(f"{src_base}/**/*.jpg", recursive=True)
    all_files = sorted(pngs + jpgs)
    
    print(f"[*] Found {len(all_files):,} images in {src_base}")
    tasks = []
    for f in all_files:
        rel = os.path.relpath(f, src_base)
        dst = os.path.join(dst_base, os.path.splitext(rel)[0] + ".png")
        tasks.append((f, dst))
        
    start = time.time()
    print(f"[*] Extracting canonical 128x128 palm ROIs using 16 threads...")
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(process_single, tasks))
        
    elapsed = time.time() - start
    success = sum(results)
    print(f"[+] Successfully extracted {success:,} / {len(tasks):,} ROIs in {elapsed:.1f}s ({success/elapsed:.0f} imgs/s) to {dst_base}")

if __name__ == "__main__":
    main()

