import cv2
import numpy as np
import glob

def find_valleys_and_crop(img_path, target_size=(128, 128)):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
        
    if img.shape[2] == 4:
        a = img[:, :, 3]
        mask = (a > 50).astype(np.uint8) * 255
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        
    # Find largest contour (the hand)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return cv2.resize(gray, target_size)
        
    hand_cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(hand_cnt) < 1000:
        return cv2.resize(gray, target_size)
        
    # Convex hull and defects
    hull = cv2.convexHull(hand_cnt, returnPoints=False)
    if len(hull) < 4:
        return cv2.resize(gray, target_size)
        
    defects = cv2.convexityDefects(hand_cnt, hull)
    if defects is None:
        return cv2.resize(gray, target_size)
        
    # Distance transform for palm center
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # Mask out edges
    min_v, max_v, min_l, max_l = cv2.minMaxLoc(dist)
    cx, cy = max_l
    radius = max_v
    
    # Inscribed palm circle crop:
    # Side length proportional to radius
    side = int(radius * 2.2)
    x1 = max(0, int(cx - side // 2))
    x2 = min(gray.shape[1], int(cx + side // 2))
    y1 = max(0, int(cy - side // 2))
    y2 = min(gray.shape[0], int(cy + side // 2))
    
    crop = gray[y1:y2, x1:x2]
    return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)

files = glob.glob("data/PalmMatchDB/**/*.png", recursive=True)
print("Testing 10 samples with distance-transform palm center crop...")
for i in range(10):
    roi = find_valleys_and_crop(files[i * 50])
    print(f"Sample {i}: shape={roi.shape}, mean={np.mean(roi):.2f}, min={np.min(roi)}, max={np.max(roi)}")

