import cv2
import numpy as np
import glob
import math


def extract_canonical_palm_roi(img_path, target_size=(128, 128)):
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
        
    # Check if image is already a cropped ROI (like PolyU: high mean, filled mask)
    if np.count_nonzero(mask) > 0.85 * mask.size:
        # Already a cropped square ROI!
        return cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
        
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return cv2.resize(gray, target_size)
        
    hand_cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(hand_cnt) < 1000:
        return cv2.resize(gray, target_size)
        
    # Maximum inscribed circle via distance transform
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    min_v, max_v, min_l, max_l = cv2.minMaxLoc(dist)
    cx, cy = max_l
    radius = max_v
    
    if radius < 15:
        return cv2.resize(gray, target_size)
        
        
        
        .or PCA on hand contour points
    moments = cv2.moments(hand_cnt)
    if moments['m00'] > 0:
        mu20 = moments['mu20'] / moments['m00']
        mu02 = moments['mu02'] / moments['m00']
        mu11 = moments['mu11'] / moments['m00']
        angle_rad = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
        angle_deg = math.degrees(angle_rad)
        # We want the major axis to be vertical (fingers up)
        # If angle is near horizontal (e.g. within 45 deg of 0 or 180), rotate to 90
        rot_angle = angle_deg - 90.0
        # Restrict to [-45, 45] rotation to avoid flipping upside down
        while rot_angle > 45.0:
            rot_angle -= 90.0
        while rot_angle < -45.0:
            rot_angle += 90.0
    else:
        rot_angle = 0.0
        
    # Rotate around palm center
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), rot_angle, 1.0)
    rotated_gray = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    # Crop square of size 2.2 * radius around (cx, cy)
    side = int(radius * 2.2)
    x1 = int(cx - side // 2)
    x2 = int(cx + side // 2)
    y1 = int(cy - side // 2)
    y2 = int(cy + side // 2)
    
    # Safe boundary pad if needed
    h, w = rotated_gray.shape
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)
    
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        padded = cv2.copyMakeBorder(rotated_gray, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
        crop = padded[y1 + pad_top : y2 + pad_top, x1 + pad_left : x2 + pad_left]
    else:
        crop = rotated_gray[y1:y2, x1:x2]
        
    return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)

files = glob.glob("data/PalmMatchDB/**/*.png", recursive=True)
print("Testing 10 samples with canonical orientation + inscribed palm disc crop...")
for i in range(10):
    roi = extract_canonical_palm_roi(files[i * 100])
    print(f"Sample {i}: shape={roi.shape}, mean={np.mean(roi):.2f}, min={np.min(roi)}, max={np.max(roi)}")
