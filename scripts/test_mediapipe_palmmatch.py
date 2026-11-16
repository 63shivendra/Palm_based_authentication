import cv2
import numpy as np
import mediapipe as mp

print("MediaPipe version:", mp.__version__)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

import glob
files = glob.glob("data/PalmMatchDB/**/*.png", recursive=True)
print(f"Testing on 10 samples from PalmMatchDB...")
detected = 0
for i, f in enumerate(files[:10]):
    img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 4:
        # Composite against white background for mediapipe detector
        b, g, r, a = cv2.split(img)
        rgb = cv2.merge([r, g, b])
        alpha = a.astype(float) / 255.0
        white_bg = np.ones_like(rgb) * 255
        rgb_comp = (rgb * alpha[:, :, None] + white_bg * (1 - alpha[:, :, None])).astype(np.uint8)
    else:
        rgb_comp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    results = hands.process(rgb_comp)
    if results.multi_hand_landmarks:
        detected += 1
        lm = results.multi_hand_landmarks[0].landmark
        # wrist: 0, index_mcp: 5, pinky_mcp: 17
        print(f"Sample {i}: DETECTED! Wrist=({lm[0].x:.3f}, {lm[0].y:.3f}), IndexMCP=({lm[5].x:.3f}, {lm[5].y:.3f}), PinkyMCP=({lm[17].x:.3f}, {lm[17].y:.3f})")
    else:
        print(f"Sample {i}: Not detected by mediapipe")

print(f"Detection rate on first 10: {detected}/10")
hands.close()

