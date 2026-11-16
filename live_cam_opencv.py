#!/usr/bin/env python3
"""
live_cam_opencv.py - Direct desktop OpenCV camera window for Palm Authentication.
Controls:
  [E] - Enroll current palm (saves template)
  [V] - Verify current palm against enrolled template
  [R] - Reset enrolled templates
  [Q] or [ESC] - Quit
"""
import sys
import cv2
import numpy as np
import palm_match

def render_compcode(code, weight):
    h, w = code.shape
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 0] = (code * 30).astype(np.uint8)
    hsv[:, :, 1] = 220
    hsv[:, :, 2] = (weight * 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam (index 0). If you have multiple cameras, try index 1.")
        sys.exit(1)

    print("====================================================")
    print("Palm Authentication - OpenCV Desktop Camera")
    print("Commands:")
    print("  [E] - Enroll palm")
    print("  [V] - Verify palm")
    print("  [R] - Reset templates")
    print("  [Q] - Quit")
    print("====================================================")

    enrolled_templates = []
    status_text = "Show palm to camera. Press [E] to Enroll."
    status_color = (255, 255, 255)
    last_roi = None
    last_compcode = None
    last_distance = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)  # Mirror view
        h, w = frame.shape[:2]
        annotated = frame.copy()

        current_roi = None
        current_code = None
        current_weight = None

        try:
            pts = palm_match.detect_landmarks(frame)
            # Draw landmarks
            for pt in pts:
                cv2.circle(annotated, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
            # Draw palm center
            p0, p9 = pts[0], pts[9]
            cv2.line(annotated, (int(p0[0]), int(p0[1])), (int(p9[0]), int(p9[1])), (255, 200, 0), 2)
            current_roi = palm_match.align_palm(frame, pts)
        except Exception:
            current_roi = palm_match.extract_palm_opencv(frame)

        if current_roi is not None:
            current_code, current_weight = palm_match.encode_roi(current_roi)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('r'), ord('R')):
            enrolled_templates.clear()
            last_distance = None
            last_roi = None
            last_compcode = None
            status_text = "Templates reset. Press [E] to Enroll."
            status_color = (0, 255, 255)
        elif key in (ord('e'), ord('E')):
            if current_roi is not None:
                enrolled_templates.append((current_code, current_weight))
                last_roi = current_roi
                last_compcode = render_compcode(current_code, current_weight)
                status_text = f"Enrolled successfully! (Total: {len(enrolled_templates)}) Now press [V] to Verify."
                status_color = (0, 255, 0)
            else:
                status_text = "Cannot enroll: palm not detected or poor quality."
                status_color = (0, 0, 255)
        elif key in (ord('v'), ord('V')):
            if not enrolled_templates:
                status_text = "Please enroll at least 1 palm template first [E]!"
                status_color = (0, 165, 255)
            elif current_roi is not None:
                last_roi = current_roi
                last_compcode = render_compcode(current_code, current_weight)
                probe = [palm_match.encode_roi(current_roi, r) for r in palm_match.ROTATIONS]
                dist = palm_match.distance(enrolled_templates, probe)
                last_distance = dist
                is_match = dist <= palm_match.DEFAULT_THRESHOLD
                if is_match:
                    status_text = f"MATCH (GENUINE)! Dist: {dist:.3f} <= {palm_match.DEFAULT_THRESHOLD:.2f}"
                    status_color = (0, 255, 0)
                else:
                    status_text = f"NO MATCH (IMPOSTOR)! Dist: {dist:.3f} > {palm_match.DEFAULT_THRESHOLD:.2f}"
                    status_color = (0, 0, 255)
            else:
                status_text = "Cannot verify: palm not detected or poor quality."
                status_color = (0, 0, 255)

        # Draw HUD overlays
        cv2.rectangle(annotated, (0, 0), (w, 60), (20, 20, 20), -1)
        cv2.putText(annotated, f"Enrolled: {len(enrolled_templates)} | [E] Enroll | [V] Verify | [R] Reset | [Q] Quit",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(annotated, status_text, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)

        # Inset preview for ROI and Competitive Code
        if last_roi is not None and last_compcode is not None:
            roi_bgr = cv2.cvtColor(last_roi, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(annotated, (w - 270, h - 145), (w - 10, h - 10), (0, 0, 0), -1)
            annotated[h - 138:h - 10, w - 265:w - 137] = roi_bgr
            annotated[h - 138:h - 10, w - 135:w - 7] = last_compcode
            cv2.putText(annotated, "ROI", (w - 260, h - 145 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(annotated, "CompCode", (w - 130, h - 145 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("Palm Authentication Matcher", annotated)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

