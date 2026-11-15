#!/usr/bin/env python3
"""
palm_quality.py - capture quality check for palm photos, with a retake instruction for every problem.

Used by the v4 matcher (palm_match_v4.py) and by the web app's live guidance. palm_match.py is not changed.

Measured on the aligned 512 x 512 grey palm crop (every input), and on the 21 MediaPipe hand landmarks when the
input is a whole-hand photo:

  exposure    share of clipped-white pixels, share of clipped-black pixels, mean brightness
  creases     contrast of crease-scale structure (band-pass energy / brightness). Faint creases = flat light,
              dirty lens, or a palm too far away
  focus       fine-detail energy relative to crease-scale energy; blur removes fine detail first
  noise       Immerkaer (1996) pixel-noise estimate; high in dark photos
  pose        palm width in pixels (resolution), palm length / width against the average hand (a palm tilted
              forward or backward looks short), fingers open and straight

Each check returns a code, a message for the user, and the measured value. The thresholds below were set on the
DEV identities only (benchmarks/tune_dev.py writes the evidence to benchmarks/results/tuning/).

  python palm_quality.py photo.jpg            # print the measures and the decision for one photo
  python palm_quality.py --no-detect roi.png  # already-cropped palm image
"""
import argparse
import sys

import cv2
import numpy as np

import palm_match as pm

# ----------------------------------------------------------------------------- thresholds (set on DEV identities)
# Set by benchmarks/gate_dev.py (benchmarks/results/tuning/*-gate/gate_dev.md): every limit rejects 0% of the natural DEV
# photos; over-exposure, focus and crease limits send back photos that mostly would fail. The v4 matcher handles dark and
# noisy photos (smoothing + CLAHE: 0% failures at x0.25 brightness and at noise sigma 12), so those limits only catch
# extreme cases.
MAX_CLIPPED_WHITE = 0.30   # share of crop pixels >= 250 (74% of photos above it fail to match)
MAX_CLIPPED_BLACK = 0.30   # share of crop pixels <= 5
MIN_BRIGHTNESS, MAX_BRIGHTNESS = 15.0, 235.0
MIN_CREASE_CONTRAST = 0.008  # every DEV photo below it failed to match
MIN_FOCUS = 0.10             # every DEV photo below it failed to match
MAX_NOISE = 15.0             # grey levels
MIN_PALM_PIXELS = 100      # palm width (index to little-finger knuckle) in the original photo: gives a crop of about
                           # the resolution of Tongji's 128 px palm images, which match at 0% EER on DEV (was 160, an
                           # uncalibrated placeholder that rejected most PalmMatchDB photos)
MIN_LENGTH_RATIO, MAX_LENGTH_RATIO = 1.05, 2.10   # wrist-to-middle-knuckle / palm width; 11k Hands DEV: 1.08-1.80
MIN_FINGERS_OPEN = 3       # of the 4 long fingers, how many must be straight (11k Hands DEV: all 4)

MESSAGES = {
    "too_bright": "too bright: move out of direct light or away from the lamp",
    "too_dark": "too dark: move to a brighter place",
    "faint_lines": "palm lines are faint: use even light and hold the palm flat and closer",
    "blurry": "blurry: hold still for a moment",
    "noisy": "too dark (grainy photo): add light",
    "too_small": "palm too far away: move the hand closer to the camera",
    "tilted": "palm tilted: hold it flat, facing the camera",
    "fingers_closed": "open the hand and straighten the fingers",
}

_LAPLACIAN = np.float32([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])


def noise_level(gray):
    """Immerkaer (1996) noise standard deviation in grey levels."""
    r = cv2.filter2D(gray.astype(np.float32), -1, _LAPLACIAN)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(r).mean() / 6)


def crop_measures(gray):
    """Photometric measures of an aligned grey palm crop (any size; computed at 256 x 256)."""
    g = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    mean = float(g.mean())
    b1, b2, b8 = (cv2.GaussianBlur(g, (0, 0), s) for s in (0.8, 2.0, 8.0))
    fine, crease = (b1 - b2).std(), (b2 - b8).std()
    return dict(clipped_white=float((gray >= 250).mean()), clipped_black=float((gray <= 5).mean()),
                brightness=mean, crease_contrast=float(crease / (mean + 1.0)),
                focus=float(fine / (crease + 1e-3)), noise=noise_level(gray))


def pose_measures(pts, image_width):
    """Hand-pose measures from the 21 landmarks of a whole-hand photo."""
    palm_w = float(np.linalg.norm(pts[17] - pts[5]))
    palm_len = float(np.linalg.norm(pts[9] - pts[0]))
    open_ = 0
    for mcp, pip, tip in ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)):
        a, b = pts[pip] - pts[mcp], pts[tip] - pts[pip]
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))
        open_ += cos > 0.6                      # finger bent less than ~53 degrees at the middle joint
    return dict(palm_pixels=palm_w, palm_fraction=palm_w / image_width,
                length_ratio=palm_len / (palm_w + 1e-6), fingers_open=int(open_))


def check(measures):
    """measures (crop and, if present, pose) -> list of (code, message) problems; empty = good photo."""
    m, out = measures, []
    if m["clipped_white"] > MAX_CLIPPED_WHITE or m["brightness"] > MAX_BRIGHTNESS:
        out.append("too_bright")
    if m["clipped_black"] > MAX_CLIPPED_BLACK or m["brightness"] < MIN_BRIGHTNESS:
        out.append("too_dark")
    if m["noise"] > MAX_NOISE and "too_dark" not in out:
        out.append("noisy")
    if m["focus"] < MIN_FOCUS:
        out.append("blurry")
    if m["crease_contrast"] < MIN_CREASE_CONTRAST and not out:
        out.append("faint_lines")
    if "palm_pixels" in m:
        if m["palm_pixels"] < MIN_PALM_PIXELS:
            out.append("too_small")
        if not (MIN_LENGTH_RATIO <= m["length_ratio"] <= MAX_LENGTH_RATIO):
            out.append("tilted")
        if m["fingers_open"] < MIN_FINGERS_OPEN:
            out.append("fingers_closed")
    return [(c, MESSAGES[c]) for c in out]


def score(measures):
    """0-100 summary for display: 100 = every measure comfortably inside its limit."""
    m = measures
    parts = [1 - m["clipped_white"] / MAX_CLIPPED_WHITE, 1 - m["clipped_black"] / MAX_CLIPPED_BLACK,
             m["crease_contrast"] / MIN_CREASE_CONTRAST - 1, m["focus"] / MIN_FOCUS - 1, 1 - m["noise"] / MAX_NOISE]
    if "palm_pixels" in m:
        parts.append(m["palm_pixels"] / MIN_PALM_PIXELS - 1)
    return float(100 * np.clip(min(np.clip(p, 0, 1) for p in parts) * 0.5 + 0.5 * np.mean(np.clip(parts, 0, 1)), 0, 1))


def assess(img, detect=True):
    """Full check of one photo -> dict(ok, score, issues, measures, crop). Raises pm.QualityError only when no hand
    is found or it cannot be aligned (nothing to measure)."""
    bgr = pm.load_bgr(img)
    m = {}
    if detect:
        pts = pm.detect_landmarks(bgr)
        m.update(pose_measures(pts, bgr.shape[1]))
        crop = pm.align_palm_v2(bgr, pts, normalise=False)
    else:
        crop = pm.palm_crop_v3(bgr, False)
    m.update(crop_measures(crop))
    issues = check(m)
    return dict(ok=not issues, score=score(m), issues=issues, measures=m, crop=crop)


def main():
    ap = argparse.ArgumentParser(description="palm photo quality check")
    ap.add_argument("--no-detect", action="store_true", help="image is already a cropped palm")
    ap.add_argument("image")
    a = ap.parse_args()
    try:
        r = assess(a.image, not a.no_detect)
    except pm.QualityError as e:
        print("REJECTED PHOTO:", e)
        sys.exit(2)
    for k, v in r["measures"].items():
        print(f"  {k:16s} {v:.3f}" if isinstance(v, float) else f"  {k:16s} {v}")
    print(f"quality {r['score']:.0f}/100 -> " + ("GOOD" if r["ok"] else "RETAKE: " + "; ".join(m for _, m in r["issues"])))
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
