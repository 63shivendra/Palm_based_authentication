#!/usr/bin/env python3
"""
palm_match_robust.py - the compact matcher (v3c) plus two robustness additions (method "v3cr").
palm_match.py and palm_match_compact.py are not changed by this file. Enrolment files are the same
compact (v3c) format, so they work with both programs.

  ADDITION 1 - noise-adaptive smoothing
    The noise level of the aligned palm crop is estimated (Immerkaer 1996). Above NOISE_GATE the crop is
    smoothed (Gaussian, sigma 2) before keypoints are extracted; DISK otherwise places its keypoints on
    noise grains. Below the gate nothing is done, so a clean photo gives exactly the v3c template.

  ADDITION 2 - retry when the first comparison fails
    The probe is tried again (a) smoothed and (b) rotated by +15 and -15 degrees. A retry has to reach
    RETRY_FACTOR x the threshold, to offset the extra attempts this gives an impostor. A comparison that
    passes first time is not touched.

Commands
  python palm_match_robust.py enroll  me.npz  a.jpg b.jpg c.jpg
  python palm_match_robust.py verify  me.npz  new.jpg
  Add --no-detect if the images are already cropped palm close-ups.
"""
import argparse
import sys

import cv2
import numpy as np

import palm_match as pm
import palm_match_compact as pmc

THRESHOLD = pmc.THRESHOLD
NOISE_GATE = 4.0            # grey levels. Clean crops of every benchmark dataset measure 0.2 - 2.8
SMOOTH_SIGMA = 2.0
RETRY_ANGLES = (15, -15)    # degrees; the pose check accepts up to 25, so this covers about +/-40
RETRY_FACTOR = 1.5          # PROVISIONAL: a retry must score 1.5 x the threshold

_LAPLACIAN = np.float32([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])


def noise_level(gray):
    """Immerkaer (1996) fast noise estimate: standard deviation of the pixel noise, in grey levels."""
    r = cv2.filter2D(gray.astype(np.float32), -1, _LAPLACIAN)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(r).mean() / 6)


def smooth(gray):
    return cv2.GaussianBlur(gray, (0, 0), SMOOTH_SIGMA)


def condition(gray):
    """ADDITION 1: smooth the crop only if it is noisy."""
    return smooth(gray) if noise_level(gray) > NOISE_GATE else gray


def rotate(gray, angle):
    m = cv2.getRotationMatrix2D((gray.shape[1] / 2, gray.shape[0] / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, gray.shape[::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def template_of_crop(gray):
    """Enrolment template of one aligned crop (compact v3c format)."""
    return pmc.features(condition(gray))


def probe_of_crop(gray):
    """Probe = [first attempt, smoothed retry, rotated retries], each a compact template."""
    g = condition(gray)
    return [pmc.features(g), pmc.features(smooth(g))] + [pmc.features(rotate(g, a)) for a in RETRY_ANGLES]


def make_template(img, detect=True):
    return template_of_crop(pm.palm_crop_v3(img, detect))


def make_probe(img, detect=True):
    return probe_of_crop(pm.palm_crop_v3(img, detect))


def match(enrolled, probe):
    """ADDITION 2: one enrolled photo vs one probe -> dict(inliers, cells, score, attempt)."""
    best = dict(pmc.match(enrolled, probe[0]), attempt=0)
    if best["score"] >= THRESHOLD:
        return best
    for n, variant in enumerate(probe[1:], 1):
        r = pmc.match(enrolled, variant)
        r["score"] /= RETRY_FACTOR
        if r["score"] > best["score"]:
            best = dict(r, attempt=n)
    return best


def score(templates, probe):
    return max(match(t, probe)["score"] for t in templates)


def main():
    ap = argparse.ArgumentParser(description="palm verification, compact templates with robustness additions (v3cr)")
    ap.add_argument("--no-detect", action="store_true", help="images are already cropped palm close-ups")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enroll"); e.add_argument("out"); e.add_argument("images", nargs="+")
    v = sub.add_parser("verify"); v.add_argument("template"); v.add_argument("image")
    a = ap.parse_args()
    detect = not a.no_detect
    try:
        if a.cmd == "enroll":
            pmc.save_templates(a.out, [make_template(f, detect) for f in a.images])
            print(f"saved {a.out} ({len(a.images)} photo(s), compact v3c format)")
        else:
            crop = pm.palm_crop_v3(a.image, detect)
            probe = probe_of_crop(crop)
            per = [match(t, probe) for t in pmc.load_templates(a.template)]
            best = max(per, key=lambda r: r["score"])
            ok = best["score"] >= THRESHOLD
            how = ["first attempt", "smoothed retry"] + [f"retry rotated {x:+d} deg" for x in RETRY_ANGLES]
            print(f"score={best['score']:.1f} ({how[best['attempt']]}, noise level {noise_level(crop):.1f}, "
                  f"best of {len(per)} enrolled photo(s)) threshold={THRESHOLD:.1f} -> {'MATCH' if ok else 'NO MATCH'}")
            sys.exit(0 if ok else 1)
    except pm.QualityError as err:
        print("REJECTED PHOTO:", err)
        sys.exit(2)


if __name__ == "__main__":
    main()
