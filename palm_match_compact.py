#!/usr/bin/env python3
"""
palm_match_compact.py - v3 palm matcher with a COMPACT template (v3c). palm_match.py is not changed by this file.

Same pipeline as palm_match.py v3 (MediaPipe hand detection, 7-landmark alignment, quality gates,
DISK + LightGlue, pose-checked RANSAC, coverage-weighted score, multi-photo enrolment), with one change:

  NOVELTY 3 - compact template
    * keep the 1,024 strongest keypoints instead of 2,048
    * store each descriptor value in 1 byte (uint8) instead of 4 (float32)
    * store keypoint positions as 2 x uint16 in 1/64 pixel
    -> about 97 KB per enrolled photo (saved file) instead of about 1,066 KB (11x smaller).
  Benchmark EER, 1 enrolled photo, v3c vs v3: Tongji 0.05% vs 0.06%, PalmMatchDB 1.11% vs 1.32%,
  Gabor dataset 0.32% vs 0.29% (benchmarks/results/REPORT.md).

Commands
  python palm_match_compact.py enroll  me.npz  a.jpg b.jpg c.jpg     # 3 photos recommended
  python palm_match_compact.py verify  me.npz  new.jpg
  Add --no-detect if the images are already cropped palm close-ups.
"""
import argparse
import sys

import numpy as np

import palm_match as pm

KEYPOINTS = 1024
THRESHOLD = 12.0       # PROVISIONAL coverage-weighted score; see benchmarks/results for how it was set


def features(gray):
    """Compact template of one aligned palm crop: (K x 2 uint16 positions in 1/64 px, K x 128 uint8 descriptors)."""
    m = pm._v3_models(); torch = m["torch"]
    x = torch.from_numpy(np.ascontiguousarray(gray)).float()[None, None].repeat(1, 3, 1, 1).to(m["dev"]) / 255
    with torch.no_grad():
        f = m["disk"](x, KEYPOINTS, pad_if_not_divisible=True)[0]      # DISK returns the K strongest keypoints
    kp = np.clip(np.round(f.keypoints.cpu().numpy() * 64), 0, 65535).astype(np.uint16)
    des = np.clip(np.round((f.descriptors.cpu().numpy() + 1) * 127.5), 0, 255).astype(np.uint8)
    return kp, des


def unpack(t):
    """Compact template -> the float (positions, descriptors) pair that palm_match.match_v3 expects."""
    kp, des = t
    d = des.astype(np.float32) / 127.5 - 1
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    return kp.astype(np.float32) / 64, d


def make_template(img, detect=True):
    return features(pm.palm_crop_v3(img, detect))


def match(a, b):
    """One enrolled photo vs one probe (both compact) -> dict(inliers, cells, score)."""
    return pm.match_v3(unpack(a), unpack(b))


def score(templates, probe):
    if isinstance(templates, tuple):
        templates = [templates]
    return max(match(t, probe)["score"] for t in templates)


def save_templates(path, templates):
    arrays = {"method": np.array("v3c")}
    for i, (p, d) in enumerate(templates):
        arrays[f"kp{i}"], arrays[f"des{i}"] = p, d
    np.savez_compressed(path, **arrays)


def load_templates(path):
    z = np.load(path)
    if "method" not in z or str(z["method"]) != "v3c":
        sys.exit("not a compact (v3c) template file - use palm_match.py for v1/v2/v3 templates")
    n = sum(1 for k in z.files if k.startswith("kp"))
    return [(z[f"kp{i}"], z[f"des{i}"]) for i in range(n)]


def main():
    ap = argparse.ArgumentParser(description="palm verification with compact templates (v3c)")
    ap.add_argument("--no-detect", action="store_true", help="images are already cropped palm close-ups")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enroll"); e.add_argument("out"); e.add_argument("images", nargs="+")
    v = sub.add_parser("verify"); v.add_argument("template"); v.add_argument("image")
    v.add_argument("--score-threshold", type=float, default=THRESHOLD)
    a = ap.parse_args()
    detect = not a.no_detect
    try:
        if a.cmd == "enroll":
            save_templates(a.out, [make_template(f, detect) for f in a.images])
            print(f"saved {a.out} ({len(a.images)} photo(s), method v3c)")
        else:
            probe = make_template(a.image, detect)
            per = [match(t, probe) for t in load_templates(a.template)]
            best = max(per, key=lambda r: r["score"])
            ok = best["score"] >= a.score_threshold
            print(f"score={best['score']:.1f} (inliers {best['inliers']}, palm cells {best['cells']}/64, "
                  f"best of {len(per)} enrolled photo(s)) threshold={a.score_threshold:.1f} -> "
                  f"{'MATCH' if ok else 'NO MATCH'}")
            sys.exit(0 if ok else 1)
    except pm.QualityError as err:
        print("REJECTED PHOTO:", err)
        sys.exit(2)


if __name__ == "__main__":
    main()
