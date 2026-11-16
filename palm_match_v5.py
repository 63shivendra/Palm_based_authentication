#!/usr/bin/env python3
"""
palm_match_v5.py - palm matcher v5. palm_match.py, palm_match_compact.py and palm_match_robust.py are not
changed by this file; enrolment files use the same compact (v3c) format, so they work with all three.

Same capture, alignment, compact template (1,024 DISK keypoints, 1 byte per value) and robustness additions
(noise-adaptive smoothing, retry on failure) as palm_match_robust.py. What is new is the match stage:

  1. LightGlue at full accuracy: early stopping and point pruning switched off (they are speed shortcuts).
  2. Affine geometric check instead of similarity: allows the slight stretch a tilted palm produces, with
     limits on scale, stretch, rotation and shift so that two aligned palms can only differ a little.
  3. Confidence-weighted score: each geometrically consistent match counts with LightGlue's own confidence,
     times the share of the palm the matches cover (8 x 8 grid), instead of every match counting as 1.
  4. Blur limit 8 instead of 15 (variance of Laplacian on the 128 px crop). The old limit mainly rejected
     low-resolution photos whose palm lines still match; a blurry photo can only lower a score, not raise an
     impostor's. On PalmMatchDB this cut rejected photos from 3.3% to 0.5% with no false accepts.

Items 1-3 were chosen on Tongji images 5-8, which the benchmark does not use (benchmark: images 0-2). On that
development set the weakest 1% of genuine scores was 4.3 x the strongest impostor score with v3c, 13.8 x with v5.

Commands
  python palm_match_v5.py enroll  me.npz  a.jpg b.jpg c.jpg
  python palm_match_v5.py verify  me.npz  new.jpg
  Add --no-detect if the images are already cropped palm close-ups.
"""
import argparse
import sys

import cv2
import numpy as np

import palm_match as pm
import palm_match_compact as pmc
import palm_match_robust as pmr

THRESHOLD = 10.0            # set on the development set: impostor max 4.2, genuine 1st percentile 58
LG_CONF = dict(depth_confidence=-1, width_confidence=-1, filter_threshold=0.1)
REPROJ = 8.0
MIN_SHARPNESS = 8.0
RETRY_FACTOR = pmr.RETRY_FACTOR
_LG = None


def _matcher():
    global _LG
    if _LG is None:
        m = pm._v3_models()
        _LG = m["KF"].LightGlueMatcher("disk", dict(LG_CONF)).eval().to(m["dev"])
    return _LG


def _pose_ok(m):
    """Two landmark-aligned palms may differ only a little: per-axis scale, stretch, rotation, shift."""
    sv = np.linalg.svd(m[:, :2], compute_uv=False)
    if not (sv.min() > 0.7 and sv.max() < 1.45 and sv.max() / sv.min() < 1.35):
        return False
    scale = np.hypot(m[0, 0], m[1, 0])
    rot = np.degrees(np.arctan2(m[1, 0], m[0, 0]))
    c = np.array([pm.BIG / 2, pm.BIG / 2])
    shift = np.abs(m[:, 2] - (np.eye(2) - m[:, :2]) @ c).max()
    return 0.75 < scale < 1.33 and abs(rot) < 25 and shift < 160 * pm.BIG / 512


def match_crops(a, b):
    """One enrolled template vs one probe template, both unpacked -> dict(inliers, cells, score)."""
    m = pm._v3_models(); torch, KF, dev = m["torch"], m["KF"], m["dev"]
    (pa, da), (pb, db) = a, b
    zero = dict(inliers=0, cells=0, score=0.0)
    if len(pa) < 8 or len(pb) < 8:
        return zero
    ka, kb = torch.from_numpy(pa).to(dev), torch.from_numpy(pb).to(dev)
    la = KF.laf_from_center_scale_ori(ka[None], torch.ones(1, len(ka), 1, 1, device=dev))
    lb = KF.laf_from_center_scale_ori(kb[None], torch.ones(1, len(kb), 1, 1, device=dev))
    hw = torch.tensor([pm.BIG, pm.BIG], device=dev)
    with torch.no_grad():
        conf, idx = _matcher()(torch.from_numpy(da).to(dev), torch.from_numpy(db).to(dev), la, lb, hw1=hw, hw2=hw)
    idx, conf = idx.cpu().numpy(), conf.cpu().numpy().ravel()
    if len(idx) < 6:
        return zero
    qa, qb = pa[idx[:, 0]].astype(np.float32), pb[idx[:, 1]].astype(np.float32)
    t, inl = cv2.estimateAffine2D(qb, qa, method=cv2.RANSAC, ransacReprojThreshold=REPROJ, maxIters=5000, confidence=0.999)
    if t is None or not _pose_ok(t):
        return zero
    inl = inl.ravel().astype(bool)
    if not inl.any():
        return zero
    cells = min(pm._covered_cells(qa[inl]), pm._covered_cells(qb[inl]))
    return dict(inliers=int(inl.sum()), cells=int(cells), score=float(conf[inl].sum()) * cells / pm.V3_GRID ** 2)


def template_of_crop(gray):
    return pmr.template_of_crop(gray)


def probe_of_crop(gray):
    """[first attempt, smoothed retry, rotated retries] as compact templates (same as palm_match_robust)."""
    return pmr.probe_of_crop(gray)


def palm_crop(img, detect=True):
    """palm_match.palm_crop_v3 with the v5 blur limit (palm_match.py itself is left unchanged)."""
    old, pm.MIN_SHARPNESS = pm.MIN_SHARPNESS, MIN_SHARPNESS
    try:
        return pm.palm_crop_v3(img, detect)
    finally:
        pm.MIN_SHARPNESS = old


def make_template(img, detect=True):
    return template_of_crop(palm_crop(img, detect))


def make_probe(img, detect=True):
    return probe_of_crop(palm_crop(img, detect))


def match(enrolled, probe):
    """One enrolled photo (compact) vs one probe (list from probe_of_crop) -> dict(inliers, cells, score, attempt).
    Retries run only if the first attempt fails, and must reach RETRY_FACTOR x the threshold."""
    e = pmc.unpack(enrolled)
    best = dict(match_crops(e, pmc.unpack(probe[0])), attempt=0)
    if best["score"] >= THRESHOLD:
        return best
    for n, variant in enumerate(probe[1:], 1):
        r = match_crops(e, pmc.unpack(variant))
        r["score"] /= RETRY_FACTOR
        if r["score"] > best["score"]:
            best = dict(r, attempt=n)
    return best


def score(templates, probe):
    return max(match(t, probe)["score"] for t in templates)


def main():
    ap = argparse.ArgumentParser(description="palm verification, matcher v5")
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
            probe = make_probe(a.image, detect)
            per = [match(t, probe) for t in pmc.load_templates(a.template)]
            best = max(per, key=lambda r: r["score"])
            ok = best["score"] >= THRESHOLD
            how = ["first attempt", "smoothed retry"] + [f"retry rotated {x:+d} deg" for x in pmr.RETRY_ANGLES]
            print(f"score={best['score']:.1f} ({how[best['attempt']]}, best of {len(per)} enrolled photo(s)) "
                  f"threshold={THRESHOLD:.1f} -> {'MATCH' if ok else 'NO MATCH'}")
            sys.exit(0 if ok else 1)
    except pm.QualityError as err:
        print("REJECTED PHOTO:", err)
        sys.exit(2)


if __name__ == "__main__":
    main()
