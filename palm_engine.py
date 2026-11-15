#!/usr/bin/env python3
"""
palm_engine.py - the complete palm pipeline ("v4") used by the kiosk web app (webapp/kiosk.py).
palm_match.py, palm_match_compact.py and palm_match_robust.py are not changed by this file.

  frame      MediaPipe landmarks -> capture quality check with a retake instruction (palm_quality.py)
  liveness   random finger-count challenge, checked on the server (palm_pad.ActiveLiveness), and the passive
             screen/print check on every captured photo (palm_pad.recapture_score)
  template   the aligned crop is smoothed if noisy (v3cr) and contrast-equalised (CLAHE); v3c keypoints + descriptors of
             it, plus compact v2 SIFT features of the crop; score = v3c score / 12 + v2 inliers / 4, accepted at 1.5
             (settings in V4, chosen on DEV identities by benchmarks/tune_dev.py); plus the HOL vector (palm_embed.py)
             for search
  storage    keypoint templates encrypted at rest (palm_protect.py); for search only the IoM hashes of the HOL vector
             (cancellable: a new key gives new, unlinkable templates)
  search     protected-vector shortlist of SHORTLIST people -> matcher score with early exit
"""
import os
import time

import cv2
import numpy as np

import palm_embed as pe
import palm_match as pm
import palm_match_compact as pmc
import palm_match_robust as pmr
import palm_pad as pad
import palm_protect as pp
import palm_quality as pq

# Scoring settings of v4 (benchmarks/tune_dev.py, DEV identities only; see benchmarks/results/tuning/).
# Chosen on 40 DEV identities each of BJTU, Tongji and 11k Hands (benchmarks/results/tuning/): CLAHE + fusion lowered the
# BJTU EER from 8.1% to 4.7% and kept 0% on Tongji and 11k Hands; the highest of 1,358 DEV impostor scores was 1.05.
V4 = dict(
    preprocess="clahe",      # crop look before DISK: raw | clahe
    fusion=True,             # add the v2 (SIFT) score: score = v3c score / 12 + v2 inliers / 4
    threshold=1.5,           # 1.4 x the highest DEV impostor score; BJTU DEV FNMR 12% (v3c at 12: 30%)
)
SHORTLIST = 3                # people re-checked with the matcher after the protected-vector search


def preprocess(crop):
    crop = pmr.condition(crop)            # v3cr's noise-adaptive smoothing: only crops noisier than 4 grey levels
    if V4["preprocess"] == "clahe":
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(crop)
    return crop


# ----------------------------------------------------------------------------- one camera frame
def analyse(bgr, need_open_hand=True):
    """Landmarks, aligned crop and quality of one frame. need_open_hand=False skips the pose checks (the
    liveness challenge asks for folded fingers)."""
    t = time.perf_counter()
    try:
        pts = pm.detect_landmarks(bgr)
    except pm.QualityError:
        return dict(hand=False, issues=[("no_hand", "show your open palm to the camera")], ms=_ms(t))
    out = dict(hand=True, pts=pts, count=pad.finger_count(pts), spread=pad.finger_spread(pts))
    m = pq.pose_measures(pts, bgr.shape[1])
    try:
        crop = pm.align_palm_v2(bgr, pts, normalise=False)
    except pm.QualityError as e:
        msg = str(e)
        code = "too_small" if "far" in msg else "blurry"
        return dict(out, crop=None, measures=m, issues=[(code, pq.MESSAGES.get(code, msg))], ms=_ms(t))
    m.update(pq.crop_measures(crop))
    issues = pq.check(m)
    if not need_open_hand:
        issues = [(c, s) for c, s in issues if c not in ("fingers_closed", "tilted")]
    return dict(out, crop=crop, measures=m, issues=issues, quality=pq.score(m), ms=_ms(t))


def _ms(t):
    return round((time.perf_counter() - t) * 1000)


# ----------------------------------------------------------------------------- templates and scores
def make_template(crop):
    """Everything stored for one photo: v3c (compact), HOL vector, and v2 features when fusion is on."""
    t = dict(v3c=pmc.features(preprocess(crop)), hol=pe.hol(crop))
    if V4["fusion"]:
        kp, des = pm.features_v2(pm.illum_normalise(crop))
        # compact storage: positions in 1/16 px (uint16), RootSIFT values (0..1) in 1/255 steps (uint8)
        t["v2"] = (np.clip(np.round(kp * 16), 0, 65535).astype(np.uint16),
                   np.clip(np.round(des * 255), 0, 255).astype(np.uint8))
    return t


def _v2(t):
    kp, des = t
    if des.dtype == np.uint8:
        return kp.astype(np.float32) / 16, des.astype(np.float32) / 255
    return kp, des


def score(enrolled, probe):
    """v4 score of one enrolled photo vs one probe photo (higher = more alike)."""
    s = pm.match_v3(pmc.unpack(enrolled["v3c"]), pmc.unpack(probe["v3c"]))["score"]
    if not V4["fusion"]:
        return float(s)
    if s / pmc.THRESHOLD >= 2 * V4["threshold"]:           # clearly the same palm: the SIFT part cannot change that
        return float(s / pmc.THRESHOLD)
    e2, p2 = _v2(enrolled["v2"]), _v2(probe["v2"])
    v2 = min(pm._inliers(e2, p2), pm._inliers(p2, e2))
    return float(s / pmc.THRESHOLD + v2 / pm.MIN_INLIERS)


def best_score(photos, probe, threshold=None):
    """Best score over a person's enrolled photos, stopping at the first photo that passes."""
    thr = V4["threshold"] if threshold is None else threshold
    best = 0.0
    for e in photos:
        best = max(best, score(e, probe))
        if best >= thr:
            break
    return best


# ----------------------------------------------------------------------------- protected search
def search(probe, people, key, shortlist=SHORTLIST):
    """people: {id: dict(hashes=(n, IOM_K) uint8, photos=[templates])}. Returns dict(id or None, score, candidates,
    timings). The protected hashes pick the shortlist; the matcher decides."""
    t0 = time.perf_counter()
    if not people:
        return dict(id=None, score=0.0, candidates=[], search_ms=0, match_ms=0, comparisons=0)
    h = pp.protect(probe["hol"], key)
    ids = list(people)
    sims = np.array([pp.similarity_matrix(h[None], people[i]["hashes"]).max() for i in ids])
    order = [ids[j] for j in np.argsort(-sims)[:shortlist]]
    t1 = time.perf_counter()
    cands, found, n = [], None, 0
    for i in order:
        s = 0.0
        for e in people[i]["photos"]:
            s = max(s, score(e, probe))
            n += 1
            if s >= V4["threshold"]:
                break
        cands.append(dict(id=i, vector_similarity=round(float(sims[ids.index(i)]), 3), score=round(s, 2)))
        if s >= V4["threshold"]:
            found = i
            break
    t2 = time.perf_counter()
    best = max(cands, key=lambda c: c["score"])
    return dict(id=found, score=best["score"] if found is None else cands[-1]["score"], candidates=cands,
                search_ms=round((t1 - t0) * 1000), match_ms=round((t2 - t1) * 1000), comparisons=n)


def recapture(bgr, pts):
    return pad.recapture_score(bgr, pts)


def warm_up():
    pm._landmarker()
    pm.features_v3(np.zeros((pm.BIG, pm.BIG), np.uint8))
    pe.hol(np.zeros((pm.BIG, pm.BIG), np.uint8))


def describe():
    return dict(preprocess=V4["preprocess"], fusion=V4["fusion"], threshold=V4["threshold"], shortlist=SHORTLIST,
                iom=f"{pp.IOM_K} x {pp.IOM_M}", recapture_threshold=pad.RECAPTURE_THRESHOLD,
                device=pm._v3_models()["dev"])


def env_ok():
    return os.path.exists(pm.MODEL_PATH)
