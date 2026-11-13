#!/usr/bin/env python3
"""
benchmarks/robustness_test.py - how v3 holds up when the probe photo is degraded.

Enrolled photo: clean. Probe photo: the same degradation applied to genuine and impostor probes.
Conditions are fixed in advance. Reports FAR / FRR at the shipped threshold and EER per condition.

  python benchmarks/robustness_test.py --root data/Tongji --ids 200
"""
import argparse
import json
import os
import random
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palm_match as pm  # noqa: E402
import palm_match_compact as pmc  # noqa: E402
import palm_match_robust as pmr  # noqa: E402
import evaluate_palm_match as E  # noqa: E402


def warp(g, angle=0.0, scale=1.0):
    m = cv2.getRotationMatrix2D((g.shape[1] / 2, g.shape[0] / 2), angle, scale)
    return cv2.warpAffine(g, m, g.shape[::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def jpeg(g, q):
    return cv2.imdecode(cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_GRAYSCALE)


def noise(g, sigma):
    return np.clip(g.astype(np.float32) + np.random.RandomState(0).normal(0, sigma, g.shape), 0, 255).astype(np.uint8)


CONDITIONS = {
    "clean (baseline)": lambda g: g,
    "darker (x0.5)": lambda g: (g * 0.5).astype(np.uint8),
    "brighter (x1.5)": lambda g: np.clip(g * 1.5, 0, 255).astype(np.uint8),
    "low contrast (x0.4)": lambda g: np.clip((g.astype(np.float32) - 128) * 0.4 + 128, 0, 255).astype(np.uint8),
    "uneven light (gradient)": lambda g: np.clip(g * np.linspace(0.5, 1.4, g.shape[1])[None, :], 0, 255).astype(np.uint8),
    "blur (sigma 2)": lambda g: cv2.GaussianBlur(g, (0, 0), 2),
    "blur (sigma 4)": lambda g: cv2.GaussianBlur(g, (0, 0), 4),
    "rotation 10 deg": lambda g: warp(g, angle=10),
    "rotation 20 deg": lambda g: warp(g, angle=20),
    "closer (x1.15)": lambda g: warp(g, scale=1.15),
    "farther (x0.85)": lambda g: warp(g, scale=0.85),
    "sensor noise (sigma 12)": lambda g: noise(g, 12),
    "JPEG quality 20": lambda g: jpeg(g, 20),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "data", "Tongji"))
    ap.add_argument("--ids", type=int, default=200)
    ap.add_argument("--impostors", type=int, default=1500)
    ap.add_argument("--method", default="v3", choices=["v3", "v3c", "v3cr"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.out = a.out or os.path.join(REPO, "benchmarks", "results",
                                  "robustness_tongji.json" if a.method == "v3" else f"robustness_tongji_{a.method}.json")
    feat, pfeat, match, thr = {"v3": (pm.features_v3, pm.features_v3, pm.match_v3, pm.V3_THRESHOLD),
                               "v3c": (pmc.features, pmc.features, pmc.match, pmc.THRESHOLD),
                               "v3cr": (pmr.template_of_crop, pmr.probe_of_crop, pmr.match, pmr.THRESHOLD)}[a.method]
    samples, _, _ = E.ds_tongji(root=a.root, ids=a.ids, per_id=1)
    gray = {}
    for ident, name, path in samples:                       # one enrolled (session1) + one probe (session2) per palm
        g = cv2.resize(cv2.cvtColor(E.load_any(path), cv2.COLOR_BGR2GRAY), (pm.BIG, pm.BIG), interpolation=cv2.INTER_CUBIC)
        gray[ident, name.split("/")[0]] = g
    ids = sorted({k[0] for k in gray})
    enrol = {i: feat(gray[i, "session1"]) for i in ids}
    imp_pairs = random.Random(0).sample([(x, y) for x in ids for y in ids if x != y], a.impostors)
    report = {}
    print(f"{len(ids)} palms, {len(ids)} genuine + {len(imp_pairs)} impostor comparisons per condition, "
          f"method {a.method}, threshold {thr}")
    print(f"{'condition':26s} {'FRR':>7s} {'FAR':>7s} {'EER':>7s}  genuine median score")
    for name, fn in CONDITIONS.items():
        probe = {i: pfeat(fn(gray[i, "session2"])) for i in ids}
        gen = np.array([match(enrol[i], probe[i])["score"] for i in ids])
        imp = np.array([match(enrol[x], probe[y])["score"] for x, y in imp_pairs])
        far, frr = E.rates(gen, imp, thr)
        eer = E.eer(gen, imp)[0]
        report[name] = dict(FRR=frr, FAR=far, EER=eer, genuine_median=float(np.median(gen)),
                            impostor_max=float(imp.max()), genuine=len(gen), impostor=len(imp))
        print(f"{name:26s} {frr * 100:6.2f}% {far * 100:6.2f}% {eer * 100:6.2f}%  {np.median(gen):8.1f}", flush=True)
    json.dump(report, open(a.out, "w"), indent=2)
    print("saved", a.out)


if __name__ == "__main__":
    main()
