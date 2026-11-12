#!/usr/bin/env python3
"""
benchmarks/evaluate_palm_match.py

ISO/IEC 19795-1 style evaluation of training-free palm matchers on several palm datasets:
  v1 = palm_match.py Competitive Code
  v2 = palm_match.py 7-landmark alignment + Tan-Triggs + crease-scale SIFT/RANSAC
  v3 = same alignment, then DISK keypoints + LightGlue transformer matcher (pretrained, zero-shot; needs kornia)

Datasets (--dataset):
  real        real palm images/  (flat folder; 1000132768 is the LEFT hand, the rest the right hand)
  palmmatchdb data/PalmMatchDB   (whole-hand photos; identity = subject + hand across the 5/8/12 MP cameras;
              byte-duplicates and files that appear under 2+ palms removed)
  synthetic   in-house procedural palms (common/generators/procedural_palm.py), generated on the fly
  folder      <root>/<identity>/<image>       e.g. PCE-SynthPalm / Diff-Palm output, any custom set
  tongji      <root>/session1|session2/*.bmp  Tongji contactless ROI set: 600 palms x 10 per session,
              consecutive blocks of 10 files = one palm. Protocol: enrol session1, probe session2.
  casia       <root>/*.jpg named SSSS_g_h_NN.jpg (CASIA palmprint); identity = subject + hand

ROI-only datasets (tongji, synthetic, most PCE outputs) have no full hand to detect, so they are
matched with --no-detect behaviour automatically.

Metrics (ISO/IEC 19795-1 vocabulary)
  FTE     failure-to-enrol rate (quality gate rejected the image)
  FMR/FNMR at the operating threshold (algorithm level);  FAR/FRR include FTE (system level)
  EER     where FMR == FNMR
  TAR@FMR TAR = 1 - FNMR at FMR = 1e-2, 1e-3, 1e-4  (reported only when there are >= 3/FMR impostor
          comparisons - "rule of 3"; otherwise marked n/a because the rate cannot be resolved)
  Rank-1  closed-set identification: first usable image of each identity is the gallery
  d'      decidability index, AUC area under the ROC

Usage
  python benchmarks/evaluate_palm_match.py --dataset real
  python benchmarks/evaluate_palm_match.py --dataset palmmatchdb --methods v1 v2
  python benchmarks/evaluate_palm_match.py --dataset synthetic --ids 100 --per-id 5
  python benchmarks/evaluate_palm_match.py --dataset tongji --root D:/datasets/Tongji --ids 100
"""
import argparse
import collections
import glob
import hashlib
import itertools
import json
import os
import random
import re
import sys
import time

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import palm_match as pm  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ============================================================================ dataset loaders
# Each loader returns (samples, detect, protocol) where samples = list of (identity, name, image_or_path)
def load_any(path):
    x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if x is None:
        raise FileNotFoundError(path)
    if x.ndim == 3 and x.shape[2] == 4:                       # transparent PNG -> white background
        a = x[:, :, 3:4] / 255.0
        x = (x[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    if x.ndim == 2:
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    return x


def ds_real(root, **_):
    root = root or os.path.join(REPO, "real palm images")
    s = []
    for f in sorted(glob.glob(os.path.join(root, "*.jpg"))):
        k = os.path.basename(f)[:-4]
        s.append(("left_hand" if k == "1000132768" else "right_hand", k, f))
    return s, True, "all-vs-all"


def _pmdb_identity(rel_parts):
    """PalmMatchDB path: <SEX> PALMPRINTS/<age group>/<SUBJECT(age)>/<camera 5|8|12 MP>/<left|right>/file.
    The camera folders hold the SAME palm photographed with different cameras, so identity = subject + hand."""
    dirs = rel_parts[:-1]
    subj = next((d for d in dirs if re.match(r"^[FM]\s*\d+.*\(", d)), None)
    hand = next(("left" if "left" in d.lower() else "right" for d in dirs
                 if "left" in d.lower() or "right" in d.lower()), None)
    cam = next((d for d in dirs if re.search(r"\d+\s*m[bp]", d.lower())), "cam?")
    return (f"{subj}_{hand}" if subj and hand else None), cam


def ds_palmmatchdb(root, ids, per_id, seed, **_):
    root = root or os.path.join(REPO, "data", "PalmMatchDB", "PalmMatchDB")
    files = [f for f in glob.glob(os.path.join(root, "**", "*"), recursive=True) if f.lower().endswith(IMG_EXT)]
    owners, first, unparsed = collections.defaultdict(set), {}, 0
    for f in files:
        ident, cam = _pmdb_identity(os.path.relpath(f, root).replace("\\", "/").split("/"))
        if ident is None:
            unparsed += 1
            continue
        h = hashlib.md5(open(f, "rb").read()).hexdigest()
        owners[h].add(ident)
        first.setdefault((h, ident), (cam, f))                # one copy of each distinct image per identity
    by_id = collections.defaultdict(lambda: collections.defaultdict(list))
    conflicts = sum(len(v) > 1 for v in owners.values())
    for (h, ident), (cam, f) in first.items():
        if len(owners[h]) == 1:                                # drop images that sit under 2+ identities
            by_id[ident][cam].append(f)
    keys = sorted(k for k, v in by_id.items() if sum(len(x) for x in v.values()) >= 2)
    print(f"[palmmatchdb] {len(files)} files -> {len(owners)} distinct images, {len(by_id)} palms (subject+hand), "
          f"{len(keys)} with 2+ images, {conflicts} images dropped (appear under 2+ palms), {unparsed} unparsed")
    random.Random(seed).shuffle(keys)
    s = []
    for k in keys[:ids]:                                       # round-robin over cameras -> cross-camera pairs
        cams = [sorted(v) for _, v in sorted(by_id[k].items())]
        picked = [f for group in itertools.zip_longest(*cams) for f in group if f][:per_id]
        s += [(k, os.path.relpath(f, root), f) for f in picked]
    return s, True, "all-vs-all"


def ds_synthetic(ids, per_id, seed, **_):
    from common.generators.procedural_palm import ProceduralPalmGenerator
    gen = ProceduralPalmGenerator(image_size=(256, 256))
    rng = np.random.RandomState(seed)
    s = []
    for i in range(ids):
        mel = rng.uniform(0.2, 0.45)
        for j in range(per_id):
            p = gen.generate_pair(identity_seed=seed * 100000 + i, presentation_seed=seed * 7 + i * 1000 + j,
                                  melanin_fraction=mel, cupping_angle_deg=rng.uniform(0, 15),
                                  pitch_deg=rng.uniform(-12, 12), roll_deg=rng.uniform(-12, 12),
                                  yaw_deg=rng.uniform(-8, 8))
            img = (p["rgb"][:, :, ::-1] * 255).astype(np.uint8)  # RGB -> BGR
            s.append((f"syn{i:04d}", f"syn{i:04d}_{j}", img))
    return s, False, "all-vs-all"


def ds_folder(root, ids, per_id, seed, no_detect=False, **_):
    if not root:
        sys.exit("--root is required for --dataset folder")
    keys = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    random.Random(seed).shuffle(keys)
    s = []
    for k in keys:
        fs = sorted(f for f in glob.glob(os.path.join(root, k, "*")) if f.lower().endswith(IMG_EXT))
        if len(fs) >= 2:
            s += [(k, os.path.basename(f), f) for f in fs[:per_id]]
        if len({x[0] for x in s}) >= ids:
            break
    return s, not no_detect, "all-vs-all"


def ds_tongji(root, ids, per_id, **_):
    if not root:
        sys.exit("--root is required for --dataset tongji (folder containing session1/ and session2/)")
    s = []
    for sess in ("session1", "session2"):
        fs = sorted(f for f in glob.glob(os.path.join(root, sess, "*")) if f.lower().endswith(IMG_EXT))
        if not fs:
            sys.exit(f"no images in {os.path.join(root, sess)}")
        for idx, f in enumerate(fs):
            palm = idx // 10
            if palm < ids and idx % 10 < per_id:
                s.append((f"palm{palm:04d}", f"{sess}/{os.path.basename(f)}", f))
    return s, False, "cross-session"


def ds_casia(root, ids, per_id, seed, no_detect=False, **_):
    if not root:
        sys.exit("--root is required for --dataset casia")
    by_id = collections.defaultdict(list)
    for f in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        b = os.path.basename(f)
        parts = b.split("_")
        if f.lower().endswith(IMG_EXT) and len(parts) >= 4:
            by_id[f"{parts[0]}_{parts[2]}"].append(f)
    keys = sorted(k for k, v in by_id.items() if len(v) >= 2)
    random.Random(seed).shuffle(keys)
    s = [(k, os.path.basename(f), f) for k in keys[:ids] for f in sorted(by_id[k])[:per_id]]
    return s, not no_detect, "all-vs-all"


LOADERS = dict(real=ds_real, palmmatchdb=ds_palmmatchdb, synthetic=ds_synthetic, folder=ds_folder,
               tongji=ds_tongji, casia=ds_casia)


# ============================================================================ feature extraction
def extract(sample, method, detect):
    img = sample[2]
    if isinstance(img, str):
        img = load_any(img)
    if method == "v3":
        return pm.make_template_v3(img, detect)
    if method == "v2":
        return pm.make_template_v2(img, detect)
    roi = pm.palm_roi(img, detect)
    return pm.encode_roi(roi), [pm.encode_roi(roi, r) for r in pm.ROTATIONS]


# ============================================================================ fast v2 scoring (GPU)
class V2Scorer:
    """Same decision logic as palm_match._inliers / score_v2, with the ratio-test kNN done on the GPU."""

    def __init__(self, feats):
        import torch
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.pts = [f[0] for f in feats]
        self.des = [torch.from_numpy(f[1]).to(self.dev) for f in feats]

    def _one_way(self, a, b, ratio=0.85, reproj=8.0):
        torch = self.torch
        pa, pb, da, db = self.pts[a], self.pts[b], self.des[a], self.des[b]
        if len(pa) < 10 or len(pb) < 10:
            return 0
        d, j = torch.cdist(db, da).topk(2, dim=1, largest=False)
        ok = (d[:, 0] < ratio * d[:, 1]).cpu().numpy()
        if ok.sum() < 6:
            return 0
        jj = j[:, 0].cpu().numpy()[ok]
        src, dst = pb[ok], pa[jj]
        m, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=reproj,
                                             maxIters=5000, confidence=0.999)
        if m is None:
            return 0
        scale = np.hypot(m[0, 0], m[1, 0])
        rot = np.degrees(np.arctan2(m[1, 0], m[0, 0]))
        c = np.array([pm.BIG / 2, pm.BIG / 2])
        shift = np.abs(m[:, 2] - (np.eye(2) - m[:, :2]) @ c).max()
        if not (0.75 < scale < 1.33 and abs(rot) < 25 and shift < 160 * pm.BIG / 512):
            return 0
        return int(inl.sum())

    def __call__(self, a, b):
        return float(min(self._one_way(a, b), self._one_way(b, a)))


class V3Scorer:
    """palm_match.match_v3 (coverage-weighted score) on precomputed templates: a = enrolled photo, b = probe."""

    def __init__(self, feats):
        self.f = feats

    def __call__(self, a, b):
        return float(pm.match_v3(self.f[a], self.f[b])["score"])


def make_scorer(method, feats):
    if method == "v3":
        return V3Scorer(feats)
    if method == "v2":
        return V2Scorer(feats)
    return lambda a, b: -pm.distance(feats[a][0], feats[b][1])   # similarity = -distance


# ============================================================================ metrics
def rates(gen, imp, t):
    """similarity scores: accept when score >= t"""
    return float(np.mean(imp >= t)), float(np.mean(gen < t))   # FMR, FNMR


def eer(gen, imp):
    ts = np.unique(np.concatenate([gen, imp]))
    best = (1.0, None, None)
    for t in ts:
        fmr, fnmr = rates(gen, imp, t)
        if abs(fmr - fnmr) < best[0]:
            best = (abs(fmr - fnmr), (fmr + fnmr) / 2, t)
    # a gap between the classes -> EER 0 at any threshold inside the gap
    if gen.min() > imp.max():
        return 0.0, float((gen.min() + imp.max()) / 2)
    return float(best[1]), float(best[2])


def tar_at_fmr(gen, imp, target):
    if len(imp) < 3 / target:
        return None                                            # not resolvable with this many impostors
    ts = np.unique(np.concatenate([gen, imp]))[::-1]
    best = 0.0
    for t in ts:
        fmr, fnmr = rates(gen, imp, t)
        if fmr <= target:
            best = 1 - fnmr
        else:
            break
    return float(best)


def auc(gen, imp):
    g, i = np.asarray(gen), np.asarray(imp)
    order = np.argsort(np.concatenate([g, i]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    allv = np.concatenate([g, i])
    for v in np.unique(allv):
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[:len(g)].sum() - len(g) * (len(g) + 1) / 2) / (len(g) * len(i)))


def dprime(gen, imp):
    return float(abs(gen.mean() - imp.mean()) / np.sqrt((gen.var() + imp.var()) / 2 + 1e-12))


# ============================================================================ main evaluation
def capture_time(name):
    """Capture time (unix seconds) from a PalmMatchDB file name, or nan."""
    b = os.path.basename(name)
    m = re.search(r"IMG_(\d{8})_(\d{6})", b)
    if m:
        return time.mktime(time.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S"))
    m = re.search(r"ei_(\d{13})", b)
    return int(m.group(1)) / 1000 if m else float("nan")


TRUST_SAME, TRUST_DIFF = 600, 86400   # PalmMatchDB label filter: same-palm pairs < 10 min apart, others > 1 day


def metrics(method, gen, imp, threshold, fta):
    fmr, fnmr = rates(gen, imp, threshold)
    e, et = eer(gen, imp)
    return dict(threshold=threshold, FMR=fmr, FNMR=fnmr, FAR=fmr * (1 - fta), FRR=fta + fnmr * (1 - fta),
                EER=e, EER_threshold=et, TAR_at_FMR_1e2=tar_at_fmr(gen, imp, 1e-2),
                TAR_at_FMR_1e3=tar_at_fmr(gen, imp, 1e-3), TAR_at_FMR_1e4=tar_at_fmr(gen, imp, 1e-4),
                dprime=dprime(gen, imp), AUC=auc(gen, imp),
                genuine_median=float(np.median(gen)), impostor_median=float(np.median(imp)),
                impostor_max=float(imp.max()), genuine_min=float(gen.min()),
                genuine_comparisons=len(gen), impostor_comparisons=len(imp))


def evaluate_enrol(keep, ids, sess, ts, score, n_enrol, protocol, trusted, max_imp, seed):
    """Multi-photo enrolment: template = first n_enrol usable photos of a palm (session1 for cross-session);
    probes = the palm's remaining photos (session2). Score = best match over the enrolled photos."""
    by = collections.defaultdict(list)
    for i, k in enumerate(ids):
        by[k].append(i)
    enrol, probes = {}, {}
    for k, ix in by.items():
        if protocol == "cross-session":
            e = [i for i in ix if sess[i] == "session1"][:n_enrol]
            p = [i for i in ix if sess[i] == "session2"]
        else:
            e, p = ix[:n_enrol], ix[n_enrol:]
        if len(e) == n_enrol:
            enrol[k] = e
        probes[k] = p if len(e) == n_enrol else ix          # palms that cannot enrol still serve as impostors

    def gap(k, p):
        return float(np.nanmin(np.abs(ts[enrol[k]] - ts[p]))) if trusted else 0.0

    gen = [(k, p) for k in enrol for p in probes[k] if not trusted or gap(k, p) < TRUST_SAME]
    imp = [(k, p) for k in enrol for o, ps in probes.items() if o != k for p in ps
           if not trusted or gap(k, p) >= TRUST_DIFF]
    if len(imp) > max_imp:
        imp = random.Random(seed).sample(imp, max_imp)
    sc = lambda k, p: max(score(e, p) for e in enrol[k])
    return (np.array([sc(k, p) for k, p in gen]), np.array([sc(k, p) for k, p in imp]), len(enrol))


def evaluate(samples, method, detect, protocol, threshold, max_imp, seed, max_probes=150, trusted=False,
             n_enrol=1):
    t0 = time.time()
    feats, keep, fte = [], [], collections.Counter()
    for s in samples:
        try:
            feats.append(extract(s, method, detect))
            keep.append(s)
        except pm.QualityError as e:
            fte[str(e).split(" - ")[0]] += 1
        except Exception as e:  # unreadable file etc.
            fte[type(e).__name__] += 1
    n_fte = sum(fte.values())
    ids = [s[0] for s in keep]
    sess = [s[1].split("/")[0] if protocol == "cross-session" else "" for s in keep]
    raw_score, cache = make_scorer(method, feats), {}

    def score(a, b):
        if (a, b) not in cache:
            cache[a, b] = raw_score(a, b)
        return cache[a, b]

    idx = range(len(keep))
    if protocol == "cross-session":
        gen_pairs = [(a, b) for a in idx for b in idx if ids[a] == ids[b] and sess[a] == "session1" and sess[b] == "session2"]
        imp_pairs = [(a, b) for a in idx for b in idx if ids[a] != ids[b] and sess[a] == "session1" and sess[b] == "session2"]
    else:
        gen_pairs = [(a, b) for a, b in itertools.combinations(idx, 2) if ids[a] == ids[b]]
        imp_pairs = [(a, b) for a, b in itertools.combinations(idx, 2) if ids[a] != ids[b]]
    ts = np.array([capture_time(s[1]) for s in keep])
    if trusted:                                   # PalmMatchDB: keep only pairs whose label can be trusted
        gen_pairs = [(a, b) for a, b in gen_pairs if abs(ts[a] - ts[b]) < TRUST_SAME]
        imp_pairs = [(a, b) for a, b in imp_pairs if abs(ts[a] - ts[b]) >= TRUST_DIFF]
    if len(imp_pairs) > max_imp:
        imp_pairs = random.Random(seed).sample(imp_pairs, max_imp)
    gen = np.array([score(a, b) for a, b in gen_pairs])
    imp = np.array([score(a, b) for a, b in imp_pairs])
    if len(gen) == 0 or len(imp) == 0:
        return dict(error="not enough usable images for genuine and impostor comparisons", fte=dict(fte))

    # Rank-1 (closed set): gallery = first usable image per identity (session1 for cross-session)
    gallery = {}
    for i in idx:
        if (protocol != "cross-session" or sess[i] == "session1") and ids[i] not in gallery:
            gallery[ids[i]] = i
    probes = [i for i in idx if i not in gallery.values() and ids[i] in gallery and
              (protocol != "cross-session" or sess[i] == "session2")]
    if len(probes) > max_probes:
        probes = random.Random(seed).sample(probes, max_probes)
    if trusted:
        probes = []                               # rank-1 needs every label to be right: not possible here
    hits = 0
    for p in probes:
        sc = {k: score(g, p) for k, g in gallery.items()}
        right = sc[ids[p]]
        hits += right > max(v for k, v in sc.items() if k != ids[p]) if len(sc) > 1 else 0

    if threshold is None:
        threshold = {"v2": pm.MIN_INLIERS, "v3": pm.V3_THRESHOLD}.get(method, -pm.DEFAULT_THRESHOLD)
    fta = n_fte / max(len(samples), 1)
    base = dict(method=method, protocol=protocol + (" (trusted-label pairs)" if trusted else ""),
                images=len(samples), usable=len(keep), identities=len(set(ids)),
                fte_rate=fta, fte_reasons=dict(fte), rank1_probes=len(probes), gallery=len(gallery))
    out = dict(base, rank1=hits / len(probes) if probes else None, **metrics(method, gen, imp, threshold, fta),
               seconds=round(time.time() - t0, 1))
    results = [(method, out, gen, imp)]
    if n_enrol > 1:
        g2, i2, n_tpl = evaluate_enrol(keep, ids, sess, ts, score, n_enrol, protocol, trusted, max_imp, seed)
        if len(g2) and len(i2):
            o2 = dict(base, protocol=f"{base['protocol']}, {n_enrol}-photo enrolment ({n_tpl} palms enrolled)",
                      rank1=None, rank1_probes=0, **metrics(method, g2, i2, threshold, fta),
                      seconds=round(time.time() - t0, 1))
            results.append((f"{method}_enrol{n_enrol}", o2, g2, i2))
    return results


def fmt(v, pct=True):
    if v is None:
        return "n/a"
    return f"{v * 100:.2f}%" if pct else f"{v:.3f}"


def print_report(name, r):
    if "error" in r:
        print(f"[{name}] {r['error']}  FTE reasons: {r['fte_reasons'] if 'fte_reasons' in r else r.get('fte')}")
        return
    print(f"\n=== {name} | method {r['method']} | protocol {r['protocol']} ===")
    print(f"images {r['images']}  usable {r['usable']}  identities {r['identities']}  FTE {fmt(r['fte_rate'])} {r['fte_reasons'] or ''}")
    print(f"comparisons: genuine {r['genuine_comparisons']}  impostor {r['impostor_comparisons']}")
    unit = {"v3": "score", "v2": "inliers"}.get(r['method'])
    thr = f"{r['threshold']:.1f} {unit}" if unit else f"distance {-r['threshold']:.3f}"
    print(f"at operating threshold ({thr}): FMR {fmt(r['FMR'])}  FNMR {fmt(r['FNMR'])}  |  system FAR {fmt(r['FAR'])}  FRR {fmt(r['FRR'])}")
    et = f"{r['EER_threshold']:.1f} {unit}" if unit else f"distance {-r['EER_threshold']:.3f}"
    print(f"EER {fmt(r['EER'])} (at {et})   TAR@FMR=1e-2 {fmt(r['TAR_at_FMR_1e2'])}  1e-3 {fmt(r['TAR_at_FMR_1e3'])}  1e-4 {fmt(r['TAR_at_FMR_1e4'])}")
    print(f"Rank-1 {fmt(r['rank1'])} ({r['rank1_probes']} probes, gallery {r['gallery']})   d' {r['dprime']:.2f}   AUC {r['AUC']:.4f}")
    sign = 1 if r['method'] != 'v1' else -1
    print(f"scores: genuine median {sign * r['genuine_median']:.3f} (worst {sign * r['genuine_min']:.3f})   "
          f"impostor median {sign * r['impostor_median']:.3f} (most similar {sign * r['impostor_max']:.3f})   [{r['seconds']} s]")


def save_plot(path, title, gen, imp, method):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sign = 1 if method != "v1" else -1
    lo, hi = min(gen.min(), imp.min()), max(gen.max(), imp.max())
    bins = np.linspace(sign * lo, sign * hi, 40) if sign > 0 else np.linspace(sign * hi, sign * lo, 40)
    ax[0].hist(sign * imp, bins=bins, alpha=0.6, label=f"impostor (n={len(imp)})", color="#d9534f", density=True)
    ax[0].hist(sign * gen, bins=bins, alpha=0.6, label=f"genuine (n={len(gen)})", color="#5cb85c", density=True)
    ax[0].set_xlabel("score (higher = same palm)" if method != "v1" else "distance (lower = same palm)")
    ax[0].legend(); ax[0].set_title("score distributions")
    ts = np.unique(np.concatenate([gen, imp]))
    fm = np.array([rates(gen, imp, t)[0] for t in ts]); fn = np.array([rates(gen, imp, t)[1] for t in ts])
    floor = 1.0 / (len(imp) + 1)
    ax[1].loglog(np.maximum(fm, floor), np.maximum(fn, 1.0 / (len(gen) + 1)), "-o", ms=3)
    ax[1].set_xlabel("FMR"); ax[1].set_ylabel("FNMR"); ax[1].set_title("DET curve"); ax[1].grid(True, which="both", alpha=0.3)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    ap.add_argument("--root", default=None)
    ap.add_argument("--methods", nargs="+", default=["v3"], choices=["v1", "v2", "v3"])
    ap.add_argument("--enrol", type=int, default=1, help="also report N-photo enrolment (e.g. 3)")
    ap.add_argument("--all-pairs", action="store_true",
                    help="palmmatchdb: use raw labels (default keeps only trustworthy-label pairs)")
    ap.add_argument("--max-probes", type=int, default=150, help="rank-1: max probe images")
    ap.add_argument("--ids", type=int, default=100, help="max identities (palms)")
    ap.add_argument("--per-id", type=int, default=5, help="max images per identity (per session for tongji)")
    ap.add_argument("--max-impostor", type=int, default=30000)
    ap.add_argument("--threshold", type=float, default=None, help="operating threshold (v2: inliers)")
    ap.add_argument("--no-detect", action="store_true", help="folder/casia: images are palm ROIs already")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO, "benchmarks", "results"))
    a = ap.parse_args()
    samples, detect, protocol = LOADERS[a.dataset](root=a.root, ids=a.ids, per_id=a.per_id, seed=a.seed,
                                                   no_detect=a.no_detect)
    print(f"[{a.dataset}] {len(samples)} images, {len({s[0] for s in samples})} identities, "
          f"hand detection {'on' if detect else 'off (ROI images)'}, protocol {protocol}")
    os.makedirs(a.out, exist_ok=True)
    report = {}
    for m in a.methods:
        trusted = a.dataset == "palmmatchdb" and not a.all_pairs
        res = evaluate(samples, m, detect, protocol, a.threshold if m != "v1" else None, a.max_impostor, a.seed,
                       a.max_probes, trusted=trusted, n_enrol=a.enrol)
        if isinstance(res, dict):
            print_report(a.dataset, res); report[m] = res; continue
        for key, r, gen, imp in res:
            print_report(a.dataset, r); report[key] = r
            save_plot(os.path.join(a.out, f"{a.dataset}_{key}.png"), f"{a.dataset} - palm_match {key}", gen, imp, m)
    jpath = os.path.join(a.out, f"{a.dataset}.json")
    if os.path.exists(jpath):                                  # keep results of methods not re-run this time
        report = {**json.load(open(jpath)), **report}
    with open(jpath, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved {a.out}/{a.dataset}.json and score/DET plots")
    if any(isinstance(r, dict) and r.get("identities", 0) < 20 for r in report.values()):
        print("WARNING: fewer than 20 identities - smoke test only, not an accuracy measurement.")
    print("Note: claiming FAR <= 1e-5 needs >= 300,000 independent impostor comparisons (rule of 3).")


if __name__ == "__main__":
    main()
