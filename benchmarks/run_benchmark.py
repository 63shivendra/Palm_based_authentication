#!/usr/bin/env python3
"""
benchmarks/run_benchmark.py - zero-shot benchmark of the palm matchers on every downloaded dataset,
with a live terminal dashboard, a plain-text log, CSV/JSON results, charts and a markdown report.

  python scripts/fetch_datasets.py all            # once: download the datasets
  python benchmarks/run_benchmark.py              # everything on disk, all methods (takes a while)
  python benchmarks/run_benchmark.py --quick      # small smoke run (a few minutes)
  python benchmarks/run_benchmark.py --datasets tongji palmmatchdb --methods v3c v3cr
  python benchmarks/run_benchmark.py --resume benchmarks/results/runs/<run>   # continue a stopped run

Zero-shot protocol
  No model has seen a palm image: DISK and LightGlue are pretrained on outdoor scenes, MediaPipe only finds the
  hand. Each dataset with >= 40 identities is split by identity once, on the full-size sample (seeded): 20% DEV
  identities, 80% TEST identities (global_split). DEV identities are the only ones any tuning script has used, and
  here they only pick alternative thresholds; every reported number is measured on TEST identities, in every profile. Three operating
  points are reported:
    shipped    the threshold in the code (v3 26, v3c/v3cr 12): the true "out of the box" behaviour
    dev-EER    the threshold where FAR = FRR on DEV, applied to TEST
    dev-FAR    the threshold giving FAR <= 0.1% on DEV, applied to TEST
  Small datasets (< 40 identities) have no DEV split: all identities are TEST, shipped threshold only.

Metrics (what the NPCI brief asks for, ISO/IEC 19795-1 vocabulary)
  Accuracy, EER, FAR, FRR (FAR/FRR include photos rejected by the quality gates), TAR at FAR 1%/0.1%/0.01%,
  AUC, d', Rank-1, failure-to-enrol, latency per stage (GPU here + CPU), template size, robustness under
  simulated lighting/blur/noise/pose, and spoof detection (not measured: there is no anti-spoofing yet).

Output: benchmarks/results/runs/<YYYYmmdd-HHMM>/  run.log, results.json, metrics.csv, latency.csv,
        robustness.csv, datasets.csv, scores/*.npz, plots/*.png, REPORT.md
"""
import argparse
import collections
import datetime
import io
import itertools
import json
import logging
import os
import random
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = os.path.join(REPO, ".bench_env")
if os.path.isdir(ENV):                     # benchmark-only CUDA torch (see .gitignore); falls back to the CPU torch
    sys.path.insert(0, ENV)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from rich.console import Console, Group  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn,  # noqa: E402
                           TimeElapsedColumn, TimeRemainingColumn)
from rich.table import Table  # noqa: E402

import palm_match as pm  # noqa: E402
import palm_match_compact as pmc  # noqa: E402
import palm_match_robust as pmr  # noqa: E402
import palm_engine as eng  # noqa: E402
import palm_quality as pq  # noqa: E402
import evaluate_palm_match as E  # noqa: E402

DATA = os.path.join(REPO, "data")
CACHE = os.path.join(DATA, "_cache")
con = Console()
log = logging.getLogger("bench")

METHODS = {
    "v4":   dict(label="v4 (v3c + quality gate, DEV-tuned)", threshold=float(eng.V4["threshold"]), unit="score"),
    "v3":   dict(label="v3 (DISK+LightGlue, full)", threshold=float(pm.V3_THRESHOLD), unit="score"),
    "v3c":  dict(label="v3c (compact template)", threshold=float(pmc.THRESHOLD), unit="score"),
    "v3cr": dict(label="v3cr (compact + robust)", threshold=float(pmr.THRESHOLD), unit="score"),
    "v2":   dict(label="v2 (SIFT + RANSAC baseline)", threshold=float(pm.MIN_INLIERS), unit="inliers"),
    "v1":   dict(label="v1 (CompCode baseline)", threshold=-float(pm.DEFAULT_THRESHOLD), unit="-distance"),
}


# ============================================================================ datasets
def ds_flat(root, ids, per_id, seed, sessions=None, **_):
    """<root>/<class>_<n>.jpg (PolyU II, XINHUA). sessions=k: images 1..k are session1, the rest session2."""
    by = collections.defaultdict(list)
    for f in os.listdir(root):
        stem, ext = os.path.splitext(f)
        if ext.lower() in E.IMG_EXT and "_" in stem:
            c, n = stem.rsplit("_", 1)
            if n.isdigit():
                by[c].append((int(n), os.path.join(root, f)))
    keys = sorted(k for k, v in by.items() if len(v) >= 2)
    random.Random(seed).shuffle(keys)
    s = []
    for k in keys[:ids]:
        imgs = sorted(by[k])
        if sessions:
            for sess, part in (("session1", [x for x in imgs if x[0] <= sessions]),
                               ("session2", [x for x in imgs if x[0] > sessions])):
                s += [(k, f"{sess}/{os.path.basename(f)}", f) for _, f in part[:per_id]]
        else:
            s += [(k, os.path.basename(f), f) for _, f in imgs[:per_id]]
    return s


def ds_11k(root, ids, per_id, seed, **_):
    """11k Hands, palmar photos only; identity = person + hand side."""
    import csv
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(root, "HandInfo.csv"), encoding="utf-8")):
        aspect = r["aspectOfHand"].lower()
        if aspect.startswith("palmar") and r.get("irregularities", "0") == "0":
            by[f"{r['id']}_{aspect.split()[-1]}"].append(r["imageName"])
    img_dir = next((os.path.join(dp) for dp, _, fs in os.walk(root) if any(f.lower().endswith(".jpg") for f in fs)), root)
    keys = sorted(k for k, v in by.items() if len(v) >= 2)
    random.Random(seed).shuffle(keys)
    return [(k, f, os.path.join(img_dir, f)) for k in keys[:ids] for f in sorted(by[k])[:per_id]]


def ds_bjtu(root, ids, per_id, seed, **_):
    """BJTU palm v2: <person>/<person>_<F|S>_<L|R><n>.jpg. Identity = person + hand; F = session 1, S = session 2."""
    import re
    by = collections.defaultdict(list)
    for dp, _, fs in os.walk(root):
        for f in fs:
            m = re.match(r"^(\d+)_([FS])_([LR])(\d+)\.", f)
            if m:
                by[f"{m.group(1)}_{m.group(3)}"].append(("session1" if m.group(2) == "F" else "session2", int(m.group(4)),
                                                         os.path.join(dp, f)))
    keys = sorted(k for k, v in by.items() if {x[0] for x in v} == {"session1", "session2"})
    random.Random(seed).shuffle(keys)
    s = []
    for k in keys[:ids]:
        for sess in ("session1", "session2"):
            part = sorted(x for x in by[k] if x[0] == sess)[:per_id]
            s += [(k, f"{sess}/{os.path.basename(f)}", f) for _, _, f in part]
    return s


def find_dir(*parts):
    p = os.path.join(DATA, *parts)
    return p if os.path.isdir(p) else None


def pmdb_root():
    d = find_dir("PalmMatchDB")
    if not d:
        return None
    for dp, ds, _ in os.walk(d):
        if any("PALMPRINTS" in x.upper() for x in ds):
            return dp
    return None


# key -> (display name, licence use, loader(ids, per_id, seed) -> samples | None, protocol, trusted pairs only,
#         hand detection: True = whole-hand photos (MediaPipe + alignment), False = pre-cropped palm images)
DATASETS = {
    "tongji": ("Tongji contactless (ROI)", "eval-only",
               lambda **k: E.ds_tongji(root=find_dir("Tongji"), **k)[0] if find_dir("Tongji", "session1") else None,
               "cross-session", False, False),
    "palmmatchdb": ("PalmMatchDB (phone photos)", "permissive",
                    lambda **k: E.ds_palmmatchdb(root=pmdb_root(), **k)[0] if pmdb_root() else None,
                    "all-vs-all", True, True),
    "11khands": ("11k Hands (palmar photos)", "eval-only",
                 lambda **k: ds_11k(find_dir("11kHands"), **k)
                 if find_dir("11kHands") and os.path.exists(os.path.join(DATA, "11kHands", "HandInfo.csv")) else None,
                 "all-vs-all", False, True),
    "bjtu2": ("BJTU palm v2 (ROI)", "eval-only",
              lambda **k: ds_bjtu(find_dir("HewelXX", "BJTU_V2"), **k) if find_dir("HewelXX", "BJTU_V2") else None,
              "cross-session", False, False),
    "xinhua": ("XINHUA (ROI)", "eval-only",
               lambda **k: ds_flat(find_dir("HewelXX", "XINHUA"), **k) if find_dir("HewelXX", "XINHUA") else None,
               "all-vs-all", False, False),
    "gabor": ("Gabor palm dataset (raw ROI)", "permissive",
              lambda **k: E.ds_folder(root=find_dir("GaborPalm", "raw"), no_detect=True, **k)[0]
              if find_dir("GaborPalm", "raw") else None, "all-vs-all", False, False),
    "real": ("Own phone photos", "own data", lambda **k: E.ds_real(None)[0], "all-vs-all", False, True),
}
# per-dataset sample sizes (identities, images per identity [per session]) for the full and the --quick run
SIZES = dict(tongji=(600, 3), palmmatchdb=(400, 6), gabor=(100, 8), xinhua=(100, 6), bjtu2=(296, 3),
             real=(10, 10), **{"11khands": (370, 5)})
QUICK = dict(tongji=(40, 2), palmmatchdb=(60, 3), gabor=(15, 4), xinhua=(30, 3), bjtu2=(40, 2),
             real=(10, 10), **{"11khands": (30, 3)})
KEY_CONDITIONS = ["clean (baseline)", "darker (x0.5)", "brighter (x1.5)", "uneven light (gradient)",
                  "overexposed (x2.0, clipped)", "hard shadow (half palm x0.4)", "blur (sigma 4)", "rotation 20 deg",
                  "sensor noise (sigma 12)", "JPEG quality 20"]
# full = sized for a GPU (the RTX 5080 desktop runs it in a few hours); quick = sized for a laptop CPU
PROFILES = {
    "full": dict(sizes=SIZES, max_genuine=6000, max_impostor=20000, rank1_probes=50, robust_ids=150,
                 robust_impostors=1000, conditions=None, methods=["v4", "v3c", "v3cr", "v3", "v2", "v1"],
                 v3cr_impostors=0.5, dev_divisor=2),
    "quick": dict(sizes=QUICK, max_genuine=200, max_impostor=500, rank1_probes=0, robust_ids=20,
                  robust_impostors=80, conditions=KEY_CONDITIONS, methods=["v4", "v3c", "v3cr", "v2", "v1"],
                  v3cr_impostors=0.25, dev_divisor=4),
}


# ============================================================================ fast metrics (vectorised)
def curve(gen, imp):
    g, i = np.sort(gen), np.sort(imp)
    ts = np.unique(np.concatenate([g, i, [max(g[-1], i[-1]) + 1]]))
    fmr = 1 - np.searchsorted(i, ts, "left") / len(i)          # impostor accepted: score >= t
    fnmr = np.searchsorted(g, ts, "left") / len(g)             # genuine rejected: score < t
    return ts, fmr, fnmr


def eer(gen, imp):
    if gen.min() > imp.max():
        return 0.0, float((gen.min() + imp.max()) / 2)
    ts, fmr, fnmr = curve(gen, imp)
    k = int(np.argmin(np.abs(fmr - fnmr)))
    return float((fmr[k] + fnmr[k]) / 2), float(ts[k])


def rates(gen, imp, t):
    return float(np.mean(imp >= t)), float(np.mean(gen < t))


def tar_at(gen, imp, target):
    if len(imp) < 3 / target:                                  # rule of 3: not resolvable
        return None
    ts, fmr, fnmr = curve(gen, imp)
    ok = fmr <= target
    return float((1 - fnmr[ok]).max()) if ok.any() else 0.0


def thr_at_fmr(gen, imp, target):
    ts, fmr, _ = curve(gen, imp)
    return float(ts[fmr <= target].min())


def auc(gen, imp):
    i = np.sort(imp)
    lo, hi = np.searchsorted(i, gen, "left"), np.searchsorted(i, gen, "right")
    return float(((lo + 0.5 * (hi - lo)) / len(i)).mean())


def dprime(gen, imp):
    return float(abs(gen.mean() - imp.mean()) / np.sqrt((gen.var() + imp.var()) / 2 + 1e-12))


def full_metrics(gen, imp, thr, fte):
    fmr, fnmr = rates(gen, imp, thr)
    e, et = eer(gen, imp)
    n = len(gen) + len(imp)
    return dict(threshold=thr, accuracy=1 - (fnmr * len(gen) + fmr * len(imp)) / n, FMR=fmr, FNMR=fnmr,
                FAR=fmr * (1 - fte), FRR=fte + fnmr * (1 - fte), EER=e, EER_threshold=et,
                TAR_at_FAR_1e2=tar_at(gen, imp, 1e-2), TAR_at_FAR_1e3=tar_at(gen, imp, 1e-3),
                TAR_at_FAR_1e4=tar_at(gen, imp, 1e-4), AUC=auc(gen, imp), dprime=dprime(gen, imp),
                genuine_median=float(np.median(gen)), genuine_min=float(gen.min()),
                impostor_median=float(np.median(imp)), impostor_max=float(imp.max()),
                genuine_comparisons=len(gen), impostor_comparisons=len(imp))


def at_threshold(gen, imp, thr, fte):
    fmr, fnmr = rates(gen, imp, thr)
    return dict(threshold=thr, FAR=fmr * (1 - fte), FRR=fte + fnmr * (1 - fte),
                accuracy=1 - (fnmr * len(gen) + fmr * len(imp)) / (len(gen) + len(imp)))


# ============================================================================ feature store (disk cached)
class Store:
    """Aligned crops and per-method templates of one dataset. Templates are cached on disk under data/_cache/
    (a full v3 template is ~1 MB, too many to hold in RAM) with a small in-memory LRU."""

    def __init__(self, key, samples, detect, lru=384):
        self.key, self.samples, self.detect = key, samples, detect
        self.dir = os.path.join(CACHE, key)
        self.lru, self.lru_n = collections.OrderedDict(), lru
        self.usable, self.fte = [], collections.Counter()
        self.crops = {}

    def _path(self, kind, i):
        name = self.samples[i][1].replace("/", "__").replace("\\", "__")
        return os.path.join(self.dir, kind, os.path.splitext(name)[0] + ".npz")

    def crop(self, i):
        if i in self.crops:
            return self.crops[i]
        img = self.samples[i][2]
        img = E.load_any(img) if isinstance(img, str) else img
        return pm.palm_crop_v3(img, self.detect)

    def prepare(self, on_step, on_event, timings):
        """Detect + align every image once; record failures to enrol. Crops (and the 21 landmarks, for the v1
        baseline) of detected hands are cached on disk."""
        rej_file = os.path.join(self.dir, "rejected.json")
        rejected = json.load(open(rej_file)) if self.detect and os.path.exists(rej_file) else {}
        for i, s in enumerate(self.samples):
            if self.detect:
                p = self._path("crop", i)
                if s[1] in rejected:
                    self.fte[rejected[s[1]]] += 1
                elif os.path.exists(p):
                    self.usable.append(i)
                else:
                    t = time.perf_counter()
                    try:
                        bgr = E.load_any(s[2]) if isinstance(s[2], str) else s[2]
                        pts = pm.detect_landmarks(bgr)
                        c = pm.align_palm_v2(bgr, pts, normalise=False)
                        timings["detect_align"].append((time.perf_counter() - t) * 1000)
                        os.makedirs(os.path.dirname(p), exist_ok=True)
                        np.savez(p, crop=c, pts=pts)
                        self.usable.append(i)
                    except pm.QualityError as e:
                        reason = str(e).split(" - ")[0]
                        rejected[s[1]] = reason
                        self.fte[reason] += 1
                        on_event(f"[yellow]rejected[/] {os.path.basename(s[1])}: {reason}")
                    except Exception as e:  # unreadable file
                        rejected[s[1]] = type(e).__name__
                        self.fte[type(e).__name__] += 1
            else:
                self.usable.append(i)
            on_step()
        if self.detect:
            os.makedirs(self.dir, exist_ok=True)
            json.dump(rejected, open(rej_file, "w"))

    def get_crop(self, i):
        if self.detect:
            return np.load(self._path("crop", i))["crop"]
        return self.crop(i)

    # ---------------------------------------------------------------- templates
    def compute(self, i, method):
        """Template of photo i. Never runs hand detection: detected datasets use the cached crop / landmarks,
        so the same alignment as palm_match.py is reproduced without loading MediaPipe in this process."""
        if method == "v2":
            if not self.detect:
                return E.extract(self.samples[i], "v2", False)
            return pm.features_v2(pm.illum_normalise(self.get_crop(i)))   # == palm_roi_v2 on the detected hand
        if method == "v1":
            if not self.detect:
                return E.extract(self.samples[i], "v1", False)
            img = self.samples[i][2]
            bgr = E.load_any(img) if isinstance(img, str) else img
            roi = pm.align_palm(bgr, np.load(self._path("crop", i))["pts"])  # v1's own 2-landmark crop
            return pm.encode_roi(roi), [pm.encode_roi(roi, r) for r in pm.ROTATIONS]
        c = self.get_crop(i)
        if method == "v4":                                  # quality gate first: a rejected photo is a failure to enrol
            m = pq.crop_measures(c)
            if self.detect:
                m.update(pq.pose_measures(np.load(self._path("crop", i))["pts"], 1.0))
            issues = pq.check(m)
            if issues:
                raise pm.QualityError("quality gate: " + ", ".join(code for code, _ in issues))
            return eng.make_template(c)
        if method == "v3":
            return pm.features_v3(c)
        if method == "v3c":
            return pmc.features(c)
        return pmr.probe_of_crop(c)                         # v3cr: [first attempt, retries...]; [0] = enrol template

    @staticmethod
    def pack(method, f):
        if method == "v4":
            out = {"kp": f["v3c"][0], "des": f["v3c"][1]}
            if "v2" in f:
                out["v2kp"], out["v2des"] = f["v2"]
            return out
        if method == "v3cr":
            return {f"{k}{j}": a for j, t in enumerate(f) for k, a in zip(("kp", "des"), t)}
        if method == "v1":
            code, probes = f
            out = {"code": code[0], "weight": code[1]}
            for j, (c, w) in enumerate(probes):
                out[f"pc{j}"], out[f"pw{j}"] = c, w
            return out
        return {"kp": f[0], "des": f[1]}

    @staticmethod
    def unpack(method, z):
        if method == "v4":
            f = {"v3c": (z["kp"], z["des"])}
            if "v2kp" in z.files:
                f["v2"] = (z["v2kp"], z["v2des"])
            return f
        if method == "v3cr":
            n = sum(1 for k in z.files if k.startswith("kp"))
            return [(z[f"kp{j}"], z[f"des{j}"]) for j in range(n)]
        if method == "v1":
            n = sum(1 for k in z.files if k.startswith("pc"))
            return (z["code"], z["weight"]), [(z[f"pc{j}"], z[f"pw{j}"]) for j in range(n)]
        return z["kp"], z["des"]

    def ensure(self, i, method, timings):
        """Compute and cache the template; False if this method's own quality gate rejects the photo (v1 only)."""
        p = self._path(method, i)
        if os.path.exists(p):
            return True
        bad = p + ".rejected"
        if os.path.exists(bad):
            return False
        t = time.perf_counter()
        try:
            f = self.compute(i, method)
        except Exception as e:  # QualityError of the method's own gate, or an unreadable file
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(bad, "w").write(str(e))
            return False
        timings[f"extract_{method}"].append((time.perf_counter() - t) * 1000)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.savez(p, **self.pack(method, f))
        return True

    def raw(self, i, method):
        with np.load(self._path(method, i)) as z:
            return self.unpack(method, z)

    def get(self, i, method):
        k = (i, method)
        if k in self.lru:
            self.lru.move_to_end(k)
            return self.lru[k]
        with np.load(self._path(method, i)) as z:
            f = self.unpack(method, z)
        if method == "v3c":
            f = pmc.unpack(f)
        self.lru[k] = f
        if len(self.lru) > self.lru_n:
            self.lru.popitem(last=False)
        return f


def make_score(store, method):
    """score(a, b): a = enrolled photo, b = probe photo. Higher = more similar."""
    if method == "v4":
        return lambda a, b: eng.score(store.get(a, method), store.get(b, method))
    if method in ("v3", "v3c"):
        return lambda a, b: float(pm.match_v3(store.get(a, method), store.get(b, method))["score"])
    if method == "v3cr":
        return lambda a, b: float(pmr.match(store.get(a, method)[0], store.get(b, method))["score"])
    if method == "v2":
        return lambda a, b: float(min(pm._inliers(store.get(a, method), store.get(b, method)),
                                      pm._inliers(store.get(b, method), store.get(a, method))))
    return lambda a, b: -pm.distance(store.get(a, method)[0], store.get(b, method)[1])


def template_bytes(store, method, i):
    """Size of one enrolled photo saved in each program's own template format."""
    buf = io.BytesIO()
    f = store.raw(i, method)
    if method == "v3cr":
        f = f[0]                                            # the enrolled template is the first attempt only
        pmc.save_templates(buf, [f])
    elif method == "v3c":
        pmc.save_templates(buf, [f])
    elif method == "v4":
        np.savez_compressed(buf, **Store.pack("v4", f))
    elif method == "v1":
        pm.save_templates(buf, [f[0]])
    else:
        np.savez(buf, kp=f[0], des=f[1])
    return buf.tell()


# ============================================================================ live dashboard
class Dashboard:
    def __init__(self, run_dir, device):
        self.run_dir, self.device = run_dir, device
        self.progress = Progress(SpinnerColumn(), TextColumn("[bold]{task.description}", justify="left"), BarColumn(bar_width=36),
                                 MofNCompleteColumn(), TextColumn("[cyan]{task.fields[rate]}"), TimeElapsedColumn(),
                                 TextColumn("eta"), TimeRemainingColumn(), console=con, expand=False)
        self.overall = self.progress.add_task("all jobs", total=1, rate="")
        self.stage = self.progress.add_task("starting", total=1, rate="")
        self.events = collections.deque(maxlen=7)
        self.live_job, self.live_stats = "", None
        self._t, self._n = time.time(), 0

    def set_stage(self, desc, total):
        self.progress.reset(self.stage, total=max(total, 1), completed=0, description=desc, rate="")
        self._t, self._n = time.time(), 0

    def step(self, n=1, unit="/s"):
        self._n += n
        dt = time.time() - self._t
        self.progress.update(self.stage, advance=n, rate=f"{self._n / dt:.1f}{unit}" if dt > 0.5 else "")

    def event(self, msg):
        self.events.append(f"[dim]{time.strftime('%H:%M:%S')}[/] {msg}")
        log.info(strip(msg))

    def __rich__(self):
        head = Table.grid(padding=(0, 2))
        head.add_row(f"[bold]Palm benchmark (zero-shot)[/]", f"device [green]{self.device}[/]",
                     f"log [dim]{os.path.relpath(os.path.join(self.run_dir, 'run.log'), REPO)}[/]")
        parts = [head, self.progress]
        if self.live_stats:
            s = self.live_stats
            t = Table(title=f"live: {self.live_job}", title_justify="left", expand=False, show_edge=False, pad_edge=False)
            for c in ("genuine", "impostor", "EER so far", "FAR @ shipped", "FRR @ shipped", "genuine median (min)",
                      "impostor median (max)", "separation"):
                t.add_column(c, justify="right")
            t.add_row(*s)
            parts.append(t)
        parts.append(Panel("\n".join(self.events) or "[dim]...[/]", title="events", title_align="left", border_style="dim"))
        return Group(*parts)


def strip(msg):
    import re
    return re.sub(r"\[/?[a-z ]*\]", "", msg)


def live_row(gen, imp, thr):
    g, i = np.asarray(gen, float), np.asarray(imp, float)
    cells = [f"{len(g):,}", f"{len(i):,}"]
    if len(g) and len(i):
        e, _ = eer(g, i)
        fmr, fnmr = rates(g, i, thr)
        if g.min() > i.max():                               # no overlap: say how wide the gap is
            sep = f"[green]gap {g.min() / i.max():.2f}x[/]" if i.max() > 0 and g.min() > 0 else "[green]no overlap[/]"
        else:
            sep = f"[yellow]overlap: {int((i >= g.min()).sum())} imp[/]"
        cells += [pct(e), pct(fmr), pct(fnmr), f"{np.median(g):.2f} ({g.min():.2f})", f"{np.median(i):.2f} ({i.max():.2f})", sep]
    else:
        cells += ["-"] * 6
    return cells


def pct(v):
    return "n/a" if v is None else f"{v * 100:.2f}%"


# ============================================================================ evaluation of one dataset x method
def pair_sets(idx, ids, sess, ts, protocol, trusted):
    if protocol == "cross-session":
        s1 = [a for a in idx if sess[a] == "session1"]
        s2 = [b for b in idx if sess[b] == "session2"]
        gen = [(a, b) for a in s1 for b in s2 if ids[a] == ids[b]]
        imp = ((a, b) for a in s1 for b in s2 if ids[a] != ids[b])
    else:
        gen = [(a, b) for a, b in itertools.combinations(idx, 2) if ids[a] == ids[b]]
        imp = ((a, b) for a, b in itertools.combinations(idx, 2) if ids[a] != ids[b])
    if trusted:                                   # PalmMatchDB: keep only pairs whose label can be trusted
        gen = [(a, b) for a, b in gen if abs(ts[a] - ts[b]) < E.TRUST_SAME]
        imp = ((a, b) for a, b in imp if abs(ts[a] - ts[b]) >= E.TRUST_DIFF)
    return gen, list(imp)


def enrol_sets(idx, ids, sess, ts, protocol, trusted, n_enrol):
    by = collections.defaultdict(list)
    for i in idx:
        by[ids[i]].append(i)
    enrol, probes = {}, {}
    for k, ix in by.items():
        if protocol == "cross-session":
            e, p = [i for i in ix if sess[i] == "session1"][:n_enrol], [i for i in ix if sess[i] == "session2"]
        else:
            e, p = ix[:n_enrol], ix[n_enrol:]
        if len(e) == n_enrol:
            enrol[k] = e
        probes[k] = p if len(e) == n_enrol else ix
    gap = lambda k, p: float(np.nanmin(np.abs(ts[enrol[k]] - ts[p]))) if trusted else 0.0
    gen = [(k, p) for k in enrol for p in probes[k] if not trusted or gap(k, p) < E.TRUST_SAME]
    imp = [(k, p) for k in enrol for o, ps in probes.items() if o != k for p in ps
           if not trusted or gap(k, p) >= E.TRUST_DIFF]
    return enrol, gen, imp


def score_pairs(pairs, score, dash, desc, thr, other=None, live=True, kind="gen"):
    """Score a list of pairs with a live progress bar and live metrics. other = scores of the opposite class."""
    dash.set_stage(desc, len(pairs))
    out, last = [], 0.0
    for a, b in sorted(pairs):                                 # sorted: the enrolled template stays in the LRU cache
        out.append(score(a, b))
        dash.step(unit=" cmp/s")
        if live and time.time() - last > 0.5:
            g, i = (out, other or []) if kind == "gen" else (other or [], out)
            dash.live_stats = live_row(g, i, thr)
            last = time.time()
    return np.array(out, float)


def run_job(ds_key, store, method, a, dash, timings, split):
    m = METHODS[method]
    thr = m["threshold"]
    ids = [s[0] for s in store.samples]
    sess = [s[1].split("/")[0] for s in store.samples]
    ts = np.array([E.capture_time(s[1]) for s in store.samples])
    protocol, trusted = DATASETS[ds_key][3], DATASETS[ds_key][4]
    rng = random.Random(a.seed)
    score_raw, cache = make_score(store, method), {}

    def score(x, y):
        if (x, y) not in cache:
            t = time.perf_counter()
            cache[x, y] = score_raw(x, y)
            timings[f"match_{method}"].append((time.perf_counter() - t) * 1000)
        return cache[x, y]

    # 1. templates for every usable photo
    dash.set_stage(f"{ds_key} · {method} · templates", len(store.usable))
    usable, own_fte = [], 0
    for i in store.usable:
        if store.ensure(i, method, timings):
            usable.append(i)
        else:
            own_fte += 1
        dash.step(unit=" img/s")
    if own_fte:
        dash.event(f"[yellow]{ds_key} · {method}: its own quality gate rejected {own_fte} more photos[/]")
    fte = (sum(store.fte.values()) + own_fte) / max(len(store.samples), 1)
    store_usable = usable
    usable = set(usable)
    dev_idx = [i for i in store_usable if ids[i] in split["dev"]]
    test_idx = [i for i in store_usable if ids[i] in split["test"]]
    max_imp = a.max_impostor if method != "v3cr" else int(a.max_impostor * PROFILES[a.profile]["v3cr_impostors"])
    job = f"{ds_key} · {method}"
    dash.live_job = job

    # 2. DEV: pick thresholds (only when the dataset is large enough to split)
    dev = None
    if dev_idx and split["dev"]:
        g_pairs, i_pairs = pair_sets(dev_idx, ids, sess, ts, protocol, trusted)
        g_pairs = rng.sample(g_pairs, min(len(g_pairs), a.max_genuine // PROFILES[a.profile]["dev_divisor"]))
        i_pairs = rng.sample(i_pairs, min(len(i_pairs), max_imp // PROFILES[a.profile]["dev_divisor"]))
        if g_pairs and i_pairs:
            dg = score_pairs(g_pairs, score, dash, f"{job} · DEV genuine", thr)
            di = score_pairs(i_pairs, score, dash, f"{job} · DEV impostor", thr, other=list(dg), kind="imp")
            dev = dict(eer_threshold=eer(dg, di)[1], far_threshold=thr_at_fmr(dg, di, 1e-3),
                       genuine=len(dg), impostor=len(di), EER=eer(dg, di)[0])
            dash.event(f"{job}: DEV thresholds  EER-point {dev['eer_threshold']:.1f}  FAR0.1%-point {dev['far_threshold']:.1f}")

    # 3. TEST, 1 enrolled photo
    g_pairs, i_pairs = pair_sets(test_idx, ids, sess, ts, protocol, trusted)
    g_pairs = rng.sample(g_pairs, min(len(g_pairs), a.max_genuine))
    i_pairs = rng.sample(i_pairs, min(len(i_pairs), max_imp))
    if not g_pairs or not i_pairs:
        dash.event(f"[red]{job}: not enough usable photos for genuine and impostor comparisons[/]")
        return None
    tg = score_pairs(g_pairs, score, dash, f"{job} · TEST genuine", thr)
    ti = score_pairs(i_pairs, score, dash, f"{job} · TEST impostor", thr, other=list(tg), kind="imp")
    res = dict(dataset=ds_key, method=method, enrol=1, protocol=protocol + (" (trusted-label pairs)" if trusted else ""),
               images=len(store.samples), usable=len(usable), identities=len(set(ids)),
               test_identities=len(split["test"]), dev_identities=len(split["dev"]), detect=store.detect,
               fte_rate=fte, fte_reasons=dict(store.fte), **full_metrics(tg, ti, thr, fte))
    res["at_dev_eer"] = at_threshold(tg, ti, dev["eer_threshold"], fte) if dev else None
    res["at_dev_far"] = at_threshold(tg, ti, dev["far_threshold"], fte) if dev else None
    res["dev"] = dev
    scores = {"genuine": tg, "impostor": ti}

    # 4. Rank-1 identification on TEST (closed set; not for PalmMatchDB, whose labels are unreliable)
    res["rank1"], res["rank1_probes"] = None, 0
    if a.rank1_probes and not trusted and method != "v3cr":
        gallery = {}
        for i in test_idx:
            if (protocol != "cross-session" or sess[i] == "session1") and ids[i] not in gallery:
                gallery[ids[i]] = i
        probes = [i for i in test_idx if i not in gallery.values() and ids[i] in gallery and
                  (protocol != "cross-session" or sess[i] == "session2")]
        probes = rng.sample(probes, min(len(probes), a.rank1_probes))
        if probes and len(gallery) > 1:
            dash.set_stage(f"{job} · TEST rank-1 ({len(gallery)} in gallery)", len(probes) * len(gallery))
            hits = 0
            for p in probes:
                sc = {}
                for k, gi in gallery.items():
                    sc[k] = score(gi, p)
                    dash.step(unit=" cmp/s")
                hits += sc[ids[p]] > max(v for k, v in sc.items() if k != ids[p])
            res["rank1"], res["rank1_probes"] = hits / len(probes), len(probes)

    out = [res]
    # 5. TEST, N enrolled photos (score = best over the enrolled photos)
    if a.enrol > 1:
        enrol, eg, ei = enrol_sets(test_idx, ids, sess, ts, protocol, trusted, a.enrol)
        ei = rng.sample(ei, min(len(ei), max_imp))
        if eg and ei:
            best = lambda k, p: max(score(e, p) for e in enrol[k])
            dash.set_stage(f"{job} · TEST {a.enrol}-photo enrolment genuine", len(eg))
            g2 = []
            for k, p in eg:
                g2.append(best(k, p)); dash.step(unit=" cmp/s")
            dash.set_stage(f"{job} · TEST {a.enrol}-photo enrolment impostor", len(ei))
            i2, last = [], 0
            for k, p in sorted(ei):
                i2.append(best(k, p)); dash.step(unit=" cmp/s")
                if time.time() - last > 0.5:
                    dash.live_stats = live_row(g2, i2, thr); last = time.time()
            g2, i2 = np.array(g2), np.array(i2)
            r2 = dict(res, enrol=a.enrol, enrolled_palms=len(enrol), rank1=None, rank1_probes=0,
                      **full_metrics(g2, i2, thr, fte))
            r2["at_dev_eer"] = at_threshold(g2, i2, dev["eer_threshold"], fte) if dev else None
            r2["at_dev_far"] = at_threshold(g2, i2, dev["far_threshold"], fte) if dev else None
            out.append(r2)
            scores.update(genuine_enrol=g2, impostor_enrol=i2)
    return out, scores


# ============================================================================ robustness and latency
def robustness(a, dash, methods):
    """Clean enrolled photo vs degraded probe photo (simulated lighting, blur, noise, pose)."""
    import robustness_test as RT
    conds = dict(RT.CONDITIONS)
    conds.update({
        "overexposed (x2.0, clipped)": lambda g: np.clip(g.astype(np.float32) * 2.0, 0, 255).astype(np.uint8),
        "gamma 0.5 (washed out)": lambda g: (255 * (g / 255.0) ** 0.5).astype(np.uint8),
        "gamma 2.0 (dim)": lambda g: (255 * (g / 255.0) ** 2.0).astype(np.uint8),
        "hard shadow (half palm x0.4)": lambda g: np.concatenate([g[:, : g.shape[1] // 2],
                                                                  (g[:, g.shape[1] // 2:] * 0.4).astype(np.uint8)], 1),
    })
    if a.conditions:
        conds = {k: conds[k] for k in a.conditions}
    root = find_dir("Tongji")
    if not (root and find_dir("Tongji", "session1")):
        dash.event("[yellow]robustness skipped: needs the Tongji dataset[/]")
        return None
    test_ids = set(global_split("tongji", a.seed)[1])     # TEST identities only: the gate limits were set on DEV
    samples, _, _ = E.ds_tongji(root=root, ids=SIZES["tongji"][0], per_id=1)
    keep = sorted({x[0] for x in samples} & test_ids)[: a.robust_ids]
    samples = [x for x in samples if x[0] in set(keep)]
    gray = {}
    for ident, name, path in samples:
        gray[ident, name.split("/")[0]] = cv2.resize(cv2.cvtColor(E.load_any(path), cv2.COLOR_BGR2GRAY),
                                                     (pm.BIG, pm.BIG), interpolation=cv2.INTER_CUBIC)
    ids = sorted({k[0] for k in gray})
    imp_pairs = random.Random(a.seed).sample([(x, y) for x in ids for y in ids if x != y], min(a.robust_impostors, len(ids) * (len(ids) - 1)))
    v4_match = lambda e, p: dict(score=eng.score(e, p))
    fns = {"v4": (eng.make_template, eng.make_template, v4_match),
           "v3": (pm.features_v3, pm.features_v3, pm.match_v3), "v3c": (pmc.features, pmc.features, pmc.match),
           "v3cr": (pmr.template_of_crop, pmr.probe_of_crop, pmr.match)}
    rows = []
    for method in [m for m in methods if m in fns]:
        feat, pfeat, match = fns[method]
        thr = METHODS[method]["threshold"]
        enrol = {i: feat(gray[i, "session1"]) for i in ids}
        dash.live_job = f"robustness · {method}"
        for cname, fn in conds.items():
            dash.set_stage(f"robustness · {method} · {cname}", len(ids) + len(imp_pairs))
            degraded = {i: fn(gray[i, "session2"]) for i in ids}
            gated = {i: method == "v4" and bool(pq.check(pq.crop_measures(degraded[i]))) for i in ids}   # retake
            probe = {i: pfeat(degraded[i]) for i in ids if not gated[i]}
            gen = []
            for i in ids:
                gen.append(0.0 if gated[i] else match(enrol[i], probe[i])["score"]); dash.step(unit=" cmp/s")
            imp, last = [], 0
            for x, y in imp_pairs:
                imp.append(0.0 if gated[y] else match(enrol[x], probe[y])["score"]); dash.step(unit=" cmp/s")
                if time.time() - last > 0.5:
                    dash.live_stats = live_row(gen, imp, thr); last = time.time()
            gen, imp = np.array(gen), np.array(imp)
            far, frr = rates(gen, imp, thr)
            rows.append(dict(method=method, condition=cname, FRR=frr, FAR=far, EER=eer(gen, imp)[0],
                             gate_rejects=float(np.mean([gated[i] for i in ids])),
                             genuine_median=float(np.median(gen)), impostor_max=float(imp.max()),
                             genuine=len(gen), impostor=len(imp)))
            dash.event(f"robustness {method} {cname}: FRR {pct(frr)} FAR {pct(far)}")
    return rows


def latency_pass(n=20):
    """Per-stage latency on whole-hand photos (own photos + PalmMatchDB if present), current device."""
    files = [s[2] for s in E.ds_real(None)[0]]
    root = pmdb_root()
    if root:
        import glob
        files += sorted(f for f in glob.glob(os.path.join(root, "**", "*.jp*g"), recursive=True))[::97][:n]
    files = files[:n]
    t = collections.defaultdict(list)
    crops = []
    pm.detect_landmarks(E.load_any(files[0]))                     # warm up: the first call loads the model
    for f in files:
        bgr = E.load_any(f)
        s = time.perf_counter()
        try:
            pts = pm.detect_landmarks(bgr)
            t1 = time.perf_counter()
            c = pm.align_palm_v2(bgr, pts, normalise=False)
            t2 = time.perf_counter()
        except pm.QualityError:
            continue
        t["detect"].append((t1 - s) * 1000); t["align"].append((t2 - t1) * 1000); crops.append(c)
    pm.features_v3(crops[0]); pmc.features(crops[0])               # warm up
    feats = {}
    for name, fx in (("v4", eng.make_template), ("v3", pm.features_v3), ("v3c", pmc.features), ("v3cr", pmr.template_of_crop)):
        feats[name] = []
        for c in crops:
            s = time.perf_counter(); feats[name].append(fx(c)); t[f"extract_{name}"].append((time.perf_counter() - s) * 1000)
    for name, mf in (("v4", eng.score), ("v3", pm.match_v3), ("v3c", pmc.match), ("v3cr", pmc.match)):
        f = feats[name]                                     # v3cr: first attempt; retries only run after a failure
        for k in range(1, len(f)):
            s = time.perf_counter(); mf(f[0], f[k]); t[f"match_{name}"].append((time.perf_counter() - s) * 1000)
    return {k: dict(median=float(np.median(v)), p95=float(np.percentile(v, 95)), n=len(v)) for k, v in t.items()}


def device_name():
    import torch
    if torch.cuda.is_available():
        return f"GPU {torch.cuda.get_device_name(0)}"
    return f"CPU ({os.cpu_count()} threads)"


_SPLITS = {}


def global_split(k, seed):
    """DEV and TEST identities of a dataset, fixed once from the full-size sample (seeded shuffle, first 20% = DEV,
    only for datasets with >= 40 identities). Every tuning script (tune_dev.py, gate_dev.py, embed_eval.py,
    pad_eval.py, speed_dev.py) uses only these DEV identities, and every profile reports only TEST identities, so no
    reported number comes from an identity that a design choice has seen."""
    if (k, seed) not in _SPLITS:
        full = DATASETS[k][2](ids=SIZES[k][0], per_id=SIZES[k][1], seed=seed) or []
        ids = sorted({s[0] for s in full})
        random.Random(seed).shuffle(ids)
        n_dev = len(ids) // 5 if len(ids) >= 40 else 0
        _SPLITS[k, seed] = (ids[:n_dev], ids[n_dev:])
    return _SPLITS[k, seed]


def plan_samples(k, sizes, seed):
    """Photos of one dataset for a profile. A smaller profile keeps the DEV/TEST split of the full one: it takes
    20% of its identities from the DEV list and the rest from the TEST list."""
    ids, per = sizes[k]
    if sizes is SIZES:
        return DATASETS[k][2](ids=ids, per_id=per, seed=seed)
    dev, test = global_split(k, seed)
    if not dev:
        return DATASETS[k][2](ids=ids, per_id=per, seed=seed)
    full = DATASETS[k][2](ids=SIZES[k][0], per_id=per, seed=seed) or []
    keep = set(dev[: ids // 5]) | set(test[: ids - ids // 5])
    return [x for x in full if x[0] in keep]


def prepare_child(k, profile, seed):
    """Child process: hand detection for one dataset. Prints S (one photo done) / E <event> lines for the parent's
    dashboard. Runs apart from the matcher because MediaPipe's threads slow PyTorch down by ~1.7x on CPU."""
    samples = plan_samples(k, PROFILES[profile]["sizes"], seed)
    store = Store(k, samples, True)
    pm._landmarker()
    store.prepare(lambda: print("S", flush=True), lambda m: print("E", m, flush=True), collections.defaultdict(list))


def prepare_detected(k, store, profile, seed, dash):
    """Parent side: run the child, mirror its progress, then read the cached crops."""
    todo = [i for i, s in enumerate(store.samples) if not os.path.exists(store._path("crop", i))]
    if todo:
        dash.set_stage(f"{k} · detect + align (MediaPipe, separate process)", len(store.samples))
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--prepare", k, "--profile", profile,
                              "--seed", str(seed)], cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            if line.startswith("S"):
                dash.step(unit=" img/s")
            elif line.startswith("E "):
                dash.event(line[2:].rstrip())
        p.wait()
    dash.set_stage(f"{k} · load cached crops", len(store.samples))
    store.prepare(lambda: dash.step(unit=" img/s"), dash.event, collections.defaultdict(list))


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=None, help=f"default: all on disk ({', '.join(DATASETS)})")
    ap.add_argument("--profile", choices=list(PROFILES), default=None,
                    help="full (GPU-sized) or quick (laptop CPU). Default: full on a GPU, quick on CPU")
    ap.add_argument("--quick", action="store_true", help="same as --profile quick")
    ap.add_argument("--methods", nargs="+", default=None, choices=list(METHODS), help="default: from the profile")
    ap.add_argument("--max-genuine", type=int, default=None)
    ap.add_argument("--max-impostor", type=int, default=None, help="v3cr uses half: it retries every failed comparison")
    ap.add_argument("--enrol", type=int, default=3, help="also report N-photo enrolment")
    ap.add_argument("--rank1-probes", type=int, default=None, help="0 = skip rank-1")
    ap.add_argument("--robust-ids", type=int, default=None)
    ap.add_argument("--robust-impostors", type=int, default=None)
    ap.add_argument("--no-robustness", action="store_true")
    ap.add_argument("--no-cpu-latency", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="run folder to continue (finished jobs are skipped)")
    ap.add_argument("--latency-only", default=None, help=argparse.SUPPRESS)   # internal: CPU latency subprocess
    ap.add_argument("--prepare", default=None, help=argparse.SUPPRESS)        # internal: hand-detection subprocess
    a = ap.parse_args()

    if a.latency_only:
        json.dump(dict(device=device_name(), stages=latency_pass()), open(a.latency_only, "w"), indent=2)
        return
    if a.prepare:
        prepare_child(a.prepare, a.profile or "full", a.seed)
        return

    import torch
    a.profile = "quick" if a.quick else (a.profile or ("full" if torch.cuda.is_available() else "quick"))
    prof = PROFILES[a.profile]
    for k in ("methods", "max_genuine", "max_impostor", "rank1_probes", "robust_ids", "robust_impostors"):
        if getattr(a, k) is None:
            setattr(a, k, prof[k])
    a.conditions = prof["conditions"]
    run_dir = a.resume or os.path.join(REPO, "benchmarks", "results", "runs",
                                       datetime.datetime.now().strftime("%Y%m%d-%H%M") + f"-{a.profile}")
    os.makedirs(os.path.join(run_dir, "jobs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "scores"), exist_ok=True)
    logging.basicConfig(filename=os.path.join(run_dir, "run.log"), level=logging.INFO, encoding="utf-8",
                        format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    log.info("command: %s", " ".join(sys.argv))

    con.print("[bold]loading models[/] (DISK, LightGlue; hand detection runs in a separate process) ...")
    pm.features_v3(np.zeros((pm.BIG, pm.BIG), np.uint8))
    dev_name = device_name()
    log.info("device: %s  profile: %s", dev_name, a.profile)
    keys = a.datasets or list(DATASETS)
    plan = []
    for k in keys:
        if k not in DATASETS:
            con.print(f"[red]unknown dataset {k}[/]"); continue
        samples = plan_samples(k, prof["sizes"], a.seed)
        if not samples:
            con.print(f"[dim]- {k}: not on disk (python scripts/fetch_datasets.py {k})[/]")
            continue
        plan.append((k, samples))
    if not plan:
        con.print("[red]no dataset found. Run: python scripts/fetch_datasets.py all[/]"); sys.exit(1)

    overview = Table(title="benchmark plan", title_justify="left")
    for c in ("dataset", "licence use", "images", "identities", "protocol", "hand detection"):
        overview.add_column(c)
    det = {}
    for k, s in plan:
        det[k] = DATASETS[k][5]
        overview.add_row(DATASETS[k][0], DATASETS[k][1], f"{len(s):,}", f"{len({x[0] for x in s}):,}", DATASETS[k][3],
                         "MediaPipe" if det[k] else "no (pre-cropped palms)")
    con.print(overview)
    con.print(f"profile [bold]{a.profile}[/]   methods: {', '.join(a.methods)}   device: [green]{dev_name}[/]   "
              f"comparisons per job: up to {a.max_genuine:,} genuine / {a.max_impostor:,} impostor   "
              f"results: {os.path.relpath(run_dir, REPO)}")
    if a.profile == "full" and "CPU" in dev_name:
        con.print("[yellow]full profile on CPU: expect a very long run (about 1 s per comparison). "
                  "Use --profile quick, or run on the GPU machine.[/]")

    timings = collections.defaultdict(list)
    results, all_rows = {}, []
    dash = Dashboard(run_dir, dev_name)
    dash.progress.update(dash.overall, total=len(plan) * len(a.methods))
    with Live(dash, console=con, refresh_per_second=6, transient=False) as live:
        for k, samples in plan:
            ids_all = {s[0] for s in samples}
            g_dev = set(global_split(k, a.seed)[0])         # fixed DEV identities (see global_split)
            split = dict(dev=ids_all & g_dev, test=ids_all - g_dev)
            store = Store(k, samples, det[k])
            dash.live_stats = None
            if det[k]:
                prepare_detected(k, store, a.profile, a.seed, dash)
            else:
                dash.set_stage(f"{k} · load", len(samples))
                store.prepare(lambda: dash.step(unit=" img/s"), dash.event, timings)
            dash.event(f"{k}: {len(store.usable)}/{len(samples)} photos usable, FTE {pct(sum(store.fte.values()) / len(samples))}"
                       f" {dict(store.fte) if store.fte else ''}; split {len(split['dev'])} DEV / {len(split['test'])} TEST identities")
            for method in a.methods:
                jpath = os.path.join(run_dir, "jobs", f"{k}__{method}.json")
                if os.path.exists(jpath):
                    rows = json.load(open(jpath))
                    dash.event(f"{k} · {method}: already done (resume)")
                else:
                    t0 = time.time()
                    out = run_job(k, store, method, a, dash, timings, split)
                    if out is None:
                        dash.progress.advance(dash.overall); continue
                    rows, scores = out
                    for r in rows:
                        r["seconds"] = round(time.time() - t0, 1)
                    np.savez_compressed(os.path.join(run_dir, "scores", f"{k}__{method}.npz"), **scores)
                    json.dump(rows, open(jpath, "w"), indent=2, default=float)
                all_rows += rows
                r = rows[0]
                r3 = rows[1] if len(rows) > 1 else None
                live.console.print(
                    f"[green]✓[/] [bold]{DATASETS[k][0]}[/] · {METHODS[method]['label']}: "
                    f"EER [bold]{pct(r['EER'])}[/]  FAR {pct(r['FAR'])}  FRR {pct(r['FRR'])}  "
                    f"acc {pct(r['accuracy'])}  TAR@FAR0.1% {pct(r['TAR_at_FAR_1e3'])}  rank-1 {pct(r['rank1'])}"
                    + (f"  | {a.enrol}-photo EER [bold]{pct(r3['EER'])}[/] FRR {pct(r3['FRR'])}" if r3 else "")
                    + f"  [dim]({r['genuine_comparisons']:,} gen / {r['impostor_comparisons']:,} imp)[/]")
                log.info("RESULT %s", json.dumps({x: r[x] for x in ("dataset", "method", "EER", "FAR", "FRR", "accuracy")}, default=float))
                dash.progress.advance(dash.overall)
            # template size per method (one enrolled photo)
            results.setdefault("template_bytes", {})
            for method in a.methods:
                if method in results["template_bytes"]:
                    continue
                for i in store.usable:
                    if os.path.exists(store._path(method, i)):
                        results["template_bytes"][method] = template_bytes(store, method, i)
                        break
            del store

        rob = None
        rpath = os.path.join(run_dir, "robustness.json")
        if not a.no_robustness:
            if os.path.exists(rpath):
                rob = json.load(open(rpath))
            else:
                rob = robustness(a, dash, a.methods)
                if rob:
                    json.dump(rob, open(rpath, "w"), indent=2)

        dash.set_stage("latency on this device (whole-hand photos)", 1)
        lat = dict(this_device=dict(device=dev_name, stages=latency_pass()))
        dash.step()
        if not a.no_cpu_latency and "GPU" in dev_name:
            dash.set_stage("latency on CPU (separate process, GPU hidden)", 1)
            cpu_file = os.path.join(run_dir, "latency_cpu.json")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
            subprocess.run([sys.executable, os.path.abspath(__file__), "--latency-only", cpu_file], env=env,
                           cwd=REPO, capture_output=True)
            if os.path.exists(cpu_file):
                lat["cpu"] = json.load(open(cpu_file))
            dash.step()

    results.update(rows=all_rows, robustness=rob, latency=lat, run_timings={
        k: dict(median=float(np.median(v)), p95=float(np.percentile(v, 95)), n=len(v)) for k, v in timings.items() if v},
        config=vars(a), device=dev_name, finished=datetime.datetime.now().isoformat(timespec="seconds"),
        datasets={k: dict(name=DATASETS[k][0], use=DATASETS[k][1], protocol=DATASETS[k][3]) for k, _ in plan})
    json.dump(results, open(os.path.join(run_dir, "results.json"), "w"), indent=2, default=float)
    log.info("results.json written")

    import bench_report
    bench_report.write_all(run_dir, con)


if __name__ == "__main__":
    main()
