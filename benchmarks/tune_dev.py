#!/usr/bin/env python3
"""
benchmarks/tune_dev.py - choose matcher changes using DEV identities only (never TEST identities).

The identity split is the one run_benchmark.py uses for its full profile (seeded shuffle, first 20% = DEV), so
nothing chosen here has seen a TEST identity. Compared on DEV:

  preprocessing   what the aligned crop looks like before DISK: raw (v3c today), CLAHE, local contrast normalisation
  fusion          v3c-style score + v2 (SIFT) inlier score, each divided by its shipped threshold, then added
  quality gate    palm_quality.check() on every DEV photo: how many photos it rejects and how many genuine failures
                  it removes
  lighting        clean enrolled photo vs a degraded probe (over-exposure, brighter, hard shadow, dim, noise),
                  Tongji DEV identities

  python benchmarks/tune_dev.py                                  # all of the above (CPU: about 1-2 hours)
  python benchmarks/tune_dev.py --variants raw clahe --datasets bjtu2

Output: benchmarks/results/tuning/<YYYYmmdd-HHMM>/ dev_tuning.md, dev_tuning.csv, dev_quality.csv, dev_lighting.csv
Templates and pair scores are cached under data/_cache/<dataset>/tune_*, so a second run only re-does the analysis.
"""
import argparse
import collections
import csv
import datetime
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_benchmark as RB  # noqa: E402  (puts the repo and .bench_env on sys.path)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn  # noqa: E402
from rich.table import Table  # noqa: E402

import palm_match as pm  # noqa: E402
import palm_match_compact as pmc  # noqa: E402
import palm_quality as pq  # noqa: E402

con = RB.con
V3C_THR, V2_THR = float(pmc.THRESHOLD), float(pm.MIN_INLIERS)


def clahe(g):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def localnorm(g, sigma=16.0):
    x = g.astype(np.float32)
    m = cv2.GaussianBlur(x, (0, 0), sigma)
    s = np.sqrt(cv2.GaussianBlur((x - m) ** 2, (0, 0), sigma))
    return np.clip(128 + 48 * (x - m) / (s + 4.0), 0, 255).astype(np.uint8)


VARIANTS = {"raw": lambda g: g, "clahe": clahe, "localnorm": localnorm}

CONDITIONS = {
    "clean": lambda g: g,
    "brighter (x1.5)": lambda g: np.clip(g.astype(np.float32) * 1.5, 0, 255).astype(np.uint8),
    "overexposed (x2.0, clipped)": lambda g: np.clip(g.astype(np.float32) * 2.0, 0, 255).astype(np.uint8),
    "hard shadow (half palm x0.4)": lambda g: np.concatenate([g[:, : g.shape[1] // 2],
                                                              (g[:, g.shape[1] // 2:] * 0.4).astype(np.uint8)], 1),
    "gamma 2.0 (dim)": lambda g: (255 * (g / 255.0) ** 2.0).astype(np.uint8),
    "sensor noise (sigma 12)": lambda g: np.clip(g + np.random.RandomState(0).normal(0, 12, g.shape), 0, 255).astype(np.uint8),
}


def dev_split(k, ids_cap, seed=0):
    samples = RB.plan_samples(k, RB.SIZES, seed)
    ids_all = sorted({s[0] for s in samples})
    sh = ids_all[:]
    random.Random(seed).shuffle(sh)
    if len(sh) < 40:
        return None, None
    dev = sh[: len(sh) // 5][:ids_cap]
    return samples, set(dev)


class Tuner:
    def __init__(self, k, samples, dev, progress):
        self.k, self.samples, self.dev = k, samples, dev
        self.detect = RB.DATASETS[k][5]
        self.store = RB.Store(k, samples, self.detect)
        self.progress = progress
        self.feat = {}
        self.score_file = os.path.join(RB.CACHE, k, "tune_scores.json")
        self.scores = json.load(open(self.score_file)) if os.path.exists(self.score_file) else {}

    def save(self):
        os.makedirs(os.path.dirname(self.score_file), exist_ok=True)
        json.dump(self.scores, open(self.score_file, "w"))

    def usable(self):
        out = []
        for i, s in enumerate(self.samples):
            if s[0] not in self.dev:
                continue
            if self.detect and not os.path.exists(self.store._path("crop", i)):
                continue
            out.append(i)
        return out

    def features(self, i, variant):
        key = (i, variant)
        if key in self.feat:
            return self.feat[key]
        p = self.store._path(f"tune_{variant}", i)
        if os.path.exists(p):
            with np.load(p) as z:
                f = (z["kp"], z["des"])
        else:
            crop = self.store.get_crop(i)
            f = pm.features_v2(pm.illum_normalise(crop)) if variant == "v2" else pmc.features(VARIANTS[variant](crop))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            np.savez(p, kp=f[0], des=f[1])
        f = f if variant == "v2" else pmc.unpack(f)
        if len(self.feat) > 900:
            self.feat.pop(next(iter(self.feat)))
        self.feat[key] = f
        return f

    def score(self, a, b, variant):
        key = f"{variant}|{a}|{b}"
        if key not in self.scores:
            fa, fb = self.features(a, variant), self.features(b, variant)
            if variant == "v2":
                self.scores[key] = float(min(pm._inliers(fa, fb), pm._inliers(fb, fa)))
            else:
                self.scores[key] = float(pm.match_v3(fa, fb)["score"])
        return self.scores[key]

    def pairs(self, n_gen, n_imp, seed=0):
        idx = self.usable()
        ids = [s[0] for s in self.samples]
        sess = [s[1].split("/")[0] for s in self.samples]
        ts = np.array([RB.E.capture_time(s[1]) for s in self.samples])
        g, i = RB.pair_sets(idx, ids, sess, ts, RB.DATASETS[self.k][3], RB.DATASETS[self.k][4])
        rng = random.Random(seed)
        return sorted(rng.sample(g, min(n_gen, len(g)))), sorted(rng.sample(i, min(n_imp, len(i))))

    def run_pairs(self, pairs, variant, desc):
        t = self.progress.add_task(desc, total=len(pairs))
        out = []
        for n, (a, b) in enumerate(pairs):
            out.append(self.score(a, b, variant))
            self.progress.advance(t)
            if n % 50 == 49:
                self.save()
        self.save()
        self.progress.remove_task(t)
        return np.array(out)


def summarise(gen, imp, thr):
    e, et = RB.eer(gen, imp)
    fmr, fnmr = RB.rates(gen, imp, thr)
    ts, fm, fn = RB.curve(gen, imp)
    ok = fm <= 0.01
    return dict(EER=e, FNMR_shipped=fnmr, FMR_shipped=fmr, FNMR_at_FMR1pct=float(fn[ok].min()) if ok.any() else 1.0,
                genuine_median=float(np.median(gen)), genuine_p05=float(np.percentile(gen, 5)),
                impostor_max=float(imp.max()), impostor_p99=float(np.percentile(imp, 99)),
                genuine=len(gen), impostor=len(imp))


def pct(v):
    return f"{v * 100:.2f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["bjtu2", "tongji", "11khands"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--ids", type=int, default=40, help="DEV identities used per dataset (cap)")
    ap.add_argument("--gen", type=int, default=150)
    ap.add_argument("--imp", type=int, default=450)
    ap.add_argument("--lighting-ids", type=int, default=40)
    ap.add_argument("--no-lighting", action="store_true")
    a = ap.parse_args()

    out_dir = os.path.join(RB.REPO, "benchmarks", "results", "tuning", datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    os.makedirs(out_dir, exist_ok=True)
    con.print(f"[bold]DEV-only tuning[/]  datasets {a.datasets}  variants {a.variants} (+ fusion with v2)  -> "
              f"{os.path.relpath(out_dir, RB.REPO)}")
    pm.features_v3(np.zeros((pm.BIG, pm.BIG), np.uint8))       # load the models before the progress bars start
    rows, qrows, lrows = [], [], []
    prog = Progress(TextColumn("[bold]{task.description}"), BarColumn(bar_width=36), MofNCompleteColumn(),
                    TimeElapsedColumn(), TextColumn("eta"), TimeRemainingColumn(), console=con)
    with prog:
        for k in a.datasets:
            samples, dev = dev_split(k, a.ids)
            if samples is None:
                con.print(f"[yellow]{k}: fewer than 40 identities, no DEV split - skipped[/]")
                continue
            T = Tuner(k, samples, dev, prog)
            gp, ip = T.pairs(a.gen, a.imp)
            con.print(f"[bold]{k}[/]: {len(dev)} DEV identities, {len(T.usable())} photos, {len(gp)} genuine / {len(ip)} impostor pairs")
            sc = {}
            for v in a.variants + ["v2"]:
                g = T.run_pairs(gp, v, f"{k} · {v} · genuine")
                i = T.run_pairs(ip, v, f"{k} · {v} · impostor")
                sc[v] = (g, i)
                thr = V2_THR if v == "v2" else V3C_THR
                rows.append(dict(dataset=k, variant=v, threshold=thr, **summarise(g, i, thr)))
                con.print(f"  {k} · {v:10s} EER {pct(rows[-1]['EER'])}  FNMR@12 {pct(rows[-1]['FNMR_shipped'])}  "
                          f"FMR@12 {pct(rows[-1]['FMR_shipped'])}  impostor max {rows[-1]['impostor_max']:.1f}")
            for v in a.variants:                            # fusion: threshold-normalised sum, accept at 2.0
                g = sc[v][0] / V3C_THR + sc["v2"][0] / V2_THR
                i = sc[v][1] / V3C_THR + sc["v2"][1] / V2_THR
                rows.append(dict(dataset=k, variant=f"{v}+v2", threshold=2.0, **summarise(g, i, 2.0)))
                con.print(f"  {k} · {v + '+v2':10s} EER {pct(rows[-1]['EER'])}  FNMR@2 {pct(rows[-1]['FNMR_shipped'])}  "
                          f"FMR@2 {pct(rows[-1]['FMR_shipped'])}")

            # quality measures of every DEV photo, with its weakest genuine score (raw variant)
            weakest = collections.defaultdict(lambda: np.inf)
            for (x, y), s in zip(gp, sc["raw"][0] if "raw" in sc else sc[a.variants[0]][0]):
                weakest[x] = min(weakest[x], s)
                weakest[y] = min(weakest[y], s)
            t = prog.add_task(f"{k} · quality measures", total=len(T.usable()))
            for i in T.usable():
                crop = T.store.get_crop(i)
                m = pq.crop_measures(crop)
                if T.detect:
                    pts = np.load(T.store._path("crop", i))["pts"]
                    img = samples[i][2]
                    w = (RB.E.load_any(img) if isinstance(img, str) else img).shape[1]
                    m.update(pq.pose_measures(pts, w))
                issues = [c for c, _ in pq.check(m)]
                qrows.append(dict(dataset=k, photo=samples[i][1], weakest_genuine=weakest.get(i, np.nan),
                                  issues=" ".join(issues), **m))
                prog.advance(t)
            prog.remove_task(t)

        # lighting robustness on Tongji DEV identities
        if not a.no_lighting and "tongji" in a.datasets:
            samples, dev = dev_split("tongji", a.lighting_ids)
            T = Tuner("tongji", samples, dev, prog)
            first = {}
            for i in T.usable():
                first.setdefault((samples[i][0], samples[i][1].split("/")[0]), i)
            ids = sorted({s for s, _ in first if (s, "session1") in first and (s, "session2") in first})
            imp_pairs = random.Random(0).sample([(x, y) for x in ids for y in ids if x != y], min(120, len(ids) * (len(ids) - 1)))
            for v in a.variants:
                enrol = {s: T.features(first[s, "session1"], v) for s in ids}
                for cname, fn in CONDITIONS.items():
                    t = prog.add_task(f"lighting · {v} · {cname}", total=len(ids) + len(imp_pairs))
                    probe, gated = {}, {}
                    for s in ids:
                        g = fn(T.store.get_crop(first[s, "session2"]))
                        probe[s] = pmc.unpack(pmc.features(VARIANTS[v](g)))
                        gated[s] = bool(pq.check(pq.crop_measures(g)))
                    gen = []
                    for s in ids:
                        gen.append(pm.match_v3(enrol[s], probe[s])["score"]); prog.advance(t)
                    imp = []
                    for x, y in imp_pairs:
                        imp.append(pm.match_v3(enrol[x], probe[y])["score"]); prog.advance(t)
                    prog.remove_task(t)
                    gen, imp = np.array(gen), np.array(imp)
                    q = np.array([gated[s] for s in ids])
                    fmr, fnmr = RB.rates(gen, imp, V3C_THR)
                    kept = gen[~q]
                    lrows.append(dict(variant=v, condition=cname, FNMR=fnmr, FMR=fmr, genuine_median=float(np.median(gen)),
                                      impostor_max=float(imp.max()), gate_rejects=float(q.mean()),
                                      FNMR_after_gate=float(np.mean(kept < V3C_THR)) if len(kept) else float("nan"),
                                      genuine=len(gen), impostor=len(imp)))
                    con.print(f"  lighting · {v:10s} {cname:30s} FNMR {pct(fnmr)}  FMR {pct(fmr)}  gate rejects "
                              f"{pct(q.mean())}  FNMR of photos passing the gate {pct(lrows[-1]['FNMR_after_gate']) if len(kept) else 'n/a'}")

    write(out_dir, rows, qrows, lrows, a)


def write(out_dir, rows, qrows, lrows, a):
    def dump(name, rs):
        if not rs:
            return
        with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rs for k in r)))   # whole-hand rows add pose fields
            w.writeheader()
            w.writerows(rs)
    dump("dev_tuning.csv", rows)
    dump("dev_quality.csv", qrows)
    dump("dev_lighting.csv", lrows)

    t = Table(title="DEV identities only: matcher variants", title_justify="left")
    for c in ("dataset", "variant", "EER", "FNMR @ shipped thr", "FMR @ shipped thr", "FNMR @ FMR 1%", "impostor max"):
        t.add_column(c, justify="right")
    md = ["# DEV-only tuning", "", f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} by `benchmarks/tune_dev.py`. "
          "Only DEV identities of the benchmark's identity split are used (TEST identities are never touched). "
          f"Up to {a.ids} DEV identities per dataset, {a.gen} genuine / {a.imp} impostor pairs.", "",
          "## Matcher variants", "", "Shipped threshold: 12 for the v3c-style variants, 4 inliers for v2, 2.0 for fusion "
          "(each score divided by its own threshold, then added).", "",
          "| Dataset | Variant | EER | FNMR at shipped threshold | FMR at shipped threshold | FNMR at FMR 1% | Impostor max | Genuine / impostor |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t.add_row(r["dataset"], r["variant"], pct(r["EER"]), pct(r["FNMR_shipped"]), pct(r["FMR_shipped"]),
                  pct(r["FNMR_at_FMR1pct"]), f"{r['impostor_max']:.1f}")
        md.append(f"| {r['dataset']} | {r['variant']} | {pct(r['EER'])} | {pct(r['FNMR_shipped'])} | {pct(r['FMR_shipped'])} | "
                  f"{pct(r['FNMR_at_FMR1pct'])} | {r['impostor_max']:.1f} | {r['genuine']} / {r['impostor']} |")
    con.print(t)
    if qrows:
        md += ["", "## Quality gate on DEV photos", "",
               "A photo counts as a genuine failure when its weakest genuine comparison (raw variant) is below 12.", "",
               "| Dataset | Photos | Rejected by the gate | Photos with a genuine failure | Of those, rejected by the gate | Rejection reasons |",
               "|---|---|---|---|---|---|"]
        by = collections.defaultdict(list)
        for q in qrows:
            by[q["dataset"]].append(q)
        for k, qs in by.items():
            rej = [q for q in qs if q["issues"]]
            fail = [q for q in qs if q["weakest_genuine"] < V3C_THR]
            reasons = collections.Counter(c for q in rej for c in q["issues"].split())
            md.append(f"| {k} | {len(qs)} | {len(rej)} ({pct(len(rej) / len(qs))}) | {len(fail)} | "
                      f"{sum(1 for q in fail if q['issues'])} | {dict(reasons)} |")
    if lrows:
        md += ["", "## Lighting conditions (Tongji DEV identities)", "",
               "Clean enrolled photo (session 1) vs degraded probe (session 2). Threshold 12.", "",
               "| Variant | Condition | FNMR | FMR | Gate rejects | FNMR of probes passing the gate | Genuine median |",
               "|---|---|---|---|---|---|---|"]
        for r in lrows:
            md.append(f"| {r['variant']} | {r['condition']} | {pct(r['FNMR'])} | {pct(r['FMR'])} | {pct(r['gate_rejects'])} | "
                      f"{pct(r['FNMR_after_gate']) if r['FNMR_after_gate'] == r['FNMR_after_gate'] else 'n/a'} | {r['genuine_median']:.0f} |")
    open(os.path.join(out_dir, "dev_tuning.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    con.print(f"written {os.path.relpath(out_dir, RB.REPO)}/dev_tuning.md")


if __name__ == "__main__":
    main()
