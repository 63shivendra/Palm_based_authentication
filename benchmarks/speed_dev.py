#!/usr/bin/env python3
"""
benchmarks/speed_dev.py - make the matcher faster without losing accuracy, measured on DEV identities only.

Variants of the v3c matcher (all pure PyTorch, no new binaries):
  base        DISK on the 512 px crop, 1,024 keypoints, LightGlue float32 (= v3c today; LightGlue's early stopping and
              point pruning are already on)
  kp512       keep the 512 strongest keypoints (LightGlue attention cost grows with the square of the count)
  px384       run DISK on the crop resized to 384 px (44% fewer pixels), keypoints scaled back to 512 px
  px384_kp512 both
  int8        LightGlue's linear layers quantised to int8 (torch dynamic quantisation), base features
  px384_kp512_int8

Reported per variant: median extraction and comparison time on this machine (run it with nothing else busy), EER,
FNMR at FMR 1%, the highest impostor score (to place the threshold) and the genuine median.

  python benchmarks/speed_dev.py [--datasets tongji bjtu2 11khands] [--gen 100] [--imp 300]
Output: benchmarks/results/speed/<YYYYmmdd-HHMM>/ speed_dev.md, speed_dev.csv
"""
import argparse
import csv
import datetime
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_benchmark as RB  # noqa: E402
import tune_dev as TD  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import palm_match as pm  # noqa: E402

con = RB.con
VARIANTS = {
    "base": dict(px=512, kp=1024, int8=False),
    "kp512": dict(px=512, kp=512, int8=False),
    "px384": dict(px=384, kp=1024, int8=False),
    "px384_kp512": dict(px=384, kp=512, int8=False),
    "int8": dict(px=512, kp=1024, int8=True),
    "px384_kp512_int8": dict(px=384, kp=512, int8=True),
}
_Q = {}


def features(crop, px, kp):
    m = pm._v3_models()
    torch = m["torch"]
    g = crop if px == pm.BIG else cv2.resize(crop, (px, px), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(np.ascontiguousarray(g)).float()[None, None].repeat(1, 3, 1, 1).to(m["dev"]) / 255
    with torch.no_grad():
        f = m["disk"](x, kp, pad_if_not_divisible=True)[0]
    k = f.keypoints.cpu().numpy().astype(np.float32) * (pm.BIG / px)
    d = f.descriptors.cpu().numpy()
    d = np.clip(np.round((d + 1) * 127.5), 0, 255).astype(np.uint8).astype(np.float32) / 127.5 - 1   # v3c storage
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    return k, d


def matcher(int8):
    """The LightGlue matcher used by palm_match.match_v3, optionally with int8 linear layers."""
    m = pm._v3_models()
    if not int8:
        return m["lg"]
    if "lg8" not in _Q:
        import copy
        torch = m["torch"]
        lg = copy.deepcopy(m["lg"])
        q = lambda mod: torch.ao.quantization.quantize_dynamic(mod, {torch.nn.Linear}, dtype=torch.qint8)
        lg.matcher.transformers = q(lg.matcher.transformers)        # the attention layers: nearly all the work
        lg.matcher.log_assignment = q(lg.matcher.log_assignment)    # posenc stays float (kornia reads its dtype)
        _Q["lg8"] = lg
    return _Q["lg8"]


def match(a, b, int8):
    m = pm._v3_models()
    saved = m["lg"]
    m["lg"] = matcher(int8)
    try:
        return pm.match_v3(a, b)["score"]
    finally:
        m["lg"] = saved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["tongji", "bjtu2", "11khands"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--ids", type=int, default=40)
    ap.add_argument("--gen", type=int, default=100)
    ap.add_argument("--imp", type=int, default=300)
    a = ap.parse_args()
    pm.features_v3(np.zeros((pm.BIG, pm.BIG), np.uint8))
    rows, timing = [], {v: dict(extract=[], match=[]) for v in a.variants}
    for k in a.datasets:
        samples, dev = TD.dev_split(k, a.ids)
        if samples is None:
            continue
        store = RB.Store(k, samples, RB.DATASETS[k][5])
        T = TD.Tuner(k, samples, dev, None)
        gp, ip = T.pairs(a.gen, a.imp)
        need = sorted({i for p in gp + ip for i in p})
        con.print(f"[bold]{k}[/]: {len(gp)} genuine / {len(ip)} impostor DEV pairs, {len(need)} photos")
        for v in a.variants:
            cfg = VARIANTS[v]
            feats = {}
            for n, i in enumerate(need):
                crop = store.get_crop(i)
                t = time.perf_counter()
                feats[i] = features(crop, cfg["px"], cfg["kp"])
                if n < 40:
                    timing[v]["extract"].append((time.perf_counter() - t) * 1000)
            match(feats[need[0]], feats[need[1]], cfg["int8"])          # warm up
            sc = {}
            for n, (x, y) in enumerate(gp + ip):
                t = time.perf_counter()
                sc[x, y] = match(feats[x], feats[y], cfg["int8"])
                if n % 4 == 0:
                    timing[v]["match"].append((time.perf_counter() - t) * 1000)
            g = np.array([sc[p] for p in gp])
            i = np.array([sc[p] for p in ip])
            r = TD.summarise(g, i, 12.0)
            rows.append(dict(dataset=k, variant=v, **r))
            con.print(f"  {k:9s} {v:17s} EER {TD.pct(r['EER'])}  FNMR@FMR1% {TD.pct(r['FNMR_at_FMR1pct'])}  impostor max "
                      f"{r['impostor_max']:.1f}  genuine median {r['genuine_median']:.0f}  "
                      f"extract {np.median(timing[v]['extract']):.0f} ms  match {np.median(timing[v]['match']):.0f} ms")
    out = os.path.join(RB.REPO, "benchmarks", "results", "speed", datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    os.makedirs(out, exist_ok=True)
    for r in rows:
        r["extract_ms"] = float(np.median(timing[r["variant"]]["extract"]))
        r["match_ms"] = float(np.median(timing[r["variant"]]["match"]))
    with open(os.path.join(out, "speed_dev.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import torch
    md = ["# Speed variants (DEV identities only)", "",
          f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} by `benchmarks/speed_dev.py` on {RB.device_name()} "
          f"({torch.get_num_threads()} PyTorch threads). Times are medians on this machine.", "",
          "| Variant | DISK input | Keypoints | LightGlue | Extract (ms) | Compare (ms) | Dataset | EER | FNMR at FMR 1% | Impostor max | Genuine median |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c = VARIANTS[r["variant"]]
        md.append(f"| {r['variant']} | {c['px']} px | {c['kp']} | {'int8' if c['int8'] else 'float32'} | {r['extract_ms']:.0f} | "
                  f"{r['match_ms']:.0f} | {r['dataset']} | {TD.pct(r['EER'])} | {TD.pct(r['FNMR_at_FMR1pct'])} | "
                  f"{r['impostor_max']:.1f} | {r['genuine_median']:.0f} |")
    open(os.path.join(out, "speed_dev.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    con.print(f"[green]written[/] {os.path.relpath(out, RB.REPO)}/speed_dev.md")


if __name__ == "__main__":
    main()
