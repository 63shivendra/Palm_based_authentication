#!/usr/bin/env python3
"""
benchmarks/gate_dev.py - set the capture quality gate (palm_quality.py) on DEV identities only.

A gate is useful when the photos it sends back would have failed to match, and harmful when it sends back photos the
matcher would have accepted (the user retakes for nothing). For each quality measure and candidate limit this reports
  rejected       share of photos the limit sends back
  would fail     share of those that indeed fail to match their own palm (v4 score below its threshold)
on two sets of DEV photos:
  natural    every DEV photo of BJTU, Tongji and 11k Hands used by tune_dev.py (no degradation)
  degraded   Tongji DEV probes made brighter, over-exposed, darker, shadowed, blurred or noisy (session 2 vs session 1)

  python benchmarks/gate_dev.py
Output: benchmarks/results/tuning/<YYYYmmdd-HHMM>-gate/ gate_dev.md, gate_dev.csv
"""
import csv
import datetime
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_benchmark as RB  # noqa: E402
import tune_dev as TD  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import palm_engine as eng  # noqa: E402
import palm_quality as pq  # noqa: E402

con = RB.con
CONDITIONS = dict(TD.CONDITIONS)
CONDITIONS.update({
    "darker (x0.5)": lambda g: (g * 0.5).astype(np.uint8),
    "very dark (x0.25)": lambda g: (g * 0.25).astype(np.uint8),
    "blur (sigma 4)": lambda g: cv2.GaussianBlur(g, (0, 0), 4),
    "blur (sigma 8)": lambda g: cv2.GaussianBlur(g, (0, 0), 8),
    "overexposed (x3.0, clipped)": lambda g: np.clip(g.astype(np.float32) * 3.0, 0, 255).astype(np.uint8),
})
CANDIDATES = {
    "clipped_white": ("max", [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]),
    "clipped_black": ("max", [0.05, 0.10, 0.20, 0.30]),
    "brightness_high": ("max", [215, 225, 235, 245]),
    "brightness_low": ("min", [25, 40, 55]),
    "focus": ("min", [0.08, 0.10, 0.12, 0.15, 0.20]),
    "noise": ("max", [4, 6, 8, 10]),
    "crease_contrast": ("min", [0.005, 0.008, 0.010, 0.014]),
}


def value(m, name):
    return m["brightness"] if name.startswith("brightness") else m[name]


def main():
    thr = eng.V4["threshold"]
    rows = []                                      # one per photo: set, condition, measures, weakest genuine v4 score
    prog = TD.Progress(TD.TextColumn("[bold]{task.description}"), TD.BarColumn(bar_width=36), TD.MofNCompleteColumn(),
                       TD.TimeElapsedColumn(), console=con)
    eng.warm_up()
    with prog:
        # natural DEV photos: measures + weakest genuine fused score from the tune_dev cache (clahe + v2)
        for k in ("bjtu2", "tongji", "11khands"):
            samples, dev = TD.dev_split(k, 40)
            T = TD.Tuner(k, samples, dev, prog)
            gp, _ = T.pairs(150, 0)
            weak = {}
            for a, b in gp:
                ka, kb = f"clahe|{a}|{b}", f"v2|{a}|{b}"
                if ka in T.scores and kb in T.scores:
                    s = T.scores[ka] / 12 + T.scores[kb] / 4
                    weak[a], weak[b] = min(weak.get(a, 9e9), s), min(weak.get(b, 9e9), s)
            for i, s in weak.items():
                m = pq.crop_measures(T.store.get_crop(i))
                rows.append(dict(set="natural", dataset=k, condition="none", score=s, **m))
        # degraded Tongji DEV probes vs a clean enrolled photo, scored with v4
        samples, dev = TD.dev_split("tongji", 40)
        T = TD.Tuner("tongji", samples, dev, prog)
        first = {}
        for i in T.usable():
            first.setdefault((samples[i][0], samples[i][1].split("/")[0]), i)
        ids = sorted({s for s, _ in first if (s, "session1") in first and (s, "session2") in first})
        enrol = {s: eng.make_template(T.store.get_crop(first[s, "session1"])) for s in ids}
        for cname, fn in CONDITIONS.items():
            t = prog.add_task(f"degraded · {cname}", total=len(ids))
            for s in ids:
                g = fn(T.store.get_crop(first[s, "session2"]))
                sc = eng.score(enrol[s], eng.make_template(g))
                rows.append(dict(set="degraded", dataset="tongji", condition=cname, score=sc, **pq.crop_measures(g)))
                prog.advance(t)
            prog.remove_task(t)

    out = os.path.join(RB.REPO, "benchmarks", "results", "tuning", datetime.datetime.now().strftime("%Y%m%d-%H%M") + "-gate")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "gate_dev.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = ["# Quality gate limits (DEV identities only)", "",
          f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} by `benchmarks/gate_dev.py`. A photo 'would fail' when its "
          f"v4 score against its own palm is below the v4 threshold ({thr}).", "",
          "## Matching failures per condition (no gate)", "", "| Set | Condition | Photos | Would fail |", "|---|---|---|---|"]
    for (st, c) in dict.fromkeys((r["set"], r["condition"]) for r in rows):
        rs = [r for r in rows if r["set"] == st and r["condition"] == c]
        md.append(f"| {st} | {c} | {len(rs)} | {TD.pct(np.mean([r['score'] < thr for r in rs]))} |")
    md += ["", "## Candidate limits", "", "| Measure | Limit | Natural photos rejected | Degraded photos rejected | "
           "Rejected photos that would fail | Failing photos caught |", "|---|---|---|---|---|---|"]
    allfail = [r for r in rows if r["score"] < thr]
    for name, (kind, cands) in CANDIDATES.items():
        for c in cands:
            rej = lambda r: value(r, name) > c if kind == "max" else value(r, name) < c
            nat = [r for r in rows if r["set"] == "natural"]
            deg = [r for r in rows if r["set"] == "degraded"]
            rr = [r for r in rows if rej(r)]
            md.append(f"| {name} | {'>' if kind == 'max' else '<'} {c} | {TD.pct(np.mean([rej(r) for r in nat]))} | "
                      f"{TD.pct(np.mean([rej(r) for r in deg]))} | "
                      f"{TD.pct(np.mean([r['score'] < thr for r in rr])) if rr else 'n/a'} | "
                      f"{TD.pct(np.mean([rej(r) for r in allfail])) if allfail else 'n/a'} |")
    open(os.path.join(out, "gate_dev.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    con.print("\n".join(md))
    con.print(f"[green]written[/] {os.path.relpath(out, RB.REPO)}/gate_dev.md")


if __name__ == "__main__":
    main()
