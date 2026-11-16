#!/usr/bin/env python3
"""
benchmarks/plot_literature.py - our matchers against published palmprint results, on the same datasets.

  python benchmarks/plot_literature.py [--run benchmarks/results/runs/20261115-0239-quick]

Two charts in benchmarks/results/plots/:
  literature_tongji.png     Tongji EER: published methods (from literature_comparison.csv) and ours. Log scale.
                            Papers use different protocols (train/test splits, sessions), so this is context.
  like_for_like.png         every dataset, same test people, same protocol: v4 and v3c against two published methods
                            re-implemented here and run on exactly the same comparisons: CompCode (Kong & Zhang 2004,
                            our v1) and SIFT + RANSAC (Wu, Zhao & Bu 2014, our v2). No published numbers were found
                            for PalmMatchDB, Gabor, 11k Hands (palm side), BJTU v2 or XINHUA.
Also appends nothing to reports: COMPARISON.md links the charts.
"""
import argparse
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_report as BR  # noqa: E402

RES = os.path.join(HERE, "results")
NAMES = {"tongji": "Tongji", "bjtu2": "BJTU", "gabor": "Gabor", "11khands": "11k Hands", "xinhua": "XINHUA",
         "palmmatchdb": "PalmMatchDB", "real": "Own photos"}
ORDER = ["tongji", "bjtu2", "gabor", "11khands", "xinhua", "palmmatchdb"]
GROUP_COLOR = {"trained on Tongji": "#6f6d66", "trained on another dataset": "#c3c2b7",
               "hand-crafted, no training": "#eda100", "ours, no palm training": "#2a78d6"}


def literature_tongji():
    rows = []
    for r in csv.DictReader(open(os.path.join(RES, "literature_comparison.csv"), encoding="utf-8")):
        if r["dataset"] != "Tongji" or r["method"].startswith("OURS") or not r["EER_pct"]:
            continue
        nums = [float(x) for x in re.findall(r"[\d.]+", r["EER_pct"])]
        t = r["trained_on_palm_images"].lower()
        group = ("trained on another dataset" if "another" in t else "trained on Tongji" if t.startswith("yes") or "partly" in t
                 else "hand-crafted, no training")
        rows.append(dict(label=r["method"], eer=min(nums) / 100, hi=max(nums) / 100, group=group, note=r["protocol"]))
    return rows


def ours_large():
    out = []
    for r in csv.DictReader(open(os.path.join(RES, "benchmark_results.csv"), encoding="utf-8")):
        if r["dataset"].startswith("Tongji") and r["enrolled_photos"] == "1" and r["method"] in ("v3", "v3c"):
            out.append(dict(label=f"OURS {r['method']} (600 palms, 30,000 impostor comparisons)", eer=float(r["EER_pct"]) / 100,
                            hi=float(r["EER_pct"]) / 100, group="ours, no palm training", note="cross-session"))
    return out


def fmt_pct(v):
    """0.0000004 -> '0.00004%', 0.000544 -> '0.0544%' (no scientific notation)."""
    p = v * 100
    return (f"{p:.6f}".rstrip("0").rstrip(".") if p < 0.01 else f"{p:.3g}") + "%"


def chart_tongji(path):
    plt = BR._plt()
    from matplotlib.patches import Patch
    rows = sorted(literature_tongji() + ours_large(), key=lambda r: r["eer"])
    fig, ax = plt.subplots(figsize=(10, 1.6 + 0.42 * len(rows)), dpi=150)
    fig.subplots_adjust(left=0.42, right=0.95, top=1 - 1.45 / (1.6 + 0.42 * len(rows)), bottom=0.09)
    for i, r in enumerate(rows):
        col = GROUP_COLOR[r["group"]]
        ax.barh(i, r["eer"], color=col, height=0.62, edgecolor=BR.SURFACE, linewidth=1)
        if r["hi"] > r["eer"]:
            ax.plot([r["eer"], r["hi"]], [i, i], color=BR.INK2, lw=1.2)
        txt = f"{fmt_pct(r['eer'])}" + (f" to {fmt_pct(r['hi'])}" if r["hi"] > r["eer"] else "")
        ax.text(r["hi"] * 1.15, i, txt, va="center", fontsize=8, color=BR.INK, fontweight="bold" if r["group"].startswith("ours") else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8.5)
    for t, r in zip(ax.get_yticklabels(), rows):
        if r["group"].startswith("ours"):
            t.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(2e-7, 0.12)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v * 100:g}%")
    ax.set_xlabel("Equal error rate on Tongji (log scale, lower is better)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=0)
    fig.legend(handles=[Patch(color=c, label=g) for g, c in GROUP_COLOR.items()], loc="upper left", ncol=4,
               bbox_to_anchor=(0.012, 1 - 0.72 / fig.get_size_inches()[1]), frameon=False, fontsize=8.5, labelcolor=BR.INK2,
               handlelength=1, columnspacing=1.4)
    BR._title(fig, "Tongji contactless palmprints: published results and ours",
              "Protocols differ between papers. The fair comparison for a zero-shot matcher is with models trained on another "
              "dataset. v4 at this scale: full GPU run pending.")
    fig.savefig(path)
    plt.close(fig)


def chart_like_for_like(run, path):
    plt = BR._plt()
    from matplotlib.patches import Patch
    m = {}
    for r in csv.DictReader(open(os.path.join(run, "metrics.csv"), encoding="utf-8")):
        if r["enrolled_photos"] == "1":
            m[r["dataset"], r["method"]] = (float(r["EER"]), int(r["test_identities"]), int(r["impostor_comparisons"]))
    series = [("v4", "ours: v4", BR.COLOR["v4"]), ("v3c", "ours: v3c", BR.COLOR["v3c"]),
              ("v2", "SIFT + RANSAC (Wu et al. 2014), our implementation", BR.COLOR["v2"]),
              ("v1", "CompCode (Kong & Zhang 2004), our implementation", BR.COLOR["v1"])]
    ds = [d for d in ORDER if (d, "v4") in m]
    fig, ax = plt.subplots(figsize=(10, 1.5 + 0.75 * len(ds)), dpi=150)
    fig.subplots_adjust(left=0.2, right=0.95, top=1 - 1.6 / (1.5 + 0.75 * len(ds)), bottom=0.1)
    h = 0.8 / len(series)
    xmax = max(m[d, s][0] for d in ds for s, _, _ in series if (d, s) in m) * 1.2
    for j, (s, _, col) in enumerate(series):
        for i, d in enumerate(ds):
            if (d, s) not in m:
                continue
            v = m[d, s][0]
            y = i + (j - 1.5) * h
            ax.barh(y, max(v, xmax * 0.003), height=h * 0.86, color=col, edgecolor=BR.SURFACE, linewidth=1)
            ax.text(max(v, xmax * 0.003) + xmax * 0.008, y, f"{v * 100:.1f}%", va="center", fontsize=7.5, color=BR.INK2)
    ax.set_yticks(range(len(ds)))
    ax.set_yticklabels([f"{NAMES[d]}\n{m[d, 'v4'][1]} people, {m[d, 'v4'][2]:,} imp." for d in ds], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v * 100:g}%")
    ax.set_xlabel("Equal error rate (lower is better); same test people and comparisons for every method")
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=0)
    fig.legend(handles=[Patch(color=c, label=l) for _, l, c in series], loc="upper left", ncol=2,
               bbox_to_anchor=(0.012, 1 - 0.7 / fig.get_size_inches()[1]), frameon=False, fontsize=8.5, labelcolor=BR.INK2,
               handlelength=1, columnspacing=1.4)
    BR._title(fig, "Like for like: ours against published methods on the same people",
              f"Quick run {os.path.basename(run)}, 1 enrolled photo. 0.0% = no error among up to 500 impostor comparisons.")
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=os.path.join(RES, "runs", "20261115-0239-quick"))
    a = ap.parse_args()
    os.makedirs(os.path.join(RES, "plots"), exist_ok=True)
    chart_tongji(os.path.join(RES, "plots", "literature_tongji.png"))
    chart_like_for_like(a.run, os.path.join(RES, "plots", "like_for_like.png"))
    print("written plots/literature_tongji.png and plots/like_for_like.png")


if __name__ == "__main__":
    main()
