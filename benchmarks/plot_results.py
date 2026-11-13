#!/usr/bin/env python3
"""benchmarks/plot_results.py - charts of the benchmark results, written to benchmarks/results/plots/.

  python benchmarks/plot_results.py
  python benchmarks/plot_results.py --skip tongji:v3c tongji:v3c_enrol3     # leave out runs that are not finished

  accuracy.png     EER and FRR per dataset, v3 next to v3c, 1 and 3 enrolled photos
  robustness.png   share of genuine users rejected under each simulated capture condition
  size_speed.png   template size, time per verification, model files on disk
The numbers behind every chart are in the CSV files next to this folder.
"""
import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, PathPatch  # noqa: E402
from matplotlib.path import Path  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_csv import R, keep, load  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {"v3": ("#2a78d6", "v3 (full template)"), "v3c": ("#eb6834", "v3c (compact template)"),
          "v3cr": ("#1baf7a", "v3c + noise smoothing and retry")}
DPI = 200
PX = DPI / 96                                   # one CSS pixel, in device pixels
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
OUT = os.path.join(R, "plots")


def figure(w, h, title, subtitle, series=()):
    fig = plt.figure(figsize=(w, h), dpi=DPI, facecolor=SURFACE)
    fig.text(0.3 / w, 1 - 0.28 / h, title, fontsize=13, fontweight="semibold", color=INK, va="top")
    fig.text(0.3 / w, 1 - 0.60 / h, subtitle, fontsize=9.5, color=INK2, va="top")
    if series:
        fig.legend(handles=[Patch(color=SERIES[s][0], label=SERIES[s][1]) for s in series], loc="upper left",
                   bbox_to_anchor=(0.3 / w, 1 - 0.92 / h), ncol=len(series), frameon=False, fontsize=9, labelcolor=INK2,
                   handlelength=1.0, handleheight=0.9, columnspacing=1.8, borderaxespad=0, borderpad=0)
    return fig


def axes(fig, left, bottom, width, height, xmax, ticks, fmt, ys, title=None):
    """Axes placed in inches; horizontal bars grow from a baseline on the left."""
    w, h = fig.get_size_inches()
    ax = fig.add_axes([left / w, bottom / h, width / w, height / h], facecolor=SURFACE)
    ax.set_xlim(0, xmax)
    ax.set_ylim(max(ys) + 0.62, min(ys) - 0.62)
    ax.set_xticks(ticks)
    ax.set_xticklabels([fmt(t) for t in ticks])
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.tick_params(axis="x", length=0, colors=MUTED, labelsize=8.5)
    ax.tick_params(axis="y", length=0, colors=INK2, labelsize=9.5, pad=8)
    if title:
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="semibold", color=INK, pad=10)
    return ax


def scale(ax):
    (x0, y0), (x1, y1) = ax.transData.transform([(0, 0), (1, 1)])
    return abs(x1 - x0), abs(y1 - y0)


def bar(ax, y, w, color, thick=12):
    """Horizontal bar: square at the baseline, 4 px rounded at the data end."""
    if w <= 0:
        return
    sx, sy = scale(ax)
    h = thick * PX / sy
    rx, ry = min(4 * PX / sx, w), min(4 * PX / sy, h / 2)
    a, b = y - h / 2, y + h / 2
    verts = [(0, a), (w - rx, a), (w, a), (w, a + ry), (w, b - ry), (w, b), (w - rx, b), (0, b), (0, a)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO,
             Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def group(ax, y, values, fmt, thick=12, missing="re-run in progress"):
    """values = [(colour, value or None), ...] drawn top to bottom around y, 2 px apart, value printed at the tip."""
    sx, sy = scale(ax)
    step = (thick + 2) * PX / sy
    for i, (colour, v) in enumerate(values):
        yy = y + (i - (len(values) - 1) / 2) * step
        if v is None:
            ax.text(6 * PX / sx, yy, missing, va="center", ha="left", fontsize=8, color=MUTED, style="italic")
            continue
        bar(ax, yy, v, colour, thick)
        ax.text(v + 6 * PX / sx, yy, fmt(v), va="center", ha="left", fontsize=8.5, color=INK2, zorder=4,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2))


def nice(maxv, room=1.3):
    """Axis end and ticks with room for the value label after the longest bar."""
    for step in (0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100, 200, 250, 500):
        if maxv * room / step <= 5.5:
            break
    n = int(maxv * room / step) + 1
    return step * n, [round(step * i, 4) for i in range(n + 1)]


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


# ---------------------------------------------------------------------------------------------- 1. accuracy
def plot_accuracy(skip):
    sets = [("tongji", "Tongji"), ("palmmatchdb", "PalmMatchDB"), ("gabor", "Gabor dataset")]
    rows, pending = [], False                      # (y, dataset key, photos, {method: result or None})
    for d, (key, _) in enumerate(sets):
        data = load(key)
        for e, n in enumerate((1, 3)):
            res = {}
            for m in ("v3", "v3c", "v3cr"):
                k = m + ("" if n == 1 else f"_enrol{n}")
                r = data.get(k)
                res[m] = r if keep(key, k, r) and f"{key}:{k}" not in skip else None
                pending |= res[m] is None and m != "v3cr"
            rows.append((d * 2.5 + e, key, n, res))
    ys = [r[0] for r in rows]
    ms = ("v3", "v3c", "v3cr") if all(res["v3cr"] for *_, res in rows) else ("v3", "v3c")   # v3cr only once complete
    shown = [res[m] for *_, res in rows for m in ms if res[m]]
    far = max(r["FAR"] for r in shown) * 100
    fh = 5.9 if len(ms) == 2 else 6.6
    fig = figure(11.6, fh, "Accuracy of v3 and v3c on each dataset" if len(ms) == 2 else "Accuracy on each dataset",
                 "No palm image was used for training. Lower is better in both panels.", ms)
    panels = [("EER", "Equal error rate (EER)", 3.15), ("FRR", "Genuine users rejected (FRR)", 7.45)]
    for metric, title, left in panels:
        xmax, ticks = nice(max(r[metric] for r in shown) * 100)
        ax = axes(fig, left, 0.95, 3.75, 3.55 if len(ms) == 2 else 4.25, xmax, ticks, lambda t: f"{t:g}%", ys, title)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{n} photo{'s' if n > 1 else ''} enrolled" for _, _, n, _ in rows] if metric == "EER" else [])
        for y, key, n, res in rows:
            group(ax, y, [(SERIES[m][0], None if res[m] is None else res[m][metric] * 100) for m in ms],
                  lambda v: f"{v:.2f}%", thick=12 if len(ms) == 2 else 10)
        if metric == "EER":                        # dataset names to the left of the row labels
            for d, (key, name) in enumerate(sets):
                ids = next(res[m]["identities"] for _, k, _, res in rows if k == key for m in ms if res[m])
                ax.text(-0.52, d * 2.5 + 0.28, name, transform=ax.get_yaxis_transform(), ha="right", va="center",
                        fontsize=10.5, fontweight="semibold", color=INK)
                ax.text(-0.52, d * 2.5 + 0.78, f"{ids} palms", transform=ax.get_yaxis_transform(), ha="right", va="center",
                        fontsize=9, color=MUTED)
    note = (f"False accepts (FAR): {far:.2f}% at most in the tests shown. " if far > 0 else
            "False accepts (FAR): 0.00% in every test shown. ")
    note += "FRR includes photos that fail the quality checks.\n"
    note += "PalmMatchDB: pairs with trustworthy labels only. Tongji: enrol in session 1, verify in session 2."
    if pending:
        note += " Runs marked 're-run in progress' are left out until they finish."
    fig.text(0.3 / 11.6, 0.42 / fh, note, fontsize=8.5, color=MUTED, va="top", linespacing=1.5)
    save(fig, "accuracy.png")


# ---------------------------------------------------------------------------------------------- 2. robustness
def plot_robustness():
    reps = {m: load(n) for m, n in (("v3", "robustness_tongji"), ("v3c", "robustness_tongji_v3c"),
                                    ("v3cr", "robustness_tongji_v3cr"))}
    full = len(reps["v3"])
    reps = {m: r for m, r in reps.items() if r and len(r) == full}        # only finished runs
    conds = list(reps["v3"])
    worst = {c: max(r[c]["FRR"] for r in reps.values()) for c in conds}
    hit = sorted((c for c in conds if worst[c] > 0), key=lambda c: -worst[c])
    fine = [c for c in conds if worst[c] == 0]
    far = max(r[c]["FAR"] for r in reps.values() for c in conds) * 100
    n = next(iter(reps["v3"].values()))
    ys = list(range(len(hit)))
    row = 0.24 + 0.16 * len(reps)                  # inches per condition
    h = 1.4 + row * len(hit) + 1.0
    fig = figure(10.4, h, "Robustness: genuine users rejected when the new photo is degraded",
                 f"Tongji, {n['genuine']} palms. Enrolled photo clean, new photo degraded by simulation. Lower is better.",
                 tuple(reps))
    ax = axes(fig, 2.55, 1.0, 7.3, row * len(hit), 100, [0, 25, 50, 75, 100], lambda t: f"{t:g}%", ys)
    ax.set_yticks(ys)
    ax.set_yticklabels([c[0].upper() + c[1:] for c in hit])
    for y, c in zip(ys, hit):
        group(ax, y, [(SERIES[m][0], r[c]["FRR"] * 100) for m, r in reps.items()],
              lambda v: f"{v:.0f}%" if v >= 10 or v == 0 else f"{v:.1f}%")
    names = ", ".join(c.split(" (")[0] if "baseline" in c else c for c in fine)
    note = f"No genuine user rejected under: {names}.\n"
    note += (f"False accepts: 0% in every condition ({n['impostor']:,} impostor comparisons each)." if far == 0 else
             f"False accepts: {far:.2f}% at most.")
    fig.text(0.3 / 10.4, 0.62 / h, note, fontsize=8.5, color=MUTED, va="top", linespacing=1.5, wrap=True)
    save(fig, "robustness.png")


# ---------------------------------------------------------------------------------------------- 3. size and speed
def plot_size_speed():
    size, lat = load("size"), load("latency")
    tot = {("v3c" if "v3c" in s else "v3"): [int(re.search(r"\d+", x).group()) for x in (g, c)]
           for s, g, c in lat["rows"] if s.startswith("Total")}
    fig = figure(12.6, 4.5, "Size and speed of v3 and v3c",
                 f"Measured on a {lat['machine']}. Not measured on a phone or payment terminal.", ("v3", "v3c"))
    b, hgt = 0.75, 2.05
    # template size
    t = size["template_kb"]
    xmax, ticks = nice(max(t.values()))
    ax = axes(fig, 0.45, b, 3.0, hgt, xmax, ticks, lambda v: f"{v:,.0f}", [0], "Stored template per enrolled photo (KB)")
    ax.set_yticks([])
    group(ax, 0, [(SERIES[m][0], t[m]) for m in ("v3", "v3c")], lambda v: f"{v:,.0f} KB", thick=14)
    # time per verification
    xmax, ticks = nice(max(max(v) for v in tot.values()))
    ax = axes(fig, 4.75, b, 2.7, hgt, xmax, ticks, lambda v: f"{v:,.0f}", [0, 1], "Time for one verification (ms)")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["GPU", "CPU"])
    for y in (0, 1):
        group(ax, y, [(SERIES[m][0], tot[m][y]) for m in ("v3", "v3c")], lambda v: f"{v:,.0f} ms", thick=14)
    # model files (the same for both versions, so not in a series colour)
    models = size["models"]
    xmax, ticks = nice(max(m[1] for m in models), room=1.9)
    ax = axes(fig, 9.55, b, 2.55, hgt, xmax, ticks, lambda v: f"{v:g}", range(len(models)), "Model files on disk (MB)")
    ax.set_yticks(range(len(models))); ax.set_yticklabels([m[0].replace(" (", "\n(") for m in models])
    for y, (name, mb, params) in enumerate(models):
        group(ax, y, [(MUTED, mb)], lambda v, p=params: f"{v:g} MB" + (f", {p / 1e6:.1f} M parameters" if p else ""), thick=14)
    fig.text(0.3 / 12.6, 0.36 / 4.5, "Time = hand detection + alignment + feature extraction + one match. "
             f"The three models are the same for v3 and v3c: {sum(m[1] for m in models):.1f} MB in total.", fontsize=8.5, color=MUTED, va="top")
    save(fig, "size_speed.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[], help="dataset:method entries to leave out, e.g. tongji:v3c")
    a = ap.parse_args()
    plot_accuracy(set(a.skip))
    plot_robustness()
    plot_size_speed()
