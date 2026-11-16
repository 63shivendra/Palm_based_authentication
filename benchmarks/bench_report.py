#!/usr/bin/env python3
"""
benchmarks/bench_report.py - CSV files, charts and REPORT.md for one run of benchmarks/run_benchmark.py.
Called automatically at the end of a run; can be re-run on any run folder:

  python benchmarks/bench_report.py benchmarks/results/runs/<run>
"""
import csv
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = os.path.join(REPO, ".bench_env")
if os.path.isdir(ENV):
    sys.path.insert(0, ENV)

# categorical slots in fixed order, one per method (never reassigned by rank)
COLOR = {"v3": "#2a78d6", "v3c": "#eb6834", "v3cr": "#1baf7a", "v2": "#eda100", "v1": "#e87ba4", "v4": "#7356c9"}
LABEL = {"v4": "v4 (v3c + quality gate, DEV-tuned)", "v3": "v3 DISK+LightGlue (full)", "v3c": "v3c compact",
         "v3cr": "v3cr compact + robust",
         "v2": "v2 SIFT baseline", "v1": "v1 CompCode baseline"}
ORDER = ["v4", "v3c", "v3cr", "v3", "v2", "v1"]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def pct(v, nd=2):
    return "n/a" if v is None else f"{v * 100:.{nd}f}%"


def load(run_dir):
    return json.load(open(os.path.join(run_dir, "results.json"), encoding="utf-8"))


# ============================================================================ CSV
def write_csvs(run_dir, R):
    rows = R["rows"]
    cols = ["dataset", "dataset_name", "licence_use", "method", "method_label", "enrolled_photos", "protocol",
            "images", "usable", "identities", "test_identities", "dev_identities", "FTE", "genuine_comparisons",
            "impostor_comparisons", "threshold_shipped", "accuracy", "EER", "FAR", "FRR", "FAR_at_devEER", "FRR_at_devEER",
            "FAR_at_devFAR0.1%", "FRR_at_devFAR0.1%", "TAR_at_FAR_1%", "TAR_at_FAR_0.1%", "TAR_at_FAR_0.01%", "AUC",
            "dprime", "rank1", "genuine_median", "genuine_min", "impostor_median", "impostor_max", "seconds"]
    with open(os.path.join(run_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            d = R["datasets"].get(r["dataset"], {})
            de, df = r.get("at_dev_eer") or {}, r.get("at_dev_far") or {}
            w.writerow([r["dataset"], d.get("name"), d.get("use"), r["method"], LABEL.get(r["method"]), r["enrol"],
                        r["protocol"], r["images"], r["usable"], r["identities"], r["test_identities"], r["dev_identities"],
                        r["fte_rate"], r["genuine_comparisons"], r["impostor_comparisons"], r["threshold"], r["accuracy"],
                        r["EER"], r["FAR"], r["FRR"], de.get("FAR"), de.get("FRR"), df.get("FAR"), df.get("FRR"),
                        r["TAR_at_FAR_1e2"], r["TAR_at_FAR_1e3"], r["TAR_at_FAR_1e4"], r["AUC"], r["dprime"],
                        r.get("rank1"), r["genuine_median"], r["genuine_min"], r["impostor_median"], r["impostor_max"],
                        r.get("seconds")])
    with open(os.path.join(run_dir, "latency.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["where", "device", "stage", "median_ms", "p95_ms", "n"])
        for where, L in (R.get("latency") or {}).items():
            for st, v in L["stages"].items():
                w.writerow([where, L["device"], st, round(v["median"], 1), round(v["p95"], 1), v["n"]])
    if R.get("robustness"):
        with open(os.path.join(run_dir, "robustness.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["method", "condition", "FRR", "FAR", "EER", "genuine_median", "impostor_max", "genuine", "impostor"])
            for r in R["robustness"]:
                w.writerow([r[k] for k in ("method", "condition", "FRR", "FAR", "EER", "genuine_median", "impostor_max",
                                           "genuine", "impostor")])
    man_path = os.path.join(REPO, "data", "MANIFEST.json")
    man = json.load(open(man_path, encoding="utf-8")) if os.path.exists(man_path) else {}
    with open(os.path.join(run_dir, "datasets.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "dataset", "use", "licence", "source", "content", "note", "images_on_disk", "downloaded",
                    "archive_sha256"])
        for k, m in man.items():
            w.writerow([k, m.get("name"), m.get("use"), m.get("licence"), m.get("source"), m.get("content"), m.get("note"),
                        m.get("images"), m.get("downloaded"), m.get("archive_sha256", "")])
    return man


# ============================================================================ charts
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _title(fig, title, sub):
    """Title and subtitle placed in inches from the top, so they never collide whatever the figure height."""
    h = fig.get_size_inches()[1]
    fig.text(0.012, 1 - 0.14 / h, title, fontsize=13, fontweight="semibold", color=INK, va="top")
    fig.text(0.012, 1 - 0.44 / h, sub, fontsize=9.5, color=INK2, va="top")


def bar_panel(ax, rows, key, datasets, methods, xlabel, fmt=lambda v: pct(v)):
    """Horizontal grouped bars: one group per dataset, one bar per method; value label at each bar end."""
    h = 0.8 / max(len(methods), 1)
    vals = [r[key] for r in rows if r[key] is not None]
    xmax = max(vals + [0.01]) * 1.25
    for j, m in enumerate(methods):
        for i, d in enumerate(datasets):
            r = next((x for x in rows if x["dataset"] == d and x["method"] == m), None)
            if r is None or r[key] is None:
                continue
            y = i + (j - (len(methods) - 1) / 2) * h
            v = r[key]
            ax.barh(y, max(v, xmax * 0.004), height=h * 0.86, color=COLOR[m], edgecolor=SURFACE, linewidth=1)
            ax.text(max(v, xmax * 0.004) + xmax * 0.01, y, fmt(v), va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v * 100:.1f}%")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=0)


def legend(fig, methods, y=0.885):
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=COLOR[m], label=LABEL[m]) for m in methods], loc="upper left", ncol=len(methods),
               bbox_to_anchor=(0.012, y), frameon=False, fontsize=8.5, labelcolor=INK2, handlelength=1, columnspacing=1.6)


def plots(run_dir, R):
    plt = _plt()
    out = os.path.join(run_dir, "plots")
    os.makedirs(out, exist_ok=True)
    made = []
    rows = R["rows"]
    methods = [m for m in ORDER if any(r["method"] == m for r in rows)]
    names = {k: v["name"] for k, v in R["datasets"].items()}
    for enrol in sorted({r["enrol"] for r in rows}):
        sub = [dict(r, dataset=names.get(r["dataset"], r["dataset"])) for r in rows if r["enrol"] == enrol]
        ds = list(dict.fromkeys(r["dataset"] for r in sub))
        if not ds:
            continue
        hgt = 1.75 + 0.22 * len(ds) * max(1, len(methods))
        fig, axs = plt.subplots(1, 3, figsize=(15, hgt), dpi=150)
        fig.subplots_adjust(left=0.15, right=0.98, top=1 - 1.15 / hgt, bottom=0.6 / hgt, wspace=0.35)
        bar_panel(axs[0], sub, "EER", ds, methods, "EER (lower is better)")
        bar_panel(axs[1], sub, "FAR", ds, methods, "FAR at shipped threshold")
        bar_panel(axs[2], sub, "FRR", ds, methods, "FRR at shipped threshold (incl. rejected photos)")
        for ax in axs[1:]:
            ax.set_yticklabels([])
        _title(fig, f"Accuracy on held-out test identities, {enrol} enrolled photo{'s' if enrol > 1 else ''}",
               "Zero-shot: no palm image was used to build any matcher. FAR and FRR at the threshold shipped in the code.")
        legend(fig, methods, y=1 - 0.7 / hgt)
        p = os.path.join(out, f"accuracy_enrol{enrol}.png"); fig.savefig(p); plt.close(fig); made.append(p)

    # DET curves + score distributions per dataset (1 enrolled photo)
    for d in dict.fromkeys(r["dataset"] for r in rows):
        curves = []
        for m in methods:
            f = os.path.join(run_dir, "scores", f"{d}__{m}.npz")
            if os.path.exists(f):
                z = np.load(f)
                curves.append((m, z["genuine"], z["impostor"]))
        if not curves:
            continue
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)
        fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.12, wspace=0.25)
        for m, g, i in curves:
            ts = np.unique(np.concatenate([g, i]))
            fmr = np.array([np.mean(i >= t) for t in ts]); fnmr = np.array([np.mean(g < t) for t in ts])
            axs[0].loglog(np.maximum(fmr, 1 / (len(i) + 1)), np.maximum(fnmr, 1 / (len(g) + 1)), color=COLOR[m], lw=2)
        lim = 1 / (max(len(c[2]) for c in curves) + 1)
        axs[0].plot([lim, 1], [lim, 1], color=MUTED, lw=0.8, ls="--")
        axs[0].set_xlabel("FAR (false accept rate)"); axs[0].set_ylabel("FRR (false reject rate)")
        axs[0].set_title("DET curve (closer to the bottom-left is better)", fontsize=10, loc="left")
        best = curves[0]
        m, g, i = best
        lo, hi = min(g.min(), i.min()), max(g.max(), i.max())
        bins = np.linspace(lo, hi, 45)
        axs[1].hist(i, bins=bins, density=True, color="#898781", alpha=0.75, label=f"different palms (n={len(i):,})")
        axs[1].hist(g, bins=bins, density=True, color=COLOR[m], alpha=0.85, label=f"same palm (n={len(g):,})")
        thr = next((r["threshold"] for r in rows if r["dataset"] == d and r["method"] == m and r["enrol"] == 1), None)
        if thr is not None:
            axs[1].axvline(thr, color=INK, lw=1.2)
            axs[1].text(thr, axs[1].get_ylim()[1] * 0.95, f"  threshold {thr:g}", fontsize=8, color=INK)
        axs[1].set_yscale("log")
        axs[1].set_xlabel(f"{LABEL[m]} score"); axs[1].set_title(f"Score distributions, {LABEL[m]}", fontsize=10, loc="left")
        axs[1].legend(frameon=False, fontsize=8.5, labelcolor=INK2)
        _title(fig, names.get(d, d), "Test identities only, 1 enrolled photo. The gap between the two distributions is what "
                                     "makes the threshold safe.")
        legend(fig, [c[0] for c in curves], y=1 - 0.7 / 4.8)
        p = os.path.join(out, f"det_{d}.png"); fig.savefig(p); plt.close(fig); made.append(p)

    # latency: stacked stages per method, one panel per device
    lat = R.get("latency") or {}
    if lat:
        fig, axs = plt.subplots(1, len(lat), figsize=(8.5 * len(lat), 3.6), dpi=150, squeeze=False)
        fig.subplots_adjust(left=0.24 / len(lat), right=0.97, top=0.72, bottom=0.17, wspace=0.55)
        stages = [("detect", "#c3c2b7"), ("align", "#898781"), ("extract", None), ("match", None)]
        for ax, (where, L) in zip(axs[0], lat.items()):
            st = L["stages"]
            ms = [m for m in ("v4", "v3c", "v3cr", "v3") if f"extract_{m}" in st]
            for y, m in enumerate(ms):
                left = 0
                for s, c in stages:
                    key = s if s in ("detect", "align") else f"{s}_{m}"
                    if key not in st:
                        continue
                    v = st[key]["median"]
                    col = c or COLOR[m]
                    ax.barh(y, v, left=left, color=col, alpha=1.0 if s != "match" else 0.55, edgecolor=SURFACE, lw=1.5, height=0.6)
                    left += v
                ax.text(left, y, f"  {left:,.0f} ms", va="center", fontsize=8.5, color=INK)
            ax.set_yticks(range(len(ms))); ax.set_yticklabels([LABEL[m] for m in ms]); ax.invert_yaxis()
            ax.set_xlabel("milliseconds per verification (median): detect | align | extract | match")
            ax.set_title(L["device"], fontsize=10, loc="left"); ax.grid(axis="y", visible=False); ax.tick_params(length=0)
            ax.set_xlim(0, ax.get_xlim()[1] * 1.18)
        _title(fig, "Time for one verification", "Detection + alignment (grey), feature extraction (solid), "
                                                 "match vs one enrolled photo (light). Whole-hand phone photos.")
        p = os.path.join(out, "latency.png"); fig.savefig(p); plt.close(fig); made.append(p)

    # robustness heatmap: FRR per condition x method (sequential, one hue)
    rob = R.get("robustness") or []
    if rob:
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])
        conds = list(dict.fromkeys(r["condition"] for r in rob))
        ms = [m for m in ORDER if any(r["method"] == m for r in rob)]
        M = np.array([[next((r["FRR"] for r in rob if r["condition"] == c and r["method"] == m), np.nan) for m in ms]
                      for c in conds])
        fig, ax = plt.subplots(figsize=(3 + 1.6 * len(ms), 1.4 + 0.36 * len(conds)), dpi=150)
        fig.subplots_adjust(left=0.42, right=0.97, top=1 - 1.1 / (1.4 + 0.36 * len(conds)), bottom=0.05)
        ax.imshow(M, cmap=cmap, vmin=0, vmax=max(0.2, np.nanmax(M)), aspect="auto")
        for y in range(len(conds)):
            for x in range(len(ms)):
                v = M[y, x]
                ax.text(x, y, pct(v, 1), ha="center", va="center", fontsize=8,
                        color="#ffffff" if v > 0.5 * max(0.2, np.nanmax(M)) else INK)
        ax.set_xticks(range(len(ms))); ax.set_xticklabels([LABEL[m] for m in ms], fontsize=8.5)
        ax.xaxis.tick_top(); ax.set_yticks(range(len(conds))); ax.set_yticklabels(conds, fontsize=8.5)
        ax.grid(False); ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        _title(fig, "Robustness: genuine users rejected (FRR)", "Clean enrolled photo, degraded probe (Tongji). FAR stayed "
                                                                "as listed in robustness.csv.")
        p = os.path.join(out, "robustness.png"); fig.savefig(p); plt.close(fig); made.append(p)

    tb = R.get("template_bytes") or {}
    if tb:
        ms = [m for m in ORDER if m in tb]
        fig, ax = plt.subplots(figsize=(8, 0.9 + 0.45 * len(ms)), dpi=150)
        fig.subplots_adjust(left=0.3, right=0.9, top=0.72, bottom=0.12)
        for y, m in enumerate(ms):
            ax.barh(y, tb[m] / 1000, color=COLOR[m], height=0.6)
            ax.text(tb[m] / 1000, y, f"  {tb[m] / 1000:,.0f} KB", va="center", fontsize=8.5, color=INK)
        ax.set_yticks(range(len(ms))); ax.set_yticklabels([LABEL[m] for m in ms]); ax.invert_yaxis()
        ax.set_xlim(0, max(tb.values()) / 1000 * 1.2); ax.grid(axis="y", visible=False); ax.tick_params(length=0)
        ax.set_xlabel("KB per enrolled photo (saved file)")
        _title(fig, "Template size", "What has to be stored per enrolled photo.")
        p = os.path.join(out, "template_size.png"); fig.savefig(p); plt.close(fig); made.append(p)
    return made


# ============================================================================ markdown report
def literature():
    p = os.path.join(REPO, "benchmarks", "results", "literature_comparison.csv")
    if not os.path.exists(p):
        return []
    return [r for r in csv.DictReader(open(p, encoding="utf-8")) if not r["method"].startswith("OURS")]


def report(run_dir, R, man, made):
    rows = R["rows"]
    names = {k: v["name"] for k, v in R["datasets"].items()}
    cfg = R["config"]
    L = []
    w = L.append
    w(f"# Palm recognition benchmark (zero-shot)\n")
    w(f"Run `{os.path.basename(run_dir)}`, finished {R['finished']}, on {R['device']}, profile **{cfg.get('profile')}**.\n")
    w("No palm image was used to build any matcher: DISK and LightGlue are pretrained on outdoor scenes and MediaPipe only "
      "locates the hand. Every number below is measured on **held-out test identities**; for datasets with at least 40 "
      "identities, 20% of identities (DEV) were used only to pick the alternative thresholds.\n")

    w("## 1. Summary (1 enrolled photo, shipped threshold)\n")
    w("| Dataset | Method | Test identities | Comparisons (genuine / impostor) | Accuracy | EER | FAR | FRR | TAR @ FAR 0.1% | Rank-1 | FTE |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in [x for x in rows if x["enrol"] == 1]:
        w(f"| {names.get(r['dataset'], r['dataset'])} | {LABEL[r['method']]} | {r['test_identities']} | "
          f"{r['genuine_comparisons']:,} / {r['impostor_comparisons']:,} | {pct(r['accuracy'])} | **{pct(r['EER'])}** | "
          f"{pct(r['FAR'])} | {pct(r['FRR'])} | {pct(r['TAR_at_FAR_1e3'])} | {pct(r.get('rank1'))} | {pct(r['fte_rate'])} |")
    multi = [x for x in rows if x["enrol"] > 1]
    if multi:
        w(f"\n### With {multi[0]['enrol']} enrolled photos (score = best match over the enrolled photos)\n")
        w("| Dataset | Method | Comparisons | Accuracy | EER | FAR | FRR |")
        w("|---|---|---|---|---|---|---|")
        for r in multi:
            w(f"| {names.get(r['dataset'], r['dataset'])} | {LABEL[r['method']]} | {r['genuine_comparisons']:,} / "
              f"{r['impostor_comparisons']:,} | {pct(r['accuracy'])} | **{pct(r['EER'])}** | {pct(r['FAR'])} | {pct(r['FRR'])} |")
    devrows = [x for x in rows if x["enrol"] == 1 and x.get("at_dev_eer")]
    if devrows:
        w("\n### Thresholds chosen on DEV identities, applied to TEST identities\n")
        w("| Dataset | Method | shipped thr | FAR / FRR | DEV EER-point thr | FAR / FRR | DEV FAR-0.1% thr | FAR / FRR |")
        w("|---|---|---|---|---|---|---|---|")
        for r in devrows:
            de, df = r["at_dev_eer"], r["at_dev_far"]
            w(f"| {names.get(r['dataset'], r['dataset'])} | {LABEL[r['method']]} | {r['threshold']:g} | {pct(r['FAR'])} / "
              f"{pct(r['FRR'])} | {de['threshold']:.1f} | {pct(de['FAR'])} / {pct(de['FRR'])} | {df['threshold']:.1f} | "
              f"{pct(df['FAR'])} / {pct(df['FRR'])} |")

    w("\n## 2. Datasets and licences\n")
    w("| Dataset | Licence | Use here | Images used | Identities | Protocol | Hand detection |")
    w("|---|---|---|---|---|---|---|")
    seen = set()
    for r in rows:
        if r["dataset"] in seen:
            continue
        seen.add(r["dataset"])
        key = {"bjtu2": "hewelxx", "xinhua": "hewelxx"}.get(r["dataset"], r["dataset"])
        m = man.get(key, {})
        w(f"| {names.get(r['dataset'])} | {m.get('licence', 'own photos' if r['dataset'] == 'real' else '?')} | "
          f"{R['datasets'][r['dataset']]['use']} | {r['images']:,} ({r['usable']:,} usable) | {r['identities']} | "
          f"{r['protocol']} | {'MediaPipe' if r.get('detect') else 'no (pre-cropped)'} |")
    w("\n`permissive` = licence allows research, development and redistribution. `eval-only` = no licence, research-only or "
      "unclear provenance: used only to measure a matcher that never trained on it, never redistributed. Notes per dataset "
      "are in `datasets.csv` (from `data/MANIFEST.json`).\n")

    rob = R.get("robustness")
    if rob:
        ms = [m for m in ORDER if any(x["method"] == m for x in rob)]
        w("## 3. Robustness to capture conditions (Tongji, simulated)\n")
        w(f"Clean enrolled photo, degraded probe photo; {rob[0]['genuine']} genuine and {rob[0]['impostor']} impostor "
          f"comparisons per condition, shipped threshold. FRR / FAR:\n")
        w("| Condition | " + " | ".join(LABEL[m] for m in ms) + " |")
        w("|---|" + "---|" * len(ms))
        for c in dict.fromkeys(x["condition"] for x in rob):
            cells = []
            for m in ms:
                x = next((y for y in rob if y["condition"] == c and y["method"] == m), None)
                gate = f" ({pct(x['gate_rejects'], 1)} retake)" if x and x.get("gate_rejects") else ""
                cells.append(f"{pct(x['FRR'], 1)} / {pct(x['FAR'], 1)}{gate}" if x else "-")
            w(f"| {c} | " + " | ".join(cells) + " |")
        w("\nDegradations are applied to the cropped image, not captured with a real camera. For v4, FRR counts every "
          "genuine attempt that did not pass, including photos its quality gate sent back for a retake (in brackets); "
          "a live system asks those users to retake the photo instead of rejecting them.\n")

    lat = R.get("latency") or {}
    if lat:
        w("## 4. Latency per verification (median, ms)\n")
        devs = list(lat.items())
        w("| Stage | " + " | ".join(L_["device"] for _, L_ in devs) + " |")
        w("|---|" + "---|" * len(devs))
        stages = list(dict.fromkeys(s for _, L_ in devs for s in L_["stages"]))
        for s in stages:
            w(f"| {s} | " + " | ".join(f"{L_['stages'][s]['median']:,.0f} (p95 {L_['stages'][s]['p95']:,.0f})"
                                         if s in L_["stages"] else "-" for _, L_ in devs) + " |")
        w("\nTotal for one verification = detect + align + extract + one match per enrolled photo. Measured on whole-hand "
          "phone photos. Not measured on a phone or payment terminal.\n")

    tb = R.get("template_bytes") or {}
    if tb:
        w("## 5. Template size (one enrolled photo, saved file)\n")
        w("| Method | Size |\n|---|---|")
        for m in [x for x in ORDER if x in tb]:
            w(f"| {LABEL[m]} | {tb[m] / 1000:,.0f} KB |")
        w("")

    w("## 6. Spoof detection\n")
    w("**Not measured.** The matchers have no liveness or presentation-attack detection yet, and no permissively "
      "licensed palm spoof dataset was found. A printed photo or a phone screen showing a palm may be accepted. "
      "Measuring APCER / BPCER (ISO/IEC 30107-3) needs our own spoof captures.\n")

    lit = literature()
    if lit:
        w("## 7. Comparison with published results\n")
        w("Published numbers are copied from the cited sources (not re-run) and use different protocols; most methods train "
          "on the dataset they are tested on. Ours is zero-shot.\n")
        w("| Dataset | Method | EER | Trained on palm images | Protocol | Source |\n|---|---|---|---|---|---|")
        for r in [x for x in rows if x["enrol"] == 1 and x["dataset"] in ("tongji", "palmmatchdb")]:
            w(f"| {names.get(r['dataset'])} | **ours: {LABEL[r['method']]}** | {pct(r['EER'])} | no (zero-shot) | "
              f"{r['protocol']}, test identities | this run |")
        for r in lit:
            eer_s = f"{r['EER_pct']}%" if r["EER_pct"] else "-"
            w(f"| {r['dataset']} | {r['method']} | {eer_s} | {r['trained_on_palm_images']} | {r['protocol']} | {r['source']} |")
        w("")

    w("## 8. How far the numbers can be trusted\n")
    small = [names.get(r["dataset"]) for r in rows if r["enrol"] == 1 and r["test_identities"] < 20]
    w(f"- Comparison counts are in the tables. With N impostor comparisons a FAR below about 3/N cannot be shown "
      f"(rule of 3); the largest test here has {max(r['impostor_comparisons'] for r in rows):,}.")
    if small:
        w(f"- Fewer than 20 test identities (smoke test only, not an accuracy measurement): {', '.join(sorted(set(small)))}.")
    w("- Tongji, BJTU, XINHUA and Gabor are pre-cropped palms, so they do not test hand detection and alignment; "
      "PalmMatchDB, 11k Hands and the own photos test the whole chain.")
    w("- 11k Hands, XINHUA and Gabor photos of a palm were taken in one sitting (same session), an easier test than "
      "Tongji and BJTU, whose two sessions are weeks apart.")
    w("- Accuracy depends on the mix of genuine and impostor comparisons (v3cr uses fewer impostor comparisons because "
      "it retries every failed one); compare methods on EER, FAR and FRR.")
    w("- PalmMatchDB labels are unreliable; only pairs with trustworthy labels are kept (same palm: photos < 10 min apart; "
      "different palm: > 1 day apart).")
    w("- Every reported identity is a TEST identity of the one global split (`global_split`). v4's settings (CLAHE, "
      "fusion, threshold 1.5, quality-gate limits) were chosen on DEV identities only "
      "(`benchmarks/results/tuning/DECISIONS.md`). The older thresholds were not: v3's 26 was set on Tongji and v3c's 12 "
      "on PalmMatchDB and the own photos, so their FAR/FRR on those sets are optimistic (EER does not depend on it).")
    w("- v4's FRR includes photos its quality gate sent back; in a live system the user would retake them.")
    if cfg.get("profile") == "quick":
        w("- **Quick profile**: small samples sized for a laptop CPU. Run `--profile full` on the GPU machine for the "
          "numbers to quote.")
    w("\n## 9. Files\n")
    w("- `metrics.csv` every metric, one row per dataset x method x enrolment; `latency.csv`, `robustness.csv`, `datasets.csv`")
    w("- `scores/*.npz` the score of every comparison; `jobs/*.json` raw results; `run.log` the full log")
    for p in made:
        w(f"- `plots/{os.path.basename(p)}`")
    w("")
    for p in made:
        w(f"![{os.path.basename(p)}](plots/{os.path.basename(p)})\n")
    open(os.path.join(run_dir, "REPORT.md"), "w", encoding="utf-8").write("\n".join(L))


def terminal_summary(R, con):
    from rich.table import Table
    names = {k: v["name"] for k, v in R["datasets"].items()}
    t = Table(title="Zero-shot results on test identities (shipped threshold)", title_justify="left")
    for c in ("dataset", "method", "enrol", "gen / imp", "accuracy", "EER", "FAR", "FRR", "TAR@FAR0.1%", "rank-1", "FTE"):
        t.add_column(c, justify="right" if c not in ("dataset", "method") else "left")
    for r in R["rows"]:
        t.add_row(names.get(r["dataset"], r["dataset"]), LABEL[r["method"]], str(r["enrol"]),
                  f"{r['genuine_comparisons']:,} / {r['impostor_comparisons']:,}", pct(r["accuracy"]),
                  f"[bold]{pct(r['EER'])}[/]", pct(r["FAR"]), pct(r["FRR"]), pct(r["TAR_at_FAR_1e3"]), pct(r.get("rank1")),
                  pct(r["fte_rate"]))
    con.print(t)


def write_all(run_dir, con=None):
    if con is None:
        from rich.console import Console
        con = Console()
    R = load(run_dir)
    man = write_csvs(run_dir, R)
    made = plots(run_dir, R)
    report(run_dir, R, man, made)
    terminal_summary(R, con)
    con.print(f"\n[bold green]done[/]  report: {os.path.relpath(os.path.join(run_dir, 'REPORT.md'), REPO)}   "
              f"charts: {len(made)} in {os.path.relpath(os.path.join(run_dir, 'plots'), REPO)}   "
              f"CSV: metrics.csv latency.csv robustness.csv datasets.csv")


if __name__ == "__main__":
    write_all(sys.argv[1] if len(sys.argv) > 1 else sorted(
        os.path.join(REPO, "benchmarks", "results", "runs", d)
        for d in os.listdir(os.path.join(REPO, "benchmarks", "results", "runs")))[-1])
