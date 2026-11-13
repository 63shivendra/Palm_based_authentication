#!/usr/bin/env python3
"""benchmarks/make_report.py - build benchmarks/results/REPORT.md from the result JSON files.
   python benchmarks/make_report.py"""
import datetime
import json
import os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_csv import DATASETS, LITERATURE, NOT_TESTED, OURS_FOR_LITERATURE  # noqa: E402

METHODS = [("v3", "v3, 1 enrolled photo"), ("v3_enrol3", "v3, 3 enrolled photos"), ("v3_enrol2", "v3, 2 enrolled photos"),
           ("v3c", "v3c compact, 1 photo"), ("v3c_enrol3", "v3c compact, 3 photos"), ("v3c_enrol2", "v3c compact, 2 photos"),
           ("v3cr", "v3cr robust, 1 photo"), ("v3cr_enrol3", "v3cr robust, 3 photos"), ("v3cr_enrol2", "v3cr robust, 2 photos"),
           ("v2", "v2 SIFT, 1 photo"), ("v1", "v1 CompCode, 1 photo")]


def pct(x, nd=2):
    return "n/a" if x is None else f"{x * 100:.{nd}f}%"


def load(name):
    p = os.path.join(R, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else {}


out = [f"# Palm verification benchmark report", "",
       f"Generated {datetime.date.today().isoformat()} by `benchmarks/make_report.py` from the JSON files in this folder.", "",
       "Matcher: `palm_match.py` v3 (MediaPipe hand detection, own alignment and quality gates, DISK + LightGlue "
       "pretrained models used without any palm training, own pose-checked verification and coverage-weighted score). "
       "`v3c` is the compact-template variant in `palm_match_compact.py`. `v3cr` is v3c plus noise-adaptive smoothing "
        "and retry on failure, in `palm_match_robust.py`.", "",
       "## 1. Test datasets", "",
       "| Dataset | Source | Licence | Content used | Protocol |", "|---|---|---|---|---|"]
for key, name, src, lic, what, proto in DATASETS:
    out.append(f"| {name} | {src} | {lic} | {what} | {proto} |")
out += ["", "Not tested:", ""] + [f"- **{n}:** {why}." for n, why in NOT_TESTED]

out += ["", "## 2. Accuracy, EER, FAR, FRR", "",
        "Accuracy = share of all comparisons decided correctly at the operating threshold. FAR and FRR include "
        "photos rejected by the quality gates (FTE). Rank-1 = identification accuracy against a gallery of all palms.", "",
        "| Dataset | Method | Genuine / impostor comparisons | Accuracy | EER | FAR | FRR | TAR at FAR 0.1% | TAR at FAR 0.01% | Rank-1 | FTE |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
for key, name, *_ in DATASETS:
    d = load(key)
    for m, label in METHODS:
        r = d.get(m)
        if not r or "error" in r or "EER" not in r:
            continue
        if key == "palmmatchdb" and "trusted" not in r.get("protocol", ""):
            continue                                    # raw-label rows are not meaningful for this dataset
        if key == "tongji" and m.startswith("v3") and not m.startswith("v3c") and r.get("threshold") != 26.0:
            continue
        ng, ni = r["genuine_comparisons"], r["impostor_comparisons"]
        acc = 1 - (r["FMR"] * ni + r["FNMR"] * ng) / (ng + ni)
        out.append(f"| {name} | {label} | {ng:,} / {ni:,} | {pct(acc)} | {pct(r['EER'])} | {pct(r['FAR'])} | {pct(r['FRR'])} | "
                   f"{pct(r.get('TAR_at_FMR_1e3'))} | {pct(r.get('TAR_at_FMR_1e4'))} | {pct(r.get('rank1'))} | {pct(r['fte_rate'])} |")
out += ["", "Notes:", "",
        "- The v3 threshold (26) was set using Tongji, so v3 FAR and FRR on Tongji are optimistic. EER does not depend on the threshold.",
        "- The v3c threshold (12) was set from PalmMatchDB and the phone photos only. v3cr uses the same threshold; "
        "a retry must reach 1.5 times it.",
        "- v3cr on Tongji uses 10,000 impostor comparisons, not 30,000, because every failed comparison is retried three times.",
        "- `n/a` means the test has too few impostor comparisons to measure that false-accept level.",
        "- The largest test has 30,000 impostor comparisons, which supports FAR claims down to about 0.01%, not the 0.001% project target.",
        "- Tongji and the Gabor dataset are pre-cropped palms, so they do not exercise hand detection and alignment. "
        "Only PalmMatchDB and the phone photos test the whole chain."]

rb = load("robustness_tongji")
if rb:
    n = next(iter(rb.values()))
    out += ["", "## 3. Robustness to capture conditions", "",
            f"Tongji, {n['genuine']} palms: clean enrolled photo, degraded probe photo, "
            f"{n['genuine']} genuine and {n['impostor']:,} impostor comparisons per condition, v3 with 1 enrolled photo.", "",
            "| Probe condition | FRR | FAR | EER | Median genuine score |", "|---|---|---|---|---|"]
    out += [f"| {k} | {pct(v['FRR'])} | {pct(v['FAR'])} | {pct(v['EER'])} | {v['genuine_median']:.0f} |" for k, v in rb.items()]
    for name, title in (("robustness_tongji_v3c", "Same test with the compact template (v3c):"),
                        ("robustness_tongji_v3cr", "Same test with v3cr (`palm_match_robust.py`: v3c plus noise-adaptive "
                         "smoothing and retry on failure). The additions were chosen on palms 300-499; this test uses palms 0-199:")):
        rc = load(name)
        if rc:
            out += ["", title, "", "| Probe condition | FRR | FAR | EER | Median genuine score |", "|---|---|---|---|---|"]
            out += [f"| {k} | {pct(v['FRR'])} | {pct(v['FAR'])} | {pct(v['EER'])} | {v['genuine_median']:.0f} |" for k, v in rc.items()]
    out += ["", "The degradations are simulated on the cropped image, not captured with a camera. "
            "Pixel noise added at this resolution breaks matching; real photos need to be tested for this."]

lat = os.path.join(R, "latency.json")
if os.path.exists(lat):
    L = json.load(open(lat))
    out += ["", "## 4. Inference latency", "", f"Measured on {L['machine']}. Median of repeated runs.", "",
            "| Stage | GPU | CPU |", "|---|---|---|"]
    out += [f"| {s} | {g} | {c} |" for s, g, c in L["rows"]]
    out += ["", L["note"]]

out += ["", "## 5. Spoof detection", "",
        "Not measured. The matcher has no liveness or anti-spoofing check; a printed photo of a palm may be accepted."]
out += ["", "## 6. Comparison with published results", "",
        "Published numbers are copied from the cited sources, not re-run here, and use different protocols "
        "(most train and test on Tongji itself). Ours is zero-shot: no palm image was used for training.", "",
        "| Dataset | Method | EER | Trained on palm images | Protocol | Source |", "|---|---|---|---|---|---|"]
for key, lname in OURS_FOR_LITERATURE:
    d = load(key)
    for m, label in METHODS:
        r = d.get(m)
        if m.startswith("v3") and r and "EER" in r and (key != "palmmatchdb" or "trusted" in r.get("protocol", "")):
            out.append(f"| {lname} | **ours: {label}** | {pct(r['EER'])} | no (zero-shot) | {r['protocol']} | this repository |")
    out += [f"| {a} | {b} | {c + '%' if c else '-'} | {d_ or '-'} | {e} | {f} |" for a, b, c, d_, e, f in LITERATURE if a == lname]
out += ["", "## 7. Files", "",
        "- `benchmark_results.csv`: every metric, one row per dataset and method",
        "- `requested_metrics.csv`: accuracy, EER, FAR, FRR, latency, robustness and spoof detection, v3 next to v3c",
        "- `robustness.csv`, `latency.csv`, `dataset_log.csv`, `literature_comparison.csv`",
        "- `benchmark_scores.csv`: the score of every single comparison",
        "- `plots/accuracy.png`, `plots/robustness.png`, `plots/size_speed.png`: summary charts (`benchmarks/plot_results.py`)",
        "- `logs/`: console output of each run; `*.json`: raw results; `*.png`: score histograms and DET curves",
        "", "Re-run v3 and v3c with `bash benchmarks/run_all.sh`, and v3cr with `bash benchmarks/run_robust.sh`."]
open(os.path.join(R, "REPORT.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
print("wrote", os.path.join(R, "REPORT.md"), len(out), "lines")
