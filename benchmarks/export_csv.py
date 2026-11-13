#!/usr/bin/env python3
"""benchmarks/export_csv.py - write the benchmark results as CSV files in benchmarks/results/.

  python benchmarks/export_csv.py

  benchmark_results.csv      one row per dataset x method: every metric the evaluator measures
  requested_metrics.csv      the metrics asked for in the project brief (accuracy, EER, FAR, FRR, latency,
                             robustness, spoof detection), v3 next to v3c and v3cr (v3c + robustness additions)
  robustness.csv             FRR / FAR / EER per simulated capture condition
  latency.csv                time per stage
  dataset_log.csv            every dataset: source, licence, what was used, protocol, or why it was not tested
  literature_comparison.csv  published results on the same datasets next to ours
  benchmark_scores.csv       every single comparison score (dataset, method, genuine/impostor, score)
"""
import csv
import datetime
import json
import os

import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TODAY = datetime.date.today().isoformat()

DATASETS = [  # key, name, source, licence, what it is, protocol
    ("tongji", "Tongji Contactless Palmprint", "cslinzhang.github.io/ContactlessPalm (official ROI images)",
     "No licence stated on the official page; treated as research-only. Used for evaluation only.",
     "600 palms, 2 sessions about 2 months apart, 12,000 pre-cropped 128x128 images; 3 images per palm per session used",
     "enrol session 1, verify session 2"),
    ("palmmatchdb", "PalmMatchDB", "huggingface.co/datasets/aspmirlab/PalmMatchDB",
     "Apache-2.0 (Hugging Face tag). Consent terms not verified.",
     "phone photos of whole hands; 10,528 files but only 531 distinct images; labels unreliable",
     "all pairs, restricted to pairs with trustworthy labels (same palm: photos < 10 min apart; different palm: > 1 day apart)"),
    ("gabor", "Palmprint Image Dataset with Gabor Filter (raw images)", "data.mendeley.com/datasets/r8hxykxnk5 (version 2)",
     "CC BY 4.0", "15 people, 103 raw 128x128 palm crops", "all pairs"),
    ("real", "Own phone photos", "real palm images/ in this repository", "own data",
     "4 photos: 3 of one right hand (2 sessions), 1 of the left hand", "all pairs"),
]
NOT_TESTED = [
    ("CASIA Palmprint", "needs registration and a signed release agreement; official portal unreachable"),
    ("SCUT-PV (palm vein)", "needs a signed agreement sent to the lab; commercial use not allowed; near-infrared images"),
    ("GenPalm", "dataset announced in the paper, no public release found"),
    ("PCE-SynthPalm (Kaggle)", "no licence stated; project policy prohibits downloading it"),
]
METHODS = [("v3", "v3", 1), ("v3_enrol3", "v3", 3), ("v3_enrol2", "v3", 2),
           ("v3c", "v3c", 1), ("v3c_enrol3", "v3c", 3), ("v3c_enrol2", "v3c", 2),
           ("v3cr", "v3cr", 1), ("v3cr_enrol3", "v3cr", 3), ("v3cr_enrol2", "v3cr", 2),
           ("v2", "v2", 1), ("v2_enrol3", "v2", 3), ("v2_enrol2", "v2", 2), ("v1", "v1", 1)]

# Published results on the datasets we hold. EER in percent, as printed in the cited source.
# These numbers were collected from survey tables, not re-run by us; protocols differ from ours.
LITERATURE = [  # dataset, method, EER %, trained on palm images, protocol, source
    ("Tongji", "CCNet", "0.00004", "yes, on Tongji", "closed-set, train and test split of Tongji",
     "survey arXiv:2501.01166; arXiv:2504.11306"),
    ("Tongji", "BOCV", "0.0056", "no (hand-crafted)", "within Tongji", "survey tables found in the literature search"),
    ("Tongji", "CompNet", "0.018 to 0.025", "yes, on Tongji", "closed-set", "arXiv:2501.01166 (0.018); arXiv:2504.11306 (0.025)"),
    ("Tongji", "PalmNet", "0.03 to 0.17", "partly (Gabor/PCA filters tuned on palms)", "within Tongji",
     "survey tables found in the literature search"),
    ("Tongji", "DHPN", "0.066", "yes", "within Tongji", "survey tables found in the literature search"),
    ("Tongji", "DHN", "0.088", "yes", "within Tongji", "survey tables found in the literature search"),
    ("Tongji", "CompCode", "0.10 to 0.11", "no (hand-crafted)", "within Tongji", "arXiv:2504.11306 and survey tables"),
    ("Tongji", "CCNet, trained on PolyU", "2.22", "yes, on another dataset", "cross-dataset (PolyU to Tongji)", "survey arXiv:2501.01166"),
    ("Tongji", "PalmBridge, trained on PolyU", "2.25", "yes, on another dataset", "cross-dataset open-set", "arXiv:2601.20351"),
    ("Tongji", "PalmBridge, trained on IITD", "2.88", "yes, on another dataset", "cross-dataset open-set", "arXiv:2601.20351"),
    ("PalmMatchDB", "no published EER found", "", "",
     "The dataset paper (Alausa et al., ICPECA 2023, IEEE Xplore 10076097) is paywalled; its results could not be read. "
     "No other paper reporting on this dataset was found.", "web search, " + TODAY),
    ("Palmprint Image Dataset with Gabor Filter", "no published result found", "", "",
     "Mendeley Data record only; no paper reporting verification results on it was found.", "web search, " + TODAY),
]
OURS_FOR_LITERATURE = [("tongji", "Tongji"), ("palmmatchdb", "PalmMatchDB"), ("gabor", "Palmprint Image Dataset with Gabor Filter")]


def load(name):
    p = os.path.join(R, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else {}


def pct(x, nd=4):
    return "" if x is None else f"{x * 100:.{nd}f}"


def accuracy(r):
    ng, ni = r["genuine_comparisons"], r["impostor_comparisons"]
    return 1 - (r["FMR"] * ni + r["FNMR"] * ng) / (ng + ni)


def keep(key, m, r):
    """Same row filter as REPORT.md."""
    if not r or "error" in r or "EER" not in r:
        return False
    if key == "palmmatchdb" and "trusted" not in r.get("protocol", ""):
        return False                                    # raw-label rows are not meaningful for this dataset
    return True


def write(name, header, rows):
    with open(os.path.join(R, name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {name}: {len(rows)} rows")


def main():
    # 1. every metric, one row per dataset x method
    head = ["date", "dataset", "method", "enrolled_photos", "protocol", "images", "usable_images", "identities",
            "genuine_comparisons", "impostor_comparisons", "threshold", "accuracy_pct", "EER_pct", "FAR_pct", "FRR_pct",
            "FMR_pct", "FNMR_pct", "FTE_pct", "TAR_at_FAR_1pct", "TAR_at_FAR_0.1pct", "TAR_at_FAR_0.01pct",
            "rank1_pct", "dprime", "AUC", "genuine_min", "genuine_median", "impostor_median", "impostor_max", "run_seconds"]
    rows, wanted = [], []
    for key, name, *_ in DATASETS:
        d = load(key)
        for mk, method, n in METHODS:
            r = d.get(mk)
            if not keep(key, mk, r):
                continue
            rows.append([TODAY, name, method, n, r["protocol"], r["images"], r["usable"], r["identities"],
                         r["genuine_comparisons"], r["impostor_comparisons"], r["threshold"], pct(accuracy(r)),
                         pct(r["EER"]), pct(r["FAR"]), pct(r["FRR"]), pct(r["FMR"]), pct(r["FNMR"]), pct(r["fte_rate"]),
                         pct(r.get("TAR_at_FMR_1e2")), pct(r.get("TAR_at_FMR_1e3")), pct(r.get("TAR_at_FMR_1e4")),
                         pct(r.get("rank1")), f"{r['dprime']:.3f}" if r["dprime"] < 1e4 else "", f"{r['AUC']:.6f}",
                         r["genuine_min"], r["genuine_median"], r["impostor_median"], r["impostor_max"], r["seconds"]])
    write("benchmark_results.csv", head, rows)

    # 2. the metrics requested in the brief, v3 next to v3c
    for key, name, *_ in DATASETS:
        d = load(key)
        for n in (1, 2, 3):
            suffix = "" if n == 1 else f"_enrol{n}"
            a, b, c = d.get("v3" + suffix), d.get("v3c" + suffix), d.get("v3cr" + suffix)
            if not (keep(key, "v3", a) or keep(key, "v3c", b)):
                continue
            for label, fn in [("Accuracy %", accuracy), ("EER %", lambda r: r["EER"]), ("FAR %", lambda r: r["FAR"]),
                              ("FRR %", lambda r: r["FRR"])]:
                wanted.append([label, name, f"{n} enrolled photo(s)", pct(fn(a), 2) if keep(key, "v3", a) else "",
                               pct(fn(b), 2) if keep(key, "v3c", b) else "", pct(fn(c), 2) if keep(key, "v3cr", c) else "", ""])
    lat = load("latency")
    for stage, g, c in lat.get("rows", []):
        if stage.startswith("Total"):
            col = 4 if "v3c" in stage else 3
            for dev, val in (("GPU", g), ("CPU", c)):
                row = ["Inference latency", lat["machine"], f"one verification, {dev}", "", "", "not measured",
                       "not measured on a phone or terminal"]
                row[col] = val
                wanted.append(row)
    ra, rb, rc = load("robustness_tongji"), load("robustness_tongji_v3c"), load("robustness_tongji_v3cr")
    for cond in ra or rb:
        cell = lambda v: f"{pct(v['FRR'], 2)} ({pct(v['FAR'], 2)})" if v else ""
        wanted.append(["Robustness: FRR % (FAR %)", "Tongji, 200 palms, simulated", cond,
                       cell(ra.get(cond)), cell(rb.get(cond)), cell(rc.get(cond)), ""])
    wanted.append(["Spoof detection", "-", "-", "not measured", "not measured", "not measured",
                   "the matcher has no liveness check"])
    write("requested_metrics.csv", ["metric", "dataset", "condition", "v3", "v3c", "v3cr", "note"], wanted)

    # 3. robustness, 4. latency
    rows = [[m, cond, pct(v["FRR"], 2), pct(v["FAR"], 2), pct(v["EER"], 2), v["genuine_median"], v["impostor_max"],
             v["genuine"], v["impostor"]]
            for m, rep in (("v3", ra), ("v3c", rb), ("v3cr", rc)) for cond, v in rep.items()]
    write("robustness.csv", ["method", "probe_condition", "FRR_pct", "FAR_pct", "EER_pct", "genuine_median_score",
                             "impostor_max_score", "genuine_comparisons", "impostor_comparisons"], rows)
    write("latency.csv", ["stage", "GPU", "CPU", "machine"], [[*r, lat["machine"]] for r in lat.get("rows", [])])

    # 5. dataset log
    rows = [[name, "tested", src, lic, what, proto] for _, name, src, lic, what, proto in DATASETS]
    rows += [[n, "not tested", "", "", "", why] for n, why in NOT_TESTED]
    write("dataset_log.csv", ["dataset", "status", "source", "licence", "content_used", "protocol_or_reason"], rows)

    # 6. literature next to ours
    rows = []
    for key, lname in OURS_FOR_LITERATURE:
        d = load(key)
        for mk, method, n in METHODS:
            r = d.get(mk)
            if method in ("v3", "v3c", "v3cr") and keep(key, mk, r):
                rows.append([lname, f"OURS {method}, {n} enrolled photo(s)", pct(r["EER"], 2), "no (zero-shot)", r["protocol"],
                             "this repository, " + TODAY])
        rows += [list(x) for x in LITERATURE if x[0] == lname]
    write("literature_comparison.csv", ["dataset", "method", "EER_pct", "trained_on_palm_images", "protocol", "source"], rows)

    # 7. every comparison score
    rows = []
    for key, name, *_ in DATASETS:
        for mk, method, n in METHODS:
            p = os.path.join(R, f"{key}_{mk}_scores.npz")
            if method not in ("v3", "v3c", "v3cr") or not os.path.exists(p):
                continue
            z = np.load(p)
            for kind in ("genuine", "impostor"):
                rows += [[name, method, n, kind, f"{s:.4f}"] for s in z[kind]]
    write("benchmark_scores.csv", ["dataset", "method", "enrolled_photos", "pair_type", "score"], rows)


if __name__ == "__main__":
    main()
