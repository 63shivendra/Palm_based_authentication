#!/usr/bin/env python3
"""
benchmarks/embed_eval.py - fixed-length palm vectors (palm_embed.py) for 1:N search, and their protected form
(palm_protect.py, IoM hashing).

  --select   DEV identities only: HOL vs DISK-VLAD. The DISK-VLAD codebook for each dataset is fitted on the DEV
             photos of the OTHER datasets (no identity labels), so the comparison is not fitted to the data it scores.
  --fit      fit the DISK-VLAD codebook and projection on the DEV photos of all datasets -> models/palm_embed_disk.npz
  --test     TEST identities of Tongji (cross-session): identification with the chosen vector alone, two-stage search
             (vector shortlist, then v3c LightGlue re-ranking), protected (IoM) vector: accuracy, unlinkability (D_sys)
             and renewability. ISO/IEC 19795-1 identification terms: FNIR (mated searches that miss), FPIR
             (non-mated searches that return someone).

  python benchmarks/embed_eval.py --select
  python benchmarks/embed_eval.py --fit
  python benchmarks/embed_eval.py --test [--embedding hol|disk]
Output: benchmarks/results/embedding/<YYYYmmdd-HHMM>/ embed_report.md, embed_metrics.csv
"""
import argparse
import collections
import csv
import datetime
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_benchmark as RB  # noqa: E402

import numpy as np  # noqa: E402
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn  # noqa: E402

import palm_embed as pe  # noqa: E402
import palm_match as pm  # noqa: E402
import palm_match_compact as pmc  # noqa: E402
import palm_protect as pp  # noqa: E402
from common.crypto.metrics import compute_system_unlinkability  # noqa: E402

con = RB.con
DEV_SETS = ["tongji", "bjtu2", "11khands"]


def split(k, seed=0):
    samples = RB.plan_samples(k, RB.SIZES, seed)
    ids = sorted({s[0] for s in samples})
    sh = ids[:]
    random.Random(seed).shuffle(sh)
    n_dev = len(sh) // 5 if len(sh) >= 40 else 0
    return samples, sh[:n_dev], sh[n_dev:]                      # same order as tune_dev.py (shared template cache)


class Data:
    """Crops, v3c templates (disk-cached) and vectors of one dataset part."""

    def __init__(self, k, part, prog, cap=None):
        self.k = k
        samples, dev, test = split(k)
        keep = set((dev if part == "dev" else test)[:cap])
        self.samples = samples
        self.store = RB.Store(k, samples, RB.DATASETS[k][5])
        self.idx = [i for i, s in enumerate(samples) if s[0] in keep and
                    (not self.store.detect or os.path.exists(self.store._path("crop", i)))]
        self.ids = {i: samples[i][0] for i in self.idx}
        self.sess = {i: samples[i][1].split("/")[0] for i in self.idx}
        self.prog = prog
        self.tpl, self.vec = {}, collections.defaultdict(dict)

    def template(self, i):
        """v3c template (compact), cached on disk; the same file tune_dev.py writes for its 'raw' variant."""
        if i not in self.tpl:
            p = self.store._path("tune_raw", i)
            if os.path.exists(p):
                with np.load(p) as z:
                    t = (z["kp"], z["des"])
            else:
                t = pmc.features(self.store.get_crop(i))
                os.makedirs(os.path.dirname(p), exist_ok=True)
                np.savez(p, kp=t[0], des=t[1])
            self.tpl[i] = t
        return self.tpl[i]

    def load(self, methods, disk_path=pe.DISK_MODEL):
        t = self.prog.add_task(f"{self.k} · templates + vectors", total=len(self.idx))
        for i in self.idx:
            if "hol" in methods:
                self.vec["hol"][i] = pe.hol(self.store.get_crop(i))
            if "disk" in methods or "fit" in methods:
                tp = self.template(i)
                if "disk" in methods:
                    self.vec["disk"][i] = pe.disk_vlad(tp, disk_path)
            self.prog.advance(t)
        self.prog.remove_task(t)

    def gallery_probes(self, n_gallery=1):
        """Cross-session: gallery = first n session-1 photos per identity, probes = session-2 photos.
        All-vs-all datasets: gallery = first n photos, probes = the rest."""
        by = collections.defaultdict(list)
        for i in self.idx:
            by[self.ids[i]].append(i)
        gal, probes = {}, []
        cross = RB.DATASETS[self.k][3] == "cross-session"
        for ident, ix in by.items():
            g = [i for i in ix if self.sess[i] == "session1"] if cross else ix
            g = g[:n_gallery]
            rest = [i for i in ix if (self.sess[i] == "session2" if cross else i not in g)]
            if len(g) == n_gallery and rest:
                gal[ident] = g
                probes += rest
        return gal, probes


def rank_metrics(sim, probe_ids, gal_ids):
    """sim: P x G similarity matrix. Returns recall@1/5/10/20 and verification EER (probe vs every gallery entry)."""
    gal_ids = np.asarray(gal_ids)
    order = np.argsort(-sim, 1)
    true = np.array([np.where(gal_ids == p)[0][0] for p in probe_ids])
    rank = np.array([np.where(order[r] == true[r])[0][0] for r in range(len(true))])
    gen = sim[np.arange(len(true)), true]
    mask = np.ones_like(sim, bool)
    mask[np.arange(len(true)), true] = False
    e, _ = RB.eer(gen, sim[mask])
    return {f"recall@{k}": float(np.mean(rank < k)) for k in (1, 5, 10, 20)} | dict(EER=e, gallery=len(gal_ids), probes=len(true))


def vectors_matrix(data, method, gal, probes, reducer=np.max):
    g_ids = list(gal)
    G = {ident: np.stack([data.vec[method][i] for i in gal[ident]]) for ident in g_ids}
    P = np.stack([data.vec[method][i] for i in probes])
    sim = np.stack([reducer(P @ G[ident].T, 1) for ident in g_ids], 1)
    return sim, [data.ids[i] for i in probes], g_ids


def pct(v):
    return f"{v * 100:.2f}%"


def progress():
    return Progress(TextColumn("[bold]{task.description}"), BarColumn(bar_width=36), MofNCompleteColumn(),
                    TimeElapsedColumn(), TextColumn("eta"), TimeRemainingColumn(), console=con)


def out_dir():
    d = os.path.join(RB.REPO, "benchmarks", "results", "embedding", datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    os.makedirs(d, exist_ok=True)
    return d


# ----------------------------------------------------------------------------- DEV selection
def select(a):
    rows = []
    with progress() as prog:
        data = {k: Data(k, "dev", prog, cap=a.cap) for k in DEV_SETS}
        for k, d in data.items():
            d.load(["hol", "fit"])
        for k, d in data.items():
            others = [(tp, o) for kk, o in data.items() if kk != k for tp in (o.template(i) for i in o.idx)]
            path = os.path.join(RB.CACHE, f"embed_select_without_{k}.npz")
            pe.fit_disk([pmc.unpack(tp) for tp, _ in others], path)
            for i in d.idx:
                d.vec["disk"][i] = pe.disk_vlad(d.template(i), path)
            gal, probes = d.gallery_probes()
            for method in ("hol", "disk"):
                sim, pid, gid = vectors_matrix(d, method, gal, probes)
                r = rank_metrics(sim, pid, gid)
                rows.append(dict(dataset=k, method=method, **r))
                con.print(f"  DEV {k:9s} {method:5s} rank-1 {pct(r['recall@1'])}  recall@10 {pct(r['recall@10'])}  "
                          f"EER {pct(r['EER'])}  ({r['gallery']} in gallery, {r['probes']} searches)")
            sim = np.mean([vectors_matrix(d, m, gal, probes)[0] for m in ("hol", "disk")], 0)
            r = rank_metrics(sim, pid, gid)
            rows.append(dict(dataset=k, method="hol+disk", **r))
            con.print(f"  DEV {k:9s} hol+disk rank-1 {pct(r['recall@1'])}  recall@10 {pct(r['recall@10'])}  EER {pct(r['EER'])}")
    d = out_dir()
    write_rows(d, "embed_select.csv", rows)
    md = ["# Palm vector selection (DEV identities only)", "",
          f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} by `benchmarks/embed_eval.py --select`. Gallery = one photo per "
          "identity (session 1 where there are sessions); every other photo (session 2) is searched against it. DISK-VLAD is "
          "fitted on the other datasets' DEV photos (no identity labels).", "",
          "| Dataset | Vector | Gallery | Searches | Rank-1 | Recall@5 | Recall@10 | Recall@20 | EER |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['dataset']} | {r['method']} | {r['gallery']} | {r['probes']} | {pct(r['recall@1'])} | {pct(r['recall@5'])} | "
                  f"{pct(r['recall@10'])} | {pct(r['recall@20'])} | {pct(r['EER'])} |")
    open(os.path.join(d, "embed_report.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    con.print(f"[green]written[/] {os.path.relpath(d, RB.REPO)}/embed_report.md")


def fit(a):
    with progress() as prog:
        tpls = []
        for k in DEV_SETS:
            d = Data(k, "dev", prog)
            d.load(["fit"])
            tpls += [pmc.unpack(d.template(i)) for i in d.idx]
    path = pe.fit_disk(tpls)
    con.print(f"[green]fitted[/] DISK-VLAD on {len(tpls)} DEV photos (no identity labels) -> {os.path.relpath(path, RB.REPO)}")


# ----------------------------------------------------------------------------- TEST report
def test(a):
    rows, md = [], []
    method = a.embedding
    keys = ["key-A (bank 1)", "key-B (bank 2)"]
    with progress() as prog:
        d = Data("tongji", "test", prog)
        d.load([method])
        gal_all, probes_all = d.gallery_probes(n_gallery=a.n_gallery)
        ids = sorted(gal_all)
        random.Random(0).shuffle(ids)
        enrolled = set(ids[: len(ids) // 2])                          # the other half never enrols: non-mated searches
        gal = {k: v for k, v in gal_all.items() if k in enrolled}
        mated = [p for p in probes_all if d.ids[p] in enrolled]
        nonmated = [p for p in probes_all if d.ids[p] not in enrolled]
        rng = random.Random(1)
        mated = rng.sample(mated, min(len(mated), a.searches))
        nonmated = rng.sample(nonmated, min(len(nonmated), a.searches))
        con.print(f"Tongji TEST: {len(gal)} enrolled palms ({a.n_gallery} photo(s) each, session 1), "
                  f"{len(mated)} mated and {len(nonmated)} non-mated searches (session 2)")

        # 1. vector alone (closed set over the enrolled gallery)
        sim, pid, gid = vectors_matrix(d, method, gal, mated)
        r = rank_metrics(sim, pid, gid)
        rows.append(dict(test="vector only, unprotected", **r))
        # 2. protected vector (IoM, key A)
        P = {i: pp.protect(d.vec[method][i], keys[0]) for i in set(mated) | set(nonmated) | {x for v in gal.values() for x in v}}
        simp = np.stack([np.max(np.stack([[pp.similarity(P[p], P[g]) for g in gal[ident]] for p in mated]), 1)
                         for ident in gid], 1)
        rp = rank_metrics(simp, pid, gid)
        rows.append(dict(test="vector only, protected (IoM)", **rp))

        # 3. unlinkability and renewability: same palm under key A vs key B
        firsts = [v[0] for v in gal_all.values()]
        sec = {d.ids[p]: p for p in probes_all}
        pairs = [(g, sec[d.ids[g]]) for g in firsts if d.ids[g] in sec]
        HA = {i: pp.protect(d.vec[method][i], keys[0]) for pr in pairs for i in pr}
        HB = {i: pp.protect(d.vec[method][i], keys[1]) for pr in pairs for i in pr}
        mated_ab = np.array([pp.similarity(HA[g], HB[p]) for g, p in pairs])
        non_ab = np.array([pp.similarity(HA[g], HB[p2]) for (g, _), (_, p2) in zip(pairs, pairs[1:] + pairs[:1])])
        same_key = np.array([pp.similarity(HA[g], HA[p]) for g, p in pairs])
        dsys = compute_system_unlinkability(1 - mated_ab, 1 - non_ab)[0]

        # 4. two-stage search: shortlist by vector (protected), re-rank with v3c LightGlue on the shortlist
        sims = {}
        n = (len(mated) + len(nonmated)) * a.shortlist
        t = prog.add_task(f"two-stage search · LightGlue re-ranking of top {a.shortlist}", total=n)
        tf = {i: pmc.unpack(d.template(i)) for i in set(mated) | set(nonmated) | {x for v in gal.values() for x in v}}
        best = {}
        for p in mated + nonmated:
            vs = np.array([max(pp.similarity(P[p], P[g]) for g in gal[ident]) for ident in gid])
            short = [gid[j] for j in np.argsort(-vs)[: a.shortlist]]
            scores = {}
            for ident in short:
                scores[ident] = max(pm.match_v3(tf[g], tf[p])["score"] for g in gal[ident])
                prog.advance(t)
            top = max(scores, key=scores.get)
            best[p] = (top, scores[top], d.ids[p] in short)
            sims[p] = scores
        prog.remove_task(t)
    thr = float(pmc.THRESHOLD)
    fnir = np.mean([not (best[p][0] == d.ids[p] and best[p][1] >= thr) for p in mated])
    fpir = np.mean([best[p][1] >= thr for p in nonmated])
    recall_short = np.mean([best[p][2] for p in mated])
    rows.append(dict(test=f"two-stage: protected vector top-{a.shortlist}, then LightGlue (threshold {thr:.0f})",
                     FNIR=fnir, FPIR=fpir, shortlist_recall=recall_short, gallery=len(gal), probes=len(mated) + len(nonmated)))
    o = out_dir()
    write_rows(o, "embed_metrics.csv", rows)
    md = ["# 1:N search and template protection (TEST identities)", "",
          f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} by `benchmarks/embed_eval.py --test`. Tongji TEST identities "
          f"(never used for any choice). Vector: **{method}**. {len(gal)} palms enrolled with {a.n_gallery} session-1 photo(s); "
          f"searches use session-2 photos about 2 months later. Half of the TEST identities never enrol and give the non-mated searches.", "",
          "## Identification", "", "| Search | Rank-1 | Recall@10 | Verification EER |", "|---|---|---|---|",
          f"| vector only, unprotected | {pct(r['recall@1'])} | {pct(r['recall@10'])} | {pct(r['EER'])} |",
          f"| vector only, protected (IoM, {pp.IOM_K} x {pp.IOM_M}) | {pct(rp['recall@1'])} | {pct(rp['recall@10'])} | {pct(rp['EER'])} |", "",
          f"**Two-stage search** (protected vector picks the top {a.shortlist}, the v3c matcher decides at threshold {thr:.0f}): "
          f"FNIR **{pct(fnir)}** ({len(mated)} mated searches; the true palm was in the shortlist for {pct(recall_short)}), "
          f"FPIR **{pct(fpir)}** ({len(nonmated)} non-mated searches). "
          f"Cost per search: {a.shortlist} LightGlue comparisons instead of {len(gal)}.", "",
          "## Template protection (ISO/IEC 24745)", "",
          f"- Protected search template: {pp.IOM_K} IoM indices = {pp.IOM_K} bytes; the raw vector is not stored.",
          f"- Same palm, same key: median {np.median(same_key):.3f} equal indices (chance level {1 / pp.IOM_M:.3f}).",
          f"- **Renewability**: same palm under a new key: median {np.median(mated_ab):.3f}, the level of a different palm "
          f"(median {np.median(non_ab):.3f}). A revoked template stops matching.",
          f"- **Unlinkability**: D_sys = **{dsys:.3f}** between key A and key B templates (0 = fully unlinkable, 1 = fully "
          f"linkable; {len(mated_ab)} mated / {len(non_ab)} non-mated pairs).",
          "- **Irreversibility** is not proven: IoM discards magnitudes, but published attacks can approximate a vector from hashes "
          "when the key is stolen too. Keys belong in a hardware keystore.",
          "- The keypoint template used by the matcher is encrypted at rest (palm_protect.py: Windows DPAPI or AES-256-GCM) and "
          "decrypted only in memory at match time."]
    open(os.path.join(o, "embed_report.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    con.print("\n".join(md))
    con.print(f"[green]written[/] {os.path.relpath(o, RB.REPO)}")


def write_rows(d, name, rows):
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("dataset", "method", "test"), k))
    with open(os.path.join(d, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--select", action="store_true")
    g.add_argument("--fit", action="store_true")
    g.add_argument("--test", action="store_true")
    ap.add_argument("--embedding", choices=("hol", "disk"), default="disk")
    ap.add_argument("--cap", type=int, default=None, help="--select: DEV identities per dataset")
    ap.add_argument("--n-gallery", type=int, default=1)
    ap.add_argument("--searches", type=int, default=150, help="--test: mated and non-mated searches each")
    ap.add_argument("--shortlist", type=int, default=5)
    a = ap.parse_args()
    pm.features_v3(np.zeros((pm.BIG, pm.BIG), np.uint8))
    (select if a.select else fit if a.fit else test)(a)


if __name__ == "__main__":
    main()
