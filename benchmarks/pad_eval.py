#!/usr/bin/env python3
"""
benchmarks/pad_eval.py - spoof (presentation attack) detection metrics for the passive check in palm_pad.py,
reported in ISO/IEC 30107-3 terms.

  APCER  attack presentations wrongly accepted as live, per attack type (species)
  BPCER  live (bona fide) presentations wrongly rejected as attacks
  D-EER  error rate where APCER = BPCER; BPCER20 / BPCER10 = BPCER at APCER 5% / 10%

Data
  bona fide   whole-hand photos at native resolution: 11k Hands (DEV identities set the threshold, TEST identities
              measure BPCER), PalmMatchDB and our own photos (TEST only)
  simulated   screen replay and printed photo, rendered from TEST bona fide photos: the image is drawn as a
              screen's RGB sub-pixel grid or as a printer's halftone dots, then "photographed" (optical blur, slight
              rotation, resampling to the camera grid, noise, JPEG). SIMULATED attacks only show that the check reacts
              to the pattern; they do not replace real attack photos
  real        if data/pad/ exists: data/pad/bona_fide/*.jpg and data/pad/<attack type>/*.jpg (e.g. print, screen),
              captured with scripts/capture_pad_set.py. These are the numbers to quote

  python benchmarks/pad_eval.py
Output: benchmarks/results/pad/<YYYYmmdd-HHMM>/ pad_report.md, pad_scores.csv, pad_scores.png
"""
import csv
import datetime
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_benchmark as RB  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn  # noqa: E402

import palm_match as pm  # noqa: E402
import palm_pad as pad  # noqa: E402

con = RB.con


# ----------------------------------------------------------------------------- simulated attacks
def _camera(canvas, n, scale, rng):
    """Photograph a high-resolution canvas: rotate, optical blur, resample to n x n camera pixels, noise, JPEG."""
    h, w = canvas.shape[:2]
    canvas = cv2.GaussianBlur(canvas, (0, 0), rng.uniform(0.35, 0.8) / scale)   # optical blur: camera px -> canvas px
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-8, 8), scale)     # scale = camera px per canvas px
    m[:, 2] += np.array([n / 2 - w / 2, n / 2 - h / 2])                         # canvas centre -> image centre
    img = cv2.warpAffine(canvas, m, (n, n), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    img = img * rng.uniform(0.85, 1.15) + rng.normal(0, 2.0, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(80, 95))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def sim_screen(win, rng, up=9):
    """Phone/monitor replay: each screen pixel = R, G, B vertical stripes with dark gaps; pitch 1.3-3.5 camera pixels."""
    n = min(win.shape[:2])
    pitch = rng.uniform(1.3, 3.5)
    k = max(int(round(n / pitch)), 8)
    shown = cv2.resize(win[:n, :n], (k, k), interpolation=cv2.INTER_AREA).astype(np.float32)
    big = np.repeat(np.repeat(shown, up, 0), up, 1)
    mask = np.zeros((up, up, 3), np.float32)
    w = up // 3
    for c, ch in enumerate((2, 1, 0)):                          # R, G, B stripes (BGR channel order), 1 px gaps
        mask[: up - 1, c * w: c * w + w - 1, ch] = 1.0
    canvas = big * np.tile(mask / mask.mean((0, 1)), (k, k, 1))   # same mean brightness as the original
    return _camera(canvas, n, n / (k * up), rng)          # no clipping before the optical blur


def sim_print(win, rng, up=6):
    """Printed photo: cyan/magenta/yellow halftone dots at screen angles 15/75/0 deg; dot pitch 1.2-3.0 camera pixels."""
    n = min(win.shape[:2])
    pitch = rng.uniform(1.2, 3.0) * up
    big = cv2.resize(win[:n, :n], (n * up, n * up), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
    yy, xx = np.mgrid[0: n * up, 0: n * up].astype(np.float32)
    out = np.ones_like(big)
    for ch, angle in ((2, 15), (1, 75), (0, 0)):                # cyan absorbs red, magenta green, yellow blue
        t = np.radians(angle)
        u, v = xx * np.cos(t) + yy * np.sin(t), -xx * np.sin(t) + yy * np.cos(t)
        cell = (np.cos(2 * np.pi * u / pitch) + np.cos(2 * np.pi * v / pitch) + 2) / 4   # 0..1 dot profile
        ink = 1 - big[..., ch]
        out[..., ch] = 1 - 0.92 * (ink > cell)
    canvas = (0.08 + 0.85 * out) * 255                         # paper white and ink black are not 255 / 0
    return _camera(canvas, n, 1 / up, rng)


def simulate_replay(n=300, fps=15):
    """Share of active-check attempts a replayed video passes (synthetic landmark sequences)."""
    hands = {c: pad.synthetic_hand(c) for c in range(1, 6)}
    out = {}
    plans = [(f"cycles 1-2-3-4-5, {p} s per pose", lambda r, p=p: [1 + i % 5 for i in range(80)], p) for p in (0.8, 1.5, 2.5, 3.5)]
    plans.append(("random order, 1.5 s per pose", lambda r: _random_counts(r), 1.5))
    for name, seq_fn, period in plans:
        passed = 0
        for k in range(n):
            ch = pad.ActiveLiveness(seed=10_000 + k)
            seq = seq_fn(random.Random(k))
            for f in range(int(60 * fps)):
                t = f / fps
                st = ch.update(hands[seq[int(t / period)]], t)
                if st["status"] != "running":
                    break
            passed += ch.status == "passed"
        out[name] = passed / n
    return out


def _random_counts(r):
    s = [5]
    while len(s) < 80:
        s.append(r.choice([c for c in range(1, 6) if c != s[-1]]))
    return s


# ----------------------------------------------------------------------------- data
def bona_fide(k, part):
    """(name, bgr, pts) of a whole-hand dataset's cached photos; part = 'dev', 'test' or 'all'."""
    samples = RB.plan_samples(k, RB.SIZES, 0)
    if not samples:
        return []
    ids = sorted({s[0] for s in samples})
    sh = ids[:]
    random.Random(0).shuffle(sh)
    dev = set(sh[: len(sh) // 5]) if len(sh) >= 40 else set()
    store = RB.Store(k, samples, True)
    out = []
    for i, s in enumerate(samples):
        if (part == "dev") != (s[0] in dev) and part != "all":
            continue
        p = store._path("crop", i)
        if os.path.exists(p):
            out.append((f"{k}/{s[1]}", s[2], np.load(p)["pts"]))
    return out


def load(img):
    return RB.E.load_any(img) if isinstance(img, str) else img


def metrics(bona, attacks, thr):
    """bona: scores of live photos; attacks: {species: scores}. Higher score = more attack-like."""
    bona = np.asarray(bona)
    out = dict(BPCER=float(np.mean(bona > thr)), bona_fide=len(bona))
    allatk = np.concatenate([np.asarray(v) for v in attacks.values()]) if attacks else np.zeros(0)
    for sp, v in attacks.items():
        out[f"APCER_{sp}"] = float(np.mean(np.asarray(v) <= thr))
        out[f"n_{sp}"] = len(v)
    if len(allatk):
        ts = np.unique(np.concatenate([bona, allatk]))
        apcer = np.array([np.mean(allatk <= t) for t in ts])
        bpcer = np.array([np.mean(bona > t) for t in ts])
        i = int(np.argmin(np.abs(apcer - bpcer)))
        out["D_EER"] = float((apcer[i] + bpcer[i]) / 2)
        for name, target in (("BPCER20", 0.05), ("BPCER10", 0.10)):
            ok = apcer <= target
            out[name] = float(bpcer[ok].min()) if ok.any() else 1.0
    return out


def pct(v):
    return f"{v * 100:.2f}%"


def main():
    out_dir = os.path.join(RB.REPO, "benchmarks", "results", "pad", datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    prog = Progress(TextColumn("[bold]{task.description}"), BarColumn(bar_width=36), MofNCompleteColumn(),
                    TimeElapsedColumn(), console=con)
    rng = np.random.RandomState(0)
    with prog:
        def score_set(items, label, species, attack=None, cap=None):
            items = items[:cap] if cap else items
            t = prog.add_task(f"{label} · {species}", total=len(items))
            res = []
            for name, img, pts in items:
                bgr = load(img)
                if attack is not None:                  # the attack image is the palm window, re-photographed
                    s = pad.recapture_score(attack(pad.palm_window(bgr, pts), rng), None)
                else:
                    s = pad.recapture_score(bgr, pts)
                res.append(s)
                rows.append(dict(set=label, species=species, photo=name, score=s))
                prog.advance(t)
            prog.remove_task(t)
            return np.array(res)

        dev = bona_fide("11khands", "dev")
        test = bona_fide("11khands", "test") + bona_fide("palmmatchdb", "all") + bona_fide("real", "all")
        con.print(f"bona fide photos: {len(dev)} DEV (threshold), {len(test)} TEST")
        dev_s = score_set(dev, "dev", "bona_fide")
        thr = float(np.percentile(dev_s, 99))
        con.print(f"threshold (99th percentile of DEV bona fide scores, BPCER 1% on DEV): [bold]{thr:.2f}[/]")
        test_s = score_set(test, "test", "bona_fide")
        random.Random(1).shuffle(test)
        sim = {"screen (simulated)": score_set(test, "test", "screen (simulated)", sim_screen, cap=300),
               "print (simulated)": score_set(test, "test", "print (simulated)", sim_print, cap=300)}
        spread = np.array([s for s in (pad.finger_spread(p) for _, _, p in dev + test) if s is not None])
        counts = np.array([c for c in (pad.finger_count(p) for _, _, p in dev + test) if c is not None])
        replay = simulate_replay()

        real = {}
        root = os.path.join(RB.DATA, "pad")
        if os.path.isdir(root):
            for sp in sorted(os.listdir(root)):
                d = os.path.join(root, sp)
                if not os.path.isdir(d):
                    continue
                items = []
                for f in sorted(os.listdir(d)):
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        bgr = cv2.imread(os.path.join(d, f))
                        try:
                            pts = pm.detect_landmarks(bgr)
                        except pm.QualityError:
                            pts = None
                        items.append((f"pad/{sp}/{f}", bgr, pts))
                real[sp] = score_set(items, "real", sp)

    m_sim = metrics(test_s, sim, thr)
    md = ["# Spoof detection (passive recapture check)", "",
          f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} by `benchmarks/pad_eval.py`. Metrics follow ISO/IEC 30107-3.", "",
          f"Threshold **{thr:.2f}**: the 99th percentile of the scores of {len(dev)} genuine photos of 11k Hands DEV "
          "identities (BPCER 1% on DEV). Everything below is measured on other photos.", "",
          "## Genuine photos (BPCER)", "", "| Set | Photos | BPCER |", "|---|---|---|"]
    for name in ("11khands", "palmmatchdb", "real"):
        v = np.array([r["score"] for r in rows if r["set"] == "test" and r["species"] == "bona_fide" and r["photo"].startswith(name + "/")])
        if len(v):
            md.append(f"| {name} (TEST) | {len(v)} | {pct(np.mean(v > thr))} |")
    md.append(f"| all TEST genuine photos | {len(test_s)} | {pct(m_sim['BPCER'])} |")
    md += ["", "## Simulated attacks (APCER)", "",
           "Rendered from TEST genuine photos. They show that the check reacts to screen and print patterns. **They do not "
           "replace real attack photos**, and real attacks may be harder (higher resolution screens, photo paper, "
           "blur hiding the pattern).", "",
           "| Attack | Presentations | APCER (accepted as live) |", "|---|---|---|"]
    for sp, v in sim.items():
        md.append(f"| {sp} | {len(v)} | {pct(m_sim[f'APCER_{sp}'])} |")
    md += ["", f"D-EER {pct(m_sim['D_EER'])}, BPCER20 {pct(m_sim['BPCER20'])}, BPCER10 {pct(m_sim['BPCER10'])} (simulated attacks).", ""]
    if real:
        bona_real = real.pop("bona_fide", np.zeros(0))
        md += ["## Real attack photos (data/pad/)", "", "| Type | Presentations | Error |", "|---|---|---|"]
        if len(bona_real):
            md.append(f"| bona fide | {len(bona_real)} | BPCER {pct(np.mean(bona_real > thr))} |")
        for sp, v in real.items():
            md.append(f"| {sp} | {len(v)} | APCER {pct(np.mean(v <= thr))} |")
    else:
        md += ["## Real attack photos", "", "Not collected yet. Capture them with `python scripts/capture_pad_set.py` "
               "(printed palm photos and palm photos shown on a phone or monitor, plus live photos of the same people), "
               "then run this script again.", ""]
    md += ["", "## Active check (random finger counts)", "",
           "**Finger counting on real hands.** The active check counts straight fingers from the MediaPipe landmarks. On "
           f"{len(counts)} genuine open-hand photos (11k Hands, PalmMatchDB, own photos) it counted 5 fingers for "
           f"{pct(np.mean(counts == 5))} and 4 for {pct(np.mean(counts == 4))} (thumb held close to the hand). "
           f"Finger spread on these photos: median {np.median(spread):.1f} deg; the final step asks for >= "
           f"{pad.OPEN_MIN_DEG:.0f} deg.", "",
           "**Replayed video (simulated landmark sequences, 300 attempts each).** A recording that shows finger counts in "
           "a fixed or random order, at 15 frames per second:", "", "| Replayed video | Accepted (APCER) |", "|---|---|"]
    for name, rate in replay.items():
        md.append(f"| {name} | {pct(rate)} |")
    md += ["", "The best possible replay has to guess each random step (1 in 3 per step, 1 in 27 for three steps, 3.7%). "
           "Lock the account for a while after 3 failed attempts. A printed photo never changes its finger count and "
           "failed every simulated attempt (tests/test_palm_pad.py)."]
    log_path = os.path.join(RB.DATA, "pad", "active_log.csv")
    if os.path.exists(log_path):
        att = list(csv.DictReader(open(log_path, encoding="utf-8")))
        md += ["", "**Real challenge attempts (data/pad/active_log.csv)**", "", "| Presentation | Attempts | Passed |", "|---|---|---|"]
        for kind in sorted({r["presentation"] for r in att}):
            rs = [r for r in att if r["presentation"] == kind]
            md.append(f"| {kind} | {len(rs)} | {pct(np.mean([r['result'] == 'passed' for r in rs]))} "
                      f"({'BPCER = 1 - this' if kind == 'bona_fide' else 'APCER'}) |")
    open(os.path.join(out_dir, "pad_report.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    with open(os.path.join(out_dir, "pad_scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["set", "species", "photo", "score"])
        w.writeheader()
        w.writerows(rows)
    plot(out_dir, test_s, sim, thr)
    con.print("\n".join(md))
    con.print(f"[green]written[/] {os.path.relpath(out_dir, RB.REPO)}")


def plot(out_dir, bona, attacks, thr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=150)
    hi = max([bona.max()] + [v.max() for v in attacks.values()])
    bins = np.linspace(0, hi * 1.02, 60)
    ax.hist(bona, bins=bins, color="#2a78d6", alpha=0.85, label=f"genuine photos (n={len(bona)})")
    for (sp, v), c in zip(attacks.items(), ("#eb6834", "#eda100")):
        ax.hist(v, bins=bins, color=c, alpha=0.75, label=f"{sp} (n={len(v)})")
    ax.axvline(thr, color="#333333", lw=1.2, ls="--")
    ax.text(thr, ax.get_ylim()[1] * 0.95, f"  threshold {thr:.1f}", va="top", fontsize=8, color="#333333")
    ax.set_xlabel("recapture score (higher = more like a screen or print)")
    ax.set_ylabel("photos")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pad_scores.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
