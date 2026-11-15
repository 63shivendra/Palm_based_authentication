#!/usr/bin/env python3
"""
scripts/capture_pad_set.py - collect a small real spoof (presentation attack) test set with a webcam.

For each consenting person:
  1. live palm photos                 press  B   (5-10 photos, vary distance and light a little)
  2. printed photo of their palm      press  P   (hold the print in front of the camera)
  3. their palm photo on a phone/monitor  press  S
  4. the active challenge             press  1 (live hand), 2 (print), 3 (screen), 4 (replayed video), then L to start;
                                      follow the prompts with the live hand, or try to pass with the attack
  Q quits.

Photos go to data/pad/<bona_fide|print|screen>/<person>_<time>.jpg (data/ is not in git) and challenge attempts to
data/pad/active_log.csv. benchmarks/pad_eval.py then reports the real APCER / BPCER. Get written consent first, and
do not share the photos: they are biometric data.

  python scripts/capture_pad_set.py --person p01 [--camera 0]
"""
import argparse
import csv
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import cv2  # noqa: E402

import palm_match as pm  # noqa: E402
import palm_pad as pad  # noqa: E402

OUT = os.path.join(REPO, "data", "pad")
TYPES = {ord("1"): "bona_fide", ord("2"): "print", ord("3"): "screen", ord("4"): "video"}
SAVE = {ord("b"): "bona_fide", ord("p"): "print", ord("s"): "screen"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--person", required=True, help="anonymous id, e.g. p01 (no names)")
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        sys.exit(f"cannot open camera {a.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    kind, challenge, msg, t_start = "bona_fide", None, "B/P/S save a photo · 1-4 attempt type · L challenge · Q quit", 0.0
    counts = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        view = frame.copy()
        try:
            pts = pm.detect_landmarks(frame)
        except pm.QualityError:
            pts = None
        line2 = "no hand"
        if pts is not None:
            for x, y in pts.astype(int):
                cv2.circle(view, (int(x), int(y)), 3, (40, 200, 255), -1)
            sp, n = pad.finger_spread(pts), pad.finger_count(pts)
            rs = pad.recapture_score(frame, pts)
            line2 = (f"fingers {n}  spread {sp:.0f} deg   recapture score {rs:.1f} "
                     f"(limit {pad.RECAPTURE_THRESHOLD:.1f})")
        if challenge is not None:
            st = challenge.update(pts, time.time())
            if st["status"] == "running":
                cv2.putText(view, st["prompt"].upper(), (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)
            else:
                os.makedirs(OUT, exist_ok=True)
                log = os.path.join(OUT, "active_log.csv")
                new = not os.path.exists(log)
                with open(log, "a", newline="") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(["person", "presentation", "result", "reason", "seconds", "sequence", "finger_motion"])
                    w.writerow([a.person, kind, st["status"], st["reason"], f"{time.time() - t_start:.1f}",
                                "-".join(map(str, st["sequence"])), f"{st['finger_motion']:.3f}"])
                msg = f"challenge ({kind}): {st['status'].upper()} - {st['reason']}"
                challenge = None
        cv2.putText(view, f"person {a.person} · attempt type: {kind}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(view, line2, (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(view, msg, (30, view.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
        cv2.imshow("palm spoof test set", view)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k in SAVE:
            d = os.path.join(OUT, SAVE[k])
            os.makedirs(d, exist_ok=True)
            name = f"{a.person}_{time.strftime('%Y%m%d-%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.jpg"
            cv2.imwrite(os.path.join(d, name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            counts[SAVE[k]] = counts.get(SAVE[k], 0) + 1
            msg = f"saved {SAVE[k]} photo ({counts[SAVE[k]]} this session)"
        elif k in TYPES:
            kind = TYPES[k]
            msg = f"attempt type set to {kind}; press L to start the challenge"
        elif k == ord("l"):
            challenge, t_start = pad.ActiveLiveness(), time.time()
            msg = "challenge running"
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
