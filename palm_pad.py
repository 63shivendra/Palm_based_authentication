#!/usr/bin/env python3
"""
palm_pad.py - presentation attack detection (spoof checks) for palm photos. No training data is used.

PASSIVE CHECK (one photo): recapture_score()
  Photographing a phone screen or a printed photo samples a regular pattern (screen pixel grid, printer halftone
  dots). The camera aliases it into narrow peaks in the 2-D spectrum (moire); live skin gives a smooth spectrum.
  Score = strength of the strongest narrow spectral peaks above the local spectrum level, in robust standard
  deviations, measured on the palm at the photo's native resolution (no resizing, which would destroy the pattern),
  in grey and in the red-blue colour difference (screen sub-pixels make coloured moire). Frequencies of the 8x8 JPEG
  block grid, the two axes and the band near the sensor's Nyquist limit are excluded.
  Threshold: set so that at most 1% of genuine photos from DEV identities are rejected (benchmarks/pad_eval.py).

ACTIVE CHECK (live camera): ActiveLiveness
  The user shows a random sequence of finger counts ("show 2 fingers", "show 4 fingers", ...) and finally the open
  hand, whose frame is the photo that is verified. Fingers are counted after mapping the landmarks into the palm's
  own frame (affine fit on 6 palm landmarks), so moving or tilting a flat printed photo does not change the count.
  Each next prompt appears after a random delay and showing it early fails the attempt, so a replayed video would
  need the right counts at timings it has never seen. Each step has a time limit.

  python palm_pad.py photo.jpg     # passive score of one photo
"""
import argparse
import random
import sys

import cv2
import numpy as np

import palm_match as pm

# ----------------------------------------------------------------------------- passive check
RECAPTURE_THRESHOLD = 3.01    # BPCER 1% point on 366 genuine photos of 11k Hands DEV identities (benchmarks/pad_eval.py)
BAND = (0.10, 0.45)           # cycles per pixel
MAX_WINDOW = 384              # pixels (native resolution) around the palm centre
MIN_WINDOW = 96
TOP_K = 6                     # spectral peaks averaged (a single moire tone gives one conjugate pair)


def palm_window(img, pts=None):
    """Square window around the palm centre at native resolution (no resizing)."""
    h, w = img.shape[:2]
    if pts is not None:
        c = pts[[0, 5, 9, 13, 17]].mean(0)
        half = int(np.clip(0.45 * np.linalg.norm(pts[17] - pts[5]), MIN_WINDOW // 2, MAX_WINDOW // 2))
    else:
        c = np.array([w / 2, h / 2])
        half = min(h, w, MAX_WINDOW) // 2
    x0 = int(np.clip(c[0] - half, 0, max(w - 2 * half, 0)))
    y0 = int(np.clip(c[1] - half, 0, max(h - 2 * half, 0)))
    return img[y0:y0 + 2 * half, x0:x0 + 2 * half]


def _peak_strength(ch, top_k=None):
    """Robust z-score of the strongest narrow peaks of one channel's spectrum."""
    n = min(ch.shape)
    n -= n % 2
    x = ch[:n, :n].astype(np.float32)
    x -= cv2.GaussianBlur(x, (0, 0), 6)                         # remove shading; keeps fine structure
    win = np.outer(np.hanning(n), np.hanning(n)).astype(np.float32)
    mag = np.log(np.abs(np.fft.fftshift(np.fft.fft2(x * win))) + 1e-3)
    resid = mag - cv2.medianBlur(mag.astype(np.float32), 5)     # peak above the local spectrum level
    f = (np.arange(n) - n // 2) / n
    fy, fx = np.meshgrid(f, f, indexing="ij")
    r = np.hypot(fx, fy)
    mask = (r >= BAND[0]) & (r <= BAND[1])
    mask &= (np.abs(fx) > 2.0 / n) & (np.abs(fy) > 2.0 / n)     # axes: leakage from the window edges
    for k in range(1, 4):                                       # JPEG 8x8 block grid: multiples of 1/8 cycle/pixel
        mask &= np.abs(np.abs(fx) - k / 8) > 1.5 / n
        mask &= np.abs(np.abs(fy) - k / 8) > 1.5 / n
    v = resid[mask]
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826 + 1e-6
    top = np.sort(v)[-(top_k or TOP_K):]                        # conjugate-symmetric: peaks come in pairs
    return float((top.mean() - med) / mad)


def recapture_score(bgr, pts=None):
    """Higher = more likely a photo of a screen or a print. Uses the palm area at native resolution."""
    w = palm_window(bgr, pts).astype(np.float32)
    grey = w @ np.float32([0.114, 0.587, 0.299])
    chroma = w[..., 2] - w[..., 0]
    return max(_peak_strength(grey), _peak_strength(chroma))


def passive_check(bgr, pts=None, threshold=RECAPTURE_THRESHOLD):
    s = recapture_score(bgr, pts)
    return dict(attack=s > threshold, score=s, threshold=threshold)


# ----------------------------------------------------------------------------- active check
PALM_IDS = [0, 1, 5, 9, 13, 17]                 # rigid part of the hand
PALM_CANON = pm.CANON[[0, 1, 3, 4, 5, 6]]       # the same landmarks in palm-width units (palm_match.CANON)
FINGER_EXTENDED = 0.45    # palm widths: a long finger counts as straight when its tip is this far beyond its knuckle
THUMB_EXTENDED = 0.70     # palm-frame x of the thumb tip (index knuckle 0.47, thumb base 0.97; open hands: 5th pct 0.79)
OPEN_MIN_DEG = 20.0       # final open hand: index-to-little-finger angle (genuine photos: see benchmarks/pad_eval.py)
STABLE_S = 0.25           # the requested pose must be held this long (and in at least 2 frames)
WRONG_S = 0.6             # a different finger count held this long fails the attempt (brief transitions are fine)
STEP_TIMEOUT_S = 5.0
HOLD_DELAY_S = (0.7, 2.0)  # random wait before the next prompt appears
PROMPTS = {1: "show 1 finger", 2: "show 2 fingers", 3: "show 3 fingers", 4: "show 4 fingers (fold the thumb)",
           5: "open your hand, fingers spread"}


def palm_frame(pts):
    """Landmarks in the palm's own frame (affine fit of the 6 palm landmarks to the average hand), right-hand layout."""
    p = pts.astype(np.float32).copy()
    v = p[9] - p[0]
    if (p[2] - p[9]) @ np.array([-v[1], v[0]]) < 0:             # left hand: mirror to the right-hand layout
        p[:, 0] = -p[:, 0]
    m, _ = cv2.estimateAffine2D(p[PALM_IDS], PALM_CANON, method=cv2.LMEDS)
    if m is None:
        return None
    return np.hstack([p, np.ones((len(p), 1), np.float32)]) @ m.T


def finger_spread(pts):
    """Angle (degrees) between the index finger and the little finger, in the palm's own frame."""
    q = palm_frame(pts)
    if q is None:
        return None
    a, b = q[8] - q[5], q[20] - q[17]
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def finger_count(pts):
    """Number of straight fingers (thumb included), measured in the palm's own frame; None if the fit fails."""
    q = palm_frame(pts)
    if q is None:
        return None
    n = sum(int(q[tip, 1] - q[mcp, 1] > FINGER_EXTENDED and q[tip, 1] > q[pip, 1])
            for mcp, pip, tip in ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)))
    return n + int(q[4, 0] > THUMB_EXTENDED)


def synthetic_hand(count=5, spread_deg=40.0, scale=180.0, centre=(320.0, 300.0)):
    """21 landmarks of an upright right hand in image pixels, for tests and attack simulations. count = straight
    fingers: 5 = all with the thumb out, 1-4 = that many long fingers (from the index) with the thumb folded."""
    p = np.zeros((21, 2), np.float32)
    for i, xy in zip(pm.CANON_IDS, pm.CANON):
        p[i] = xy
    if count == 5:
        p[3], p[4] = p[2] + (0.30, 0.25), p[2] + (0.55, 0.45)             # thumb out
    else:
        p[3], p[4] = p[2] + (-0.30, 0.10), np.float32([0.30, -0.50])      # thumb folded across the palm
    straight = 4 if count == 5 else count
    for k, (mcp, angle) in enumerate(zip((5, 9, 13, 17), np.linspace(spread_deg / 2, -spread_deg / 2, 4))):
        if k < straight:
            d = np.array([np.sin(np.radians(angle)), np.cos(np.radians(angle))], np.float32)
            for j, length in enumerate((0.45, 0.75, 0.95), 1):
                p[mcp + j] = p[mcp] + length * d
        else:                                                              # folded towards the palm
            p[mcp + 1], p[mcp + 2], p[mcp + 3] = p[mcp] + (0, 0.25), p[mcp] + (0, 0.05), p[mcp] + (0, -0.15)
    img = np.empty_like(p)
    img[:, 0] = centre[0] + scale * p[:, 0]
    img[:, 1] = centre[1] - scale * p[:, 1]                               # image y points down
    return img


def rigid_residual(pts_a, pts_b):
    """How much the fingertips moved relative to the palm between two frames (palm widths). A flat print gives ~0."""
    qa, qb = palm_frame(pts_a), palm_frame(pts_b)
    if qa is None or qb is None:
        return None
    return float(np.linalg.norm(qa[[8, 12, 16, 20]] - qb[[8, 12, 16, 20]], axis=1).max())


class ActiveLiveness:
    """Random challenge: call update(landmarks, time_s) once per camera frame until status is 'passed' or 'failed'.

    The user shows a random sequence of finger counts (1-4, never the same twice in a row), then an open hand (5
    fingers, spread); that last frame is the photo to verify, so the liveness check covers the matched photo itself.
    After each completed step the next prompt appears after a random delay. While waiting the pose must not change,
    and after a prompt the hand must go to the requested count without settling on another one. A print cannot
    change its finger count; a replayed video would have to show the right counts (36 orders for 4 steps) at random
    times it has never seen."""

    def __init__(self, n_steps=4, seed=None, timeout=STEP_TIMEOUT_S, delay=HOLD_DELAY_S):
        self.rng = random.Random(seed)
        self.steps = []
        while len(self.steps) < n_steps - 1:
            self.steps.append(self.rng.choice([c for c in (1, 2, 3, 4) if not self.steps or c != self.steps[-1]]))
        self.steps.append(5)
        self.timeout, self.delay = timeout, delay
        self.i, self.t0, self.release = 0, None, None             # release = time the next prompt appears
        self.prev = None                                         # count of the previous step (first: the starting pose)
        self.cur, self.cur_since, self.cur_frames = None, 0.0, 0  # current count and since when it is shown
        self.count, self.spread, self.first_pts, self.max_residual, self.final_pts = None, None, None, 0.0, None
        self.status, self.reason = "running", ""

    @property
    def prompt(self):
        if self.status != "running":
            return ""
        return "hold still" if self.release is not None else PROMPTS[self.steps[self.i]]

    def state(self):
        return dict(status=self.status, step=self.i, steps=len(self.steps), prompt=self.prompt, count=self.count,
                    spread=self.spread, reason=self.reason, sequence=list(self.steps), finger_motion=self.max_residual)

    def _fail(self, reason):
        self.status, self.reason = "failed", reason
        return self.state()

    def update(self, pts, t):
        if self.status != "running":
            return self.state()
        if self.t0 is None:
            self.t0 = t
        self.count = finger_count(pts) if pts is not None else None
        self.spread = finger_spread(pts) if self.count is not None else None
        if self.count is not None:
            if self.first_pts is None:
                self.first_pts = pts
            self.max_residual = max(self.max_residual, rigid_residual(self.first_pts, pts) or 0.0)
        if self.count != self.cur:
            self.cur, self.cur_since, self.cur_frames = self.count, t, 0
        self.cur_frames += 1
        held = t - self.cur_since
        if self.prev is None and self.cur is not None and held >= STABLE_S:
            self.prev = self.cur                                 # starting pose
        want = self.steps[self.i]
        if self.release is not None:                             # waiting for the next prompt: keep the pose
            if self.cur is not None and self.cur != self.prev and held >= WRONG_S:
                return self._fail("the hand changed before the next prompt appeared")
            if t >= self.release:
                self.release, self.t0 = None, t
            return self.state()
        if t - self.t0 > self.timeout:
            return self._fail(f"'{PROMPTS[want]}' not seen within {self.timeout:.0f} s")
        if self.cur is None:
            return self.state()
        if self.cur == want and held >= STABLE_S and self.cur_frames >= 2 and \
                (want != 5 or (self.spread is not None and self.spread >= OPEN_MIN_DEG)):
            self.prev = want
            self.i += 1
            if self.i == len(self.steps):
                self.status, self.reason, self.final_pts = "passed", "followed the random prompts", pts
            else:
                self.release = t + self.rng.uniform(*self.delay)
        elif self.cur not in (want, self.prev) and self.prev is not None and held >= WRONG_S:
            return self._fail(f"showed {self.cur} fingers, asked for {want}")
        return self.state()


def main():
    ap = argparse.ArgumentParser(description="passive spoof check of one palm photo")
    ap.add_argument("--no-detect", action="store_true")
    ap.add_argument("image")
    a = ap.parse_args()
    bgr = pm.load_bgr(a.image)
    pts = None if a.no_detect else pm.detect_landmarks(bgr)
    r = passive_check(bgr, pts)
    extra = f"  finger spread {finger_spread(pts):.0f} deg" if pts is not None else ""
    print(f"recapture score {r['score']:.1f} (threshold {r['threshold']:.1f}) -> "
          f"{'POSSIBLE SCREEN/PRINT' if r['attack'] else 'live'}{extra}")
    sys.exit(1 if r["attack"] else 0)


if __name__ == "__main__":
    main()
