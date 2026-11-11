#!/usr/bin/env python3
"""
palm_match.py - simple 1:1 palm verification. No training, no dataset needed to build it.

Pipeline
  1. Find the hand with MediaPipe Hand Landmarker (21 landmarks).
  2. Quality gates: hand found, hand not too far / too close, image not blurry.
  3. Align: rotate the hand upright (wrist -> middle knuckle), scale by palm width
     (index knuckle <-> little knuckle), crop the palm centre, resize to 128x128.
  4. "Competitive Code": 6 Gabor line filters, keep the strongest direction per pixel.
  5. Match: agreement of the two direction maps, best over small shifts/rotations.

Commands
  python palm_match.py enroll   alice.npz  a1.jpg a2.jpg a3.jpg   # 1-5 photos of the SAME palm
  python palm_match.py verify   alice.npz  new_photo.jpg          # prints distance + MATCH / NO MATCH
  python palm_match.py evaluate photos/                           # photos/<person>/<image>.jpg -> EER
  python palm_match.py roi      photo.jpg  roi.png                # save the aligned palm crop (debugging)
  Add --no-detect if your images are already cropped palm close-ups.

Install
  pip install opencv-python numpy mediapipe
  On first run the MediaPipe model (hand_landmarker.task, ~7.5 MB) is downloaded next to this file.

IMPORTANT LIMITS - read before using
  * Accuracy on real photos is UNKNOWN until you run `evaluate` on real photos of consenting people.
    DEFAULT_THRESHOLD is a placeholder chosen on synthetic images. Do not quote any accuracy before that.
  * Use the same hand (e.g. right palm) for enrollment and verification, palm facing the camera.
  * No liveness / anti-spoofing. A good printed photo of a palm may be accepted.
  * MediaPipe's model card lists the model as experimental and identity recognition / henna as out of
    scope. It is used here only to locate the hand. Get legal review before commercial use.
  * Templates are raw palm-line maps = biometric data. Encrypt / protect them before real storage.
"""
import argparse
import glob
import itertools
import os
import sys
import urllib.request

import cv2
import numpy as np

# ----------------------------------------------------------------------------- settings
SIZE = 128                # aligned palm crop is SIZE x SIZE
N_ORI = 6                 # number of line directions
MAX_SHIFT = 4             # residual shift search (pixels)
ROTATIONS = (-6, -3, 0, 3, 6)   # residual rotation search (degrees)
KEEP_FRACTION = 0.40      # use the 40% most line-like pixels
DEFAULT_THRESHOLD = 0.30  # PLACEHOLDER - set it with `evaluate` on real photos

ROI_BETA = 0.50           # palm centre = middle knuckle + BETA * palm width (towards wrist)
ROI_GAMMA = 0.90          # crop side = GAMMA * palm width
MIN_PALM_FRACTION = 0.12  # palm width must be 12%..85% of image width (too far / too close)
MAX_PALM_FRACTION = 0.85
MIN_SHARPNESS = 15.0      # variance of Laplacian on the crop; below = too blurry (UNTUNED - check with your camera)

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


class QualityError(Exception):
    """Photo rejected (no hand, too far, too blurry...). Ask the user to retake it."""


# ----------------------------------------------------------------------------- image loading
def load_bgr(img):
    if isinstance(img, str):
        x = cv2.imread(img, cv2.IMREAD_COLOR)
        if x is None:
            raise FileNotFoundError(f"cannot read image: {img}")
        return x
    x = np.asarray(img)
    if x.dtype != np.uint8:
        x = np.clip(x * (255.0 if x.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    if x.ndim == 2:
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    return x


# ----------------------------------------------------------------------------- hand detection
_LANDMARKER = None


def _landmarker():
    global _LANDMARKER
    if _LANDMARKER is None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError:
            sys.exit("mediapipe is not installed: pip install mediapipe   (or use --no-detect)")
        if not os.path.exists(MODEL_PATH):
            print("downloading MediaPipe hand model ...", file=sys.stderr)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        opts = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL_PATH),
                                            num_hands=1, min_hand_detection_confidence=0.6,
                                            min_hand_presence_confidence=0.6)
        _LANDMARKER = (mp, vision.HandLandmarker.create_from_options(opts))
    return _LANDMARKER


def detect_landmarks(bgr):
    """Returns 21x2 pixel coordinates of the hand landmarks, or raises QualityError."""
    mp, lm = _landmarker()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    if not res.hand_landmarks:
        raise QualityError("no hand found - show the open palm clearly to the camera")
    h, w = bgr.shape[:2]
    return np.array([[p.x * w, p.y * h] for p in res.hand_landmarks[0]], dtype=np.float32)


# ----------------------------------------------------------------------------- alignment
def align_palm(bgr, pts):
    """Rotate hand upright, crop palm centre by palm width, return SIZE x SIZE grey crop."""
    h, w = bgr.shape[:2]
    p0, p5, p9, p17 = pts[0], pts[5], pts[9], pts[17]
    palm_w = float(np.linalg.norm(p17 - p5))
    if not (MIN_PALM_FRACTION * w <= palm_w <= MAX_PALM_FRACTION * w):
        raise QualityError("hand too far or too close - keep the palm at a normal distance")
    v = p9 - p0                                            # wrist -> middle knuckle
    angle = np.degrees(np.arctan2(v[0], -v[1]))            # 0 when fingers point straight up
    m = cv2.getRotationMatrix2D((float(p9[0]), float(p9[1])), angle, 1.0)
    rot = cv2.warpAffine(bgr, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    q = np.hstack([pts, np.ones((len(pts), 1), np.float32)]) @ m.T   # landmarks after rotation
    xc = (q[5, 0] + q[17, 0]) / 2.0
    yc = q[9, 1] + ROI_BETA * palm_w                       # move from knuckles towards the wrist
    half = ROI_GAMMA * palm_w / 2.0
    src = np.float32([[xc - half, yc - half], [xc + half, yc - half], [xc - half, yc + half]])
    dst = np.float32([[0, 0], [SIZE, 0], [0, SIZE]])
    crop = cv2.warpAffine(rot, cv2.getAffineTransform(src, dst), (SIZE, SIZE),
                          flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < MIN_SHARPNESS:
        raise QualityError("photo too blurry - hold still and retake")
    return gray


def centre_crop(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return cv2.resize(gray[y0:y0 + s, x0:x0 + s], (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def palm_roi(img, detect=True):
    bgr = load_bgr(img)
    return align_palm(bgr, detect_landmarks(bgr)) if detect else centre_crop(bgr)


# ----------------------------------------------------------------------------- encoding
def _gabor_bank():
    bank = []
    for k in range(N_ORI):
        kern = cv2.getGaborKernel((21, 21), sigma=3.5, theta=np.pi * k / N_ORI, lambd=10.0,
                                  gamma=0.5, psi=0, ktype=cv2.CV_32F)
        bank.append(kern - kern.mean())
    return bank


_BANK = _gabor_bank()


def _normalise(gray):
    g = gray.astype(np.float32)
    d = g - cv2.GaussianBlur(g, (0, 0), 12)               # remove uneven lighting
    return d / (np.sqrt(cv2.GaussianBlur(d * d, (0, 0), 12)) + 1e-3)


def encode_roi(roi, rotation_deg=0.0):
    x = _normalise(roi)
    if rotation_deg:
        m = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), rotation_deg, 1.0)
        x = cv2.warpAffine(x, m, (SIZE, SIZE), borderMode=cv2.BORDER_REFLECT)
    resp = np.stack([cv2.filter2D(x, cv2.CV_32F, k) for k in _BANK])
    code = resp.argmin(0).astype(np.int8)                 # dark palm lines -> strongest negative response
    strength = -resp.min(0)
    weight = (strength > np.quantile(strength, 1 - KEEP_FRACTION)).astype(np.float32)
    return code, weight


# ----------------------------------------------------------------------------- matching
def _agreement(a, b):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    d = np.minimum(d, N_ORI - d)
    return 1.0 - d / (N_ORI / 2.0)


def _distance_one(A, B):
    (ca, wa), (cb, wb) = A, B
    h, w = ca.shape
    best = 1.0
    for dy, dx in itertools.product(range(-MAX_SHIFT, MAX_SHIFT + 1), repeat=2):
        ya, yb = slice(max(0, dy), h + min(0, dy)), slice(max(0, -dy), h + min(0, -dy))
        xa, xb = slice(max(0, dx), w + min(0, dx)), slice(max(0, -dx), w + min(0, -dx))
        wa_s, wb_s = wa[ya, xa], wb[yb, xb]
        den = np.sqrt(wa_s.sum() * wb_s.sum()) + 1e-6
        best = min(best, 1.0 - (wa_s * wb_s * _agreement(ca[ya, xa], cb[yb, xb])).sum() / den)
    return float(best)


def make_template(img, detect=True):
    return encode_roi(palm_roi(img, detect))


def make_probe(img, detect=True):
    roi = palm_roi(img, detect)
    return [encode_roi(roi, r) for r in ROTATIONS]


def distance(templates, probe):
    """templates: one (code, weight) or a list of them (multi-photo enrollment)."""
    if isinstance(templates, tuple):
        templates = [templates]
    return min(_distance_one(t, p) for t in templates for p in probe)


def verify(templates, img, threshold=DEFAULT_THRESHOLD, detect=True):
    d = distance(templates, make_probe(img, detect))
    return d <= threshold, d


def save_templates(path, templates):
    np.savez(path, codes=np.stack([t[0] for t in templates]), weights=np.stack([t[1] for t in templates]))


def load_templates(path):
    z = np.load(path)
    return list(zip(z["codes"], z["weights"]))


# ----------------------------------------------------------------------------- evaluation
def equal_error_rate(genuine, impostor):
    genuine, impostor = np.sort(genuine), np.sort(impostor)
    ts = np.unique(np.concatenate([genuine, impostor]))
    far = np.searchsorted(impostor, ts, side="right") / len(impostor)
    frr = 1.0 - np.searchsorted(genuine, ts, side="right") / len(genuine)
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(ts[i])


def evaluate_folder(folder, detect=True, max_impostor_pairs=20000, seed=0):
    people = {}
    for p in sorted(os.listdir(folder)):
        files = sorted(f for ext in ("jpg", "jpeg", "png", "bmp")
                       for f in glob.glob(os.path.join(folder, p, "*." + ext)))
        if len(files) >= 2:
            people[p] = files
    tpl, prb, skipped = {}, {}, []
    for fs in people.values():
        for f in fs:
            try:
                roi = palm_roi(f, detect)
                tpl[f] = encode_roi(roi)
                prb[f] = [encode_roi(roi, r) for r in ROTATIONS]
            except QualityError as e:
                skipped.append((f, str(e)))
    people = {p: [f for f in fs if f in tpl] for p, fs in people.items()}
    people = {p: fs for p, fs in people.items() if len(fs) >= 2}
    if skipped:
        print(f"skipped {len(skipped)} photos (quality): e.g. {skipped[0][0]}: {skipped[0][1]}")
    if len(people) < 2:
        sys.exit("need folder/<person>/<image> with at least 2 people and 2 usable images each")
    genuine = [distance(tpl[a], prb[b]) for fs in people.values() for a, b in itertools.permutations(fs, 2)]
    allf = [(p, f) for p, fs in people.items() for f in fs]
    pairs = [(a, b) for (pa, a), (pb, b) in itertools.permutations(allf, 2) if pa != pb]
    if len(pairs) > max_impostor_pairs:
        pick = np.random.RandomState(seed).choice(len(pairs), max_impostor_pairs, replace=False)
        pairs = [pairs[i] for i in pick]
    impostor = [distance(tpl[a], prb[b]) for a, b in pairs]
    eer, thr = equal_error_rate(np.array(genuine), np.array(impostor))
    print(f"people={len(people)} genuine pairs={len(genuine)} impostor pairs={len(impostor)}")
    print(f"EER = {eer * 100:.2f}%  at distance threshold {thr:.3f}  (use this as --threshold)")
    if len(people) < 20:
        print(f"WARNING: only {len(people)} people. This is a smoke test, NOT an accuracy measurement.")
    print("Note: a claim of FAR 1e-5 needs about 300,000 independent impostor comparisons.")
    return eer, thr


# ----------------------------------------------------------------------------- command line
def main():
    ap = argparse.ArgumentParser(description="simple 1:1 palm verification")
    ap.add_argument("--no-detect", action="store_true", help="images are already cropped palm close-ups")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enroll"); e.add_argument("out"); e.add_argument("images", nargs="+")
    v = sub.add_parser("verify"); v.add_argument("template"); v.add_argument("image")
    v.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ev = sub.add_parser("evaluate"); ev.add_argument("folder")
    r = sub.add_parser("roi"); r.add_argument("image"); r.add_argument("out")
    a = ap.parse_args()
    detect = not a.no_detect
    try:
        if a.cmd == "enroll":
            save_templates(a.out, [make_template(f, detect) for f in a.images])
            print(f"saved {a.out} ({len(a.images)} photo(s))")
        elif a.cmd == "verify":
            ok, d = verify(load_templates(a.template), a.image, a.threshold, detect)
            print(f"distance={d:.3f} threshold={a.threshold:.3f} -> {'MATCH' if ok else 'NO MATCH'}")
            sys.exit(0 if ok else 1)
        elif a.cmd == "evaluate":
            evaluate_folder(a.folder, detect)
        else:
            cv2.imwrite(a.out, palm_roi(a.image, detect))
            print("saved", a.out)
    except QualityError as err:
        print("REJECTED PHOTO:", err)
        sys.exit(2)


if __name__ == "__main__":
    main()
