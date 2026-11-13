#!/usr/bin/env python3
"""
palm_match.py - simple 1:1 palm verification. No training, no dataset needed to build it.

Pipeline (default method v3)
  1. Find the hand with MediaPipe Hand Landmarker (21 landmarks).
  2. Quality gates: hand found, hand not too far / too close, image not blurry.
  3. Align: least-squares fit on 7 landmarks, left hands mirrored to right-hand layout, 512x512 palm crop.
  4. DISK keypoints + LightGlue matcher (pretrained, zero-shot, no palm training data).
  5. Pose-checked RANSAC, then coverage-weighted score = inliers x palm cells covered / 64.
  6. Multi-photo enrolment: best score over the 2-3 enrolled photos.
  Older methods: --method v2 (crease-scale SIFT, OpenCV only) and --method v1 (Competitive Code).

Commands
  python palm_match.py enroll   alice.npz  a1.jpg a2.jpg a3.jpg   # 1-5 photos of the SAME palm (3 recommended)
  python palm_match.py verify   alice.npz  new_photo.jpg          # prints score + MATCH / NO MATCH
  python palm_match.py evaluate photos/                           # photos/<person>/<image>.jpg -> EER
  python palm_match.py roi      photo.jpg  roi.png                # save the aligned palm crop (debugging)
  Add --no-detect if your images are already cropped palm close-ups.
  Add --method v2 or --method v1 (before the command) for the older matchers.

Install
  pip install opencv-python numpy mediapipe torch kornia     (torch + kornia only needed for v3)
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


# ----------------------------------------------------------------------------- v2 matcher
# Why v2: on real phone photos taken in different sessions, v1 could not separate the owner's
# right palm from other palms (genuine 0.63-0.77 vs impostor 0.66-0.80). v2 changes:
#   * ROI from a least-squares similarity fit on 7 landmarks (not 2) -> far less crop drift
#   * left hands are mirrored into right-hand layout, so one canonical landmark template fits all
#   * Tan-Triggs illumination normalisation (gamma + DoG + contrast equalisation) at CREASE scale:
#     fine pore texture changes with lighting and produced only false SIFT matches across sessions
#   * SIFT keypoints + RANSAC similarity with a pose sanity check; score = number of inliers
#     (Wu, Zhao & Bu, Pattern Recognition 2014, use the same SIFT + RANSAC inlier-count score)
#     (Morales, Ferrer & Kumar, "Towards contactless palmprint authentication", IET CV 2011)
BIG = 512                          # v2 crop side (pixels); 768/1024 tested worse on real photos
BIG_GAMMA, BIG_BETA = 1.15, 0.55   # crop side = 1.15 palm widths, centre 0.55 palm widths below knuckles
CANON_IDS = [0, 1, 2, 5, 9, 13, 17]
# landmark layout of an upright right palm in palm-width units (origin between knuckles 5 and 17,
# +y towards the fingers); median over 54 hands (4 real photos + 50 PalmMatchDB)
CANON = np.float32([[0.115, -1.359], [0.636, -1.045], [0.973, -0.573], [0.472, 0.164],
                    [0.115, 0.171], [-0.193, 0.055], [-0.472, -0.164]])
CREASE_SIGMA = 2.0                 # Tan-Triggs inner sigma: crease scale, not pore scale (see illum_normalise)
MIN_INLIERS = 4                    # PLACEHOLDER: real photos gave genuine 6-41, impostors 0 (only 3 genuine pairs!)
_SIFT = None


def to_right_hand(bgr, pts):
    """Mirror a left hand so every palm has the right-hand layout. Returns (bgr, pts, was_left)."""
    v = pts[9] - pts[0]
    u = np.array([-v[1], v[0]])
    if (pts[2] - pts[9]) @ u < 0:
        pts = pts.copy()
        pts[:, 0] = bgr.shape[1] - 1 - pts[:, 0]
        return cv2.flip(bgr, 1), pts, True
    return bgr, pts, False


def illum_normalise(gray, inner=CREASE_SIGMA):
    """Tan & Triggs (2010) preprocessing chain -> uint8 image showing lines, not lighting.
    inner = DoG inner sigma. 1.0 keeps pore/ridge texture, which changes with lighting and made SIFT match
    skin pores (median keypoint 2.4 px, all matches wrong on a cross-session pair); 2.0 keeps crease-scale
    structure (wrinkles, secondary lines), which is stable between sessions."""
    x = np.power(gray.astype(np.float32) / 255.0 + 1e-3, 0.4)
    x = cv2.GaussianBlur(x, (0, 0), inner) - cv2.GaussianBlur(x, (0, 0), inner * 8)
    x = x / (np.mean(np.abs(x) ** 0.1) ** 10 + 1e-6)
    x = x / (np.mean(np.minimum(np.abs(x), 10) ** 0.1) ** 10 + 1e-6)
    x = 10 * np.tanh(x / 10)
    return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def align_palm_v2(bgr, pts, normalise=True):
    """Upright BIG x BIG palm crop (right-hand layout), illumination-normalised grey.
    normalise=False returns the plain grey crop (for learned matchers that do their own normalisation)."""
    bgr, pts, _ = to_right_hand(bgr, pts)
    palm_w = float(np.linalg.norm(pts[17] - pts[5]))
    if not (MIN_PALM_FRACTION * bgr.shape[1] <= palm_w <= MAX_PALM_FRACTION * bgr.shape[1]):
        raise QualityError("hand too far or too close - keep the palm at a normal distance")
    s = BIG / BIG_GAMMA
    dst = np.float32([[BIG / 2 + cx * s, BIG / 2 - (cy + BIG_BETA) * s] for cx, cy in CANON])
    m, _ = cv2.estimateAffinePartial2D(pts[CANON_IDS].astype(np.float32), dst, method=cv2.LMEDS)
    if m is None:
        raise QualityError("could not align the hand - open the palm fully and retake")
    crop = cv2.warpAffine(bgr, m, (BIG, BIG), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    if cv2.Laplacian(small, cv2.CV_64F).var() < MIN_SHARPNESS:
        raise QualityError("photo too blurry - hold still and retake")
    return illum_normalise(gray) if normalise else gray


def palm_roi_v2(img, detect=True):
    bgr = load_bgr(img)
    if detect:
        return align_palm_v2(bgr, detect_landmarks(bgr))
    return illum_normalise(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (BIG, BIG)))


def features_v2(roi):
    """RootSIFT keypoints of the normalised crop: (Nx2 positions, Nx128 descriptors)."""
    global _SIFT
    if _SIFT is None:
        _SIFT = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.01)   # creases are low-contrast
    kp, des = _SIFT.detectAndCompute(roi, None)
    if des is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    des = np.sqrt(des / (des.sum(1, keepdims=True) + 1e-7)).astype(np.float32)
    return np.float32([k.pt for k in kp]), des


def _inliers(a, b, ratio=0.85, reproj=8.0):
    """Geometrically consistent matches b -> a. The pose must stay near identity because both crops are
    already landmark-aligned: scale 0.75-1.33, rotation < 25 deg, shift < 160 px (at BIG=512)."""
    (pa, da), (pb, db) = a, b
    if len(da) < 10 or len(db) < 10:
        return 0
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(db, da, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < ratio * m[1].distance]
    if len(good) < 6:
        return 0
    src = pb[[m.queryIdx for m in good]]
    dst = pa[[m.trainIdx for m in good]]
    m, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=reproj,
                                         maxIters=5000, confidence=0.999)
    if m is None:
        return 0
    scale = np.hypot(m[0, 0], m[1, 0])
    rot = np.degrees(np.arctan2(m[1, 0], m[0, 0]))
    c = np.array([BIG / 2, BIG / 2])
    shift = np.abs(m[:, 2] - (np.eye(2) - m[:, :2]) @ c).max()
    if not (0.75 < scale < 1.33 and abs(rot) < 25 and shift < 160 * BIG / 512):
        return 0
    return int(inl.sum())


def score_v2(templates, probe):
    """Higher = more similar. Symmetric inlier count, best over the enrolled photos."""
    if isinstance(templates, tuple):
        templates = [templates]
    return max(min(_inliers(t, probe), _inliers(probe, t)) for t in templates)


def make_template_v2(img, detect=True):
    return features_v2(palm_roi_v2(img, detect))


def verify_v2(templates, img, min_inliers=MIN_INLIERS, detect=True):
    s = score_v2(templates, make_template_v2(img, detect))
    return s >= min_inliers, s


def save_templates_v2(path, templates):
    arrays = {"method": np.array("v2")}
    for i, (p, d) in enumerate(templates):
        arrays[f"kp{i}"], arrays[f"des{i}"] = p, d
    np.savez(path, **arrays)


# ----------------------------------------------------------------------------- v3 matcher (default)
# Same alignment and quality gates as v2, then:
#   * DISK keypoints + LightGlue transformer matcher, both pretrained on general scenes (Apache-2.0) and
#     used zero-shot: no palm training data. Needs: pip install torch kornia
#   * pose-checked RANSAC (same near-identity check as v2)
#   * NOVELTY 1 - coverage-weighted score: inliers x (palm cells covered / 64), on an 8x8 grid of the
#     aligned palm, taking the smaller coverage of the two images. False matches between different palms
#     cluster in a few cells; true matches spread over the whole palm. On Tongji (600 palms, 2 sessions)
#     this widened the gap between the weakest genuine pair and the strongest impostor from 1.32x to 1.55x
#     (5th-weakest vs 5th-strongest: 2.55x -> 5.03x), better in 500 of 500 bootstrap resamples.
#   * NOVELTY 2 - multi-photo enrolment: enrol 2-3 photos, score = best match over the enrolled photos.
#     On PalmMatchDB (trustworthy labels) single-photo EER 1.32% -> 0.00% with 3 enrolled photos
#     (benchmarks/results/REPORT.md; an early experiment with the raw inlier count measured 7.15%).
V3_KEYPOINTS = 2048
V3_GRID = 8
V3_THRESHOLD = 26.0    # PROVISIONAL, coverage-weighted score. Tongji: impostor max 21.1 / genuine min 32.8;
                       # PalmMatchDB trustworthy impostor max 11.4; real phone photos: genuine 147-198, impostor 0
_V3 = None


def _v3_models():
    global _V3
    if _V3 is None:
        try:
            import torch
            import kornia.feature as KF
        except ImportError:
            sys.exit("method v3 needs torch and kornia:  pip install torch kornia   (or use --method v2)")
        import warnings
        warnings.filterwarnings("ignore", message=".*torch.tensor.*")   # kornia LightGlue hw warning
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _V3 = dict(torch=torch, KF=KF, dev=dev, disk=KF.DISK.from_pretrained("depth").eval().to(dev),
                   lg=KF.LightGlueMatcher("disk").eval().to(dev))
    return _V3


def features_v3(gray):
    """DISK keypoints of the plain (not Tan-Triggs) aligned crop: (Nx2 positions, Nx128 descriptors) float32."""
    m = _v3_models(); torch = m["torch"]
    x = torch.from_numpy(np.ascontiguousarray(gray)).float()[None, None].repeat(1, 3, 1, 1).to(m["dev"]) / 255
    with torch.no_grad():
        f = m["disk"](x, V3_KEYPOINTS, pad_if_not_divisible=True)[0]
    return f.keypoints.cpu().numpy().astype(np.float32), f.descriptors.cpu().numpy().astype(np.float32)


def palm_crop_v3(img, detect=True):
    bgr = load_bgr(img)
    if detect:
        return align_palm_v2(bgr, detect_landmarks(bgr), normalise=False)
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (BIG, BIG), interpolation=cv2.INTER_CUBIC)


def make_template_v3(img, detect=True):
    return features_v3(palm_crop_v3(img, detect))


def _pose_checked_mask(src, dst, reproj=8.0):
    """RANSAC similarity src -> dst; inlier mask, all False if the pose is not near identity."""
    out = np.zeros(len(src), bool)
    if len(src) < 6:
        return out
    m, inl = cv2.estimateAffinePartial2D(np.float32(src), np.float32(dst), method=cv2.RANSAC,
                                         ransacReprojThreshold=reproj, maxIters=5000, confidence=0.999)
    if m is None:
        return out
    scale = np.hypot(m[0, 0], m[1, 0])
    rot = np.degrees(np.arctan2(m[1, 0], m[0, 0]))
    c = np.array([BIG / 2, BIG / 2])
    shift = np.abs(m[:, 2] - (np.eye(2) - m[:, :2]) @ c).max()
    if not (0.75 < scale < 1.33 and abs(rot) < 25 and shift < 160 * BIG / 512):
        return out
    return inl.ravel().astype(bool)


def _covered_cells(p):
    c = np.clip((p / (BIG / V3_GRID)).astype(int), 0, V3_GRID - 1)
    return np.unique(c[:, 0] * V3_GRID + c[:, 1]).size


def match_v3(a, b):
    """One template vs one probe -> dict(inliers, cells, score). score = inliers * cells / 64."""
    m = _v3_models(); torch, KF, dev = m["torch"], m["KF"], m["dev"]
    (pa, da), (pb, db) = a, b
    if len(pa) < 8 or len(pb) < 8:
        return dict(inliers=0, cells=0, score=0.0)
    ka, kb = torch.from_numpy(pa).to(dev), torch.from_numpy(pb).to(dev)
    la = KF.laf_from_center_scale_ori(ka[None], torch.ones(1, len(ka), 1, 1, device=dev))
    lb = KF.laf_from_center_scale_ori(kb[None], torch.ones(1, len(kb), 1, 1, device=dev))
    hw = torch.tensor([BIG, BIG], device=dev)
    with torch.no_grad():
        _, idx = m["lg"](torch.from_numpy(da).to(dev), torch.from_numpy(db).to(dev), la, lb, hw1=hw, hw2=hw)
    idx = idx.cpu().numpy()
    qa, qb = pa[idx[:, 0]], pb[idx[:, 1]]
    inl = _pose_checked_mask(qb, qa)
    if not inl.any():
        return dict(inliers=0, cells=0, score=0.0)
    cells = min(_covered_cells(qa[inl]), _covered_cells(qb[inl]))
    n = int(inl.sum())
    return dict(inliers=n, cells=int(cells), score=n * cells / V3_GRID ** 2)


def score_v3(templates, probe):
    """Multi-photo enrolment: best coverage-weighted score over the enrolled photos."""
    if isinstance(templates, tuple):
        templates = [templates]
    return max(match_v3(t, probe)["score"] for t in templates)


def verify_v3(templates, img, threshold=V3_THRESHOLD, detect=True):
    s = score_v3(templates, make_template_v3(img, detect))
    return s >= threshold, s


def save_templates_v3(path, templates):
    arrays = {"method": np.array("v3")}
    for i, (p, d) in enumerate(templates):
        arrays[f"kp{i}"], arrays[f"des{i}"] = p, d
    np.savez(path, **arrays)


def load_any_templates(path):
    z = np.load(path)
    if "method" in z and str(z["method"]) in ("v2", "v3"):
        n = sum(1 for k in z.files if k.startswith("kp"))
        return str(z["method"]), [(z[f"kp{i}"], z[f"des{i}"]) for i in range(n)]
    return "v1", list(zip(z["codes"], z["weights"]))


# ----------------------------------------------------------------------------- evaluation
def equal_error_rate(genuine, impostor):
    genuine, impostor = np.sort(genuine), np.sort(impostor)
    ts = np.unique(np.concatenate([genuine, impostor]))
    far = np.searchsorted(impostor, ts, side="right") / len(impostor)
    frr = 1.0 - np.searchsorted(genuine, ts, side="right") / len(genuine)
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(ts[i])


def evaluate_folder(folder, detect=True, max_impostor_pairs=20000, seed=0, method="v3"):
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
                if method == "v3":
                    tpl[f] = prb[f] = make_template_v3(f, detect)
                elif method == "v2":
                    tpl[f] = prb[f] = make_template_v2(f, detect)
                else:
                    roi = palm_roi(f, detect)
                    tpl[f] = encode_roi(roi)
                    prb[f] = [encode_roi(roi, r) for r in ROTATIONS]
            except QualityError as e:
                skipped.append((f, str(e)))
    # v2 scores are inlier counts (higher = more alike); negate so the EER code can treat them as distances
    distance_fn = {"v3": lambda t, p: -score_v3(t, p), "v2": lambda t, p: -score_v2(t, p)}.get(method, distance)
    people = {p: [f for f in fs if f in tpl] for p, fs in people.items()}
    people = {p: fs for p, fs in people.items() if len(fs) >= 2}
    if skipped:
        print(f"skipped {len(skipped)} photos (quality): e.g. {skipped[0][0]}: {skipped[0][1]}")
    if len(people) < 2:
        sys.exit("need folder/<person>/<image> with at least 2 people and 2 usable images each")
    genuine = [distance_fn(tpl[a], prb[b]) for fs in people.values() for a, b in itertools.permutations(fs, 2)]
    allf = [(p, f) for p, fs in people.items() for f in fs]
    pairs = [(a, b) for (pa, a), (pb, b) in itertools.permutations(allf, 2) if pa != pb]
    if len(pairs) > max_impostor_pairs:
        pick = np.random.RandomState(seed).choice(len(pairs), max_impostor_pairs, replace=False)
        pairs = [pairs[i] for i in pick]
    impostor = [distance_fn(tpl[a], prb[b]) for a, b in pairs]
    eer, thr = equal_error_rate(np.array(genuine), np.array(impostor))
    print(f"people={len(people)} genuine pairs={len(genuine)} impostor pairs={len(impostor)}")
    if method == "v3":
        print(f"EER = {eer * 100:.2f}%  at score {-thr:.1f}  (use this as --score-threshold)")
    elif method == "v2":
        print(f"EER = {eer * 100:.2f}%  at {-thr:.0f} inliers  (use this as --min-inliers)")
    else:
        print(f"EER = {eer * 100:.2f}%  at distance threshold {thr:.3f}  (use this as --threshold)")
    if len(people) < 20:
        print(f"WARNING: only {len(people)} people. This is a smoke test, NOT an accuracy measurement.")
    print("Note: a claim of FAR 1e-5 needs about 300,000 independent impostor comparisons.")
    return eer, thr


# ----------------------------------------------------------------------------- command line
def main():
    ap = argparse.ArgumentParser(description="simple 1:1 palm verification")
    ap.add_argument("--no-detect", action="store_true", help="images are already cropped palm close-ups")
    ap.add_argument("--method", choices=("v3", "v2", "v1"), default="v3",
                    help="v3 = DISK+LightGlue with coverage-weighted score (default, needs torch+kornia); "
                         "v2 = crease-scale SIFT; v1 = original Competitive Code")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enroll"); e.add_argument("out"); e.add_argument("images", nargs="+")
    v = sub.add_parser("verify"); v.add_argument("template"); v.add_argument("image")
    v.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="v1: max distance")
    v.add_argument("--min-inliers", type=int, default=MIN_INLIERS, help="v2: min matched keypoints")
    v.add_argument("--score-threshold", type=float, default=V3_THRESHOLD, help="v3: min coverage-weighted score")
    ev = sub.add_parser("evaluate"); ev.add_argument("folder")
    r = sub.add_parser("roi"); r.add_argument("image"); r.add_argument("out")
    a = ap.parse_args()
    detect = not a.no_detect
    try:
        if a.cmd == "enroll":
            if a.method == "v3":
                save_templates_v3(a.out, [make_template_v3(f, detect) for f in a.images])
            elif a.method == "v2":
                save_templates_v2(a.out, [make_template_v2(f, detect) for f in a.images])
            else:
                save_templates(a.out, [make_template(f, detect) for f in a.images])
            print(f"saved {a.out} ({len(a.images)} photo(s), method {a.method})")
        elif a.cmd == "verify":
            method, templates = load_any_templates(a.template)   # the template file decides the method
            if method == "v3":
                probe = make_template_v3(a.image, detect)
                per = [match_v3(t, probe) for t in templates]
                best = max(per, key=lambda r: r["score"])
                ok = best["score"] >= a.score_threshold
                print(f"score={best['score']:.1f} (inliers {best['inliers']}, palm cells {best['cells']}/64, "
                      f"best of {len(templates)} enrolled photo(s)) threshold={a.score_threshold:.1f} -> "
                      f"{'MATCH' if ok else 'NO MATCH'}")
            elif method == "v2":
                ok, s = verify_v2(templates, a.image, a.min_inliers, detect)
                print(f"inliers={s} min_inliers={a.min_inliers} -> {'MATCH' if ok else 'NO MATCH'}")
            else:
                ok, d = verify(templates, a.image, a.threshold, detect)
                print(f"distance={d:.3f} threshold={a.threshold:.3f} -> {'MATCH' if ok else 'NO MATCH'}")
            sys.exit(0 if ok else 1)
        elif a.cmd == "evaluate":
            evaluate_folder(a.folder, detect, method=a.method)
        else:
            roi = {"v3": palm_crop_v3, "v2": palm_roi_v2}.get(a.method, palm_roi)(a.image, detect)
            cv2.imwrite(a.out, roi)
            print("saved", a.out)
    except QualityError as err:
        print("REJECTED PHOTO:", err)
        sys.exit(2)


if __name__ == "__main__":
    main()
