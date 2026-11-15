"""Unit tests for palm_quality.py, palm_embed.py (HOL) and palm_protect.py on synthetic palm-like images."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import palm_embed as pe  # noqa: E402
import palm_protect as pp  # noqa: E402
import palm_quality as pq  # noqa: E402


def palm(seed=0, size=512):
    """Skin-like grey image with a few dark curved creases and fine texture."""
    rng = np.random.RandomState(seed)
    img = np.full((size, size), 150, np.float32)
    img += cv2.GaussianBlur(rng.normal(0, 30, (size, size)).astype(np.float32), (0, 0), 25)
    for _ in range(9):
        pts = np.cumsum(rng.normal(0, 18, (12, 2)), 0) + rng.uniform(60, size - 60, 2)
        cv2.polylines(img, [pts.astype(np.int32)], False, float(rng.uniform(70, 110)), int(rng.randint(2, 5)), cv2.LINE_AA)
    img += cv2.GaussianBlur(rng.normal(0, 10, (size, size)).astype(np.float32), (0, 0), 1.0)
    return np.clip(cv2.GaussianBlur(img, (0, 0), 1.2), 0, 255).astype(np.uint8)


def codes(img):
    return {c for c, _ in pq.check(pq.crop_measures(img))}


def test_clean_palm_passes_and_problems_are_named():
    g = palm()
    assert codes(g) == set()
    assert "too_bright" in codes(np.clip(g.astype(np.float32) * 2.2, 0, 255).astype(np.uint8))
    assert "too_dark" in codes((g * 0.05).astype(np.uint8))              # near black; dim photos are matched fine
    assert codes(cv2.GaussianBlur(g, (0, 0), 12)) & {"blurry", "faint_lines"}   # severe blur is sent back; mild blur matches
    assert all(c in pq.MESSAGES for c in pq.MESSAGES)


def test_pose_checks():
    m = dict(pq.crop_measures(palm()), palm_pixels=90.0, palm_fraction=0.1, length_ratio=0.8, fingers_open=1)
    assert codes_from(m) >= {"too_small", "tilted", "fingers_closed"}


def codes_from(m):
    return {c for c, _ in pq.check(m)}


def test_hol_same_palm_beats_other_palm():
    a = palm(1)
    shifted = np.roll(a, (6, -5), (0, 1))                    # small alignment error
    darker = (a * 0.6).astype(np.uint8)                       # different lighting
    same = [pe.similarity(pe.hol(a), pe.hol(x)) for x in (shifted, darker)]
    other = [pe.similarity(pe.hol(a), pe.hol(palm(s))) for s in range(2, 8)]
    assert min(same) > max(other)
    assert abs(np.linalg.norm(pe.hol(a)) - 1) < 1e-4


def test_protection_is_renewable_and_keeps_order():
    a, b = pe.hol(palm(1)), pe.hol(np.roll(palm(1), (5, 5), (0, 1)))
    c = pe.hol(palm(9))
    k1, k2 = "bank-1 key", "bank-2 key"
    same_key = pp.similarity(pp.protect(a, k1), pp.protect(b, k1))
    other_palm = pp.similarity(pp.protect(a, k1), pp.protect(c, k1))
    new_key = pp.similarity(pp.protect(a, k1), pp.protect(b, k2))
    assert same_key > other_palm + 0.15
    assert new_key < same_key - 0.15                          # a re-issued template does not match the old one
    assert pp.protect(a, k1).dtype == np.uint8 and pp.protect(a, k1).nbytes == pp.IOM_K
    m = pp.similarity_matrix(np.stack([pp.protect(a, k1)]), np.stack([pp.protect(b, k1), pp.protect(c, k1)]))
    assert m.shape == (1, 2) and m[0, 0] > m[0, 1]


@pytest.mark.skipif(sys.platform != "win32", reason="DPAPI is Windows-only; elsewhere AES-GCM needs a key")
def test_encrypted_template_round_trip(tmp_path):
    p = str(tmp_path / "t.palm")
    kp = np.arange(20, dtype=np.uint16).reshape(10, 2)
    pp.save_encrypted(p, kp=kp, des=np.ones((10, 128), np.uint8))
    raw = open(p, "rb").read()
    assert raw.startswith(pp.MAGIC_DPAPI) and kp.tobytes() not in raw
    z = pp.load_encrypted(p)
    assert (z["kp"] == kp).all() and z["des"].shape == (10, 128)
