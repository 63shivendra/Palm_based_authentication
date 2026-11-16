"""Unit tests for palm_pad.py (spoof checks), on synthetic hand landmarks and synthetic images."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import palm_pad as pad  # noqa: E402

hand = pad.synthetic_hand


def warp(pts, m):
    q = np.hstack([pts, np.ones((len(pts), 1), np.float32)]) @ m.T
    return (q[:, :2] / q[:, 2:3]).astype(np.float32) if m.shape[0] == 3 else q.astype(np.float32)


TILT = cv2.getPerspectiveTransform(np.float32([[0, 0], [640, 0], [640, 480], [0, 480]]),
                                   np.float32([[60, 30], [600, 0], [640, 480], [20, 450]]))


@pytest.mark.parametrize("s", [6.0, 20.0, 40.0])
def test_finger_spread_matches_geometry(s):
    assert abs(pad.finger_spread(hand(5, s)) - s) < 1.5


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_finger_count(n):
    assert pad.finger_count(hand(n)) == n
    assert pad.finger_count(warp(hand(n), TILT)) == n                    # also seen at an angle


def test_left_hand_is_measured_like_a_right_hand():
    pts = hand(3, 30.0)
    pts[:, 0] = 640 - pts[:, 0]
    assert pad.finger_count(pts) == 3


def test_tilted_flat_print_does_not_move_its_fingers():
    pts = hand(5, 10.0)
    assert abs(pad.finger_spread(warp(pts, TILT)) - pad.finger_spread(pts)) < 5.0
    assert pad.rigid_residual(pts, warp(pts, TILT)) < 0.08


def run(ch, frames):
    st = None
    for t, pts in frames:
        st = ch.update(pts, t)
        if st["status"] != "running":
            break
    return st


COUNT_OF = {v: k for k, v in pad.PROMPTS.items()}


def live_user(ch, fps=15, reaction=0.5):
    """Follows each prompt after a human reaction time and keeps the pose while told to hold still."""
    t, current, shown, since = 0.0, hand(5, 25.0), None, 0.0
    while t < 40 and ch.status == "running":
        p = ch.prompt
        if p != shown:
            shown, since = p, t
        if p in COUNT_OF and t - since >= reaction:
            current = hand(COUNT_OF[p])
        ch.update(current, t)
        t += 1 / fps
    return ch.state()


def test_live_hand_passes():
    for seed in range(30):
        ch = pad.ActiveLiveness(seed=seed)
        st = live_user(ch)
        assert st["status"] == "passed", (seed, st)
        assert ch.final_pts is not None and pad.finger_count(ch.final_pts) == 5


def test_sequence_is_random():
    seqs = {tuple(pad.ActiveLiveness(seed=s).steps) for s in range(200)}
    assert len(seqs) >= 30
    assert all(q[-1] == 5 and all(a != b for a, b in zip(q, q[1:])) for q in seqs)


def test_printed_photo_fails():
    """Rigid moves of a printed hand (shift, rotation, scale, tilt) never change the finger count."""
    for count in (2, 5):
        ch = pad.ActiveLiveness(seed=3)
        frames = []
        for k in range(300):
            t = k / 15
            m = cv2.getRotationMatrix2D((320, 260), 15 * np.sin(t), 1 + 0.2 * np.sin(0.7 * t))
            m[:, 2] += (30 * np.sin(1.3 * t), 20 * np.cos(t))
            frames.append((t, warp(hand(count), m)))
        assert run(ch, frames)["status"] == "failed"


@pytest.mark.parametrize("period", [0.8, 1.5, 2.5])
def test_replayed_video_fails(period):
    """A recorded hand cycling through 1-2-3-4-5 fingers cannot follow random prompts at random times."""
    passed = 0
    for seed in range(50):
        ch = pad.ActiveLiveness(seed=seed)
        frames = [(k / 15, hand(1 + int(k / 15 / period) % 5)) for k in range(900)]
        passed += run(ch, frames)["status"] == "passed"
    assert passed <= 3                       # best possible for a video guessing each random step: 1/27 = 3.7%


def test_recapture_score_reacts_to_a_screen_pattern():
    rng = np.random.RandomState(0)
    skin = cv2.GaussianBlur(rng.normal(150, 25, (320, 320, 3)).astype(np.float32), (0, 0), 3)
    skin = np.clip(skin + rng.normal(0, 2, skin.shape), 0, 255).astype(np.uint8)
    yy, xx = np.mgrid[0:320, 0:320]
    grid = (0.7 + 0.3 * (np.cos(2 * np.pi * (0.31 * xx + 0.07 * yy)) > 0))[..., None]   # aliased sub-pixel stripes
    screen = np.clip(skin * grid, 0, 255).astype(np.uint8)
    assert pad.recapture_score(screen) > pad.recapture_score(skin) + 3
