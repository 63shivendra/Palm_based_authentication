"""End-to-end test of the kiosk server (webapp/kiosk.py) with the Flask test client and our own palm photos.

Static photos cannot change their finger count, so the first test expects the liveness check to fail; the second
replaces the challenge with one that passes at once, to test registration, duplicate check and payment.
Needs the photos in 'real palm images/' (skipped otherwise); the matcher runs on the CPU (about a minute)."""
import io
import os
import sys
import time

import cv2
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "webapp"))
PHOTOS = os.path.join(ROOT, "real palm images")
RIGHT_DAY1, RIGHT_DAY2, LEFT = "1000132769.jpg", "1000132924.jpg", "1000132768.jpg"
pytestmark = pytest.mark.skipif(not os.path.isdir(PHOTOS), reason="needs the photos in 'real palm images/'")


@pytest.fixture(scope="module")
def kiosk(tmp_path_factory):
    import kiosk as K
    d = tmp_path_factory.mktemp("kiosk")
    K.DATA, K.PEOPLE_DIR = str(d), str(d / "people")
    K.AUDIT, K.KEY_FILE = str(d / "audit.jsonl"), str(d / "search.key")
    K.eng.warm_up()
    K.KEY = K.load_key()
    K.PEOPLE.clear()
    return K


def jpeg(name, scale=0.5):
    img = cv2.imread(os.path.join(PHOTOS, name))
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


def run_flow(client, mode, photo, name="", max_frames=40):
    sid = client.post("/api/kiosk/start", data=dict(mode=mode, name=name)).get_json()["session"]
    data = jpeg(photo)
    j = None
    for _ in range(max_frames):
        j = client.post("/api/kiosk/frame", data=dict(session=sid, image=(io.BytesIO(data), "f.jpg"))).get_json()
        if j["phase"] in ("captured", "done"):
            break
        time.sleep(0.35)                          # capture needs photos at least 0.3 s apart
    if j["phase"] == "captured":
        return client.post("/api/kiosk/finish", data=dict(session=sid)).get_json()["result"], j
    return j.get("result"), j


def test_static_photo_fails_liveness(kiosk):
    kiosk.STEPS["pay"] = 3
    kiosk.FAILS.clear()
    c = kiosk.app.test_client()
    sid = c.post("/api/kiosk/start", data=dict(mode="pay")).get_json()["session"]
    data = jpeg(RIGHT_DAY1)
    seen, j = set(), None
    t0 = time.time()
    while time.time() - t0 < 25:
        j = c.post("/api/kiosk/frame", data=dict(session=sid, image=(io.BytesIO(data), "f.jpg"))).get_json()
        seen.add(j["phase"])
        if j["phase"] == "done":
            break
    assert "liveness" in seen
    assert j["phase"] == "done" and j["result"]["kind"] == "liveness"
    kiosk.FAILS.clear()


class PassAtOnce:
    """Stand-in for palm_pad.ActiveLiveness that passes on the first frame with a hand."""

    def __init__(self, *a, **k):
        self.status = "running"

    def update(self, pts, t):
        if pts is not None:
            self.status = "passed"
        return self.state()

    def state(self):
        return dict(status=self.status, prompt="", step=1, steps=1, reason="", sequence=[5], finger_motion=0.0)


def test_register_duplicate_and_pay(kiosk, monkeypatch):
    monkeypatch.setattr(kiosk.pad, "ActiveLiveness", PassAtOnce)
    c = kiosk.app.test_client()
    r, _ = run_flow(c, "register", RIGHT_DAY1, name="test person")
    assert r["ok"] and r["kind"] == "registered" and r["id"] == "PALM-0001", r
    assert os.path.exists(os.path.join(kiosk.PEOPLE_DIR, "PALM-0001", "templates.palm"))
    with open(os.path.join(kiosk.PEOPLE_DIR, "PALM-0001", "templates.palm"), "rb") as f:
        assert f.read(7) in (b"PALMDP1", b"PALMAG1")              # encrypted at rest

    r, _ = run_flow(c, "register", RIGHT_DAY2)                     # same palm, other day: duplicate
    assert not r["ok"] and r["kind"] == "duplicate" and r["existing"] == "PALM-0001", r

    r, _ = run_flow(c, "pay", RIGHT_DAY2)                          # pay: identified
    assert r["ok"] and r["kind"] == "identified" and r["id"] == "PALM-0001", r

    r, _ = run_flow(c, "pay", LEFT)                                # the other hand: not recognised
    assert not r["ok"] and r["kind"] == "unknown", r

    s = c.get("/api/kiosk/state").get_json()
    assert [p["id"] for p in s["people"]] == ["PALM-0001"]
    assert s["people"][0]["search_bytes"] == 3 * 512
    outcomes = [a["outcome"] for a in s["audit"]]
    assert {"registered", "duplicate", "identified", "not recognised"} <= set(outcomes)

    s = c.post("/api/kiosk/delete", data=dict(id="PALM-0001")).get_json()
    assert s["people"] == [] and not os.path.exists(os.path.join(kiosk.PEOPLE_DIR, "PALM-0001"))
