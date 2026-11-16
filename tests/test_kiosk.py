"""End-to-end test of the palm payment kiosk (webapp/kiosk.py) with the Flask test client and our own palm photos.

Static photos cannot change their finger count, so the first test expects the liveness check to fail; the others
replace the challenge with one that passes at once, to test registration with consent, duplicate check, payments
(palm only, PIN step-up, limits), the account view, erasure and the admin console.
Needs the photos in 'real palm images/' (skipped otherwise); the matcher runs on the CPU (a few minutes)."""
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
    K.eng.warm_up()
    K.init(str(tmp_path_factory.mktemp("kiosk")))
    return K


def jpeg(name, scale=0.5):
    img = cv2.imread(os.path.join(PHOTOS, name))
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


def capture(client, sid, photo, max_frames=40):
    data, j = jpeg(photo), None
    for _ in range(max_frames):
        j = client.post("/api/kiosk/frame", data=dict(session=sid, image=(io.BytesIO(data), "f.jpg"))).get_json()
        if j["phase"] in ("captured", "done"):
            break
        time.sleep(0.35)                          # capture needs photos at least 0.3 s apart
    return j


def flow(client, mode, photo, **form):
    r = client.post("/api/kiosk/start", data=dict(mode=mode, **form))
    assert r.status_code == 200, r.get_json()
    sid = r.get_json()["session"]
    j = capture(client, sid, photo)
    if j["phase"] == "captured":
        return sid, client.post("/api/kiosk/finish", data=dict(session=sid)).get_json()["result"]
    return sid, j.get("result")


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


def test_static_photo_fails_liveness(kiosk):
    kiosk.FAILS.clear()
    c = kiosk.app.test_client()
    sid = c.post("/api/kiosk/start", data=dict(mode="pay", amount="50")).get_json()["session"]
    data, seen, j, t0 = jpeg(RIGHT_DAY1), set(), None, time.time()
    while time.time() - t0 < 25:
        j = c.post("/api/kiosk/frame", data=dict(session=sid, image=(io.BytesIO(data), "f.jpg"))).get_json()
        seen.add(j["phase"])
        if j["phase"] == "done":
            break
    assert "liveness" in seen and j["phase"] == "done" and j["result"]["kind"] == "liveness"
    kiosk.FAILS.clear()


def test_registration_needs_consent_and_valid_details(kiosk):
    c = kiosk.app.test_client()
    assert c.post("/api/kiosk/start", data=dict(mode="register", upi="asha@upi")).status_code == 400          # no consent
    assert c.post("/api/kiosk/start", data=dict(mode="register", upi="not-an-id", consent="1")).status_code == 400
    assert c.post("/api/kiosk/start", data=dict(mode="register", upi="asha@upi", pin="12", consent="1")).status_code == 400
    assert c.post("/api/kiosk/start", data=dict(mode="pay", amount="999999")).status_code == 400


def test_end_to_end(kiosk, monkeypatch):
    monkeypatch.setattr(kiosk.pad, "ActiveLiveness", PassAtOnce)
    c = kiosk.app.test_client()

    # register with consent, a demo wallet of Rs 3,000 and a PIN
    _, r = flow(c, "register", RIGHT_DAY1, name="Test Person", upi="test.person@upi", balance="3000", pin="4321", consent="1")
    assert r["ok"] and r["kind"] == "registered" and r["id"] == "PALM-0001", r
    with open(os.path.join(kiosk.PEOPLE_DIR, "PALM-0001", "templates.palm"), "rb") as f:
        assert f.read(7) in (b"PALMDP1", b"PALMAG1")                                   # encrypted at rest

    # the same palm cannot be registered twice
    _, r = flow(c, "register", RIGHT_DAY2, upi="other@upi", consent="1")
    assert not r["ok"] and r["kind"] == "duplicate" and r["existing"] == "PALM-0001", r

    # small payment: palm only
    _, r = flow(c, "pay", RIGHT_DAY2, amount="250")
    assert r["ok"] and r["kind"] == "approved" and r["txn"]["auth"] == "palm", r
    assert r["txn"]["balance"] == 2750 and r["txn"]["message"]["Simulated"] is True
    assert "test.person@upi" not in str(r)                                              # UPI ID is masked

    # large payment: PIN step-up; one wrong PIN, then the right one
    sid, r = flow(c, "pay", RIGHT_DAY2, amount="2500")
    assert r["kind"] == "pin_required" and r["id"] == "PALM-0001", r
    assert c.post("/api/kiosk/pin", data=dict(session=sid, pin="0000")).get_json() == dict(retry=True, tries_left=2)
    r = c.post("/api/kiosk/pin", data=dict(session=sid, pin="4321")).get_json()["result"]
    assert r["ok"] and r["txn"]["auth"] == "palm + PIN" and r["txn"]["balance"] == 250, r

    # over the daily limit (Rs 5,000: 2,750 spent) -> declined, balance unchanged
    sid, r = flow(c, "pay", RIGHT_DAY2, amount="1900")
    assert not r["ok"] and r["kind"] == "declined" and "insufficient" in r["txn"]["reason"], r

    # the other hand is not recognised
    _, r = flow(c, "pay", LEFT, amount="100")
    assert not r["ok"] and r["kind"] == "unknown", r

    # admin: set the PIN, see the stats and transactions, top up
    assert c.get("/api/admin/overview").status_code == 401
    tok = c.post("/api/admin/setup", data=dict(pin="20261115")).get_json()["token"]
    assert c.post("/api/admin/setup", data=dict(pin="11111111")).status_code == 403          # only once
    h = {"X-Admin-Token": tok}
    a = c.get("/api/admin/overview", headers=h).get_json()
    assert [p["id"] for p in a["people"]] == ["PALM-0001"] and a["people"][0]["search_bytes"] == 3 * 512
    assert a["stats"]["approved"] == 2 and a["stats"]["declined"] >= 2 and a["stats"]["duplicates_blocked"] == 1
    a = c.post("/api/admin/topup", data=dict(id="PALM-0001", amount="1000"), headers=h).get_json()
    assert a["people"][0]["balance"] == 1250
    assert c.get(f"/api/admin/export/transactions.csv?token={tok}").status_code == 200

    # my account, then erase everything
    sid, r = flow(c, "account", RIGHT_DAY2)
    assert r["ok"] and r["kind"] == "account" and r["balance"] == 1250 and len(r["transactions"]) >= 3, r
    assert c.post("/api/kiosk/erase", data=dict(session=sid)).get_json()["ok"]
    assert not os.path.exists(os.path.join(kiosk.PEOPLE_DIR, "PALM-0001"))
    assert c.get("/api/admin/overview", headers=h).get_json()["people"] == []
    _, r = flow(c, "pay", RIGHT_DAY2, amount="10")
    assert r["kind"] == "unknown"                                                         # erased palm no longer pays
