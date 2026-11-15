#!/usr/bin/env python3
"""
webapp/kiosk.py - palm payment kiosk: register a palm, then pay with it. Runs on this computer only.

  python webapp/kiosk.py                 ->  http://127.0.0.1:5060
  python webapp/kiosk.py --pad enforce   ->  reject photos the passive screen/print check flags (default: warn only,
                                             until the threshold is checked on this camera with
                                             scripts/capture_pad_set.py and benchmarks/pad_eval.py)

Flow (the camera streams frames to this server; every check runs here, not in the browser):
  1. position   live instructions until the palm is sharp, well lit, open and close enough (palm_quality.py)
  2. liveness   random finger counts at random times, then the open hand (palm_pad.ActiveLiveness)
  3. capture    the open-hand frames are the photos used (3 for registration, 2 for payment)
  4. register   duplicate search over everyone enrolled; a new palm gets the next free ID (PALM-0001, ...)
     pay        protected-vector shortlist, then the matcher decides (palm_engine.search)

Stored in webapp/data/kiosk/ (git-ignored): per person the keypoint templates encrypted with Windows DPAPI and the
IoM search hashes; an audit log without images. Photos are never written to disk. Delete removes everything of a
person (right to erasure).
"""
import argparse
import csv
import datetime
import io
import json
import os
import secrets
import shutil
import sys
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import palm_engine as eng  # noqa: E402
import palm_pad as pad  # noqa: E402
import palm_protect as pp  # noqa: E402

DATA = os.path.join(HERE, "data", "kiosk")
PEOPLE_DIR = os.path.join(DATA, "people")
AUDIT = os.path.join(DATA, "audit.jsonl")
KEY_FILE = os.path.join(DATA, "search.key")
STEPS = {"register": 4, "pay": 3}                  # liveness steps (the last one is always the open hand)
PHOTOS = {"register": 3, "pay": 2}
GOOD_FRAMES_TO_START = 3
FLOW_TIMEOUT_S = 60
LOCKOUT = dict(failures=3, window_s=120, lock_s=60)

app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
LOCK = threading.Lock()
PEOPLE = {}          # id -> dict(name, created, hashes, photos)
SESSIONS = {}
FAILS = []           # times of failed liveness attempts (kiosk-wide lockout)
CFG = dict(pad="warn")
KEY = b""


# ------------------------------------------------------------------------------------------------ storage
def load_key():
    os.makedirs(DATA, exist_ok=True)
    if os.path.exists(KEY_FILE):
        return pp.decrypt(open(KEY_FILE, "rb").read())
    k = secrets.token_bytes(32)
    with open(KEY_FILE, "wb") as f:
        f.write(pp.encrypt(k))
    return k


def save_person(pid):
    p = PEOPLE[pid]
    d = os.path.join(PEOPLE_DIR, pid)
    os.makedirs(d, exist_ok=True)
    arrays = {}
    for j, t in enumerate(p["photos"]):
        arrays[f"kp{j}"], arrays[f"des{j}"] = t["v3c"]
        if "v2" in t:
            arrays[f"v2kp{j}"], arrays[f"v2des{j}"] = t["v2"]
    pp.save_encrypted(os.path.join(d, "templates.palm"), **arrays)
    np.save(os.path.join(d, "search_hashes.npy"), p["hashes"])
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(dict(id=pid, name=p["name"], created=p["created"], photos=len(p["photos"])), f)


def load_people():
    if not os.path.isdir(PEOPLE_DIR):
        return
    for pid in sorted(os.listdir(PEOPLE_DIR)):
        d = os.path.join(PEOPLE_DIR, pid)
        try:
            meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
            z = pp.load_encrypted(os.path.join(d, "templates.palm"))
            photos = []
            for j in range(meta["photos"]):
                t = dict(v3c=(z[f"kp{j}"], z[f"des{j}"]))
                if f"v2kp{j}" in z:
                    t["v2"] = (z[f"v2kp{j}"], z[f"v2des{j}"])
                photos.append(t)
            PEOPLE[pid] = dict(name=meta.get("name", ""), created=meta["created"], photos=photos,
                               hashes=np.load(os.path.join(d, "search_hashes.npy")))
        except Exception as e:   # a damaged folder must not stop the kiosk
            print(f"skipping {d}: {e}", file=sys.stderr)


def next_id():
    n = 1 + max([int(p.split("-")[1]) for p in PEOPLE] + [0])
    return f"PALM-{n:04d}"


def audit(**entry):
    entry["time"] = datetime.datetime.now().isoformat(timespec="seconds")
    os.makedirs(DATA, exist_ok=True)
    with open(AUDIT, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def read_audit(n=200):
    if not os.path.exists(AUDIT):
        return []
    lines = open(AUDIT, encoding="utf-8").read().splitlines()[-n:]
    return [json.loads(x) for x in reversed(lines)]


# ------------------------------------------------------------------------------------------------ sessions
def locked_for():
    now = time.time()
    recent = [t for t in FAILS if now - t < LOCKOUT["window_s"]]
    FAILS[:] = recent
    if len(recent) >= LOCKOUT["failures"]:
        return max(0, int(LOCKOUT["lock_s"] - (now - recent[-1])))
    return 0


class Session:
    def __init__(self, mode, name):
        self.mode, self.name = mode, name
        self.t0 = time.time()
        self.phase = "position"
        self.good = 0
        self.challenge = None
        self.photos = []           # (bgr, pts, crop, recapture score)
        self.last_photo_t = 0.0
        self.capture_t0 = 0.0
        self.result = None
        self.events = []

    def frame(self, bgr):
        now = time.time()
        if now - self.t0 > FLOW_TIMEOUT_S and self.phase not in ("captured", "done"):
            self.phase = "done"
            self.result = dict(ok=False, kind="timeout", message="Took too long. Please start again.")
            audit(action=self.mode, outcome="timeout")
        if self.phase in ("captured", "done"):
            return self.view(None)
        a = eng.analyse(bgr, need_open_hand=self.phase != "liveness")
        if self.phase == "position":
            self.good = self.good + 1 if a["hand"] and not a["issues"] else 0
            if self.good >= GOOD_FRAMES_TO_START:
                self.phase = "liveness"
                self.challenge = pad.ActiveLiveness(n_steps=STEPS[self.mode], seed=secrets.randbits(32))
        elif self.phase == "liveness":
            st = self.challenge.update(a.get("pts"), now)
            if st["status"] == "failed":
                FAILS.append(now)
                self.phase = "done"
                self.result = dict(ok=False, kind="liveness", message=f"Liveness check failed: {st['reason']}.")
                audit(action=self.mode, outcome="liveness failed", reason=st["reason"], sequence=st["sequence"])
            elif st["status"] == "passed":
                self.phase, self.capture_t0 = "capture", now
                self.take(bgr, a, now)
        elif self.phase == "capture":
            if a["hand"] and not a["issues"] and a["count"] == 5 and now - self.last_photo_t > 0.3:
                self.take(bgr, a, now)
            if self.phase == "capture" and now - max(self.last_photo_t, self.capture_t0) > 6:
                self.phase = "done"
                self.result = dict(ok=False, kind="capture", message="Could not take a clear photo. Please start again.")
        return self.view(a)

    def take(self, bgr, a, now):
        if a.get("crop") is None or a["issues"]:
            return                    # the open-hand frame was not clean enough: keep holding, the next one is used
        self.photos.append(dict(crop=a["crop"], recapture=float(eng.recapture(bgr, a["pts"])), quality=a.get("quality")))
        self.last_photo_t = now
        if len(self.photos) >= PHOTOS[self.mode]:
            self.phase = "captured"

    def view(self, a):
        h = None
        if a and a.get("hand"):
            h = dict(pts=a["pts"].round(1).tolist(), count=a.get("count"), spread=round(a.get("spread") or 0, 1),
                     quality=round(a.get("quality") or 0))
        issues = [m for _, m in (a or {}).get("issues", [])]
        out = dict(phase=self.phase, hand=h, issues=issues, elapsed=round(time.time() - self.t0, 1),
                   photos=len(self.photos), photos_needed=PHOTOS[self.mode], ms=(a or {}).get("ms"))
        if self.challenge is not None:
            st = self.challenge.state()
            out["liveness"] = dict(prompt=st["prompt"], step=st["step"], steps=st["steps"], status=st["status"])
        if self.phase == "position":
            out["prompt"] = issues[0] if issues else ("hold still" if self.good else "show your open palm to the camera")
        elif self.phase == "liveness":
            out["prompt"] = out["liveness"]["prompt"]
        elif self.phase == "capture":
            out["prompt"] = "hold your open hand still"
        elif self.phase == "captured":
            out["prompt"] = "processing..."
        if self.result:
            out["result"] = self.result
        return out


# ------------------------------------------------------------------------------------------------ processing
def process(s):
    """Templates from the captured photos, passive spoof check, then registration or payment."""
    t0 = time.perf_counter()
    flags = [p["recapture"] for p in s.photos]
    flagged = any(r > pad.RECAPTURE_THRESHOLD for r in flags)
    if flagged and CFG["pad"] == "enforce":
        audit(action=s.mode, outcome="spoof suspected", recapture=max(flags))
        return dict(ok=False, kind="spoof", message="This looks like a photo of a screen or a print. Use your real hand.",
                    recapture=max(flags))
    tpls = [eng.make_template(p["crop"]) for p in s.photos]
    t1 = time.perf_counter()
    timing = dict(templates_ms=round((t1 - t0) * 1000))
    pad_info = dict(recapture=round(max(flags), 2), threshold=pad.RECAPTURE_THRESHOLD, flagged=flagged, mode=CFG["pad"])
    people = {i: dict(hashes=p["hashes"], photos=p["photos"]) for i, p in PEOPLE.items()}
    if s.mode == "register":
        dup = eng.search(tpls[0], people, KEY)
        timing.update(search_ms=dup["search_ms"], match_ms=dup["match_ms"], comparisons=dup["comparisons"])
        if dup["id"]:
            audit(action="register", outcome="duplicate", existing=dup["id"], score=dup["score"])
            return dict(ok=False, kind="duplicate", existing=dup["id"], name=PEOPLE[dup["id"]]["name"], score=dup["score"],
                        message=f"This palm is already registered as {dup['id']}.", timing=timing, pad=pad_info)
        pid = next_id()
        PEOPLE[pid] = dict(name=s.name, created=datetime.date.today().isoformat(), photos=tpls,
                           hashes=pp.protect_many(np.stack([t["hol"] for t in tpls]), KEY))
        save_person(pid)
        timing["total_ms"] = round((time.perf_counter() - t0) * 1000)
        audit(action="register", outcome="registered", id=pid, photos=len(tpls), recapture=pad_info["recapture"], **timing)
        return dict(ok=True, kind="registered", id=pid, name=s.name, message=f"Registered as {pid}.", timing=timing,
                    pad=pad_info)
    best = None
    for t in tpls:                          # payment: either captured photo may identify the person
        r = eng.search(t, people, KEY)
        if best is None or (r["id"] and not best["id"]) or (bool(r["id"]) == bool(best["id"]) and r["score"] > best["score"]):
            best = r
        if r["id"]:
            break
    timing.update(search_ms=best["search_ms"], match_ms=best["match_ms"], comparisons=best["comparisons"],
                  total_ms=round((time.perf_counter() - t0) * 1000))
    if best["id"]:
        audit(action="pay", outcome="identified", id=best["id"], score=best["score"], recapture=pad_info["recapture"], **timing)
        return dict(ok=True, kind="identified", id=best["id"], name=PEOPLE[best["id"]]["name"], score=best["score"],
                    message=f"Payment approved for {best['id']}.", candidates=best["candidates"], timing=timing, pad=pad_info)
    audit(action="pay", outcome="not recognised", best=best["score"], recapture=pad_info["recapture"], **timing)
    return dict(ok=False, kind="unknown", score=best["score"], candidates=best["candidates"],
                message="Palm not recognised. Register first, or try again.", timing=timing, pad=pad_info)


# ------------------------------------------------------------------------------------------------ routes
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "kiosk.html")


@app.get("/api/kiosk/state")
def api_state():
    people = [dict(id=i, name=p["name"], created=p["created"], photos=len(p["photos"]),
                   search_bytes=int(p["hashes"].nbytes),
                   template_bytes=os.path.getsize(os.path.join(PEOPLE_DIR, i, "templates.palm")))
              for i, p in sorted(PEOPLE.items())]
    return jsonify(people=people, audit=read_audit(), config=dict(eng.describe(), pad=CFG["pad"]),
                   locked=locked_for())


@app.post("/api/kiosk/start")
def api_start():
    mode = request.form.get("mode", "pay")
    if mode not in STEPS:
        return jsonify(error="unknown mode"), 400
    wait = locked_for()
    if wait:
        return jsonify(error=f"Too many failed attempts. Try again in {wait} s.", locked=wait), 429
    sid = secrets.token_hex(8)
    with LOCK:
        for k in [k for k, v in SESSIONS.items() if time.time() - v.t0 > 300]:
            SESSIONS.pop(k)
        SESSIONS[sid] = Session(mode, request.form.get("name", "").strip()[:40])
    return jsonify(session=sid, photos_needed=PHOTOS[mode], steps=STEPS[mode])


@app.post("/api/kiosk/frame")
def api_frame():
    s = SESSIONS.get(request.form.get("session", ""))
    if s is None:
        return jsonify(error="session expired"), 404
    f = request.files.get("image")
    bgr = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR) if f else None
    if bgr is None:
        return jsonify(error="no frame"), 400
    with LOCK:
        return jsonify(s.frame(bgr))


@app.post("/api/kiosk/finish")
def api_finish():
    sid = request.form.get("session", "")
    s = SESSIONS.get(sid)
    if s is None:
        return jsonify(error="session expired"), 404
    with LOCK:
        if s.phase == "captured":
            s.result = process(s)
            s.phase = "done"
        SESSIONS.pop(sid, None)
        return jsonify(result=s.result or dict(ok=False, kind="incomplete", message="Not finished."))


@app.post("/api/kiosk/cancel")
def api_cancel():
    SESSIONS.pop(request.form.get("session", ""), None)
    return jsonify(ok=True)


@app.post("/api/kiosk/delete")
def api_delete():
    pid = request.form.get("id", "")
    with LOCK:
        if pid in PEOPLE:
            PEOPLE.pop(pid)
            shutil.rmtree(os.path.join(PEOPLE_DIR, pid), ignore_errors=True)
            audit(action="delete", id=pid, outcome="deleted (templates and search hashes erased)")
    return api_state()


@app.get("/api/kiosk/audit.csv")
def api_audit_csv():
    rows = read_audit(100000)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("time", "action", "outcome", "id"), k))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
    return Response(buf.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=palm_audit.csv"})


def main():
    global KEY
    ap = argparse.ArgumentParser(description="palm payment kiosk")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5060)
    ap.add_argument("--pad", choices=("warn", "enforce"), default="warn",
                    help="passive screen/print check: warn = show the score only, enforce = reject flagged photos")
    a = ap.parse_args()
    CFG["pad"] = a.pad
    print("loading models ...", flush=True)
    eng.warm_up()
    KEY = load_key()
    load_people()
    print(f"ready on http://{a.host}:{a.port}   ({eng.describe()['device']}, {len(PEOPLE)} registered, "
          f"passive spoof check: {a.pad})", flush=True)
    app.run(host=a.host, port=a.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
