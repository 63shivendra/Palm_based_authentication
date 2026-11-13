#!/usr/bin/env python3
"""
webapp/app.py - live test dashboard for the palm matchers. Runs on this computer only.

  python webapp/app.py                  ->  http://127.0.0.1:5050
  python webapp/app.py --host 0.0.0.0   ->  also reachable from a phone on the same network
                                            (photo upload works; browsers allow the live camera only on
                                             localhost or https)

Enrol a person with 1-3 palm photos, then verify new photos. Every photo goes through all three matchers
and the dashboard shows their decisions, scores and timings side by side:

  v3    palm_match.py           full template (2,048 keypoints)
  v3c   palm_match_compact.py   compact template (1,024 keypoints, 1 byte per value)
  v3cr  palm_match_robust.py    v3c + noise-adaptive smoothing + retry on failure

Only templates are stored, in webapp/data/ (git-ignored). Photos are never written to disk.
There is NO spoof detection: a photo of a palm can be accepted.
"""
import argparse
import base64
import datetime
import json
import os
import re
import shutil
import sys
import threading
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import palm_match as pm  # noqa: E402
import palm_match_compact as pmc  # noqa: E402
import palm_match_robust as pmr  # noqa: E402

DATA = os.path.join(HERE, "data")
USERS_DIR = os.path.join(DATA, "users")
HISTORY_FILE = os.path.join(DATA, "history.json")
SAMPLES_DIR = os.path.join(ROOT, "real palm images")
MAX_PHOTOS = 3
MAX_HISTORY = 200
MODELS = [
    dict(key="v3", name="v3", detail="Full template, 2,048 keypoints", threshold=float(pm.V3_THRESHOLD)),
    dict(key="v3c", name="v3c", detail="Compact template, 1,024 keypoints", threshold=float(pmc.THRESHOLD)),
    dict(key="v3cr", name="v3cr", detail="v3c + noise smoothing and retry", threshold=float(pmr.THRESHOLD)),
]
THRESHOLD = {m["key"]: m["threshold"] for m in MODELS}
ATTEMPT = ["first attempt", "smoothed retry"] + [f"retry rotated {a:+d}°" for a in pmr.RETRY_ANGLES]

app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
LOCK = threading.Lock()          # one photo at a time: the models and the hand detector are not thread-safe
USERS = {}                       # slug -> dict(name, created, tpl={model: [templates]}, bytes={model: file size})
HISTORY = []
THUMBS = {}
# what the sample photos in "real palm images/" show (all four are the same person)
SAMPLE_NOTES = {"1000132768.jpg": "left hand", "1000132769.jpg": "right hand, day 1",
                "1000132924.jpg": "right hand, day 2", "1000132925.jpg": "right hand, day 2"}


# ------------------------------------------------------------------------------------------------ storage
def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40]


def save_user(s):
    u = USERS[s]
    d = os.path.join(USERS_DIR, s)
    os.makedirs(d, exist_ok=True)
    pm.save_templates_v3(os.path.join(d, "v3.npz"), u["tpl"]["v3"])
    pmc.save_templates(os.path.join(d, "v3c.npz"), u["tpl"]["v3c"])
    pmc.save_templates(os.path.join(d, "v3cr.npz"), u["tpl"]["v3cr"])
    u["bytes"] = {m["key"]: os.path.getsize(os.path.join(d, m["key"] + ".npz")) for m in MODELS}
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(dict(name=u["name"], created=u["created"]), f)


def load_users():
    if not os.path.isdir(USERS_DIR):
        return
    for s in sorted(os.listdir(USERS_DIR)):
        d = os.path.join(USERS_DIR, s)
        try:
            meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
            tpl = {"v3": pm.load_any_templates(os.path.join(d, "v3.npz"))[1],
                   "v3c": pmc.load_templates(os.path.join(d, "v3c.npz")),
                   "v3cr": pmc.load_templates(os.path.join(d, "v3cr.npz"))}
            USERS[s] = dict(meta, tpl=tpl, bytes={m["key"]: os.path.getsize(os.path.join(d, m["key"] + ".npz"))
                                                  for m in MODELS})
        except Exception as e:  # a half-written folder must not stop the app
            print(f"skipping {d}: {e}", file=sys.stderr)


def log(entry):
    entry["time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    HISTORY.insert(0, entry)
    del HISTORY[MAX_HISTORY:]
    os.makedirs(DATA, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(HISTORY, f)


def state():
    users = [dict(id=s, name=u["name"], created=u["created"], photos=len(u["tpl"]["v3"]), bytes=u["bytes"])
             for s, u in sorted(USERS.items(), key=lambda kv: kv[1]["name"].lower())]
    samples = sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))) \
        if os.path.isdir(SAMPLES_DIR) else []
    samples = [dict(file=f, note=SAMPLE_NOTES.get(f, "")) for f in samples]
    return dict(models=MODELS, users=users, history=HISTORY, samples=samples, max_photos=MAX_PHOTOS,
                device=pm._v3_models()["dev"])


# ------------------------------------------------------------------------------------------------ pipeline
def read_photo():
    """The uploaded photo (or a named sample photo) as a BGR image, and a short label of where it came from."""
    sample = request.form.get("sample", "")
    if sample:
        path = os.path.join(SAMPLES_DIR, os.path.basename(sample))
        if not os.path.isfile(path):
            raise ValueError("sample photo not found")
        return cv2.imread(path, cv2.IMREAD_COLOR), "sample " + os.path.basename(sample)
    f = request.files.get("image")
    if f is None:
        raise ValueError("no photo received")
    bgr = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("the file is not a readable image")
    return bgr, request.form.get("source", "upload")


def palm_crop(bgr):
    """Detect the hand, align and crop the palm. Returns (512x512 grey crop, quality info)."""
    t = time.perf_counter()
    pts = pm.detect_landmarks(bgr)
    left = bool(pm.to_right_hand(bgr, pts)[2])
    crop = pm.align_palm_v2(bgr, pts, normalise=False)
    small = cv2.resize(crop, (pm.SIZE, pm.SIZE), interpolation=cv2.INTER_AREA)
    info = dict(hand="left (mirrored)" if left else "right", noise=round(pmr.noise_level(crop), 2),
                sharpness=round(float(cv2.Laplacian(small, cv2.CV_64F).var()), 1),
                noisy=pmr.noise_level(crop) > pmr.NOISE_GATE, width=int(bgr.shape[1]), height=int(bgr.shape[0]),
                detect_ms=round((time.perf_counter() - t) * 1000))
    return crop, info


def preview(crop):
    ok, buf = cv2.imencode(".jpg", cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def best_of(results):
    i = max(range(len(results)), key=lambda k: results[k]["score"])
    return dict(results[i], photo=i + 1)


def run_models(crop, claimed):
    """Every matcher against every enrolled person. Timings are for the claimed person only."""
    order = [claimed] + [s for s in USERS if s != claimed]
    out, scores = {}, {s: {} for s in order}

    def timed(key, extract, match_user):
        t0 = time.perf_counter()
        probe = extract()
        t1 = time.perf_counter()
        r = match_user(probe, claimed)
        t2 = time.perf_counter()
        thr = THRESHOLD[key]
        out[key] = dict(match=bool(r["score"] >= thr), score=round(float(r["score"]), 1), threshold=thr,
                        inliers=int(r["inliers"]), cells=int(r["cells"]), photo=r["photo"],
                        attempt=ATTEMPT[r.get("attempt", 0)], extract_ms=round((t1 - t0) * 1000),
                        match_ms=round((t2 - t1) * 1000), total_ms=round((t2 - t0) * 1000))
        scores[claimed][key] = out[key]["score"]
        for s in order[1:]:
            scores[s][key] = round(float(match_user(probe, s)["score"]), 1)

    timed("v3", lambda: pm.features_v3(crop),
          lambda p, s: best_of([pm.match_v3(t, p) for t in USERS[s]["tpl"]["v3"]]))
    timed("v3c", lambda: pmc.features(crop),
          lambda p, s: best_of([pmc.match(t, p) for t in USERS[s]["tpl"]["v3c"]]))

    retries = {}                                   # the retry variants are only extracted if a first attempt fails

    def robust(first, s):
        r = best_of([dict(pmc.match(t, first), attempt=0) for t in USERS[s]["tpl"]["v3cr"]])
        if r["score"] >= THRESHOLD["v3cr"]:
            return r
        if "probe" not in retries:
            retries["probe"] = pmr.probe_of_crop(crop)
        return best_of([pmr.match(t, retries["probe"]) for t in USERS[s]["tpl"]["v3cr"]])

    timed("v3cr", lambda: pmc.features(pmr.condition(crop)), robust)
    others = [dict(id=s, name=USERS[s]["name"], claimed=s == claimed, scores=scores[s]) for s in order]
    return out, others


# ------------------------------------------------------------------------------------------------ routes
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/state")
def api_state():
    return jsonify(state())


@app.get("/api/sample/<name>")
def api_sample(name):
    """Small thumbnail of a sample photo, for the buttons."""
    name = os.path.basename(name)
    if name not in THUMBS:
        img = cv2.imread(os.path.join(SAMPLES_DIR, name), cv2.IMREAD_COLOR)
        if img is None:
            return "not found", 404
        k = 200 / max(img.shape[:2])
        ok, buf = cv2.imencode(".jpg", cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA))
        THUMBS[name] = buf.tobytes()
    return app.response_class(THUMBS[name], mimetype="image/jpeg")


@app.post("/api/enroll")
def api_enroll():
    name = re.sub(r"\s+", " ", request.form.get("name", "")).strip()[:40]
    s = slug(name)
    if not s:
        return jsonify(ok=False, error="Enter a name for the person being enrolled."), 400
    with LOCK:
        if s in USERS and len(USERS[s]["tpl"]["v3"]) >= MAX_PHOTOS:
            return jsonify(ok=False, error=f"{USERS[s]['name']} already has {MAX_PHOTOS} photos. Delete the person to start again."), 400
        try:
            bgr, source = read_photo()
            t = time.perf_counter()
            crop, info = palm_crop(bgr)
            tpl = {"v3": pm.features_v3(crop), "v3c": pmc.features(crop), "v3cr": pmr.template_of_crop(crop)}
        except pm.QualityError as e:
            log(dict(action="enrol", person=name, rejected=str(e)))
            return jsonify(ok=False, rejected=True, error=str(e), state=state())
        except ValueError as e:
            return jsonify(ok=False, error=str(e)), 400
        u = USERS.setdefault(s, dict(name=name, created=datetime.date.today().isoformat(),
                                     tpl={m["key"]: [] for m in MODELS}, bytes={}))
        for k, v in tpl.items():
            u["tpl"][k].append(v)
        save_user(s)
        n = len(u["tpl"]["v3"])
        log(dict(action="enrol", person=u["name"], photo=n, source=source, total_ms=round((time.perf_counter() - t) * 1000)))
        return jsonify(ok=True, person=u["name"], photo=n, crop=preview(crop), quality=info, state=state())


@app.post("/api/verify")
def api_verify():
    claimed = request.form.get("person", "")
    with LOCK:
        if claimed not in USERS:
            return jsonify(ok=False, error="Choose an enrolled person to verify against."), 400
        name = USERS[claimed]["name"]
        try:
            bgr, source = read_photo()
            t = time.perf_counter()
            crop, info = palm_crop(bgr)
            results, others = run_models(crop, claimed)
        except pm.QualityError as e:
            log(dict(action="verify", person=name, rejected=str(e)))
            return jsonify(ok=False, rejected=True, error=str(e), state=state())
        except ValueError as e:
            return jsonify(ok=False, error=str(e)), 400
        total = round((time.perf_counter() - t) * 1000)
        log(dict(action="verify", person=name, source=source, total_ms=total, detect_ms=info["detect_ms"],
                 results={k: dict(match=r["match"], score=r["score"], ms=r["total_ms"]) for k, r in results.items()}))
        return jsonify(ok=True, person=name, crop=preview(crop), quality=info, results=results, others=others,
                       total_ms=total, state=state())


@app.post("/api/delete_person")
def api_delete_person():
    s = request.form.get("person", "")
    with LOCK:
        if s in USERS:
            name = USERS.pop(s)["name"]
            shutil.rmtree(os.path.join(USERS_DIR, s), ignore_errors=True)
            log(dict(action="delete", person=name))
        return jsonify(ok=True, state=state())


@app.post("/api/clear_history")
def api_clear_history():
    with LOCK:
        HISTORY.clear()
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        return jsonify(ok=True, state=state())


def main():
    ap = argparse.ArgumentParser(description="live test dashboard for the palm matchers")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to allow other devices on the network")
    ap.add_argument("--port", type=int, default=5050)
    a = ap.parse_args()
    print("loading models ...", flush=True)
    pm._landmarker()
    pm.features_v3(np.zeros((pm.BIG, pm.BIG), np.uint8))          # warm up DISK so the first photo is not slow
    load_users()
    if os.path.exists(HISTORY_FILE):
        try:
            HISTORY.extend(json.load(open(HISTORY_FILE, encoding="utf-8")))
        except Exception:
            pass
    print(f"ready on http://{'127.0.0.1' if a.host == '0.0.0.0' else a.host}:{a.port}   "
          f"({pm._v3_models()['dev']}, {len(USERS)} enrolled)", flush=True)
    app.run(host=a.host, port=a.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
