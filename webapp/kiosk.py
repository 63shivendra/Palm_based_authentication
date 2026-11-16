#!/usr/bin/env python3
"""
webapp/kiosk.py - palm payment kiosk, end to end: register a palm with a demo wallet, pay a merchant with the palm
alone (1:N search), PIN step-up for larger amounts, self-service account view and erasure, admin console.

  python webapp/kiosk.py                       ->  http://127.0.0.1:5060
  python webapp/kiosk.py --https --host 0.0.0.0 ->  https://<this-pc>:5060 from a phone on the same network
                                                   (self-signed certificate made with openssl; the browser warns once)
  python webapp/kiosk.py --pad enforce         ->  reject photos the passive screen/print check flags (default: warn)

SIMULATED PAYMENTS: wallets, balances and payment messages are a demo. Nothing is sent to a bank, NPCI or UPI.

Flow (the camera streams frames to this server; every check runs here, not in the browser):
  1. position   live instructions until the palm is sharp, well lit, open and close enough (palm_quality.py)
  2. liveness   random finger counts at random times, then the open hand (palm_pad.ActiveLiveness)
  3. capture    the open-hand frames are the photos used (3 for registration, 2 otherwise)
  4. register   consent recorded, duplicate search, next free palm ID, demo wallet with optional PIN
     pay        protected-vector shortlist + matcher -> limits -> PIN above the step-up amount -> receipt
     account    identify -> balance and recent payments, or erase everything stored for this palm

Stored in webapp/data/kiosk/ (git-ignored): palmpay.db (people, wallets, transactions, audit; PINs as PBKDF2 hashes),
per person the keypoint templates encrypted with Windows DPAPI and the cancellable IoM search hashes. Photos are never
written to disk.
"""
import argparse
import csv
import io
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import palm_engine as eng  # noqa: E402
import palm_pad as pad  # noqa: E402
import palm_protect as pp  # noqa: E402
import palmpay_db as pdb  # noqa: E402

DATA = os.path.join(HERE, "data", "kiosk")
PEOPLE_DIR = os.path.join(DATA, "people")
KEY_FILE = os.path.join(DATA, "search.key")
DB_FILE = os.path.join(DATA, "palmpay.db")
STEPS = {"register": 4, "pay": 3, "account": 3}       # liveness steps (the last one is always the open hand)
PHOTOS = {"register": 3, "pay": 2, "account": 2}
GOOD_FRAMES_TO_START = 3
FLOW_TIMEOUT_S = 60
LOCKOUT = dict(failures=3, window_s=120, lock_s=60)
RULES = dict(step_up=2000.0, max_payment=10000.0, daily_limit=5000.0, opening_balance=5000.0)
MERCHANT = dict(name="Demo Kirana Store", upi="demo.kirana@upi")
UPI_RE = re.compile(r"^[a-z0-9._-]{2,64}@[a-z]{2,32}$")
ADMIN_TTL_S = 1800

app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
LOCK = threading.Lock()
TEMPLATES = {}       # palm_id -> dict(hashes, photos): decrypted in memory only
SESSIONS = {}
FAILS = []           # times of failed liveness / PIN attempts (kiosk-wide lockout)
ADMIN = dict(tokens={}, fails=[])
CFG = dict(pad="warn")
KEY = b""
DB = None


# ------------------------------------------------------------------------------------------------ storage
def load_key():
    os.makedirs(DATA, exist_ok=True)
    if os.path.exists(KEY_FILE):
        return pp.decrypt(open(KEY_FILE, "rb").read())
    k = secrets.token_bytes(32)
    with open(KEY_FILE, "wb") as f:
        f.write(pp.encrypt(k))
    return k


def save_templates(pid, photos, hashes):
    d = os.path.join(PEOPLE_DIR, pid)
    os.makedirs(d, exist_ok=True)
    arrays = {}
    for j, t in enumerate(photos):
        arrays[f"kp{j}"], arrays[f"des{j}"] = t["v3c"]
        if "v2" in t:
            arrays[f"v2kp{j}"], arrays[f"v2des{j}"] = t["v2"]
    pp.save_encrypted(os.path.join(d, "templates.palm"), **arrays)
    np.save(os.path.join(d, "search_hashes.npy"), hashes)


def load_templates():
    TEMPLATES.clear()
    for pid in DB.palm_ids():
        d = os.path.join(PEOPLE_DIR, pid)
        try:
            z = pp.load_encrypted(os.path.join(d, "templates.palm"))
            photos = []
            for j in range(sum(1 for k in z if k.startswith("kp"))):
                t = dict(v3c=(z[f"kp{j}"], z[f"des{j}"]))
                if f"v2kp{j}" in z:
                    t["v2"] = (z[f"v2kp{j}"], z[f"v2des{j}"])
                photos.append(t)
            TEMPLATES[pid] = dict(photos=photos, hashes=np.load(os.path.join(d, "search_hashes.npy")))
        except Exception as e:     # a damaged folder must not stop the kiosk
            print(f"skipping {pid}: {e}", file=sys.stderr)


def erase(pid):
    """Right to erasure: templates, search hashes, wallet and person record. Transactions keep a masked payer ID."""
    TEMPLATES.pop(pid, None)
    shutil.rmtree(os.path.join(PEOPLE_DIR, pid), ignore_errors=True)
    DB.delete_person(pid)


def next_id():
    used = [int(p.split("-")[1]) for p in DB.palm_ids()] + [int(p.split("-")[1]) for p in TEMPLATES]
    return f"PALM-{1 + max(used + [0]):04d}"


# ------------------------------------------------------------------------------------------------ sessions
def locked_for():
    now = time.time()
    FAILS[:] = [t for t in FAILS if now - t < LOCKOUT["window_s"]]
    if len(FAILS) >= LOCKOUT["failures"]:
        return max(0, int(LOCKOUT["lock_s"] - (now - FAILS[-1])))
    return 0


class Session:
    def __init__(self, mode, form):
        self.mode, self.form = mode, form
        self.t0 = time.time()
        self.phase = "position"
        self.good = 0
        self.challenge = None
        self.photos = []
        self.last_photo_t = self.capture_t0 = 0.0
        self.result = None
        self.pending = None          # pay: identified person waiting for the PIN; account: person shown
        self.pin_tries = 0
        self.until = 0.0             # follow-up (PIN entry, account view) allowed until this time

    def frame(self, bgr):
        now = time.time()
        if now - self.t0 > FLOW_TIMEOUT_S and self.phase in ("position", "liveness", "capture"):
            self.phase = "done"
            self.result = dict(ok=False, kind="timeout", message="Took too long. Please start again.")
            DB.audit(self.mode, "timeout")
        if self.phase not in ("position", "liveness", "capture"):
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
                DB.audit(self.mode, "liveness failed", reason=st["reason"], sequence=st["sequence"])
            elif st["status"] == "passed":
                self.phase, self.capture_t0 = "capture", now
                self.take(bgr, a, now)
        elif self.phase == "capture":
            if a["hand"] and not a["issues"] and a["count"] == 5 and now - self.last_photo_t > 0.3:
                self.take(bgr, a, now)
            if self.phase == "capture" and now - max(self.last_photo_t, self.capture_t0) > 6:
                self.phase = "done"
                self.result = dict(ok=False, kind="capture", message="Could not take a clear photo. Please start again.")
                DB.audit(self.mode, "no clear photo")
        return self.view(a)

    def take(self, bgr, a, now):
        if a.get("crop") is None or a["issues"]:
            return                    # the open-hand frame was not clean enough: keep holding, the next one is used
        self.photos.append(dict(crop=a["crop"], recapture=float(eng.recapture(bgr, a["pts"]))))
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
        prompts = dict(position=issues[0] if issues else ("hold still" if self.good else "show your open palm to the camera"),
                       liveness=out.get("liveness", {}).get("prompt", ""), capture="hold your open hand still",
                       captured="processing...")
        out["prompt"] = prompts.get(self.phase, "")
        if self.result:
            out["result"] = self.result
        return out


# ------------------------------------------------------------------------------------------------ processing
def people_index():
    return {i: dict(hashes=t["hashes"], photos=t["photos"]) for i, t in TEMPLATES.items()}


def receipt(txn, extra=None):
    ok = txn["status"] == "approved"
    return dict(ok=ok, kind="approved" if ok else "declined", txn=txn,
                message=("Payment approved." if ok else f"Payment declined: {txn['reason']}."), **(extra or {}))


def process(s):
    """Templates from the captured photos, passive spoof check, then registration, payment or account lookup."""
    t0 = time.perf_counter()
    flags = [p["recapture"] for p in s.photos]
    flagged = any(r > pad.RECAPTURE_THRESHOLD for r in flags)
    pad_info = dict(recapture=round(max(flags), 2), threshold=pad.RECAPTURE_THRESHOLD, flagged=flagged, mode=CFG["pad"])
    if flagged and CFG["pad"] == "enforce":
        DB.audit(s.mode, "spoof suspected", recapture=max(flags))
        return dict(ok=False, kind="spoof", message="This looks like a photo of a screen or a print. Use your real hand.",
                    pad=pad_info)
    tpls = [eng.make_template(p["crop"]) for p in s.photos]
    timing = dict(templates_ms=round((time.perf_counter() - t0) * 1000))
    f = s.form

    if s.mode == "register":
        dup = eng.search(tpls[0], people_index(), KEY)
        timing.update(search_ms=dup["search_ms"], match_ms=dup["match_ms"], comparisons=dup["comparisons"])
        if dup["id"]:
            DB.audit("register", "duplicate", dup["id"], score=dup["score"])
            return dict(ok=False, kind="duplicate", existing=dup["id"], score=dup["score"], timing=timing, pad=pad_info,
                        message=f"This palm is already registered as {dup['id']}.")
        pid = next_id()
        hashes = pp.protect_many(np.stack([t["hol"] for t in tpls]), KEY)
        save_templates(pid, tpls, hashes)
        DB.add_person(pid, f["name"], len(tpls), f["upi"], pdb.paise(f["balance"]), pdb.paise(RULES["daily_limit"]),
                      f["pin"] or None)
        TEMPLATES[pid] = dict(photos=tpls, hashes=hashes)
        timing["total_ms"] = round((time.perf_counter() - t0) * 1000)
        DB.audit("register", "registered", pid, consent=True, pin=bool(f["pin"]), recapture=pad_info["recapture"], **timing)
        return dict(ok=True, kind="registered", id=pid, name=f["name"], upi=pdb.mask_upi(f["upi"]),
                    balance=float(f["balance"]), message=f"Registered as {pid}.", timing=timing, pad=pad_info)

    best = None
    for t in tpls:                          # either captured photo may identify the person
        r = eng.search(t, people_index(), KEY)
        if best is None or (r["id"] and not best["id"]) or (bool(r["id"]) == bool(best["id"]) and r["score"] > best["score"]):
            best = r
        if r["id"]:
            break
    timing.update(search_ms=best["search_ms"], match_ms=best["match_ms"], comparisons=best["comparisons"],
                  total_ms=round((time.perf_counter() - t0) * 1000))
    common = dict(timing=timing, pad=pad_info, score=best["score"], candidates=best["candidates"])

    if s.mode == "account":
        if not best["id"]:
            DB.audit("account", "not recognised", best=best["score"])
            return dict(ok=False, kind="unknown", message="Palm not recognised.", **common)
        s.phase, s.pending, s.until = "account", best["id"], time.time() + 120
        DB.audit("account", "viewed", best["id"], score=best["score"])
        return dict(ok=True, kind="account", **account_view(best["id"]), **common)

    amount = pdb.paise(f["amount"])
    if not best["id"]:
        DB.record_failed_payment(amount, MERCHANT["name"], MERCHANT["upi"], "palm not recognised", score=best["score"],
                                 latency_ms=timing["total_ms"])
        DB.audit("pay", "not recognised", best=best["score"], amount=f["amount"], **timing)
        return dict(ok=False, kind="unknown", message="Palm not recognised. Register first, or try again.", **common)
    pid = best["id"]
    DB.audit("pay", "identified", pid, score=best["score"], amount=f["amount"], **timing)
    person = DB.person(pid)
    if float(f["amount"]) > RULES["step_up"]:
        if not person["pin_hash"]:
            reason = f"a PIN is needed above Rs {RULES['step_up']:,.0f} and this palm ID has no PIN"
            txn_id = DB.record_failed_payment(amount, MERCHANT["name"], MERCHANT["upi"], reason, palm_id=pid,
                                              score=best["score"], latency_ms=timing["total_ms"])
            DB.audit("payment", "declined", pid, txn=txn_id, amount=f["amount"], reason=reason)
            txn = dict(txn_id=txn_id, status="declined", reason=reason, amount=float(f["amount"]),
                       balance=pdb.rupees(person["balance_paise"]), payer=pdb.mask_upi(person["upi_id"]),
                       merchant=MERCHANT["name"], auth="palm", message=None)
            return dict(receipt(txn, dict(id=pid, name=person["name"])), **common)
        s.phase, s.pending, s.until = "pin", dict(pid=pid, score=best["score"], latency=timing["total_ms"]), time.time() + 60
        return dict(ok=True, kind="pin_required", id=pid, name=person["name"], amount=float(f["amount"]),
                    message=f"Amount above Rs {RULES['step_up']:,.0f}: enter the PIN of {pid}.", **common)
    txn = DB.pay(pid, amount, MERCHANT["name"], MERCHANT["upi"], "palm", best["score"], timing["total_ms"],
                 pdb.paise(RULES["max_payment"]))
    DB.audit("payment", txn["status"], pid, txn=txn["txn_id"], amount=f["amount"], reason=txn["reason"])
    return dict(receipt(txn, dict(id=pid, name=person["name"])), **common)


def account_view(pid):
    p = DB.person(pid)
    return dict(id=pid, name=p["name"], upi=pdb.mask_upi(p["upi_id"]), balance=pdb.rupees(p["balance_paise"]),
                daily_limit=pdb.rupees(p["daily_limit_paise"]), spent_today=pdb.rupees(DB.spent_today(pid)),
                registered=p["created"], has_pin=bool(p["pin_hash"]), transactions=DB.transactions(pid, 10))


# ------------------------------------------------------------------------------------------------ kiosk routes
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "kiosk.html")


@app.get("/api/kiosk/state")
def api_state():
    return jsonify(config=dict(eng.describe(), pad=CFG["pad"]), registered=len(TEMPLATES), locked=locked_for(),
                   rules=RULES, merchant=MERCHANT, admin_setup=DB.setting("admin_pin") is not None, simulated=True)


def bad(msg, code=400):
    return jsonify(error=msg), code


@app.post("/api/kiosk/start")
def api_start():
    mode = request.form.get("mode", "pay")
    if mode not in STEPS:
        return bad("unknown mode")
    wait = locked_for()
    if wait:
        return jsonify(error=f"Too many failed attempts. Try again in {wait} s.", locked=wait), 429
    g = lambda k, d="": request.form.get(k, d).strip()
    form = {}
    if mode == "register":
        if g("consent") != "1":
            return bad("Please read and accept the consent to register.")
        upi, pin = g("upi").lower(), g("pin")
        if not UPI_RE.match(upi):
            return bad("Enter a UPI ID like name@bank (demo only).")
        if pin and not re.fullmatch(r"\d{4,6}", pin):
            return bad("The PIN must be 4 to 6 digits, or left empty.")
        try:
            balance = float(g("balance", str(RULES["opening_balance"])))
        except ValueError:
            return bad("Opening balance must be a number.")
        if not 0 <= balance <= 100000:
            return bad("Opening balance (demo) must be between 0 and 1,00,000.")
        form = dict(name=g("name")[:40], upi=upi, pin=pin, balance=balance)
    elif mode == "pay":
        try:
            amount = float(g("amount"))
        except ValueError:
            return bad("Enter the amount to pay.")
        if not 0 < amount <= RULES["max_payment"]:
            return bad(f"The amount must be between 1 and {RULES['max_payment']:,.0f}.")
        form = dict(amount=round(amount, 2))
    sid = secrets.token_hex(12)
    with LOCK:
        for k in [k for k, v in SESSIONS.items() if time.time() - v.t0 > 300]:
            SESSIONS.pop(k)
        SESSIONS[sid] = Session(mode, form)
    return jsonify(session=sid, photos_needed=PHOTOS[mode], steps=STEPS[mode])


@app.post("/api/kiosk/frame")
def api_frame():
    s = SESSIONS.get(request.form.get("session", ""))
    if s is None:
        return bad("session expired", 404)
    f = request.files.get("image")
    bgr = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR) if f else None
    if bgr is None:
        return bad("no frame")
    with LOCK:
        return jsonify(s.frame(bgr))


@app.post("/api/kiosk/finish")
def api_finish():
    sid = request.form.get("session", "")
    s = SESSIONS.get(sid)
    if s is None:
        return bad("session expired", 404)
    with LOCK:
        if s.phase == "captured":
            s.phase = "done"
            s.result = process(s)
        if s.phase not in ("pin", "account"):          # follow-up steps keep the session for a short time
            SESSIONS.pop(sid, None)
        return jsonify(result=s.result or dict(ok=False, kind="incomplete", message="Not finished."))


@app.post("/api/kiosk/pin")
def api_pin():
    sid = request.form.get("session", "")
    s = SESSIONS.get(sid)
    if s is None or s.phase != "pin" or time.time() > s.until:
        SESSIONS.pop(sid, None)
        return bad("This payment has expired. Please start again.", 404)
    with LOCK:
        p = s.pending
        person = DB.person(p["pid"])
        if not pdb.check_pin(request.form.get("pin", ""), person["pin_hash"]):
            s.pin_tries += 1
            DB.audit("pin", "wrong PIN", p["pid"], tries=s.pin_tries)
            if s.pin_tries >= 3:
                FAILS.append(time.time())
                SESSIONS.pop(sid, None)
                DB.record_failed_payment(pdb.paise(s.form["amount"]), MERCHANT["name"], MERCHANT["upi"], "wrong PIN 3 times",
                                         palm_id=p["pid"], score=p["score"], latency_ms=p["latency"])
                return jsonify(result=dict(ok=False, kind="declined", message="Payment declined: wrong PIN 3 times."))
            return jsonify(retry=True, tries_left=3 - s.pin_tries)
        txn = DB.pay(p["pid"], pdb.paise(s.form["amount"]), MERCHANT["name"], MERCHANT["upi"], "palm + PIN", p["score"],
                     p["latency"], pdb.paise(RULES["max_payment"]))
        DB.audit("payment", txn["status"], p["pid"], txn=txn["txn_id"], amount=s.form["amount"], reason=txn["reason"], pin=True)
        SESSIONS.pop(sid, None)
        return jsonify(result=receipt(txn, dict(id=p["pid"], name=person["name"])))


@app.post("/api/kiosk/erase")
def api_erase():
    sid = request.form.get("session", "")
    s = SESSIONS.get(sid)
    if s is None or s.phase != "account" or time.time() > s.until:
        return bad("Please show your palm again to erase your data.", 404)
    with LOCK:
        pid = s.pending
        erase(pid)
        SESSIONS.pop(sid, None)
        DB.audit("erase", "erased by the person", pid)
        return jsonify(ok=True, id=pid)


@app.post("/api/kiosk/cancel")
def api_cancel():
    SESSIONS.pop(request.form.get("session", ""), None)
    return jsonify(ok=True)


# ------------------------------------------------------------------------------------------------ admin routes
def admin_ok():
    tok = request.headers.get("X-Admin-Token") or request.args.get("token", "")
    exp = ADMIN["tokens"].get(tok)
    if exp and exp > time.time():
        ADMIN["tokens"][tok] = time.time() + ADMIN_TTL_S
        return True
    return False


@app.post("/api/admin/setup")
def api_admin_setup():
    if DB.setting("admin_pin"):
        return bad("The admin PIN is already set.", 403)
    pin = request.form.get("pin", "")
    if not re.fullmatch(r"\d{6,8}", pin):
        return bad("The admin PIN must be 6 to 8 digits.")
    DB.set_setting("admin_pin", pdb.hash_pin(pin))
    DB.audit("admin", "admin PIN set")
    return api_admin_login()


@app.post("/api/admin/login")
def api_admin_login():
    now = time.time()
    ADMIN["fails"] = [t for t in ADMIN["fails"] if now - t < 300]
    if len(ADMIN["fails"]) >= 5:
        return bad("Too many wrong admin PINs. Wait 5 minutes.", 429)
    if not pdb.check_pin(request.form.get("pin", ""), DB.setting("admin_pin")):
        ADMIN["fails"].append(now)
        DB.audit("admin", "wrong admin PIN")
        return bad("Wrong admin PIN.", 403)
    tok = secrets.token_urlsafe(24)
    ADMIN["tokens"][tok] = now + ADMIN_TTL_S
    DB.audit("admin", "login")
    return jsonify(token=tok)


@app.get("/api/admin/overview")
def api_admin_overview():
    if not admin_ok():
        return bad("admin login required", 401)
    people = []
    for p in DB.people():
        d = os.path.join(PEOPLE_DIR, p["palm_id"], "templates.palm")
        people.append(dict(id=p["palm_id"], name=p["name"], created=p["created"], consent=p["consent_at"],
                           photos=p["photos"], upi=pdb.mask_upi(p["upi_id"]), balance=pdb.rupees(p["balance_paise"]),
                           daily_limit=pdb.rupees(p["daily_limit_paise"]), spent_today=pdb.rupees(DB.spent_today(p["palm_id"])),
                           has_pin=bool(p["has_pin"]), template_bytes=os.path.getsize(d) if os.path.exists(d) else 0,
                           search_bytes=int(TEMPLATES[p["palm_id"]]["hashes"].nbytes) if p["palm_id"] in TEMPLATES else 0))
    return jsonify(people=people, transactions=DB.transactions(limit=200), audit=DB.audit_log(300), stats=DB.stats(24),
                   config=dict(eng.describe(), pad=CFG["pad"], rules=RULES, merchant=MERCHANT), locked=locked_for())


@app.post("/api/admin/delete")
def api_admin_delete():
    if not admin_ok():
        return bad("admin login required", 401)
    pid = request.form.get("id", "")
    with LOCK:
        if pid in TEMPLATES or DB.person(pid):
            erase(pid)
            DB.audit("erase", "erased by the admin", pid)
    return api_admin_overview()


@app.post("/api/admin/topup")
def api_admin_topup():
    if not admin_ok():
        return bad("admin login required", 401)
    try:
        amount = float(request.form.get("amount", ""))
    except ValueError:
        return bad("amount must be a number")
    pid = request.form.get("id", "")
    if not DB.person(pid) or not 0 < amount <= 100000:
        return bad("unknown palm ID or amount out of range")
    DB.top_up(pid, pdb.paise(amount))
    DB.audit("admin", "demo top-up", pid, amount=amount)
    return api_admin_overview()


@app.get("/api/admin/export/<kind>.csv")
def api_admin_export(kind):
    if not admin_ok():
        return bad("admin login required", 401)
    rows = DB.audit_log(100000) if kind == "audit" else DB.transactions(limit=100000) if kind == "transactions" else None
    if rows is None:
        return bad("unknown export", 404)
    keys = list(dict.fromkeys(k for r in rows for k in r)) or ["empty"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=palmpay_{kind}.csv"})


# ------------------------------------------------------------------------------------------------ start-up
def https_files():
    """Self-signed certificate for phone access (browsers allow the camera only on https or localhost)."""
    d = os.path.join(HERE, "data", "cert")
    cert, key = os.path.join(d, "cert.pem"), os.path.join(d, "key.pem")
    if not (os.path.exists(cert) and os.path.exists(key)):
        os.makedirs(d, exist_ok=True)
        exe = shutil.which("openssl") or next((p for p in (r"C:\Program Files\Git\usr\bin\openssl.exe",
                                                           r"C:\Program Files\Git\mingw64\bin\openssl.exe") if os.path.exists(p)), None)
        if not exe:
            sys.exit("--https needs openssl (it comes with Git for Windows) to make a certificate")
        subprocess.run([exe, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", key, "-out", cert, "-days", "365",
                        "-subj", "/CN=palm-kiosk"], check=True, capture_output=True)
    return cert, key


def init(data_dir=None):
    """Open the database, the search key and the templates (also used by the tests with a temporary folder)."""
    global DATA, PEOPLE_DIR, KEY_FILE, DB_FILE, KEY, DB
    if data_dir:
        DATA = data_dir
        PEOPLE_DIR, KEY_FILE, DB_FILE = (os.path.join(DATA, "people"), os.path.join(DATA, "search.key"),
                                         os.path.join(DATA, "palmpay.db"))
    KEY = load_key()
    DB = pdb.Store(DB_FILE)
    load_templates()


def main():
    ap = argparse.ArgumentParser(description="palm payment kiosk (simulated payments)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5060)
    ap.add_argument("--https", action="store_true", help="serve over https with a self-signed certificate (phone cameras)")
    ap.add_argument("--pad", choices=("warn", "enforce"), default="warn",
                    help="passive screen/print check: warn = show the score only, enforce = reject flagged photos")
    ap.add_argument("--step-up", type=float, default=RULES["step_up"], help="ask for the PIN above this amount (Rs)")
    a = ap.parse_args()
    CFG["pad"], RULES["step_up"] = a.pad, a.step_up
    print("loading models ...", flush=True)
    eng.warm_up()
    init()
    ssl = https_files() if a.https else None
    print(f"ready on http{'s' if ssl else ''}://{a.host}:{a.port}   ({eng.describe()['device']}, {len(TEMPLATES)} registered, "
          f"passive spoof check: {a.pad}, PIN above Rs {RULES['step_up']:,.0f})", flush=True)
    app.run(host=a.host, port=a.port, debug=False, threaded=True, ssl_context=ssl)


if __name__ == "__main__":
    main()
