"""
webapp/palmpay_db.py - SQLite store of the palm payment kiosk: people, demo wallets, transactions, audit log, settings.

SIMULATED PAYMENTS: every account is a demo wallet with a made-up balance. Nothing here talks to a bank, NPCI or UPI.
The payment message attached to each transaction follows the shape of an ISO 20022 pacs.008 credit transfer so the
flow can be shown end to end; it is marked SIMULATED and carries an HMAC-SHA256 integrity tag with a local key, not a
bank signature.

Biometric data is not in this database: the palm templates are encrypted files next to it (palm_protect.py) and the
search hashes are cancellable IoM hashes. PINs are stored as PBKDF2-SHA256 hashes.
"""
import datetime
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading

SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    palm_id TEXT PRIMARY KEY, name TEXT, created TEXT NOT NULL, consent_at TEXT NOT NULL, photos INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS accounts (
    palm_id TEXT PRIMARY KEY REFERENCES people(palm_id) ON DELETE CASCADE, upi_id TEXT NOT NULL,
    balance_paise INTEGER NOT NULL, daily_limit_paise INTEGER NOT NULL, pin_hash TEXT);
CREATE TABLE IF NOT EXISTS transactions (
    txn_id TEXT PRIMARY KEY, time TEXT NOT NULL, palm_id TEXT, payer_upi TEXT, merchant TEXT NOT NULL,
    merchant_upi TEXT NOT NULL, amount_paise INTEGER NOT NULL, status TEXT NOT NULL, reason TEXT, auth TEXT,
    score REAL, latency_ms INTEGER, message TEXT);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT NOT NULL, action TEXT NOT NULL, outcome TEXT, palm_id TEXT,
    detail TEXT);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""
PBKDF2_ROUNDS = 200_000


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def paise(rupees):
    return int(round(float(rupees) * 100))


def rupees(p):
    return round(p / 100, 2)


def mask_upi(upi):
    """asha.k@upi -> as****@upi (receipts and the admin view never show the full ID)."""
    if not upi or "@" not in upi:
        return upi or ""
    user, bank = upi.split("@", 1)
    return f"{user[:2]}{'*' * max(len(user) - 2, 2)}@{bank}"


def hash_pin(pin, salt=None):
    salt = salt or secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", pin.encode(), bytes.fromhex(salt), PBKDF2_ROUNDS).hex()
    return f"pbkdf2${PBKDF2_ROUNDS}${salt}${h}"


def check_pin(pin, stored):
    if not stored:
        return False
    _, rounds, salt, h = stored.split("$")
    test = hashlib.pbkdf2_hmac("sha256", pin.encode(), bytes.fromhex(salt), int(rounds)).hex()
    return hmac.compare_digest(test, h)


class Store:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self.db.commit()
        if self.setting("message_key") is None:
            self.set_setting("message_key", secrets.token_hex(32))

    # ---------------------------------------------------------------- settings
    def setting(self, key, default=None):
        r = self.db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return r["value"] if r else default

    def set_setting(self, key, value):
        with self.lock:
            self.db.execute("INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = ?",
                            (key, value, value))
            self.db.commit()

    # ---------------------------------------------------------------- people and accounts
    def add_person(self, palm_id, name, photos, upi_id, balance_paise, daily_limit_paise, pin):
        t = now()
        with self.lock:
            self.db.execute("INSERT INTO people VALUES(?, ?, ?, ?, ?)", (palm_id, name, t, t, photos))
            self.db.execute("INSERT INTO accounts VALUES(?, ?, ?, ?, ?)",
                            (palm_id, upi_id, balance_paise, daily_limit_paise, hash_pin(pin) if pin else None))
            self.db.commit()

    def delete_person(self, palm_id):
        with self.lock:
            self.db.execute("DELETE FROM people WHERE palm_id = ?", (palm_id,))   # account goes with it (cascade)
            self.db.commit()

    def person(self, palm_id):
        r = self.db.execute("SELECT p.*, a.upi_id, a.balance_paise, a.daily_limit_paise, a.pin_hash FROM people p "
                            "JOIN accounts a USING(palm_id) WHERE palm_id = ?", (palm_id,)).fetchone()
        return dict(r) if r else None

    def people(self):
        rows = self.db.execute("SELECT p.*, a.upi_id, a.balance_paise, a.daily_limit_paise, a.pin_hash IS NOT NULL AS has_pin "
                               "FROM people p JOIN accounts a USING(palm_id) ORDER BY palm_id").fetchall()
        return [dict(r) for r in rows]

    def palm_ids(self):
        return [r["palm_id"] for r in self.db.execute("SELECT palm_id FROM people ORDER BY palm_id")]

    def top_up(self, palm_id, amount_paise):
        with self.lock:
            self.db.execute("UPDATE accounts SET balance_paise = balance_paise + ? WHERE palm_id = ?", (amount_paise, palm_id))
            self.db.commit()

    def spent_today(self, palm_id):
        day = datetime.date.today().isoformat()
        r = self.db.execute("SELECT COALESCE(SUM(amount_paise), 0) AS s FROM transactions WHERE palm_id = ? AND "
                            "status = 'approved' AND time >= ?", (palm_id, day)).fetchone()
        return int(r["s"])

    # ---------------------------------------------------------------- payments
    def pay(self, palm_id, amount_paise, merchant, merchant_upi, auth, score, latency_ms, max_txn_paise):
        """Apply the rules and record the transaction. Returns the transaction dict (status approved / declined)."""
        with self.lock:
            p = self.person(palm_id)
            reason = None
            if amount_paise <= 0:
                reason = "invalid amount"
            elif amount_paise > max_txn_paise:
                reason = f"above the per-payment limit of Rs {rupees(max_txn_paise):,.0f}"
            elif self.spent_today(palm_id) + amount_paise > p["daily_limit_paise"]:
                reason = f"daily limit of Rs {rupees(p['daily_limit_paise']):,.0f} reached"
            elif amount_paise > p["balance_paise"]:
                reason = "insufficient balance"
            status = "declined" if reason else "approved"
            txn_id = f"TXN{datetime.datetime.now():%Y%m%d%H%M%S}{secrets.token_hex(3).upper()}"
            msg = self.payment_message(txn_id, p["upi_id"], merchant, merchant_upi, amount_paise, auth, status)
            if status == "approved":
                self.db.execute("UPDATE accounts SET balance_paise = balance_paise - ? WHERE palm_id = ?",
                                (amount_paise, palm_id))
            self.db.execute("INSERT INTO transactions VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (txn_id, now(), palm_id, p["upi_id"], merchant, merchant_upi, amount_paise, status, reason,
                             auth, score, latency_ms, json.dumps(msg)))
            self.db.commit()
            return dict(txn_id=txn_id, status=status, reason=reason, amount=rupees(amount_paise),
                        balance=rupees(self.person(palm_id)["balance_paise"]), payer=mask_upi(p["upi_id"]),
                        merchant=merchant, auth=auth, message=msg)

    def record_failed_payment(self, amount_paise, merchant, merchant_upi, reason, palm_id=None, score=None, latency_ms=None):
        txn_id = f"TXN{datetime.datetime.now():%Y%m%d%H%M%S}{secrets.token_hex(3).upper()}"
        with self.lock:
            self.db.execute("INSERT INTO transactions VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (txn_id, now(), palm_id, None, merchant, merchant_upi, amount_paise, "declined", reason,
                             None, score, latency_ms, None))
            self.db.commit()
        return txn_id

    def payment_message(self, txn_id, payer_upi, merchant, merchant_upi, amount_paise, auth, status):
        """pacs.008-shaped credit transfer, marked SIMULATED, with an HMAC-SHA256 tag (local key, not a bank signature)."""
        msg = {
            "Simulated": True,
            "MessageDefinition": "pacs.008-like credit transfer (simulation, not sent anywhere)",
            "MessageId": f"SIM-{secrets.token_hex(8).upper()}",
            "CreationDateTime": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "TransactionId": txn_id,
            "Amount": {"Currency": "INR", "Value": rupees(amount_paise)},
            "Debtor": {"VirtualPaymentAddress": mask_upi(payer_upi)},
            "Creditor": {"Name": merchant, "VirtualPaymentAddress": merchant_upi},
            "Authentication": {"Method": auth, "Biometric": "contactless palm (1:N search), liveness checked",
                               "BiometricDataIncluded": False},
            "Status": status,
        }
        body = json.dumps(msg, sort_keys=True).encode()
        msg["IntegrityTag"] = {"Algorithm": "HMAC-SHA256 (local demo key)",
                               "Value": hmac.new(bytes.fromhex(self.setting("message_key")), body, hashlib.sha256).hexdigest()}
        return msg

    def transactions(self, palm_id=None, limit=200):
        q = "SELECT * FROM transactions" + (" WHERE palm_id = ?" if palm_id else "") + " ORDER BY time DESC, txn_id DESC LIMIT ?"
        rows = self.db.execute(q, ((palm_id, limit) if palm_id else (limit,))).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["amount"] = rupees(d.pop("amount_paise"))
            d["payer_upi"] = mask_upi(d["payer_upi"])
            d.pop("message", None)
            out.append(d)
        return out

    # ---------------------------------------------------------------- audit and statistics
    def audit(self, action, outcome=None, palm_id=None, **detail):
        with self.lock:
            self.db.execute("INSERT INTO audit(time, action, outcome, palm_id, detail) VALUES(?, ?, ?, ?, ?)",
                            (now(), action, outcome, palm_id, json.dumps(detail, default=str)))
            self.db.commit()

    def audit_log(self, limit=300):
        rows = self.db.execute("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d.update(json.loads(d.pop("detail") or "{}"))
            out.append(d)
        return out

    def stats(self, hours=24):
        since = (datetime.datetime.now() - datetime.timedelta(hours=hours)).isoformat(timespec="seconds")
        a = self.db.execute("SELECT action, outcome, COUNT(*) AS n FROM audit WHERE time >= ? GROUP BY action, outcome",
                            (since,)).fetchall()
        count = {(r["action"], r["outcome"]): r["n"] for r in a}
        t = self.db.execute("SELECT status, COUNT(*) AS n, COALESCE(SUM(amount_paise), 0) AS s FROM transactions "
                            "WHERE time >= ? GROUP BY status", (since,)).fetchall()
        tx = {r["status"]: (r["n"], r["s"]) for r in t}
        lat = [r["latency_ms"] for r in self.db.execute(
            "SELECT latency_ms FROM transactions WHERE time >= ? AND latency_ms IS NOT NULL", (since,))]
        searches = sum(n for (act, out), n in count.items() if act == "pay" and out in ("identified", "not recognised"))
        identified = count.get(("pay", "identified"), 0)
        attempts = sum(n for (act, _), n in count.items() if act in ("pay", "register", "account"))
        live_fail = sum(n for (act, out), n in count.items() if out == "liveness failed")
        return dict(hours=hours, registered=len(self.palm_ids()), attempts=attempts, searches=searches,
                    identified=identified, identification_rate=identified / searches if searches else None,
                    liveness_failed=live_fail, spoof_flags=sum(n for (_, out), n in count.items() if out == "spoof suspected"),
                    timeouts=sum(n for (_, out), n in count.items() if out == "timeout"),
                    duplicates_blocked=count.get(("register", "duplicate"), 0),
                    approved=tx.get("approved", (0, 0))[0], approved_amount=rupees(tx.get("approved", (0, 0))[1]),
                    declined=tx.get("declined", (0, 0))[0],
                    median_latency_ms=int(sorted(lat)[len(lat) // 2]) if lat else None)
