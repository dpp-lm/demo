"""Private usage store of the demo server: sessions, UI events and one row per request (no GPU imports, unit-testable).

Kept apart from the research run log (`api_guard.RunLog`), which holds the completions and nothing about visitors.
This store holds who did what and when. It stores the client address (the limit key `req_client` returns: the visitor address
from X-Real-IP behind Caddy, IPv6 reduced to its /64, or 'shared' when there is none) in `sessions`, `events` and `requests`, and the
prompt text of each request in `requests`. Completions are not stored here. A row of `requests` points at them by `run_id` (run log)
or, for a request answered from the draw cache, by `stream_key` with `stream_offset` and `n` (the draw cache holds the texts).

Identity. The page sends a random `visitor` (kept in its localStorage) and a random `session` (per page load) with its
requests and in `POST /event`. A request without them (for example Tim's copy of the page) is grouped per UTC day under
`DailySalt.anon_id(client)`, a keyed hash of the client address whose key is random, lives in memory only and is replaced
every UTC day. The hash only groups such requests; the address is stored next to it in the `addr` columns.
Databases made before the `addr`, `prompt` and `stream_key` columns existed are migrated in place on open (`_migrate`): the columns
are added and old rows keep ''.

Database: SQLite, mode 600, in a directory of mode 700 (`DPP_USAGE_DIR`, default `usage/` next to server.py).
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import statistics
import threading
import time
import datetime
from urllib.parse import urlparse

ID_RE = re.compile(r"[A-Za-z0-9_-]{8,64}")
EVENT_TYPES = ("load", "ui")        # POST /event: one `load` per page load, `ui` events named from UI_NAMES
UI_NAMES = ("expert_on", "expert_off", "mode", "matrices_open", "spectrum_open", "groups_on", "preset")
MODE_VALUES = ("map", "sample")
PRESET_RE = re.compile(r"[A-Za-z0-9_.:-]{1,32}")
MAX_EVENTS_PER_SESSION = 500
ROUTES = ("/sample", "/embed", "/describe")
MAX_DAYS = 100_000      # `days=all` and any larger value return the whole history
SESSIONS_LIMIT = 200


def day_of(t):
    return time.strftime("%Y-%m-%d", time.gmtime(t))


def valid_id(v):
    """A client-supplied visitor or session id (8 to 64 characters of A-Za-z0-9_-), else None. A bad id is ignored, never an error."""
    return v if isinstance(v, str) and ID_RE.fullmatch(v) else None


def host_of(v, limit=100):
    """Host name (lower case, no port, no path or query) of an Origin, Referer or `document.referrer` value, or a bare host; '' if none."""
    if not isinstance(v, str) or not v.strip():
        return ""
    v = v.strip()[:500]
    try:
        h = urlparse(v if "//" in v else "//" + v).hostname or ""
    except ValueError:
        return ""
    return h.lower()[:limit]


def request_origin(headers):
    """Host of the page that made the request: the Origin header, else the Referer's host, else ''."""
    return host_of(headers.get("origin") or "") or host_of(headers.get("referer") or "")


class DailySalt:
    """A random key per UTC day, held in memory only (never written anywhere), for grouping requests that carry no ids."""

    def __init__(self, clock=time.time):
        self.clock, self.day, self.key = clock, None, None
        self.lock = threading.Lock()

    def anon_id(self, client):
        now = self.clock()
        d = day_of(now)
        with self.lock:
            if d != self.day:
                self.day, self.key = d, secrets.token_bytes(32)
            key = self.key
        return "a-" + hmac.new(key, client.encode(), hashlib.sha256).hexdigest()[:16], d


def check_password(headers, own_secret_file, password_file, env):
    """'ok', 'bad' or 'unconfigured' for a /stats request. The existing owner key (header X-Own-Secret, the contents of
    `.own_secret`) is accepted, and so is X-Stats-Password against the stats password the owner sets in `password_file`
    (first line) or in the environment variable DPP_STATS_PASSWORD. Comparisons take constant time."""
    def read(path):
        try:
            return open(path, encoding="utf-8").read().strip()
        except OSError:
            return ""
    own = read(own_secret_file)
    if own and hmac.compare_digest(headers.get("x-own-secret", "").encode(), own.encode()):
        return "ok"
    pw = read(password_file) or (env.get("DPP_STATS_PASSWORD") or "").strip()
    if not pw:
        return "unconfigured"
    given = headers.get("x-stats-password", "")
    return "ok" if given and hmac.compare_digest(given.encode(), pw.encode()) else "bad"


def clean_event(body):
    """(type, name, value, ref) of a POST /event body, or None when it is not on the allowlist. Unknown fields are dropped.
    `load`: ref is the referrer reduced to its host, value is 'expert' when the page is in expert mode. `ui`: name is one of UI_NAMES;
    value is kept for `mode` (map or sample) and `preset` (a short id) and dropped for the others."""
    t = body.get("type")
    if t == "load":
        return "load", "", "expert" if body.get("expert") is True else "", host_of(body.get("referrer") or body.get("ref") or "")
    if t == "ui" and body.get("name") in UI_NAMES:
        name, v = body["name"], body.get("value")
        if name == "mode":
            v = v if v in MODE_VALUES else ""
        elif name == "preset":
            v = v if isinstance(v, str) and PRESET_RE.fullmatch(v) else ""
        else:
            v = ""
        return "ui", name, v, ""
    return None


def _median(xs):
    return float(statistics.median(xs)) if xs else 0.0


def _dist(rows):
    c = {}
    for v in rows:
        if v is not None:
            c[v] = c.get(v, 0) + 1
    return [{"key": k, "count": n} for k, n in sorted(c.items(), key=lambda kv: kv[0])]


def _top(rows, limit=20):
    c = {}
    for v in rows:
        c[v] = c.get(v, 0) + 1
    return [{"key": k, "count": n} for k, n in sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0])))[:limit]]


class UsageStore:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS sessions (
      id TEXT PRIMARY KEY, visitor TEXT NOT NULL, first_t REAL NOT NULL, last_t REAL NOT NULL, day TEXT NOT NULL,
      origin TEXT NOT NULL DEFAULT '', referrer TEXT NOT NULL DEFAULT '', events INTEGER NOT NULL DEFAULT 0, addr TEXT NOT NULL DEFAULT '');
    CREATE INDEX IF NOT EXISTS sessions_day ON sessions(day);
    CREATE TABLE IF NOT EXISTS events (
      id INTEGER PRIMARY KEY, t REAL NOT NULL, session TEXT NOT NULL, type TEXT NOT NULL, name TEXT, value TEXT, addr TEXT NOT NULL DEFAULT '');
    CREATE INDEX IF NOT EXISTS events_session ON events(session);
    CREATE TABLE IF NOT EXISTS requests (
      id INTEGER PRIMARY KEY, t REAL NOT NULL, day TEXT NOT NULL, session TEXT NOT NULL, origin TEXT NOT NULL DEFAULT '',
      route TEXT NOT NULL, model TEXT, embedder TEXT, n INTEGER, tprime REAL, max_new INTEGER, outcome TEXT NOT NULL,
      wait_s REAL, run_s REAL, gen_s REAL, embed_s REAL, tokens INTEGER, completions INTEGER, run_id TEXT,
      cached_n INTEGER, stream_offset INTEGER, addr TEXT NOT NULL DEFAULT '', prompt TEXT NOT NULL DEFAULT '', stream_key TEXT);
    CREATE INDEX IF NOT EXISTS requests_day ON requests(day);
    CREATE INDEX IF NOT EXISTS requests_session ON requests(session);
    CREATE INDEX IF NOT EXISTS requests_run ON requests(run_id);
    """

    def __init__(self, directory, enabled=True, on_error=None):
        self.dir, self.enabled, self.on_error = directory, bool(enabled), on_error or (lambda m: None)
        self.lock = threading.Lock()
        self.db = None
        self.path = os.path.join(directory, "usage.sqlite3")

    @classmethod
    def from_env(cls, env, default_dir, on_error=None):
        off = str(env.get("DPP_USAGE", "1")).strip().lower() in ("0", "false", "off", "no")
        return cls(env.get("DPP_USAGE_DIR") or default_dir, enabled=not off, on_error=on_error)

    def _open(self):
        if self.db is None:
            os.makedirs(self.dir, mode=0o700, exist_ok=True)
            os.chmod(self.dir, 0o700)
            old = os.umask(0o077)
            try:
                self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=10)
                self.db.execute("PRAGMA journal_mode=WAL")
                self.db.execute("PRAGMA synchronous=NORMAL")
                self.db.executescript(self.SCHEMA)
                self._migrate(self.db)
            finally:
                os.umask(old)
            os.chmod(self.path, 0o600)
        return self.db

    MIGRATIONS = (("requests", "cached_n", "INTEGER"), ("requests", "stream_offset", "INTEGER"),   # before the draw cache
                  ("sessions", "addr", "TEXT NOT NULL DEFAULT ''"), ("events", "addr", "TEXT NOT NULL DEFAULT ''"),
                  ("requests", "addr", "TEXT NOT NULL DEFAULT ''"), ("requests", "prompt", "TEXT NOT NULL DEFAULT ''"),
                  ("requests", "stream_key", "TEXT"))

    @classmethod
    def _migrate(cls, db):
        """Add every column of MIGRATIONS that an older database file lacks. Old rows keep the default ('' or NULL)."""
        have = {}
        for table, col, decl in cls.MIGRATIONS:
            if table not in have:
                have[table] = {r[1] for r in db.execute(f"PRAGMA table_info({table})")}
            if col not in have[table]:
                db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")

    def _safe(self, f, *a):
        """Run a write under the lock; a failure is reported and swallowed (a usage record never fails a request)."""
        if not self.enabled:
            return None
        try:
            with self.lock:
                return f(self._open(), *a)
        except Exception as e:
            try:
                self.on_error(f"usage store failed: {type(e).__name__}: {e}")
            except Exception:
                pass
            return None

    # ---------------------------------------------------------------- writes
    @staticmethod
    def _touch(db, session, visitor, t, origin, referrer="", addr=""):
        row = db.execute("SELECT 1 FROM sessions WHERE id=?", (session,)).fetchone()
        if row is None:
            db.execute("INSERT INTO sessions(id, visitor, first_t, last_t, day, origin, referrer, addr) VALUES(?,?,?,?,?,?,?,?)",
                       (session, visitor, t, t, day_of(t), origin, referrer, addr))
        else:   # the first non-empty value of origin, referrer and addr stays
            db.execute("UPDATE sessions SET last_t=MAX(last_t, ?), origin=CASE WHEN origin='' THEN ? ELSE origin END, "
                       "referrer=CASE WHEN referrer='' THEN ? ELSE referrer END, addr=CASE WHEN addr='' THEN ? ELSE addr END WHERE id=?",
                       (t, origin, referrer, addr, session))

    def record_request(self, t, session, visitor, origin, route, outcome, model=None, embedder=None, n=None, tprime=None,
                       max_new=None, wait_s=None, run_s=None, gen_s=None, embed_s=None, tokens=None, completions=None, run_id=None,
                       cached=None, offset=None, addr="", prompt="", stream_key=None):
        def w(db):
            self._touch(db, session, visitor, t, origin, addr=addr)
            db.execute("INSERT INTO requests(t, day, session, origin, route, model, embedder, n, tprime, max_new, outcome, wait_s, run_s,"
                       " gen_s, embed_s, tokens, completions, run_id, cached_n, stream_offset, addr, prompt, stream_key) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (t, day_of(t), session, origin, route, model, embedder, n, tprime, max_new, outcome, wait_s, run_s, gen_s,
                        embed_s, tokens, completions, run_id, cached, offset, addr, prompt or "", stream_key))
        return self._safe(w)

    def record_event(self, t, session, visitor, origin, etype, name="", value="", ref="", addr=""):
        """Store a UI event. Returns 'ok', 'ignored' (the session is at its event cap) or None on failure."""
        def w(db):
            self._touch(db, session, visitor, t, origin, ref if etype == "load" else "", addr)
            (n,) = db.execute("SELECT events FROM sessions WHERE id=?", (session,)).fetchone()
            if n >= MAX_EVENTS_PER_SESSION:
                return "ignored"
            db.execute("INSERT INTO events(t, session, type, name, value, addr) VALUES(?,?,?,?,?,?)", (t, session, etype, name, value, addr))
            db.execute("UPDATE sessions SET events=events+1 WHERE id=?", (session,))
            return "ok"
        return self._safe(w)

    # ---------------------------------------------------------------- reads
    def stats(self, days, now=None, sessions_limit=SESSIONS_LIMIT):
        """The /stats body. Days are UTC days; the window is the last `days` days including today. `days` "all" (or more than MAX_DAYS)
        is the whole history: `since` is then the first day with data and `days` the number of days from it to today."""
        now = now if now is not None else time.time()
        everything = days == "all" or int(days) > MAX_DAYS
        days = MAX_DAYS if everything else max(1, int(days))
        today = day_of(now)
        since = "0000-01-01" if everything else day_of(now - (days - 1) * 86400)

        def r(db):
            ses = db.execute("SELECT id, visitor, first_t, last_t, day, origin, referrer, events, addr FROM sessions WHERE day>=? ORDER BY first_t",
                             (since,)).fetchall()
            req = db.execute("SELECT t, day, session, origin, route, model, embedder, n, tprime, max_new, outcome, wait_s, run_s, gen_s,"
                             " embed_s, tokens, completions, run_id, cached_n FROM requests WHERE day>=? ORDER BY t", (since,)).fetchall()
            return ses, req
        got = self._safe(r)
        ses, req = got if got is not None else ([], [])
        ok_runs = [q for q in req if q[4] == "/sample" and q[10] == "ok"]
        run_sessions = {q[2] for q in ok_runs}
        durations = [s[3] - s[2] for s in ses]
        per_run_sessions = {}
        for q in req:
            per_run_sessions.setdefault(q[2], [0, 0])
            per_run_sessions[q[2]][0] += 1
            per_run_sessions[q[2]][1] += q[4] == "/sample" and q[10] == "ok"

        def day_row(d):
            sd = [s for s in ses if s[4] == d]
            rd = [q for q in ok_runs if q[1] == d]
            return {"day": d, "visitors": len({s[1] for s in sd}), "sessions": len(sd), "runs": len(rd),
                    "completions": sum(q[16] or 0 for q in rd), "errors": sum(1 for q in req if q[1] == d and q[10] != "ok")}
        first = min([s[4] for s in ses] + [q[1] for q in req] + [today])
        daily, d = [], datetime.date.fromisoformat(first)
        end = datetime.date.fromisoformat(today)
        while d <= end:   # calendar arithmetic, so the series has no gap or repeat
            daily.append(day_row(d.isoformat()))
            d += datetime.timedelta(days=1)
        daily = daily[-days:]
        if everything:
            since, days = daily[0]["day"], len(daily)
        todayrow = day_row(today)
        hours = [0] * 24
        for q in req:
            hours[int(time.strftime("%H", time.gmtime(q[0])))] += 1
        errs = [q[10] for q in req if q[10] != "ok"]
        by_origin = {}
        for q in req:
            o = by_origin.setdefault(q[3] or "", {"key": q[3] or "", "requests": 0, "runs": 0, "sessions": set()})
            o["requests"] += 1
            o["runs"] += q[4] == "/sample" and q[10] == "ok"
            o["sessions"].add(q[2])
        samples = [q for q in req if q[4] == "/sample"]
        order = sorted(ses, key=lambda s: -s[3])
        shown = order[:sessions_limit]
        return {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), "days": days, "since": since,
            "totals": {
                "visitors": len({s[1] for s in ses}), "sessions": len(ses), "sessionsWithRun": len(run_sessions),
                "requests": len(req), "runs": len(ok_runs), "completions": sum(q[16] or 0 for q in ok_runs),
                "tokens": sum(q[15] or 0 for q in ok_runs), "gpuSeconds": round(sum(q[12] or 0 for q in req), 1),
                "errors": len(errs), "cachedRequests": sum(1 for q in samples if q[18] is not None and q[7] and q[18] >= q[7] and q[10] == "ok"),
                "cachedCompletions": sum(q[18] or 0 for q in samples if q[10] == "ok"), "medianSessionS": round(_median(durations)), "meanSessionS": round(sum(durations) / len(durations)) if durations else 0,
            },
            "today": dict(todayrow, requests=sum(1 for q in req if q[1] == today)),
            "daily": daily,
            "breakdowns": {
                "byOrigin": sorted(({"key": o["key"], "requests": o["requests"], "runs": o["runs"], "sessions": len(o["sessions"])}
                                    for o in by_origin.values()), key=lambda o: -o["requests"]),
                "byEmbedder": _top([q[6] for q in ok_runs if q[6]]),
                "byErrorCode": _top(errs),
                "n": _dist(q[7] for q in ok_runs), "tprime": _dist(q[8] for q in ok_runs), "maxNewTokens": _dist(q[9] for q in ok_runs),
                "medianQueueWaitS": round(_median([q[11] for q in samples if q[11] is not None]), 2),
                "medianGenerateS": round(_median([q[13] for q in ok_runs if q[13] is not None]), 2),
                "medianEmbedS": round(_median([q[14] for q in ok_runs if q[14] is not None]), 2),
                "busiestHours": [{"hour": h, "requests": c} for h, c in enumerate(hours)],
            },
            "sessions": [{"id": s[0], "visitor": s[1][:8], "start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(s[2])),
                          "durationS": round(s[3] - s[2]), "origin": s[5], "referrer": s[6],
                          "runs": per_run_sessions.get(s[0], [0, 0])[1], "requests": per_run_sessions.get(s[0], [0, 0])[0],
                          "events": s[7], "addr": s[8]} for s in shown],
            "sessionsTruncated": max(0, len(order) - len(shown)),
        }

    def session(self, sid):
        """Events and requests of one session in time order, or None when the session is unknown. A request has `_stream`, the draw-cache
        stream key (or None), which the route turns into `drawTexts` and drops."""
        def r(db):
            s = db.execute("SELECT id, visitor, first_t, last_t, origin, referrer, addr FROM sessions WHERE id=?", (sid,)).fetchone()
            if s is None:
                return None
            ev = db.execute("SELECT t, type, name, value FROM events WHERE session=? ORDER BY t, id", (sid,)).fetchall()
            rq = db.execute("SELECT t, route, model, embedder, n, tprime, max_new, outcome, wait_s, run_s, gen_s, embed_s, tokens,"
                            " completions, run_id, cached_n, stream_offset, prompt, addr, stream_key FROM requests WHERE session=? ORDER BY t, id", (sid,)).fetchall()
            return s, ev, rq
        got = self._safe(r)
        if got is None:
            return None
        s, ev, rq = got
        iso = lambda t: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))
        return {
            "session": {"id": s[0], "visitor": s[1][:8], "start": iso(s[2]), "durationS": round(s[3] - s[2]), "origin": s[4], "referrer": s[5], "addr": s[6]},
            "events": [{"time": iso(e[0]), "type": e[1], "name": e[2] or "", "value": e[3] or ""} for e in ev],
            "requests": [{"time": iso(q[0]), "route": q[1], "model": q[2], "embedder": q[3], "N": q[4], "Tprime": q[5], "maxNewTokens": q[6],
                          "outcome": q[7], "queueWaitS": q[8], "runS": q[9], "generateS": q[10], "embedS": q[11], "tokens": q[12],
                          "completions": q[13], "runId": q[14], "cached": q[15], "offset": q[16], "prompt": q[17] or None, "addr": q[18], "_stream": q[19]} for q in rq],
        }

    def run_day(self, run_id):
        """UTC day of the request that produced `run_id`, or None."""
        row = self._safe(lambda db: db.execute("SELECT day FROM requests WHERE run_id=?", (run_id,)).fetchone())
        return row[0] if row else None

    def dump_text(self):
        """Every stored value as one string (for tests that check what is on disk)."""
        def r(db):
            return json.dumps([db.execute(f"SELECT * FROM {t}").fetchall() for t in ("sessions", "events", "requests")], default=str)
        return self._safe(r) or ""
