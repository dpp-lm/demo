"""Shared draw cache: growing streams of i.i.d. completions, and cached embeddings (no GPU imports, unit-testable).

A stream holds the draws for one key (prompt as run, model key, repo, weights revision, T', max new tokens). Draw k of the stream is
completion k of that key. The i.i.d. property rests on four rules, all enforced here or by the caller:
  * only the server adds draws, with seeds it picks at random (`pick_seed` avoids any seed range an earlier batch of the stream used,
    because a completion is a function of seed + index);
  * a batch is appended whole, in generation order, in one transaction, and only at the current end of the stream (`append` refuses
    otherwise); nothing is filtered, deduplicated or reordered by content;
  * a failed or cancelled batch is never appended (the caller appends only after generation succeeded);
  * requests that carry their own seed never touch a stream.
Storage: SQLite (WAL) in a directory of mode 700, files 600. The size is capped (`max_bytes`): when it is exceeded, whole streams are
evicted least recently used first, then cached embeddings least recently used first.
"""
import hashlib
import json
import os
import random
import sqlite3
import threading
import time

MAX_STREAM = 1000   # draws per stream (the page's bufferMax)


class Conflict(Exception):
    """append() at a position that is not the end of the stream."""


def stream_key(prompt, model_key, repo, revision, t_prime, max_new):
    """Hex key of a stream. 4-bit and 16-bit weights differ in `repo`, other weights in `revision`."""
    return hashlib.sha256(json.dumps([prompt, model_key, repo, revision, repr(float(t_prime)), int(max_new)], ensure_ascii=False).encode()).hexdigest()


def emb_key(sig, text):
    return hashlib.sha256(sig.encode() + b"\0" + text.encode()).hexdigest()


class DrawCache:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS streams (
      id INTEGER PRIMARY KEY, key TEXT UNIQUE NOT NULL, model_key TEXT, repo TEXT, revision TEXT, tprime REAL, max_new INTEGER,
      length INTEGER NOT NULL DEFAULT 0, bytes INTEGER NOT NULL DEFAULT 0, created REAL, last_used REAL);
    CREATE TABLE IF NOT EXISTS draws (
      stream INTEGER NOT NULL, pos INTEGER NOT NULL, text TEXT NOT NULL, logp REAL NOT NULL, logq REAL NOT NULL, ntokens INTEGER NOT NULL,
      ended INTEGER NOT NULL, own BLOB, PRIMARY KEY (stream, pos)) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS batches (
      stream INTEGER NOT NULL, start INTEGER NOT NULL, n INTEGER NOT NULL, seed INTEGER NOT NULL, run_id TEXT, time REAL, PRIMARY KEY (stream, start));
    CREATE TABLE IF NOT EXISTS embs (
      k TEXT PRIMARY KEY, sig TEXT NOT NULL, dim INTEGER NOT NULL, vec BLOB NOT NULL, last_used REAL NOT NULL);
    CREATE INDEX IF NOT EXISTS embs_used ON embs(last_used);
    """

    def __init__(self, directory, enabled=True, max_bytes=4 * 2**30, on_error=None, clock=time.time):
        self.dir, self.enabled, self.max_bytes = directory, bool(enabled), int(max_bytes)
        self.on_error, self.clock = on_error or (lambda m: None), clock
        self.lock = threading.RLock()
        self.db = None
        self.path = os.path.join(directory, "draws.sqlite3")
        self.rng = random.SystemRandom()

    @classmethod
    def from_env(cls, env, default_dir, on_error=None):
        """DPP_DRAW_CACHE=0 turns it off, DPP_DRAW_CACHE_DIR moves it, DPP_DRAW_CACHE_MAX_GB (default 4) caps its size."""
        off = str(env.get("DPP_DRAW_CACHE", "1")).strip().lower() in ("0", "false", "off", "no")
        gb = float(env.get("DPP_DRAW_CACHE_MAX_GB") or 4)
        return cls(env.get("DPP_DRAW_CACHE_DIR") or default_dir, enabled=not off, max_bytes=int(gb * 2**30), on_error=on_error)

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
            finally:
                os.umask(old)
            os.chmod(self.path, 0o600)
        return self.db

    # ---------------------------------------------------------------- streams
    def length(self, key):
        with self.lock:
            r = self._open().execute("SELECT length FROM streams WHERE key=?", (key,)).fetchone()
            return r[0] if r else 0

    def read(self, key, start, end, touch=True):
        """Draws [start, end) as dicts {text, logp, logq, nTokens, ended}; ValueError if the stream is shorter than `end`."""
        with self.lock:
            db = self._open()
            s = db.execute("SELECT id, length FROM streams WHERE key=?", (key,)).fetchone()
            if s is None or s[1] < end:
                raise ValueError("the stream has fewer draws than asked for")
            rows = db.execute("SELECT text, logp, logq, ntokens, ended FROM draws WHERE stream=? AND pos>=? AND pos<? ORDER BY pos", (s[0], start, end)).fetchall()
            if touch:
                db.execute("UPDATE streams SET last_used=? WHERE id=?", (self.clock(), s[0]))
            return [{"text": r[0], "logp": r[1], "logq": r[2], "nTokens": r[3], "ended": bool(r[4])} for r in rows]

    def read_own(self, key, start, end):
        """Stored uncentered own states of draws [start, end) as a list of bytes (float32) or None."""
        with self.lock:
            db = self._open()
            s = db.execute("SELECT id FROM streams WHERE key=?", (key,)).fetchone()
            if s is None:
                return [None] * (end - start)
            return [r[0] for r in db.execute("SELECT own FROM draws WHERE stream=? AND pos>=? AND pos<? ORDER BY pos", (s[0], start, end))]

    def batches(self, key):
        with self.lock:
            db = self._open()
            return [(r[0], r[1], r[2]) for r in db.execute("SELECT b.start, b.n, b.seed FROM batches b JOIN streams s ON s.id=b.stream WHERE s.key=? ORDER BY b.start", (key,))]

    def pick_seed(self, key, n):
        """A random seed whose range [seed, seed + n) overlaps no earlier batch of the stream (a completion depends on seed + index only)."""
        used = [(sd, sd + k) for _, k, sd in self.batches(key)]
        while True:
            sd = self.rng.randrange(2**31 - 1000)
            if all(sd + n <= a or sd >= b for a, b in used):
                return sd

    def append(self, key, meta, expect_len, draws, owns, seed, run_id=None):
        """Append `draws` (dicts as read() returns) at position `expect_len` in one transaction; Conflict when the stream has another length.
        `owns` is a list of float32 bytes or None per draw. Returns the new length. meta: model_key, repo, revision, tprime, max_new."""
        if not self.enabled:
            raise RuntimeError("draw cache off")
        if len(draws) != len(owns) or not draws:
            raise ValueError("a batch is appended whole")
        with self.lock:
            db = self._open()
            db.execute("BEGIN IMMEDIATE")
            try:
                now = self.clock()
                s = db.execute("SELECT id, length FROM streams WHERE key=?", (key,)).fetchone()
                if s is None:
                    if expect_len != 0:
                        raise Conflict("no such stream")
                    cur = db.execute("INSERT INTO streams(key, model_key, repo, revision, tprime, max_new, length, bytes, created, last_used) VALUES(?,?,?,?,?,?,0,0,?,?)",
                                     (key, meta["model_key"], meta["repo"], meta["revision"], meta["tprime"], meta["max_new"], now, now))
                    sid, length = cur.lastrowid, 0
                else:
                    sid, length = s
                if length != expect_len:
                    raise Conflict(f"the stream has {length} draws, not {expect_len}")
                if length + len(draws) > MAX_STREAM:
                    raise ValueError("a stream holds at most %d draws" % MAX_STREAM)
                size = 0
                for i, (d, o) in enumerate(zip(draws, owns)):
                    db.execute("INSERT INTO draws(stream, pos, text, logp, logq, ntokens, ended, own) VALUES(?,?,?,?,?,?,?,?)",
                               (sid, length + i, d["text"], d["logp"], d["logq"], d["nTokens"], int(d["ended"]), o))
                    size += len(d["text"].encode()) + 64 + (len(o) if o else 0)
                db.execute("INSERT INTO batches(stream, start, n, seed, run_id, time) VALUES(?,?,?,?,?,?)", (sid, length, len(draws), seed, run_id, now))
                db.execute("UPDATE streams SET length=?, bytes=bytes+?, last_used=? WHERE id=?", (length + len(draws), size, now, sid))
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
            self.evict()
            return length + len(draws)

    # ---------------------------------------------------------------- embeddings
    def get_embs(self, sig, texts):
        """{text: float32 bytes} for the texts that are cached under `sig` (embedder id, revision, dtype)."""
        out = {}
        with self.lock:
            db = self._open()
            now = self.clock()
            for t in set(texts):
                k = emb_key(sig, t)
                r = db.execute("SELECT vec FROM embs WHERE k=?", (k,)).fetchone()
                if r:
                    out[t] = r[0]
                    db.execute("UPDATE embs SET last_used=? WHERE k=?", (now, k))
        return out

    def put_embs(self, sig, vecs):
        """Store {text: (dim, float32 bytes)}; an existing entry is kept (the first vector of a text is the text's vector). Returns the stored ones."""
        out = {}
        with self.lock:
            db = self._open()
            now = self.clock()
            db.execute("BEGIN IMMEDIATE")
            try:
                for t, (dim, vec) in vecs.items():
                    k = emb_key(sig, t)
                    db.execute("INSERT OR IGNORE INTO embs(k, sig, dim, vec, last_used) VALUES(?,?,?,?,?)", (k, sig, dim, vec, now))
                    out[t] = db.execute("SELECT vec FROM embs WHERE k=?", (k,)).fetchone()[0]
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
            self.evict()
        return out

    # ---------------------------------------------------------------- size
    def size(self):
        with self.lock:
            db = self._open()
            s = db.execute("SELECT COALESCE(SUM(bytes), 0) FROM streams").fetchone()[0]
            e = db.execute("SELECT COALESCE(SUM(LENGTH(vec) + 80), 0) FROM embs").fetchone()[0]
            return s + e

    def evict(self):
        """Over the cap: drop whole streams, least recently used first, then embeddings, least recently used first, down to 90 percent."""
        with self.lock:
            if self.size() <= self.max_bytes:
                return 0
            db = self._open()
            goal, n = int(self.max_bytes * 0.9), 0
            for sid, in db.execute("SELECT id FROM streams ORDER BY last_used").fetchall():
                if self.size() <= goal:
                    break
                db.execute("DELETE FROM draws WHERE stream=?", (sid,))
                db.execute("DELETE FROM batches WHERE stream=?", (sid,))
                db.execute("DELETE FROM streams WHERE id=?", (sid,))
                n += 1
            while self.size() > goal:
                ks = [r[0] for r in db.execute("SELECT k FROM embs ORDER BY last_used LIMIT 8")]
                if not ks:
                    break
                db.executemany("DELETE FROM embs WHERE k=?", [(k,) for k in ks])
            return n

    def stats(self):
        with self.lock:
            db = self._open()
            return {"streams": db.execute("SELECT COUNT(*) FROM streams").fetchone()[0], "draws": db.execute("SELECT COALESCE(SUM(length), 0) FROM streams").fetchone()[0],
                    "embeddings": db.execute("SELECT COUNT(*) FROM embs").fetchone()[0], "bytes": self.size()}
