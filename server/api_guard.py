"""Input validation, per-client limits and engine watchdog helpers for the demo server (no GPU imports, unit-testable).

Used by server.py. Everything here raises HTTPException with a short message a visitor can read; nothing raises a bare
exception for bad input.
"""
import asyncio
import collections
import contextlib
import hashlib
import hmac
import ipaddress
import json
import math
import os
import re
import shutil
import threading
import time

import numpy as np
from fastapi import HTTPException

LOOPBACK = ("127.0.0.1", "::1")


class ApiError(HTTPException):
    """HTTPException that carries the machine-readable `code` of docs/BACKEND.md (bad_request, auth, rate_limit, budget,
    busy, server; `starting` for a 503 while an engine restarts). Without an explicit code `error_code` derives it from the status."""

    def __init__(self, status, detail, code=None, headers=None):
        super().__init__(status, detail, headers=headers)
        self.code = code


def error_code(exc):
    """The `code` of an error body: the explicit one, else by HTTP status."""
    c = getattr(exc, "code", None)
    if c:
        return c
    st = exc.status_code
    if st in (401, 403):
        return "auth"
    if st == 429:
        return "busy"
    if st == 503:
        return "starting"
    if 400 <= st < 500:
        return "bad_request"
    return "server"


def bad(msg, status=400, headers=None, code=None):
    return ApiError(status, msg, code=code, headers=headers)


# ---------------------------------------------------------------- body

def _no_constant(name):
    raise ValueError("non-finite number")


async def read_json(req, limit):
    """Request body as a dict: Content-Type application/json, at most `limit` bytes (checked on Content-Length and while
    reading), valid JSON without NaN/Infinity literals, top level an object."""
    ct = req.headers.get("content-type", "").split(";")[0].strip().lower()
    if ct != "application/json":
        raise bad("Send the request as Content-Type: application/json.", 415)
    cl = req.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > limit:
                raise bad(f"The request body is limited to {limit} bytes.", 413)
        except ValueError:
            raise bad("Bad Content-Length header.")
    buf = bytearray()
    async for chunk in req.stream():
        buf.extend(chunk)
        if len(buf) > limit:
            raise bad(f"The request body is limited to {limit} bytes.", 413)
    try:
        obj = json.loads(bytes(buf), parse_constant=_no_constant)
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise bad("The request body is not valid JSON.")
    if not isinstance(obj, dict):
        raise bad("The request body must be a JSON object.")
    return obj


# ---------------------------------------------------------------- values

def number(v, name, integer=False, lo=None, hi=None, clamp=True):
    """A finite number (int, float, or numeric string) as float or int. Out of range: clamped (clamp=True) or 400."""
    if isinstance(v, bool) or v is None or isinstance(v, (list, dict)):
        raise bad(f"{name} must be a number.")
    try:
        if isinstance(v, int) and integer:
            f = v
        else:
            f = float(v)
            if not math.isfinite(f):
                raise bad(f"{name} must be a finite number.")
            if integer:
                f = int(f)
    except HTTPException:
        raise
    except (TypeError, ValueError, OverflowError):
        raise bad(f"{name} must be a number.")
    if lo is not None and f < lo:
        if clamp:
            f = lo
        else:
            raise bad(f"{name} must be at least {lo}.")
    if hi is not None and f > hi:
        if clamp:
            f = hi
        else:
            raise bad(f"{name} must be at most {hi}.")
    return f


def text(v, name, maxlen, required=True, strip=True, truncate=True):
    """A string, stripped. Longer than `maxlen`: cut (truncate=True) or 400 (truncate=False, the contract's limits never cap silently)."""
    if v is None:
        v = ""
    if not isinstance(v, str):
        raise bad(f"{name} must be a string.")
    if strip:
        v = v.strip()
    if required and not v:
        raise bad(f"{name} is empty.")
    if maxlen and len(v) > maxlen and not truncate:
        raise bad(f"{name} is limited to {maxlen} characters (got {len(v)}).")
    return v[:maxlen] if maxlen else v


def gram_b64(emb):
    """G of docs/BACKEND.md: inner products in float64 of the unit vectors `emb` (N x D, float32), symmetrized, entries clipped to
    [-1, 1], diagonal exactly 1, returned as base64 of little-endian float32, row-major."""
    import base64
    u = np.asarray(emb, dtype=np.float32).astype(np.float64)
    g = u @ u.T
    g = (g + g.T) / 2
    np.clip(g, -1.0, 1.0, out=g)
    np.fill_diagonal(g, 1.0)
    return base64.b64encode(np.ascontiguousarray(g, dtype="<f4").tobytes()).decode("ascii")


def embedder_list(v, known, max_len):
    """`embedders`: a list of at most `max_len` distinct known keys (duplicates dropped, order kept), or 400."""
    if v is None or v == []:
        return []
    if not isinstance(v, list) or not all(isinstance(k, str) for k in v):
        raise bad("embedders must be a list of embedder names.")
    out = list(dict.fromkeys(v))
    if len(out) > max_len:
        raise bad(f"embedders can list at most {max_len} different embedders.")
    if any(k not in known for k in out):
        raise bad(f"Unknown embedder; use one of {sorted(known)}.")
    return out


# ---------------------------------------------------------------- clients

def filter_models(models, allow, resident_key):
    """Restrict `models` (key -> config) to the comma-separated keys in `allow` (the DPP_MODELS setting).

    None or an empty string keeps every model. The allowlist must name only known keys and must include
    `resident_key`; otherwise ValueError (the service refuses to start rather than serve a wrong set).
    Returns a new dict in the original order."""
    if allow is None or not allow.strip():
        return dict(models)
    keys = [k.strip() for k in allow.split(",") if k.strip()]
    unknown = [k for k in keys if k not in models]
    if unknown:
        raise ValueError(f"DPP_MODELS names unknown model keys {unknown}; known keys: {sorted(models)}")
    if resident_key not in keys:
        raise ValueError(f"DPP_MODELS must include the resident model {resident_key!r}")
    return {k: v for k, v in models.items() if k in keys}


def client_key(peer, real_ip):
    """Limit key of a request. Behind Caddy (peer is loopback) the visitor address comes from X-Real-IP; without a valid one all
    such traffic shares the key 'shared' (limits then act as global ones). IPv6 is keyed on its /64."""
    if peer in LOOPBACK:
        if not real_ip:
            return "shared"
        try:
            ip = ipaddress.ip_address(real_ip.strip())
        except ValueError:
            return "shared"
    else:
        try:
            ip = ipaddress.ip_address(peer)
        except (ValueError, TypeError):
            return "peer:" + str(peer)
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped:
            return str(ip.ipv4_mapped)
        return str(ipaddress.IPv6Network((int(ip) >> 64 << 64, 64)))
    return str(ip)


def short(key):
    """Short hash of a client key for logs."""
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def is_owner(headers, secret_file):
    try:
        secret = open(secret_file).read().strip()
    except OSError:
        return False
    return bool(secret) and hmac.compare_digest(headers.get("x-own-secret", ""), secret)


class Bucket:
    """Token bucket: `capacity` units refilled evenly over `period` seconds."""

    def __init__(self, capacity, period):
        self.cap, self.rate = float(capacity), float(capacity) / period
        self.level, self.t = float(capacity), None

    def _fill(self, now):
        if self.t is None:
            self.t = now
        dt = max(0.0, now - self.t)
        self.level = min(self.cap, self.level + dt * self.rate)
        self.t = max(self.t, now)

    def wait_for(self, cost, now):
        """Seconds until `cost` units are available (0 when they are now)."""
        self._fill(now)
        return 0.0 if self.level >= cost else (cost - self.level) / self.rate

    def take(self, cost, now):
        self._fill(now)
        self.level -= cost

    def give(self, cost, now):
        self._fill(now)
        self.level = min(self.cap, self.level + cost)


class Limits:
    """Per-client and global limits. `check` raises 429 with Retry-After; `commit` records a request."""

    def __init__(self, tokens_per_hour, swaps_per_hour, runs_per_hour_global, tokens_per_hour_global, embeds_per_hour_client,
                 embeds_per_hour_global, swap_min_interval, max_clients=20000, describes_per_hour_client=120,
                 describes_per_hour_global=600):
        self.tph, self.sph = tokens_per_hour, swaps_per_hour
        self.rph_g, self.tph_g = runs_per_hour_global, tokens_per_hour_global
        self.eph_c, self.eph_g = embeds_per_hour_client, embeds_per_hour_global
        self.swap_gap = swap_min_interval
        self.dph_c, self.dph_g = describes_per_hour_client, describes_per_hour_global
        self.describes = collections.defaultdict(collections.deque)
        self.druns = collections.deque()
        self.buckets = {}
        self.swaps = collections.defaultdict(collections.deque)
        self.embeds = collections.defaultdict(collections.deque)
        self.runs = collections.deque()
        self.gbucket = Bucket(tokens_per_hour_global, 3600)
        self.eruns = collections.deque()
        self.last_swap = 0.0
        self.inflight = set()
        self.max_clients = max_clients

    @staticmethod
    def _trim(dq, now, window=3600):
        while dq and now - dq[0] > window:
            dq.popleft()

    def _bucket(self, client):
        b = self.buckets.get(client)
        if b is None:
            if len(self.buckets) >= self.max_clients:
                self.buckets.pop(next(iter(self.buckets)))
                self.swaps.pop(next(iter(self.swaps), None), None)
            b = self.buckets[client] = Bucket(self.tph, 3600)
        return b

    @staticmethod
    def _429(msg, wait, code="rate_limit"):
        """429 with Retry-After. `rate_limit`: this visitor's share is used up. `busy`: the server as a whole is occupied."""
        w = max(1, int(math.ceil(wait)))
        m = max(1, round(w / 60))
        return ApiError(429, msg.format(w=w, mm="a minute" if m == 1 else f"{m} minutes"), code=code, headers={"Retry-After": str(w)})

    def claim(self, client):
        """Reserve the client's one sample request (queued or running); 429 when it already holds one. Release with `unclaim`."""
        if client in self.inflight:
            raise self._429("You already have a request waiting or running. Wait for it to finish, then try again.", 5, "busy")
        self.inflight.add(client)

    def unclaim(self, client):
        self.inflight.discard(client)

    def swap_wait(self, now=None):
        """Seconds until the global gap between two slot swaps has passed (0 when it has)."""
        now = now or time.time()
        return max(0.0, self.last_swap + self.swap_gap - now)

    def check_run(self, client, cost, needs_swap, now=None, check_gap=True):
        """429 when a budget is exhausted. `check_gap=False` skips the global gap between swaps (a queued job waits it out at its start)."""
        now = now or time.time()
        self._trim(self.runs, now)
        if len(self.runs) >= self.rph_g:
            raise self._429("The server is busy this hour. Please try again in about {mm}.", 300, "busy")
        w = self.gbucket.wait_for(cost, now)
        if w > 0:
            raise self._429("The server has produced a lot of text this hour. Please try again in about {mm}.", w, "busy")
        w = self._bucket(client).wait_for(cost, now)
        if w > 0:
            raise self._429("You have used your share of generation for this hour. Please try again in about {mm}, or use fewer or shorter completions.", w)
        if needs_swap:
            gap = self.last_swap + self.swap_gap - now if check_gap else 0
            if gap > 0:
                raise self._429("The server just switched models for someone else. Please try again in {w} seconds.", gap, "busy")
            dq = self.swaps[client]
            self._trim(dq, now)
            if len(dq) >= self.sph:
                raise self._429("You have switched models several times this hour. Please try again in about {mm}, or pick the model that is already loaded.", dq[0] + 3600 - now)

    def commit_run(self, client, cost, needs_swap, now=None):
        now = now or time.time()
        self.runs.append(now)
        self._bucket(client).take(cost, now)
        self.gbucket.take(cost, now)
        if needs_swap:
            self.last_swap = now
            self.swaps[client].append(now)

    def end_run(self, client, refund=0, now=None):
        now = now or time.time()
        if refund:
            self._bucket(client).give(refund, now)
            self.gbucket.give(refund, now)

    def check_embed(self, client, now=None):
        now = now or time.time()
        self._trim(self.eruns, now)
        if len(self.eruns) >= self.eph_g:
            raise self._429("The server is busy this hour. Please try again later.", 300, "busy")
        dq = self.embeds[client]
        self._trim(dq, now)
        if len(dq) >= self.eph_c:
            raise self._429("Too many embedding requests from you this hour. Please try again in about {mm}.", dq[0] + 3600 - now)

    def commit_embed(self, client, now=None):
        now = now or time.time()
        self.eruns.append(now)
        self.embeds[client].append(now)

    def check_describe(self, client, now=None):
        now = now or time.time()
        self._trim(self.druns, now)
        if len(self.druns) >= self.dph_g:
            raise self._429("The server is busy this hour. Please try again later.", 300, "busy")
        dq = self.describes[client]
        self._trim(dq, now)
        if len(dq) >= self.dph_c:
            raise self._429("Too many naming requests from you this hour. Please try again in about {mm}.", dq[0] + 3600 - now)

    def commit_describe(self, client, now=None):
        now = now or time.time()
        self.druns.append(now)
        self.describes[client].append(now)

    def snapshot(self, now=None):
        now = now or time.time()
        self._trim(self.runs, now)
        return {"runsLastHour": len(self.runs), "clients": len(self.buckets), "globalTokensLeft": int(self.gbucket.level)}


# ---------------------------------------------------------------- engines and logs

async def first_unhealthy(engines, timeout=8):
    """First (name, error) among [(name, engine)] whose check_health fails, else None. Engines may be None."""
    for name, e in engines:
        if e is None:
            continue
        try:
            await asyncio.wait_for(e.check_health(), timeout)
        except asyncio.TimeoutError:
            return name, "check_health timed out"
        except Exception as ex:  # EngineDeadError or any failure of the engine client
            return name, f"{type(ex).__name__}: {ex}"[:200]
    return None


def rotate_log(path, max_bytes, keep=4):
    """Copy-truncate rotation (the file is appended to by systemd, so truncating in place is safe): path -> path.1 -> ... -> path.<keep>."""
    try:
        if os.path.getsize(path) <= max_bytes:
            return False
    except OSError:
        return False
    for i in range(keep, 0, -1):
        src = path if i == 1 else f"{path}.{i - 1}"
        if os.path.exists(src):
            if i == keep and os.path.exists(f"{path}.{i}"):
                os.remove(f"{path}.{i}")
            shutil.copyfile(src, f"{path}.{i}") if src == path else os.replace(src, f"{path}.{i}")
    with open(path, "r+b") as f:
        f.truncate(0)
    return True


# ---------------------------------------------------------------- request ids, queue

_RID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def request_id(v):
    """Optional `request_id` of a request: 1 to 64 characters of A-Z, a-z, 0-9, `_` and `-`; None when absent."""
    if v is None:
        return None
    if not isinstance(v, str) or not _RID.fullmatch(v):
        raise bad("request_id must be 1 to 64 characters of A-Z, a-z, 0-9, _ and -.")
    return v


class RateWindow:
    """At most `limit` events per client in any `period` seconds (for cheap read-only routes)."""

    def __init__(self, limit, period, max_clients=20000):
        self.limit, self.period, self.max_clients = limit, period, max_clients
        self.ev = {}

    def count(self, client, now=None):
        """Events of `client` within the window (without adding one)."""
        now = now or time.time()
        dq = self.ev.get(client)
        return 0 if dq is None else sum(1 for t in dq if now - t <= self.period)

    def hit(self, client, now=None):
        now = now or time.time()
        dq = self.ev.get(client)
        if dq is None:
            if len(self.ev) >= self.max_clients:
                self.ev.pop(next(iter(self.ev)))
            dq = self.ev[client] = collections.deque()
        while dq and now - dq[0] > self.period:
            dq.popleft()
        if len(dq) >= self.limit:
            raise bad("Too many requests. Please slow down.", 429, {"Retry-After": "5"}, "rate_limit")
        dq.append(now)


class _Job:
    __slots__ = ("rid", "engine", "state", "fut", "t")

    def __init__(self, rid, engine, loop):
        self.rid, self.engine, self.state, self.t = rid, engine, "waiting", time.monotonic()
        self.fut = loop.create_future()


class JobQueue:
    """FIFO admission of generating work, one running job per engine.

    `admit(engine, request_id, gone)` is an async context manager: it returns when the job is at the head of its engine's queue
    and that engine is idle, and releases the engine on exit (normal, error or cancellation). At most `max_waiting` jobs
    wait over all engines (429 with Retry-After beyond that). A waiting job leaves the queue when `gone()` (awaitable, true when
    the client disconnected; HTTP 499) or after `max_wait` seconds (HTTP 503)."""

    def __init__(self, engines, max_waiting=16, max_wait=600.0, poll=1.0):
        self.max_waiting, self.max_wait, self.poll = max_waiting, max_wait, poll
        self.q = {e: collections.deque() for e in engines}
        self.cur = {e: None for e in engines}
        self.ids = {}

    def counts(self):
        return {"running": sum(j is not None for j in self.cur.values()), "waiting": sum(len(d) for d in self.q.values()),
                "max": self.max_waiting}

    def busy(self, engine):
        """True when a job runs or waits on `engine`."""
        return self.cur[engine] is not None or bool(self.q[engine])

    def position(self, rid):
        """(state, ahead) of a request id: ahead counts the running job and the waiting jobs in front on the same engine."""
        j = self.ids.get(rid)
        if j is None:
            return "unknown", 0
        if j.state == "running" or j.engine is None:
            return "running", 0
        return "waiting", (1 if self.cur[j.engine] is not None else 0) + self.q[j.engine].index(j)

    def _use_id(self, rid):
        if rid is not None and rid in self.ids:
            raise bad("This request_id is already in use by a request that has not finished.", 409)

    @contextlib.contextmanager
    def track(self, rid):
        """Register a request that needs no queue (shown as running under its id)."""
        self._use_id(rid)
        job = _Job(rid, None, asyncio.get_running_loop())
        job.state = "running"
        if rid is not None:
            self.ids[rid] = job
        try:
            yield job
        finally:
            if rid is not None and self.ids.get(rid) is job:
                del self.ids[rid]

    def _promote(self, engine):
        d = self.q[engine]
        if self.cur[engine] is None and d:
            j = d.popleft()
            j.state, self.cur[engine] = "running", j
            if not j.fut.done():
                j.fut.set_result(True)

    def _enqueue(self, engine, rid):
        self._use_id(rid)
        job = _Job(rid, engine, asyncio.get_running_loop())
        idle = self.cur[engine] is None and not self.q[engine]
        if not idle and self.counts()["waiting"] >= self.max_waiting:
            raise bad("Many people are using the server right now. Please try again in a minute.", 429, {"Retry-After": "30"}, "busy")
        self.q[engine].append(job)
        if rid is not None:
            self.ids[rid] = job
        self._promote(engine)
        return job

    def _leave(self, job):
        if job.rid is not None and self.ids.get(job.rid) is job:
            del self.ids[job.rid]
        e = job.engine
        if job.state == "waiting":
            try:
                self.q[e].remove(job)
            except ValueError:
                pass
        elif self.cur[e] is job:
            self.cur[e] = None
            self._promote(e)
        job.state = "done"

    async def _wait(self, job, gone):
        deadline = time.monotonic() + self.max_wait
        while job.state != "running":
            left = deadline - time.monotonic()
            if left <= 0:
                raise bad("The request waited too long in the queue. Please try again in a minute.", 503, {"Retry-After": "60"}, "busy")
            await asyncio.wait({job.fut}, timeout=min(self.poll, left))
            if job.state != "running" and gone is not None and await gone():
                raise bad("Client closed the request.", 499, None, "bad_request")

    @contextlib.asynccontextmanager
    async def admit(self, engine, rid=None, gone=None):
        job = self._enqueue(engine, rid)
        try:
            await self._wait(job, gone)
            yield job
        finally:
            self._leave(job)


# ---------------------------------------------------------------- /describe

DESCRIBE_MIN_GROUPS, DESCRIBE_MAX_GROUPS, DESCRIBE_MAX_TEXTS, DESCRIBE_TEXT_CHARS, DESCRIBE_NAME_CHARS = 2, 8, 4, 240, 40


def describe_groups(v):
    """`groups`: 2 to 8 lists of 1 to 4 strings; each text is stripped and cut to 240 characters (an empty text becomes "(empty)")."""
    if not isinstance(v, list) or not DESCRIBE_MIN_GROUPS <= len(v) <= DESCRIBE_MAX_GROUPS:
        raise bad(f"groups must be a list of {DESCRIBE_MIN_GROUPS} to {DESCRIBE_MAX_GROUPS} groups.")
    out = []
    for g in v:
        if not isinstance(g, list) or not 1 <= len(g) <= DESCRIBE_MAX_TEXTS or not all(isinstance(t, str) for t in g):
            raise bad(f"Each group must be a list of 1 to {DESCRIBE_MAX_TEXTS} strings.")
        out.append([t.strip()[:DESCRIBE_TEXT_CHARS] or "(empty)" for t in g])
    return out


DESCRIBE_TEMPLATE = """A language model was given this prompt:
{prompt}

Its answers were clustered by meaning into {k} groups. Group {i} contains answers like:
{group}

Answers from the other groups look like:
{others}

Write a label of 2 to 5 words that says what sets the answers of group {i} apart from the other groups: a different topic, claim, example, or approach, or an unusual feature such as garbled text. The label must be specific enough that it would not fit the other groups. Do not just say that answers are similar or repeated. No quotes, no trailing period. Reply with the label only."""


def describe_prompts(prompt, groups, template=DESCRIBE_TEMPLATE):
    """One user message per group: the task prompt, that group's texts, and the first text of each other group."""
    def bullets(ts):
        return "\n".join("- " + t.replace("\n", " ") for t in ts)
    out = []
    for i, g in enumerate(groups):
        others = [h[0] for j, h in enumerate(groups) if j != i]
        out.append(template.format(prompt=prompt, k=len(groups), i=i + 1, group=bullets(g), others=bullets(others)))
    return out


_QUOTES = "\"'`\u201c\u201d\u2018\u2019\u00ab\u00bb"
_TRAIL = ".,;:!?\u2026"


def clean_label(s):
    """Model output to a label: first line, markdown and quotes removed, trailing punctuation dropped, whitespace collapsed,
    at most 40 characters (cut at a word boundary when possible), first letter upper case. Empty when nothing is left."""
    s = (s or "").strip().split("\n")[0]
    s = re.sub(r"^\s*(?:[-*+>]\s+|#+\s*|\d+[.)]\s+)", "", s)
    s = re.sub(r"^(?:label|name|group \d+)\s*:\s*", "", s, flags=re.I)
    s = re.sub(r"[*_`~#]+", "", s) if re.search(r"[*_`~#]", s) else s
    s = re.sub("[\"`\u201c\u201d\u00ab\u00bb]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    for _ in range(3):
        s = s.strip(_QUOTES + " ").rstrip(_TRAIL + " ")
    if len(s) > DESCRIBE_NAME_CHARS:
        cut = s[:DESCRIBE_NAME_CHARS]
        s = (cut[:cut.rfind(" ")] if " " in cut and s[DESCRIBE_NAME_CHARS] != " " else cut).rstrip(" " + _TRAIL + _QUOTES + "-,")
    words = s.split(" ")
    if len(words) > 1 and all(w[:1].isupper() for w in words if w[:1].isalpha()):   # Title Case to sentence case; acronyms stay
        s = " ".join([words[0]] + [w if (len(w) > 1 and w.isupper()) else w.lower() for w in words[1:]])
    return s[:1].upper() + s[1:]


def describe_key(model, prompt, groups, template):
    return hashlib.sha256(json.dumps([model, prompt, groups, template], ensure_ascii=False).encode()).hexdigest()


class LRU:
    def __init__(self, size=256):
        self.size, self.d = size, collections.OrderedDict()

    def get(self, k):
        if k in self.d:
            self.d.move_to_end(k)
            return self.d[k]
        return None

    def put(self, k, v):
        self.d[k] = v
        self.d.move_to_end(k)
        while len(self.d) > self.size:
            self.d.popitem(last=False)


# ---------------------------------------------------------------- run log

_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
RUNLOG_SCHEMA = 1


def utc_iso(t):
    """Seconds since the epoch as 'YYYY-MM-DDTHH:MM:SSZ' (UTC)."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def sample_record(pool, key, repo, revision, embedder_id, t, run_id=None):
    """The run-log record of a successful /sample: configuration, seed, completions with exact logp and logq, timing, UTC time.
    Nothing about the visitor (address, client id, user agent, request_id, token) and no embeddings or G (recomputable from the texts)."""
    return {
        "schema": RUNLOG_SCHEMA, "kind": "sample", "run_id": run_id, "time": utc_iso(t),
        "prompt": pool["prompt"], "modelKey": key, "model": repo, "revision": revision,
        "embedderId": embedder_id, "embedder": pool.get("embedder"),
        "N": pool["N"], "Tprime": pool["Tprime"], "maxNewTokens": pool["maxNewTokens"], "seed": pool["seed"],
        "samples": [{"text": s["text"], "logp": s["logp"], "logq": s["logq"], "nTokens": s["nTokens"], "ended": s["ended"]}
                    for s in pool["samples"]],
        "timing": pool.get("timing", {}),
    }


class RunLog:
    """Append-only JSON-lines log of runs, one file per UTC day (`runs-YYYY-MM-DD.jsonl`) in a directory of mode 700 (files 600).

    `write` is safe to call from several threads (one lock, one `os.write` of a full line to an O_APPEND descriptor) and never raises:
    a failed write is reported to `on_error` (the service log) and returns False. Disabled (`enabled=False`) it does nothing."""

    def __init__(self, directory, enabled=True, on_error=None):
        self.dir, self.enabled, self.on_error = directory, bool(enabled), on_error or (lambda msg: None)
        self.lock = threading.Lock()

    @classmethod
    def from_env(cls, env, default_dir, on_error=None):
        """DPP_RUNLOG=0 (or false, off, no) turns the log off; DPP_RUNLOG_DIR sets the directory. On by default."""
        off = str(env.get("DPP_RUNLOG", "1")).strip().lower() in ("0", "false", "off", "no")
        return cls(env.get("DPP_RUNLOG_DIR") or default_dir, enabled=not off, on_error=on_error)

    def _ensure_dir(self):
        os.makedirs(self.dir, mode=0o700, exist_ok=True)
        os.chmod(self.dir, 0o700)

    def path(self, day):
        if not isinstance(day, str) or not _DAY.fullmatch(day):
            raise bad("day must be YYYY-MM-DD.")
        return os.path.join(self.dir, f"runs-{day}.jsonl")

    def write(self, record, now=None):
        if not self.enabled:
            return False
        try:
            now = now if now is not None else time.time()
            line = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            day = time.strftime("%Y-%m-%d", time.gmtime(now))
            with self.lock:
                self._ensure_dir()
                fd = os.open(self.path(day), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                try:
                    os.fchmod(fd, 0o600)
                    n = os.write(fd, line)
                    while n < len(line):   # a short write: finish the line before another writer can interleave
                        n += os.write(fd, line[n:])
                finally:
                    os.close(fd)
            return True
        except Exception as e:   # a log failure must not fail the request
            try:
                self.on_error(f"run log write failed: {type(e).__name__}: {e}")
            except Exception:
                pass
            return False

    def find(self, run_id, day=None):
        """The record with this `run_id` (a dict) or None. Looks in the file of `day` when given, else in every file, newest first."""
        if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
            return None
        names = [f"runs-{day}.jsonl"] if day else [f"runs-{d['day']}.jsonl" for d in reversed(self.days())]
        needle = f'"run_id":"{run_id}"'.encode()
        for name in names:
            try:
                with open(os.path.join(self.dir, name), "rb") as f:
                    for line in f:
                        if needle in line:
                            return json.loads(line)
            except (OSError, ValueError):
                continue
        return None

    def days(self):
        """[{day, lines, bytes}] of the files in the directory, oldest first."""
        out = []
        try:
            names = sorted(os.listdir(self.dir))
        except OSError:
            return out
        for name in names:
            m = re.fullmatch(r"runs-(\d{4}-\d{2}-\d{2})\.jsonl", name)
            if not m:
                continue
            p = os.path.join(self.dir, name)
            with open(p, "rb") as f:
                data = f.read()
            out.append({"day": m.group(1), "lines": data.count(b"\n"), "bytes": len(data)})
        return out
