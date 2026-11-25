"""Unit tests of the run log (api_guard.RunLog) and the usage store (usage.py); no GPU: .venv/bin/python tests/test_usage.py"""
import json, os, stat, sys, tempfile, threading, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fastapi import HTTPException
import api_guard as G
import usage as U

ok = True
def check(name, cond, extra=""):
    global ok
    ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, extra)

mode = lambda p: stat.S_IMODE(os.stat(p).st_mode)
day_of = U.day_of
T0 = 1_795_000_000.0   # 2026-11-18T... UTC, a fixed clock for the fixtures

# ---------------------------------------------------------------- run log
pool = {"prompt": "Name a color.", "N": 2, "Tprime": 1.0, "maxNewTokens": 16, "seed": 7, "embedder": "Qwen/Qwen3-Embedding-0.6B",
        "timing": {"generate_s": 0.5, "embed_s": 0.1}, "D": 1024, "emb": "AAAA", "G": "BBBB", "request_id": "r-secret", "client": "203.0.113.9",
        "samples": [{"text": "Blue", "logp": -1.5, "logq": -1.25, "nTokens": 2, "ended": True, "ids": [1, 2]},
                    {"text": "Red", "logp": -2.5, "logq": -2.0, "nTokens": 2, "ended": False}]}
RID = "0123456789abcdef0123456789abcdef"
rec = G.sample_record(pool, "qwen3-14b", "Qwen/Qwen3-14B-AWQ", "abc123", "qwen3-emb", T0, RID)
blob = json.dumps(rec)
check("record has the configuration", rec["prompt"] == "Name a color." and rec["modelKey"] == "qwen3-14b" and rec["model"] == "Qwen/Qwen3-14B-AWQ"
      and rec["revision"] == "abc123" and rec["embedderId"] == "qwen3-emb" and rec["N"] == 2 and rec["Tprime"] == 1.0 and rec["maxNewTokens"] == 16 and rec["seed"] == 7)
check("record has completions with exact logp and logq", rec["samples"][0] == {"text": "Blue", "logp": -1.5, "logq": -1.25, "nTokens": 2, "ended": True})
check("record has timing, run_id, UTC time", rec["timing"] == {"generate_s": 0.5, "embed_s": 0.1} and rec["run_id"] == RID and rec["time"].endswith("Z") and rec["time"][:4] == "2026")
check("record has no embeddings, G, token ids", all(k not in rec for k in ("emb", "G", "D")) and "ids" not in rec["samples"][0] and "AAAA" not in blob)
for bad in ("r-secret", "203.0.113.9", "request_id", "visitor", "session", "client", "user_agent", "authorization"):
    check(f"record contains no visitor value or field: {bad}", bad not in blob)
check("record keys are the documented ones", set(rec) == {"schema", "kind", "run_id", "time", "prompt", "modelKey", "model", "revision", "embedderId", "embedder", "N", "Tprime", "maxNewTokens", "seed", "samples", "timing"}, sorted(rec))

d = os.path.join(tempfile.mkdtemp(), "runlog")
errs = []
L = G.RunLog(d, True, errs.append)
check("write returns True", L.write(rec, now=T0) is True)
day = time.strftime("%Y-%m-%d", time.gmtime(T0))
p = os.path.join(d, f"runs-{day}.jsonl")
check("file is runs-YYYY-MM-DD.jsonl of the UTC day", os.path.isfile(p))
check("directory mode 700, file mode 600", mode(d) == 0o700 and mode(p) == 0o600, (oct(mode(d)), oct(mode(p))))
check("one full line per record", open(p).read().count("\n") == 1 and json.loads(open(p).read()) == rec)
L.write(dict(rec, run_id="f" * 32), now=T0 + 86400)
check("next UTC day goes to the next file", len(os.listdir(d)) == 2)
check("days() counts lines", L.days() == [{"day": day, "lines": 1, "bytes": os.path.getsize(p)}, L.days()[1]] and L.days()[1]["lines"] == 1)
check("find by run_id", L.find(RID, day)["prompt"] == "Name a color." and L.find(RID)["run_id"] == RID and L.find("e" * 32) is None and L.find("../x") is None)
for badday in ("../etc", "2026-1-1", "x", None):
    try:
        L.path(badday); r = None
    except HTTPException as e:
        r = e.status_code
    check(f"path({badday!r}) refused", r == 400)
# concurrent writers: every line complete
def w(i):
    for j in range(40):
        L.write(dict(rec, run_id=f"{i:02d}{j:02d}".ljust(32, "0"), prompt="x" * 20000), now=T0 + 2 * 86400)
th = [threading.Thread(target=w, args=(i,)) for i in range(8)]
[t.start() for t in th]; [t.join() for t in th]
lines = open(os.path.join(d, f"runs-{time.strftime('%Y-%m-%d', time.gmtime(T0 + 2 * 86400))}.jsonl")).read().split("\n")
check("8 threads x 40 records: 320 intact lines", len(lines) == 321 and lines[-1] == "" and all(json.loads(x)["schema"] == 1 for x in lines[:-1]))
# off switch
d2 = os.path.join(tempfile.mkdtemp(), "off")
for v in ("0", "false", "off", "no"):
    check(f"DPP_RUNLOG={v} turns it off", G.RunLog.from_env({"DPP_RUNLOG": v}, d2).enabled is False)
check("default is on", G.RunLog.from_env({}, d2).enabled is True and G.RunLog.from_env({"DPP_RUNLOG": "1"}, d2).enabled is True)
check("DPP_RUNLOG_DIR sets the directory", G.RunLog.from_env({"DPP_RUNLOG_DIR": "/x/y"}, d2).dir == "/x/y" and G.RunLog.from_env({}, d2).dir == d2)
off = G.RunLog.from_env({"DPP_RUNLOG": "0"}, d2)
check("disabled log writes nothing and creates nothing", off.write(rec) is False and not os.path.exists(d2))
# write failure: the call returns False, reports, never raises
blocker = os.path.join(tempfile.mkdtemp(), "file"); open(blocker, "w").write("x")
bad_log = G.RunLog(os.path.join(blocker, "sub"), True, errs.append)   # a path below a regular file cannot be created
n0 = len(errs)
check("failed write returns False without raising", bad_log.write(rec) is False)
check("failed write is reported to the log callback", len(errs) == n0 + 1 and "run log write failed" in errs[-1], errs[-1:])
check("unserializable record also fails softly", L.write({"x": object()}) is False)
check("a raising error callback is swallowed", G.RunLog(os.path.join(blocker, "s"), True, lambda m: 1 / 0).write(rec) is False)

# ---------------------------------------------------------------- usage store
d3 = os.path.join(tempfile.mkdtemp(), "usage")
St = U.UsageStore(d3, True, errs.append)
NOON = time.mktime(time.strptime("2026-11-20 12:00:00", "%Y-%m-%d %H:%M:%S")) - time.timezone   # 2026-11-20T12:00:00Z
DAY = 86400
S1, V1 = "s1" * 6, "v1" * 6
S2, V2 = "s2" * 6, "v2" * 6
S3 = "d-0011223344556677"; V3 = "a-00112233445566ff"
# day -2: S1 visit, one ok sample; day 0: S1 again (same visitor, new session S2), S3 anonymous with an error
St.record_event(NOON - 2 * DAY, S1, V1, "zueri.ts.net", "load", ref="news.example.org")
St.record_request(NOON - 2 * DAY + 5, S1, V1, "zueri.ts.net", "/sample", "ok", model="qwen3-14b", embedder="qwen3-emb", n=30, tprime=1.0, max_new=64,
                  wait_s=0.0, run_s=4.0, gen_s=3.0, embed_s=1.0, tokens=900, completions=30, run_id="a" * 32)
St.record_event(NOON - 2 * DAY + 100, S1, V1, "zueri.ts.net", "ui", "mode", "sample")
St.record_event(NOON, S2, V1, "zueri.ts.net", "load", ref="")
St.record_request(NOON + 3, S2, V1, "zueri.ts.net", "/sample", "ok", model="qwen3-14b", embedder="own", n=100, tprime=1.5, max_new=128,
                  wait_s=2.0, run_s=10.0, gen_s=8.0, embed_s=2.0, tokens=5000, completions=100, run_id="b" * 32)
St.record_request(NOON + 60, S2, V1, "zueri.ts.net", "/embed", "ok", embedder="qwen3-emb", n=100, wait_s=None, run_s=0.5, embed_s=0.5)
St.record_request(NOON + 70, S3, V3, "timvieira.github.io", "/sample", "rate_limit", model="qwen3-14b", embedder="qwen3-emb", n=200, tprime=1.0, max_new=256, wait_s=0.0, run_s=0.0)
St.record_request(NOON + 80, S3, V3, "timvieira.github.io", "/sample", "ok", model="qwen3-14b", embedder="qwen3-emb", n=30, tprime=1.0, max_new=64,
                  wait_s=0.0, run_s=4.0, gen_s=3.0, embed_s=1.0, tokens=800, completions=30, run_id="c" * 32)
st = St.stats(7, now=NOON + 100)
t = st["totals"]
check("db file mode 600, directory 700", mode(St.path) == 0o600 and mode(d3) == 0o700, (oct(mode(St.path)), oct(mode(d3))))
check("totals: visitors, sessions, sessions with a run", (t["visitors"], t["sessions"], t["sessionsWithRun"]) == (3 - 1, 3, 3), t)
check("totals: runs, completions, tokens", (t["runs"], t["completions"], t["tokens"]) == (3, 160, 6700), t)
check("totals: requests and errors", (t["requests"], t["errors"]) == (5, 1), t)
check("totals: GPU seconds are the run time of every request", t["gpuSeconds"] == 18.5, t["gpuSeconds"])
check("session length: median and mean", t["medianSessionS"] == 60 and t["meanSessionS"] == round((100 + 60 + 10) / 3), (t["medianSessionS"], t["meanSessionS"]))
check("today row (a daily row plus requests)", st["today"] == {"day": "2026-11-20", "visitors": 2, "sessions": 2, "runs": 2, "completions": 130, "errors": 1, "requests": 4}, st["today"])
check("cached totals are zero without cached reads", t["cachedRequests"] == 0 and t["cachedCompletions"] == 0, t)
St.record_request(NOON + 95, S3, V3, "", "/sample", "ok", model="qwen3-14b", embedder="qwen3-emb", n=10, tprime=1.0, max_new=64, wait_s=0.0, run_s=0.01, gen_s=0.0, embed_s=0.0, tokens=0, completions=0, cached=10, offset=0)
St.record_request(NOON + 96, S3, V3, "", "/sample", "ok", model="qwen3-14b", embedder="qwen3-emb", n=10, tprime=1.0, max_new=64, wait_s=0.0, run_s=1.0, gen_s=1.0, embed_s=0.0, tokens=100, completions=5, cached=5, offset=5, run_id="e" * 32)
st2 = St.stats(7, now=NOON + 100)
check("cachedRequests counts requests served entirely from the stream; cachedCompletions the draws read from it", st2["totals"]["cachedRequests"] == 1 and st2["totals"]["cachedCompletions"] == 15, st2["totals"])
check("an empty origin is '' in byOrigin and in the sessions", "" in [o["key"] for o in st2["breakdowns"]["byOrigin"]] and "(none)" not in [o["key"] for o in st2["breakdowns"]["byOrigin"]])
check("the session lists cached and offset", [(q["cached"], q["offset"]) for q in St.session(S3)["requests"][-2:]] == [(10, 0), (5, 5)])
al = St.stats("all", now=NOON + 100)
check("days=all: since is the first day with data, days counts the days up to today", al["since"] == "2026-11-18" and al["days"] == 3 and al["totals"]["requests"] == st2["totals"]["requests"], (al["since"], al["days"]))
check("more than 366 days are allowed", St.stats(5000, now=NOON + 100)["totals"]["requests"] == st2["totals"]["requests"] and St.stats(5000, now=NOON + 100)["days"] == 5000)
St.record_request(NOON - 400 * DAY, "old1" * 4, "oldv" * 4, "x", "/sample", "ok", model="m", n=2, run_s=1.0, completions=2, tokens=4, run_id="9" * 32)
al2 = St.stats("all", now=NOON + 100)
check("days=all reaches back more than a year", al2["days"] > 400 and len(al2["daily"]) == al2["days"] and al2["since"] == day_of(NOON - 400 * DAY), al2["days"])
days = [r["day"] for r in st["daily"]]
check("daily series is contiguous, ends today, starts at the first day with data", days == ["2026-11-18", "2026-11-19", "2026-11-20"], days)
check("daily: gap day is zero", st["daily"][1] == {"day": "2026-11-19", "visitors": 0, "sessions": 0, "runs": 0, "completions": 0, "errors": 0})
check("daily: first day", st["daily"][0] == {"day": "2026-11-18", "visitors": 1, "sessions": 1, "runs": 1, "completions": 30, "errors": 0}, st["daily"][0])
b = st["breakdowns"]
check("by origin splits the two copies of the page", {o["key"]: (o["requests"], o["runs"], o["sessions"]) for o in b["byOrigin"]} == {"zueri.ts.net": (3, 2, 2), "timvieira.github.io": (2, 1, 1)}, b["byOrigin"])
check("by embedder counts ok runs", {e["key"]: e["count"] for e in b["byEmbedder"]} == {"qwen3-emb": 2, "own": 1}, b["byEmbedder"])
check("by error code", b["byErrorCode"] == [{"key": "rate_limit", "count": 1}], b["byErrorCode"])
check("distributions of N, T', max tokens", [x["key"] for x in b["n"]] == [30, 100] and [x["count"] for x in b["n"]] == [2, 1] and [x["key"] for x in b["tprime"]] == [1.0, 1.5] and [x["key"] for x in b["maxNewTokens"]] == [64, 128], b["n"])
check("medians of queue wait, generate and embed time", (b["medianQueueWaitS"], b["medianGenerateS"], b["medianEmbedS"]) == (0.0, 3.0, 1.0), (b["medianQueueWaitS"], b["medianGenerateS"], b["medianEmbedS"]))
check("busiest hours: 24 entries, 12:00 UTC has all five", len(b["busiestHours"]) == 24 and b["busiestHours"][12]["requests"] == 5 and sum(h["requests"] for h in b["busiestHours"]) == 5, b["busiestHours"][12])
ss = {s["id"]: s for s in st["sessions"]}
check("sessions: newest first, short visitor id, referrer, origin, counts", [s["id"] for s in st["sessions"]] == [S3, S2, S1] and ss[S2]["visitor"] == V1[:8]
      and ss[S1]["referrer"] == "news.example.org" and ss[S2]["origin"] == "zueri.ts.net" and ss[S2]["runs"] == 1 and ss[S2]["requests"] == 2 and ss[S1]["durationS"] == 100, st["sessions"])
check("sessionsTruncated counts the rest", St.stats(7, now=NOON + 100, sessions_limit=2)["sessionsTruncated"] == 1 and len(St.stats(7, now=NOON + 100, sessions_limit=2)["sessions"]) == 2)
check("days=1 only today", St.stats(1, now=NOON + 100)["totals"]["sessions"] == 2 and len(St.stats(1, now=NOON + 100)["daily"]) == 1)
check("empty window works", U.UsageStore(os.path.join(tempfile.mkdtemp(), "e"), True).stats(30)["totals"]["runs"] == 0)
se = St.session(S1)
check("session detail: events then requests in order", [e["type"] for e in se["events"]] == ["load", "ui"] and se["events"][1]["name"] == "mode" and se["requests"][0]["runId"] == "a" * 32
      and se["requests"][0]["N"] == 30 and se["requests"][0]["outcome"] == "ok" and se["session"]["referrer"] == "news.example.org", se)
check("unknown session", St.session("zz" * 6) is None)
check("run_day finds the day of a run", St.run_day("b" * 32) == "2026-11-20" and St.run_day("7" * 32) is None)
# events: allowlist, caps, heartbeats
check("event allowlist: load and ui only", U.clean_event({"type": "drop table"}) is None and U.clean_event({}) is None and U.clean_event({"type": "visit"}) is None and U.clean_event({"type": "ui"}) is None
      and U.clean_event({"type": "ui", "name": "evil"}) is None)
check("load: referrer reduced to its host, expert flag", U.clean_event({"type": "load", "referrer": "https://Example.org:8080/a/b?q=1", "expert": True, "extra": "x"}) == ("load", "", "expert", "example.org")
      and U.clean_event({"type": "load", "referrer": "", "expert": False}) == ("load", "", "", "") and U.clean_event({"type": "load"}) == ("load", "", "", ""))
check("ui names", all(U.clean_event({"type": "ui", "name": n}) == ("ui", n, "", "") for n in ("expert_on", "expert_off", "matrices_open", "spectrum_open", "groups_on")))
check("ui mode keeps map and sample only", U.clean_event({"type": "ui", "name": "mode", "value": "map"})[2] == "map" and U.clean_event({"type": "ui", "name": "mode", "value": "sample"})[2] == "sample" and U.clean_event({"type": "ui", "name": "mode", "value": "x"})[2] == "")
check("ui preset keeps a short id and drops anything else", U.clean_event({"type": "ui", "name": "preset", "value": "joke_1"})[2] == "joke_1" and U.clean_event({"type": "ui", "name": "preset", "value": "x" * 40})[2] == "" and U.clean_event({"type": "ui", "name": "preset", "value": "a b"})[2] == "" and U.clean_event({"type": "ui", "name": "preset", "value": {"a": 1}})[2] == "")
check("other ui events drop their value", U.clean_event({"type": "ui", "name": "expert_on", "value": "secret text"})[2] == "")
check("host_of", U.host_of("https://Timvieira.github.io/dpp-lm-demo/") == "timvieira.github.io" and U.host_of("localhost:5173") == "localhost" and U.host_of("") == "" and U.host_of(None) == "")
check("request_origin falls back to the Referer host", U.request_origin({"origin": "https://a.example", "referer": "https://b.example/x"}) == "a.example" and U.request_origin({"referer": "https://b.example/x?y"}) == "b.example" and U.request_origin({}) == "")
check("ids: 8 to 64 characters of A-Za-z0-9_-", U.valid_id("abcdefgh") and U.valid_id("a" * 64) and not U.valid_id("a" * 65) and not U.valid_id("short") and not U.valid_id("bad id!!!") and not U.valid_id(5) and not U.valid_id(None))
S4, V4 = "s4" * 6, "v4" * 6
r1 = St.record_event(NOON, S4, V4, "x", "load")
for i in range(U.MAX_EVENTS_PER_SESSION + 5):
    St.record_event(NOON + 100 + i, S4, V4, "x", "ui", "mode", "map")
check("a session stores at most MAX_EVENTS_PER_SESSION events", len(St.session(S4)["events"]) == U.MAX_EVENTS_PER_SESSION)

# ---------------------------------------------------------------- identity: the daily salt and raw addresses
clock = [NOON]
salt = U.DailySalt(lambda: clock[0])
a1, d1 = salt.anon_id("203.0.113.9"); a2, _ = salt.anon_id("203.0.113.9"); a3, _ = salt.anon_id("203.0.113.10")
check("same address, same day: same id; other address: other id", a1 == a2 and a1 != a3 and a1.startswith("a-") and len(a1) == 18)
clock[0] = NOON + DAY
a4, d2 = salt.anon_id("203.0.113.9")
check("the next UTC day gives a different id and day", a4 != a1 and d2 != d1, (a1, a4))
check("the id does not contain the address", "203" not in a1 and "113" not in a1 and "203.0.113.9" not in a1)
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "usage.py")).read()
srv = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server.py")).read()
check("the salt is never written to disk (no file write or database use of it)", "self.key" in src and "open(" not in src.split("class DailySalt")[1].split("def check_password")[0]
      and "SALT" in srv and "SALT.key" not in srv and "USAGE.record" not in srv.split("SALT =")[0])
# everything stored and everything returned is free of raw addresses
for addr in ("203.0.113.9", "198.51.100.7"):
    St.record_request(NOON + 90, "d-" + "9" * 16, salt.anon_id(addr)[0], "timvieira.github.io", "/sample", "ok", model="m", embedder="e", n=2, run_id="d" * 32)
dump = St.dump_text() + json.dumps(St.stats(30, now=NOON + 100)) + json.dumps(St.session("d-" + "9" * 16))
raw = open(St.path, "rb").read() + (open(St.path + "-wal", "rb").read() if os.path.exists(St.path + "-wal") else b"")
check("stored data and /stats output contain no raw client address", all(a not in dump and a.encode() not in raw for a in ("203.0.113.9", "198.51.100.7", "203.0.113", "198.51.100")))
check("files next to the db have mode 600", all(mode(os.path.join(d3, f)) == 0o600 for f in os.listdir(d3)), sorted(os.listdir(d3)))
check("no table has a user-agent, address or secret column", all(c not in src.lower().split("schema = ")[1].split('"""')[1] for c in ("user_agent", "agent", "address", "ip ", "secret")))

# ---------------------------------------------------------------- password
dd = tempfile.mkdtemp(); own = os.path.join(dd, "own"); pw = os.path.join(dd, "pw")
open(own, "w").write("OWNERKEY1\n")
chk = lambda h, env=None: U.check_password(h, own, pw, env or {})
check("no stats password configured: 503 case", chk({"x-stats-password": "anything"}) == "unconfigured" and chk({}) == "unconfigured")
open(pw, "w").write("hunter2\n")
check("correct password", chk({"x-stats-password": "hunter2"}) == "ok")
check("wrong, empty and missing password", chk({"x-stats-password": "hunter3"}) == "bad" and chk({"x-stats-password": ""}) == "bad" and chk({}) == "bad")
check("prefix and suffix of the password are wrong", chk({"x-stats-password": "hunter"}) == "bad" and chk({"x-stats-password": "hunter22"}) == "bad")
check("non-ASCII input does not raise", chk({"x-stats-password": "hünter2"}) == "bad")
check("the owner key is accepted too", chk({"x-own-secret": "OWNERKEY1"}) == "ok" and chk({"x-own-secret": "OWNERKEY2"}) == "bad")
os.remove(pw)
check("environment variable DPP_STATS_PASSWORD is used when there is no file", chk({"x-stats-password": "envpw"}, {"DPP_STATS_PASSWORD": "envpw"}) == "ok" and chk({"x-stats-password": "no"}, {"DPP_STATS_PASSWORD": "envpw"}) == "bad")
check("the owner key works when no stats password is set", chk({"x-own-secret": "OWNERKEY1"}) == "ok")
check("password comparison is constant-time (hmac.compare_digest)", "compare_digest" in src.split("def check_password")[1].split("def clean_event")[0])
rw = G.RateWindow(3, 300)
for _ in range(3): rw.hit("c")
check("failed attempts are counted per client for the 429", rw.count("c") == 3 and rw.count("d") == 0)
# store off / failing
check("usage off: writes are no-ops", U.UsageStore.from_env({"DPP_USAGE": "0"}, d3).enabled is False and U.UsageStore.from_env({}, d3).enabled is True
      and U.UsageStore(os.path.join(tempfile.mkdtemp(), "o"), False).record_request(1, "s", "v", "o", "/sample", "ok") is None)
bad_store = U.UsageStore(os.path.join(blocker, "u"), True, errs.append)
n0 = len(errs)
check("a failing store reports and returns None", bad_store.record_request(1, "s" * 8, "v" * 8, "o", "/sample", "ok") is None and len(errs) == n0 + 1)

print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
