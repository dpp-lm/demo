"""Route tests without a GPU: imports server.py (no engine is started, the lifespan does not run) and calls the cheap routes
through starlette's TestClient: /status contract fields, error bodies and CORS, /event, /stats*, /runlog.
Usage (needs vllm and torch importable, as the service venv has): .venv/bin/python tests/test_routes.py"""
import json, os, sys, tempfile, time
tmp = tempfile.mkdtemp()
os.environ.update(DPP_LOGQ="1", DPP_MODELS="qwen3-14b", DPP_OWN_EMB="1", DPP_RUNLOG_DIR=os.path.join(tmp, "runlog"),
                  DPP_USAGE_DIR=os.path.join(tmp, "usage"), DPP_DRAW_CACHE_DIR=os.path.join(tmp, "drawcache"), DPP_STATS_PASSWORD_FILE=os.path.join(tmp, "pw"), HF_HUB_OFFLINE="1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import server as srv
import api_guard as G
from starlette.testclient import TestClient

ok = True
def check(name, cond, extra=""):
    global ok
    ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, extra)

srv.OWN_SECRET_FILE = os.path.join(tmp, "own")
open(srv.OWN_SECRET_FILE, "w").write("OWNERKEY-for-tests\n")
srv.S.update(evrate=G.RateWindow(srv.EVENTS_PER_MINUTE, 60), authfail=G.RateWindow(srv.STATS_FAILS, srv.STATS_FAIL_WINDOW), qpoll=G.RateWindow(120, 60))
c = TestClient(srv.app, raise_server_exceptions=False, client=("127.0.0.1", 50000))   # a loopback peer, as behind Caddy: X-Real-IP is the client

# /status
st = c.get("/status").json()
check("status: version 1, modelKey, limits, auth", st["version"] == 1 and st["modelKey"] == "qwen3-14b" and st["limits"] == {"maxN": 200, "maxNewTokens": 256} and st["auth"] is False, st["limits"])
check("status: model repo and embedders {id: repo}", st["model"] == "Qwen/Qwen3-14B-AWQ" and st["embedders"]["qwen3-emb"] == "Qwen/Qwen3-Embedding-0.6B" and st["defaultEmbedder"] in st["embedders"])
check("status: old top-level fields kept", st["maxN"] == 200 and st["maxNewTokens"] == 256 and "models" in st and "queue" in st and st["describe"] is True and st["maxDescribeGroups"] == 8)
check("status: logsRuns and usage flags", st["logsRuns"] is True and st["usage"] is True)
# error bodies
r = c.post("/sample", json={"prompt": "hi", "n": 201})
check("limit refused: 400 {code: bad_request, detail}", r.status_code == 400 and r.json()["code"] == "bad_request" and "n must be at most 200" in r.json()["detail"], r.json())
check("unknown route is JSON bad_request", c.get("/nope").status_code == 404 and c.get("/nope").json()["code"] == "bad_request")
check("wrong content type: 415 bad_request", c.post("/sample", content=b"x", headers={"Content-Type": "text/plain"}).json()["code"] == "bad_request")
check("owner-only option: 401 auth", c.post("/sample", json={"prompt": "hi", "n": 2, "embedders": ["minilm"]}).json() == {"code": "auth", "detail": "The options embedders, own_raw and return_ids are for the owner's scripts."})
check("/docs is 404", c.get("/docs").status_code == 404)
# CORS
h = lambda o, **k: c.post("/sample", json={"prompt": "hi", "n": 201}, headers={"Origin": o})
check("CORS: timvieira.github.io allowed", h("https://timvieira.github.io").headers.get("access-control-allow-origin") == "https://timvieira.github.io")
check("CORS: http://localhost:5173 and http://localhost allowed", h("http://localhost:5173").headers.get("access-control-allow-origin") == "http://localhost:5173" and h("http://localhost").headers.get("access-control-allow-origin") == "http://localhost")
check("CORS: other origins not allowed", "access-control-allow-origin" not in h("https://example.org").headers and "access-control-allow-origin" not in h("http://localhost.evil.org").headers and "access-control-allow-origin" not in h("https://localhost:5173").headers)
pf = c.options("/sample", headers={"Origin": "https://timvieira.github.io", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization,content-type"})
check("CORS preflight: methods GET, POST; headers Authorization, Content-Type", pf.status_code == 200 and "POST" in pf.headers["access-control-allow-methods"] and "GET" in pf.headers["access-control-allow-methods"]
      and "authorization" in pf.headers["access-control-allow-headers"].lower() and "content-type" in pf.headers["access-control-allow-headers"].lower(), dict(pf.headers))
pf2 = c.options("/stats", headers={"Origin": "https://timvieira.github.io", "Access-Control-Request-Method": "GET", "Access-Control-Request-Headers": "x-stats-password"})
check("CORS: X-Stats-Password is not allowed for timvieira.github.io", pf2.status_code == 400, pf2.status_code)
# /event
ev = lambda **b: c.post("/event", json=b, headers={"X-Real-IP": "203.0.113.5", "Origin": "https://zueri-monstro.rufous-hamlet.ts.net"})
S1 = "sess" + "1" * 8; V1 = "vis" + "1" * 9
r = ev(session=S1, visitor=V1, type="load", referrer="news.example.org", expert=False)
check("event: load answers 204 with no body", r.status_code == 204 and r.content == b"", r.status_code)
check("event: unknown type 400", ev(session=S1, type="pwn").status_code == 400 and ev(session=S1, type="ui", name="evil").status_code == 400)
check("event: bad session 400", ev(session="x", type="load").status_code == 400)
check("event: ui with unknown fields ignored", ev(session=S1, visitor=V1, type="ui", name="mode", value="map", evil="x" * 3000).status_code == 204)
check("event: body over 4 KB refused", c.post("/event", content=json.dumps({"session": S1, "type": "ui", "name": "mode", "value": "x" * 5000}), headers={"Content-Type": "application/json"}).status_code == 413)
check("event: CORS preflight from the server's own origin and localhost", all(c.options("/event", headers={"Origin": o, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"}).status_code == 200
      for o in ("https://zueri-monstro.rufous-hamlet.ts.net", "http://localhost:8123", "https://timvieira.github.io")))
srv.S["evrate"] = G.RateWindow(3, 60)
codes = [ev(session=S1, visitor=V1, type="ui", name="groups_on").status_code for _ in range(5)]
check("event: rate limited per client (429)", codes == [204, 204, 204, 429, 429], codes)
srv.S["evrate"] = G.RateWindow(srv.EVENTS_PER_MINUTE, 60)
# a request without ids is grouped by the salted hash of the address
t0 = time.time()
fake = type("R", (), {"client": type("C", (), {"host": "127.0.0.1"})(), "headers": {"x-real-ip": "203.0.113.77"}})()
sess, vis = srv.identity(fake, None, None, t0)
srv.USAGE.record_request(t0, sess, vis, "timvieira.github.io", "/sample", "ok", model="qwen3-14b", embedder="qwen3-emb", n=4, tprime=1.0, max_new=16,
                         wait_s=0.0, run_s=1.0, gen_s=0.8, embed_s=0.2, tokens=40, completions=4, run_id="a" * 32, addr=srv.req_client(fake))
# run log record for that run
srv.RUNLOG.write(G.sample_record({"prompt": "Name a color.", "N": 1, "Tprime": 1.0, "maxNewTokens": 16, "seed": 1, "embedder": "x", "timing": {},
                                  "samples": [{"text": "Blue", "logp": -1.0, "logq": -1.0, "nTokens": 1, "ended": True}]}, "qwen3-14b", "Qwen/Qwen3-14B-AWQ", "rev", "qwen3-emb", t0, "a" * 32))
# /stats gate
check("stats: 503 while no stats password is set (and no owner key sent)", c.get("/stats").status_code == 503 and c.get("/stats").json()["code"] == "server" and "stats password" in c.get("/stats").json()["detail"])
open(srv.STATS_PASSWORD_FILE, "w").write("test-password\n")
check("stats: 401 without the header", c.get("/stats").status_code == 401 and c.get("/stats").json()["code"] == "auth")
check("stats: 401 with a wrong password", c.get("/stats", headers={"X-Stats-Password": "nope"}).status_code == 401)
g = c.get("/stats?days=7", headers={"X-Stats-Password": "test-password"})
check("stats: 200 with the password, no-store", g.status_code == 200 and g.headers["cache-control"] == "no-store", g.status_code)
j = g.json()
check("stats: shape", set(j) == {"generated", "days", "since", "totals", "today", "daily", "breakdowns", "sessions", "sessionsTruncated"} and set(j["breakdowns"]) >=
      {"byOrigin", "byEmbedder", "byErrorCode", "n", "tprime", "maxNewTokens", "medianQueueWaitS", "medianGenerateS", "medianEmbedS", "busiestHours"}, sorted(j))
check("stats: the owner key works too", c.get("/stats", headers={"X-Own-Secret": "OWNERKEY-for-tests"}).status_code == 200)
check("stats: counted the visit, the anonymous run and the refused requests", j["totals"]["sessions"] >= 2 and j["totals"]["runs"] == 1 and j["totals"]["completions"] == 4 and j["totals"]["errors"] >= 1 and {e["key"] for e in j["breakdowns"]["byErrorCode"]} >= {"bad_request", "auth"}, j["totals"])
body = json.dumps(j) + json.dumps(c.get("/stats/session?id=" + S1, headers={"X-Stats-Password": "test-password"}).json())
check("stats: the loopback peer address 127.0.0.1 is never stored (the client key is the X-Real-IP)", "127.0.0.1" not in body and "203.0.113.77" in body)
sid = sess
sd = c.get("/stats/session", params={"id": sid}, headers={"X-Stats-Password": "test-password"}).json()
check("stats/session: the run with its prompt from the run log", sd["requests"][0]["runId"] == "a" * 32 and sd["requests"][0]["prompt"] == "Name a color." and sd["requests"][0]["outcome"] == "ok", sd["requests"][:1])
s1 = c.get("/stats/session", params={"id": S1}, headers={"X-Stats-Password": "test-password"}).json()
check("stats/session: events in order with referrer host", [e["type"] for e in s1["events"]][:2] == ["load", "ui"] and s1["session"]["referrer"] == "news.example.org" and s1["session"]["origin"] == "zueri-monstro.rufous-hamlet.ts.net", s1["session"])
check("stats/session: unknown 404, bad id 400", c.get("/stats/session", params={"id": "z" * 12}, headers={"X-Stats-Password": "test-password"}).status_code == 404 and c.get("/stats/session?id=x", headers={"X-Stats-Password": "test-password"}).status_code == 400)
rr = c.get("/stats/run", params={"id": "a" * 32}, headers={"X-Stats-Password": "test-password"})
check("stats/run: the full run-log record", rr.status_code == 200 and rr.json()["samples"][0]["logp"] == -1.0 and rr.json()["run_id"] == "a" * 32)
check("stats/run: unknown 404", c.get("/stats/run", params={"id": "b" * 32}, headers={"X-Stats-Password": "test-password"}).status_code == 404)
# address, prompt and cached completions in the usage store
H = {"X-Stats-Password": "test-password"}
check("stats/session: the address is the limit key (X-Real-IP behind the proxy), on the session and the request", sd["session"]["addr"] == "203.0.113.77" and sd["requests"][0]["addr"] == "203.0.113.77", sd["session"])
check("stats/session: a row without a stored prompt gets it from the run log", sd["requests"][0]["prompt"] == "Name a color.")
s1 = c.get("/stats/session", params={"id": S1}, headers=H).json()
check("event: the address is stored on the session and /stats lists it", s1["session"]["addr"] == "203.0.113.5" and {x["id"]: x["addr"] for x in c.get("/stats?days=7", headers=H).json()["sessions"]}[S1] == "203.0.113.5")
# a failed request still stores its address and prompt (capped at MAX_PROMPT_CHARS)
r = c.post("/sample", json={"prompt": "P" * (srv.MAX_PROMPT_CHARS + 50), "n": 201, "visitor": V1, "session": "sessFAIL1"}, headers={"X-Real-IP": "203.0.113.6"})
fs = c.get("/stats/session", params={"id": "sessFAIL1"}, headers=H).json()
check("failed /sample: 400, row has address and the prompt capped at MAX_PROMPT_CHARS", r.status_code == 400 and fs["session"]["addr"] == "203.0.113.6" and fs["requests"][0]["outcome"] == "bad_request"
      and fs["requests"][0]["prompt"] == "P" * srv.MAX_PROMPT_CHARS, [(q["outcome"], len(q["prompt"] or "")) for q in fs["requests"]])
c.post("/embed", json={"texts": 5, "prompt": "embed prompt", "session": "sessFAIL2"}, headers={"X-Real-IP": "203.0.113.6"})
check("failed /embed with a prompt field stores it", c.get("/stats/session", params={"id": "sessFAIL2"}, headers=H).json()["requests"][0]["prompt"] == "embed prompt")
# a request answered from the draw cache: texts come from the stream by (key, offset, N), no run log line
skey = srv.D.stream_key("cached prompt", "qwen3-14b", "Qwen/Qwen3-14B-AWQ", "rev", 1.0, 16)
srv.DRAWS.append(skey, {"model_key": "qwen3-14b", "repo": "Qwen/Qwen3-14B-AWQ", "revision": "rev", "tprime": 1.0, "max_new": 16}, 0,
                 [{"text": f"draw {i}", "logp": -1.0 - i, "logq": -1.0, "nTokens": 2, "ended": True} for i in range(6)], [None] * 6, 5, "c" * 32)
tc = time.time()
srv.USAGE.record_request(tc, "sessCACHE1", "visCACHE01", "x", "/sample", "ok", model="qwen3-14b", n=3, cached=3, offset=2, completions=0, addr="203.0.113.8", prompt="cached prompt", stream_key=skey)
srv.USAGE.record_request(tc + 1, "sessCACHE1", "visCACHE01", "x", "/sample", "ok", model="qwen3-14b", n=3, cached=3, offset=5, completions=0, addr="203.0.113.8", prompt="cached prompt", stream_key=skey)   # past the end
cs = c.get("/stats/session", params={"id": "sessCACHE1"}, headers=H).json()["requests"]
check("stats/session: cached request has drawTexts for [offset, offset + N), prompt, no runId, no stream key in the output",
      [d["text"] for d in cs[0]["drawTexts"]] == ["draw 2", "draw 3", "draw 4"] and cs[0]["drawTexts"][0]["logp"] == -3.0 and cs[0]["prompt"] == "cached prompt" and cs[0]["runId"] is None and "_stream" not in cs[0] and skey not in json.dumps(cs), cs[0])
check("stats/session: a stream that is shorter than the read (or evicted) gives drawTexts null, other requests too", cs[1]["drawTexts"] is None and sd["requests"][0]["drawTexts"] is None)
# failed attempts are limited
srv.S["authfail"] = G.RateWindow(srv.STATS_FAILS, srv.STATS_FAIL_WINDOW)
codes = [c.get("/stats", headers={"X-Stats-Password": f"bad{i}", "X-Real-IP": "198.51.100.200"}).status_code for i in range(srv.STATS_FAILS + 2)]
check("stats: failed attempts are limited to 10, then 429 (also for the right password)", codes[:srv.STATS_FAILS] == [401] * srv.STATS_FAILS and codes[-1] == 429
      and c.get("/stats", headers={"X-Stats-Password": "test-password", "X-Real-IP": "198.51.100.200"}).status_code == 429
      and c.get("/stats", headers={"X-Stats-Password": "test-password", "X-Real-IP": "198.51.100.201"}).status_code == 200, codes)
# /runlog export (owner key only)
o = {"X-Own-Secret": "OWNERKEY-for-tests"}
check("runlog: 401 without the owner key, also with the stats password", c.get("/runlog").status_code == 401 and c.get("/runlog", headers={"X-Stats-Password": "test-password"}).status_code == 401)
lst = c.get("/runlog", headers=o).json()
check("runlog: lists days with line counts", lst["enabled"] is True and len(lst["days"]) == 1 and lst["days"][0]["lines"] == 1, lst)
dl = c.get("/runlog", params={"day": lst["days"][0]["day"]}, headers=o)
check("runlog: streams the day file", dl.status_code == 200 and dl.headers["content-type"].startswith("application/x-ndjson") and json.loads(dl.text.strip())["run_id"] == "a" * 32)
check("runlog: bad day 400, missing day 404, no traversal", c.get("/runlog?day=../x", headers=o).status_code == 400 and c.get("/runlog?day=2001-01-01", headers=o).status_code == 404)

print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
