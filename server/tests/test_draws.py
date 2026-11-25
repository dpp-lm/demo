"""Draw cache through POST /sample, no GPU: imports server.py, replaces generation and embedding by fakes, and calls the app in one event loop
with httpx's ASGI transport. Usage (vllm and torch must import, as in the service venv): CUDA_VISIBLE_DEVICES= .venv/bin/python tests/test_draws.py"""
import asyncio, base64, json, os, sys, tempfile, zlib
tmp = tempfile.mkdtemp()
os.environ.update(DPP_LOGQ="1", DPP_MODELS="qwen3-14b", DPP_OWN_EMB="1", DPP_RUNLOG_DIR=os.path.join(tmp, "runlog"), DPP_USAGE_DIR=os.path.join(tmp, "usage"),
                  DPP_DRAW_CACHE_DIR=os.path.join(tmp, "drawcache"), DPP_STATS_PASSWORD_FILE=os.path.join(tmp, "pw"), HF_HUB_OFFLINE="1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import httpx
import numpy as np
import server as srv
import api_guard as G
import drawcache as D

ok = True
def check(name, cond, extra=""):
    global ok
    ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, extra)

REV, OWN, FAIL, DELAY, CALLS, EMBCALLS = ["rev1"], [True], [False], [0.02], [], []
OWNW = 6


def vec_of(text, noise=0.0, rng=None):
    v = np.random.default_rng(zlib.crc32(text.encode())).standard_normal(8)
    if noise:
        v = v + noise * rng.standard_normal(8)
    return (v / np.linalg.norm(v)).astype(np.float32)


async def fake_sample_pool(key, engine, prompt, n, t_prime, max_new_tokens, seed, embedder, extra=(), own_raw=False, return_ids=False, with_emb=False):
    call = dict(n=n, seed=seed, embedder=embedder, extra=tuple(extra), own_raw=own_raw, done=False)
    CALLS.append(call)
    if FAIL[0]:
        raise RuntimeError("boom")
    await asyncio.sleep(DELAY[0])
    call["done"] = True
    rng = np.random.default_rng(seed)
    samples = [{"text": "dup" if i % 3 == 0 else f"s{seed}-{i}", "logp": -1.0 - i * 0.01, "logq": -1.0 - i * 0.02, "nTokens": 5 + i, "ended": i % 2 == 0} for i in range(n)]
    emb = np.stack([vec_of(s["text"], 1e-4, rng) for s in samples])   # copies of a text differ slightly, as batched embedding makes them
    pool = {"prompt": prompt, "model": "m", "N": n, "Tprime": t_prime, "maxNewTokens": max_new_tokens, "seed": seed, "embedder": "e", "samples": samples,
            "timing": {"generate_s": 1.5, "embed_s": 0.1}, "D": 8, "emb": base64.b64encode(emb.tobytes()).decode()}
    if embedder == srv.OWN_KEY or srv.OWN_KEY in extra:
        raw = rng.standard_normal((n, OWNW)).astype("<f4") + 3.0
        if own_raw:
            pool["ownRaw"] = base64.b64encode(raw.tobytes()).decode()
    return pool


def fake_embedder(k):
    def f(texts):
        EMBCALLS.append(list(texts))
        rng = np.random.default_rng(7)
        return np.stack([vec_of(t, 1e-4, rng) for t in texts])
    return f


srv.sample_pool, srv.get_embedder = fake_sample_pool, fake_embedder
srv.model_revision = lambda k: REV[0]
srv.own_available = lambda k: OWN[0]
srv.own_mu = lambda k: np.full(OWNW, 3.0)
srv.OWN_ON = True
srv.OWN_SECRET_FILE = os.path.join(tmp, "own"); open(srv.OWN_SECRET_FILE, "w").write("k-for-tests\n")
srv.STATS_PASSWORD_FILE = os.environ["DPP_STATS_PASSWORD_FILE"]; open(srv.STATS_PASSWORD_FILE, "w").write("pw-for-tests\n")
srv.S.update(engine="fake", evrate=G.RateWindow(120, 60), qpoll=G.RateWindow(120, 60), authfail=G.RateWindow(10, 300), cacherate=G.RateWindow(120, 60))


def gm(b64, n):
    return np.frombuffer(base64.b64decode(b64), "<f4").reshape(n, n)


async def main():
    srv.S["queue"] = G.JobQueue(("resident", "slot"), 16, 60)
    srv.S["limits"] = G.Limits(10**9, 3, 10**6, 10**12, 240, 1200, 30)
    srv.S["emb_lock"] = asyncio.Lock()
    srv.SLOT["cond"] = asyncio.Condition()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=srv.app), base_url="http://t", timeout=60) as ac:
        ip = [0]
        def post(body, **h):
            ip[0] += 1
            return ac.post("/sample", json=body, headers={"X-Real-IP": f"198.51.100.{ip[0] % 250 + 1}", **h})
        B = dict(prompt="Name a color.", n=4, t_prime=1.0, max_new_tokens=16, embedder="qwen3-emb")

        st = (await ac.get("/status")).json()
        check("status: drawCache true", st["drawCache"] is True)
        # first request fills the stream
        r = await post(dict(B, offset=0, seed=5)); j = r.json()
        check("offset 0: 200, 4 samples at positions 0..3, streamLength 4, cached 0", r.status_code == 200 and j["N"] == 4 and len(j["samples"]) == 4 and j["offset"] == 0 and j["streamLength"] == 4 and j["cached"] == 0, j.get("detail"))
        check("seed of a stream answer is null and the client's seed was not used", j["seed"] is None and len(CALLS) == 1 and CALLS[0]["seed"] != 5 and CALLS[0]["n"] == 4)
        check("the new draws are generated with own states captured (own_raw) when available", CALLS[0]["extra"] == (srv.OWN_KEY,) and CALLS[0]["own_raw"] is True)
        check("timing reports the generation", j["timing"]["generate_s"] == 1.5 and j["embedderId"] == "qwen3-emb" and "emb" not in j)
        g1 = gm(j["G"], 4)
        check("G meets the contract: symmetric, diagonal exactly 1, copies of a text have identical rows", (g1 == g1.T).all() and (np.diag(g1) == 1).all() and (g1[0] == g1[3]).all() and g1[0, 3] == 1.0 and g1.max() <= 1.0, g1[0, 3])
        # identical second request: from the stream, no generation, no embedding
        n_calls, n_emb = len(CALLS), len(EMBCALLS)
        r2 = await post(dict(B, offset=0)); j2 = r2.json()
        check("second identical request: same samples", r2.status_code == 200 and j2["samples"] == j["samples"])
        check("second identical request: same G (byte for byte)", j2["G"] == j["G"])
        check("second identical request: cached == n, generate_s 0, no generation, no embedding", j2["cached"] == 4 and j2["timing"]["generate_s"] == 0.0 and len(CALLS) == n_calls and len(EMBCALLS) == n_emb and j2["streamLength"] == 4)
        # prefix property
        a = (await post(dict(B, n=2, offset=0))).json(); b = (await post(dict(B, n=2, offset=2))).json(); whole = (await post(dict(B, n=4, offset=0))).json()
        check("prefix property: [0, 2) then [2, 4) equals [0, 4)", a["samples"] + b["samples"] == whole["samples"] == j["samples"])
        # extension
        e = (await post(dict(B, n=4, offset=4))).json()
        check("offset 4: generates only the 4 missing, stream grows to 8", e["cached"] == 0 and e["streamLength"] == 8 and len(CALLS) == n_calls + 1 and CALLS[-1]["n"] == 4 and e["offset"] == 4)
        w = (await post(dict(B, n=8, offset=0))).json()
        check("a slice over old and new draws is served from the stream, in order", w["samples"] == j["samples"] + e["samples"] and w["cached"] == 8 and len(CALLS) == n_calls + 1)
        gw = gm(w["G"], 8)
        check("G of [0, 8) contains the G of [0, 4) and [4, 8) (cached vectors)", np.array_equal(gw[:4, :4], g1) and np.array_equal(gw[4:, 4:], gm(e["G"], 4)))
        seeds = [c["seed"] for c in CALLS]
        check("each extension uses a fresh server-chosen seed range", len(set(seeds)) == len(seeds) and all(abs(x - y) >= 4 for i, x in enumerate(seeds) for y in seeds[:i]), seeds)
        # gap
        gap = (await post(dict(B, n=4, offset=12))).json()
        check("offset beyond the end first fills the gap: draws 8..15 generated, positions 12..15 returned", gap["streamLength"] == 16 and CALLS[-1]["n"] == 8 and gap["cached"] == 0 and gap["offset"] == 12)
        check("the gap draws are in the stream in order", (await post(dict(B, n=8, offset=8))).json()["cached"] == 8)
        far = await post(dict(B, n=4, offset=16 + 200 - 4 + 1)); fj = far.json()
        check("offset too far: 400 bad_request that names the largest allowed offset", far.status_code == 400 and fj["code"] == "bad_request" and "at most 212" in fj["detail"], fj)
        edge = await post(dict(B, n=4, offset=212))
        check("the largest allowed offset works and generates 200", edge.status_code == 200 and CALLS[-1]["n"] == 200 and edge.json()["streamLength"] == 216)
        cap = await post(dict(B, n=4, offset=997))
        check("past the cap of 1000: 400 bad_request", cap.status_code == 400 and cap.json()["code"] == "bad_request" and "1000" in cap.json()["detail"])
        n_after = len(CALLS)
        check("the cap check generated nothing", n_after == len(CALLS))
        # invalid offsets
        for v in (-1, 1.5, "3", True, [0]):
            rr = await post(dict(B, offset=v))
            check(f"offset {v!r} is refused with 400 bad_request", rr.status_code == 400 and rr.json()["code"] == "bad_request", rr.json())
        # failure appends nothing
        B2 = dict(B, prompt="Another prompt.")
        FAIL[0] = True
        rf = await post(dict(B2, offset=0)); L0 = srv.DRAWS.length(D.stream_key("Another prompt.", "qwen3-14b", "Qwen/Qwen3-14B-AWQ", "rev1", 1.0, 16))
        FAIL[0] = False
        check("a failed batch: 500 and nothing appended", rf.status_code == 500 and L0 == 0 and rf.json()["code"] == "server")
        ok1 = (await post(dict(B2, offset=0))).json()
        check("the next request starts the stream at position 0, no hole", ok1["streamLength"] == 4 and ok1["cached"] == 0)
        # a cancelled request appends nothing and releases the key
        DELAY[0] = 0.5
        B3 = dict(B, prompt="Cancelled prompt.")
        k3 = D.stream_key("Cancelled prompt.", "qwen3-14b", "Qwen/Qwen3-14B-AWQ", "rev1", 1.0, 16)
        t = asyncio.create_task(post(dict(B3, offset=0)))
        await asyncio.sleep(0.15); t.cancel()
        try:
            await t
        except BaseException:
            pass
        await asyncio.sleep(0.8)   # longer than the fake generation: an orphaned task would have appended by now
        check("a cancelled request appends nothing", srv.DRAWS.length(k3) == 0)
        DELAY[0] = 0.02
        c3 = (await post(dict(B3, offset=0))).json()
        check("the stream is usable after the cancellation (lock released, queue free)", c3["streamLength"] == 4 and not srv._KEYLOCKS and srv.S["queue"].counts()["running"] == 0, c3.get("detail"))
        # concurrency on one key
        B4 = dict(B, prompt="Concurrent prompt.")
        before = len(CALLS)
        DELAY[0] = 0.2
        r1, r2 = await asyncio.gather(post(dict(B4, n=4, offset=0)), post(dict(B4, n=4, offset=0)))
        DELAY[0] = 0.02
        j1, j2 = r1.json(), r2.json()
        check("two concurrent requests for the same range: one generation, identical answers", len(CALLS) - before == 1 and j1["samples"] == j2["samples"] and j1["G"] == j2["G"] and sorted([j1["cached"], j2["cached"]]) == [0, 4], (j1["cached"], j2["cached"]))
        B5 = dict(B, prompt="Overlap prompt.")
        before = len(CALLS)
        DELAY[0] = 0.2
        r1, r2 = await asyncio.gather(post(dict(B5, n=4, offset=0)), post(dict(B5, n=4, offset=2)))
        DELAY[0] = 0.02
        j1, j2 = r1.json(), r2.json()
        check("overlapping concurrent ranges: the second extends only past what the first appended", [c["n"] for c in CALLS[before:]] == [4, 2] and j1["samples"][2:] == j2["samples"][:2] and j2["streamLength"] == 6 and j2["cached"] == 2, (j1["streamLength"], j2["streamLength"], j2["cached"]))
        # separate streams
        before = len(CALLS)
        sep = {}
        for name, body in [("same", B), ("T'", dict(B, t_prime=1.5)), ("max tokens", dict(B, max_new_tokens=32)), ("prompt", dict(B, prompt="Name a color. "))]:
            sep[name] = (await post(dict(body, offset=0))).json()
        check("another T' or max tokens is another stream", sep["T'"]["cached"] == 0 and sep["max tokens"]["cached"] == 0 and sep["same"]["cached"] == 4)
        check("a prompt that differs only by whitespace is the same stream (the server strips it)", sep["prompt"]["cached"] == 4)
        REV[0] = "rev2"
        rv = (await post(dict(B, offset=0))).json()
        check("another weights revision is another stream", rv["cached"] == 0 and rv["streamLength"] == 4 and rv["samples"] != j["samples"])
        REV[0] = "rev1"
        check("the old revision's stream is still there", (await post(dict(B, offset=0))).json()["cached"] == 4)
        # own embedder
        o1 = (await post(dict(B, embedder="own", offset=0))).json()
        check("own: served from stored states (no generation), D is the state width, G valid", o1["D"] == OWNW and o1["cached"] == 4 and o1["embedderId"] == "own" and o1["embedder"] == "own")
        go = gm(o1["G"], 4)
        check("own: G symmetric with diagonal exactly 1", (go == go.T).all() and (np.diag(go) == 1).all())
        o2 = (await post(dict(B, embedder="own", offset=0))).json()
        check("own: identical on a second request", o2["G"] == o1["G"] and o2["samples"] == o1["samples"])
        OWN[0] = False
        B6 = dict(B, prompt="No own states prompt.")
        await post(dict(B6, offset=0))
        OWN[0] = True
        no = await post(dict(B6, embedder="own", offset=0))
        check("own on draws stored without states: 409, not a made-up vector", no.status_code == 409, no.json())
        # emb flag
        em = (await post(dict(B, offset=0, emb=True))).json()
        check("emb: true adds the unit vectors", "emb" in em and len(base64.b64decode(em["emb"])) == 4 * 8 * 4)
        # no offset: untouched
        before, ln = len(CALLS), srv.DRAWS.stats()["draws"]
        n0 = (await post(dict(B, seed=77))).json()
        check("without offset: generated with the client's seed, answer without offset fields", CALLS[-1]["seed"] == 77 and len(CALLS) == before + 1 and n0["seed"] == 77 and "offset" not in n0 and "cached" not in n0 and "streamLength" not in n0)
        check("without offset: the draws are not added to any stream", srv.DRAWS.stats()["draws"] == ln and CALLS[-1]["extra"] == () and CALLS[-1]["own_raw"] is False)
        n1 = (await post(dict(B, seed=77))).json()
        check("without offset: a repeated seed is generated again", len(CALLS) == before + 2)
        ow = await post(dict(B, offset=0, embedders=["minilm"]), **{"X-Own-Secret": "k-for-tests"})
        check("offset cannot be combined with the owner options", ow.status_code == 400)
        # rate limit of cached reads
        srv.S["cacherate"] = G.RateWindow(3, 60)
        codes = []
        for _ in range(5):
            codes.append((await ac.post("/sample", json=dict(B, offset=0), headers={"X-Real-IP": "203.0.113.200"})).status_code)
        check("cached reads have a per-client rate limit (429 rate_limit)", codes == [200, 200, 200, 429, 429], codes)
        # run log: one record per generated batch
        lines = [json.loads(x) for x in open(os.path.join(tmp, "runlog", os.listdir(os.path.join(tmp, "runlog"))[0]))]
        stream_lines = [x for x in lines if "stream" in x]
        plain = [x for x in lines if "stream" not in x]
        gen_stream_calls = sum(1 for c in CALLS if c["own_raw"] and c["done"])
        check("run log: one record per newly generated stream batch, none per cached read", len(stream_lines) == gen_stream_calls - 0 and len(plain) == 2, (len(stream_lines), gen_stream_calls, len(plain)))
        s0 = stream_lines[0]
        check("run log: stream key, positions and the server-chosen seed", s0["stream"] == {"key": D.stream_key("Name a color.", "qwen3-14b", "Qwen/Qwen3-14B-AWQ", "rev1", 1.0, 16), "start": 0, "end": 4} and s0["seed"] == CALLS[0]["seed"] and s0["N"] == 4)
        check("run log: no visitor fields", all(k not in json.dumps(s0) for k in ("203.0.113", "198.51.100", "visitor", "session", "request_id")))
        # usage: cached and offset
        sj = (await ac.get("/stats?days=7", headers={"X-Stats-Password": "pw-for-tests"})).json()
        t = sj["totals"]
        check("stats: totals.cachedRequests and cachedCompletions", t["cachedRequests"] >= 10 and t["cachedCompletions"] >= 40 and t["cachedRequests"] < t["requests"], (t["cachedRequests"], t["cachedCompletions"], t["requests"]))
        check("stats: today.requests", sj["today"]["requests"] == t["requests"], sj["today"])
        al = (await ac.get("/stats?days=all", headers={"X-Stats-Password": "pw-for-tests"})).json()
        check("stats: days=all returns the whole history; days and since follow", al["days"] == len(al["daily"]) and al["since"] == al["daily"][0]["day"] and al["totals"]["requests"] == t["requests"])
        check("stats: a very large days is the same as all", (await ac.get("/stats?days=99999999", headers={"X-Stats-Password": "pw-for-tests"})).json()["totals"]["requests"] == t["requests"])
        check("stats: days must be a positive integer or all", (await ac.get("/stats?days=0", headers={"X-Stats-Password": "pw-for-tests"})).status_code == 400 and (await ac.get("/stats?days=abc", headers={"X-Stats-Password": "pw-for-tests"})).status_code == 400)
        keys = [o["key"] for o in al["breakdowns"]["byOrigin"]]
        check("stats: an empty origin is '' in byOrigin as in the sessions", "(none)" not in keys and "" in keys and all(s["origin"] == "" for s in al["sessions"]), keys)
        sess = al["sessions"][0]["id"]
        sd = (await ac.get("/stats/session", params={"id": sess}, headers={"X-Stats-Password": "pw-for-tests"})).json()
        check("stats/session: requests carry cached and offset", all("cached" in q and "offset" in q for q in sd["requests"]))
        # every /sample request through the handlers stores address and prompt; the cached ones show their draws
        allq = []
        for s in al["sessions"]:
            allq += (await ac.get("/stats/session", params={"id": s["id"]}, headers={"X-Stats-Password": "pw-for-tests"})).json()["requests"]
        cq = [q for q in allq if q["route"] == "/sample" and q["outcome"] == "ok" and q["offset"] is not None]
        check("stats/session: stream requests store the prompt and the address (the X-Real-IP these requests send)", cq and all(q["prompt"] and q["addr"].startswith(("198.51.100.", "203.0.113.")) for q in cq) and any(q["prompt"] == B["prompt"] for q in cq), sorted({(q["addr"], q["prompt"]) for q in cq}))
        check("stats/session: stream requests list drawTexts (N texts in stream order), also those answered from the cache", all(q["drawTexts"] is not None and len(q["drawTexts"]) == q["N"] for q in cq)
              and any(q["cached"] == q["N"] for q in cq) and all("_stream" not in q for q in allq), [(q["N"], q["cached"], q["offset"]) for q in cq[:4]])
        check("stats/session: a request without offset has no drawTexts", all(q["drawTexts"] is None for q in allq if q["offset"] is None))

    # the cache off
    srv.DRAWS.enabled = False
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=srv.app), base_url="http://t", timeout=60) as ac:
        st = (await ac.get("/status")).json()
        r = await ac.post("/sample", json=dict(B, offset=0))
        check("DPP_DRAW_CACHE=0: status says false and offset is refused", st["drawCache"] is False and r.status_code == 400 and r.json()["code"] == "bad_request")
        r = await ac.post("/sample", json=dict(B, seed=3))
        check("DPP_DRAW_CACHE=0: requests without offset work as before", r.status_code == 200 and r.json()["seed"] == 3)

B = dict(prompt="Name a color.", n=4, t_prime=1.0, max_new_tokens=16, embedder="qwen3-emb")
asyncio.run(main())
mode = lambda p: oct(os.stat(p).st_mode & 0o777)
check("storage: directory 700, database 600", mode(srv.DRAWS.dir) == "0o700" and mode(srv.DRAWS.path) == "0o600", (mode(srv.DRAWS.dir), mode(srv.DRAWS.path)))
print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
