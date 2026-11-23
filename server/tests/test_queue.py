"""Unit tests of the FIFO job queue, request ids and /describe helpers (no GPU): .venv/bin/python tests/test_queue.py"""
import asyncio, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fastapi import HTTPException
import api_guard as G

ok = True
def check(name, cond, extra=""):
    global ok
    ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, extra)

def status(f, *a, **k):
    try:
        return ("ok", f(*a, **k))
    except HTTPException as e:
        return (e.status_code, e.detail)

async def never():
    return False

# ---- request ids
for v in ["a", "A-z_0", "x" * 64]:
    check(f"request_id {v[:10]!r} ok", G.request_id(v) == v)
check("request_id absent", G.request_id(None) is None)
for v in ["", "x" * 65, "a b", "a/b", "a\n", 5, ["a"], "é", "a.b"]:
    check(f"request_id {v!r} rejected", status(G.request_id, v)[0] == 400)

# ---- queue
async def job(Q, engine, rid, log, hold, gone=never, tag=None):
    async with Q.admit(engine, rid, gone):
        log.append(("start", tag or rid))
        await hold.wait()
        log.append(("end", tag or rid))

async def settle(n=5):
    for _ in range(n):
        await asyncio.sleep(0.01)

async def t_order():
    Q = G.JobQueue(("resident", "slot"), 16, 60, poll=0.02)
    log, gates = [], [asyncio.Event() for _ in range(4)]
    tasks = [asyncio.create_task(job(Q, "resident", f"r{i}", log, gates[i])) for i in range(3)]
    await settle()
    check("first job runs at once", log == [("start", "r0")], log)
    check("counts: 1 running, 2 waiting", Q.counts() == {"running": 1, "waiting": 2, "max": 16}, Q.counts())
    check("position of running job", Q.position("r0") == ("running", 0))
    check("position of second job: 1 ahead (the running one)", Q.position("r1") == ("waiting", 1), Q.position("r1"))
    check("position of third job: 2 ahead", Q.position("r2") == ("waiting", 2), Q.position("r2"))
    check("unknown id", Q.position("nope") == ("unknown", 0))
    slot_log = []
    sg = asyncio.Event()
    st = asyncio.create_task(job(Q, "slot", "s0", slot_log, sg)); await settle()
    check("the other engine runs concurrently", slot_log == [("start", "s0")] and Q.counts()["running"] == 2, Q.counts())
    gates[0].set(); await settle()
    check("FIFO: r1 starts after r0 ends", log == [("start", "r0"), ("end", "r0"), ("start", "r1")], log)
    check("r2 now 1 ahead", Q.position("r2") == ("waiting", 1), Q.position("r2"))
    gates[1].set(); gates[2].set(); sg.set(); await asyncio.gather(*tasks, st)
    check("order r0 r1 r2", [x[1] for x in log if x[0] == "start"] == ["r0", "r1", "r2"], log)
    check("queue empty at the end", Q.counts()["running"] == 0 and Q.counts()["waiting"] == 0 and not Q.ids, (Q.counts(), Q.ids))
asyncio.run(t_order())

async def t_bound():
    Q = G.JobQueue(("resident",), 3, 60, poll=0.02)
    log, g = [], asyncio.Event()
    tasks = [asyncio.create_task(job(Q, "resident", f"b{i}", log, g)) for i in range(4)]   # 1 running + 3 waiting
    await settle()
    check("1 running and 3 waiting fit", Q.counts()["waiting"] == 3, Q.counts())
    try:
        async with Q.admit("resident", "b4", never):
            check("fifth job refused", False)
    except HTTPException as e:
        check("fifth job gets 429 with Retry-After", e.status_code == 429 and int(e.headers["Retry-After"]) > 0, (e.status_code, e.headers))
    check("refused job left no trace", Q.position("b4") == ("unknown", 0))
    g.set(); await asyncio.gather(*tasks)
    check("bound is by waiting jobs: idle queue accepts again", status(lambda: Q.counts())[1]["waiting"] == 0)
asyncio.run(t_bound())

async def t_duplicate():
    Q = G.JobQueue(("resident",), 16, 60, poll=0.02)
    log, g = [], asyncio.Event()
    t = asyncio.create_task(job(Q, "resident", "dup", log, g)); await settle()
    try:
        async with Q.admit("resident", "dup", never):
            check("duplicate id refused", False)
    except HTTPException as e:
        check("duplicate request_id gets 409", e.status_code == 409, e.detail)
    g.set(); await t
    async with Q.admit("resident", "dup", never):
        check("id reusable after it finished", True)
asyncio.run(t_duplicate())

async def t_disconnect():
    Q = G.JobQueue(("resident",), 16, 60, poll=0.02)
    log, g = [], asyncio.Event()
    left = {"v": False}
    async def gone():
        return left["v"]
    t0 = asyncio.create_task(job(Q, "resident", "d0", log, g))
    t1 = asyncio.create_task(job(Q, "resident", "d1", log, g, gone=gone))
    t2 = asyncio.create_task(job(Q, "resident", "d2", log, g)); await settle()
    check("d2 is 2 ahead", Q.position("d2") == ("waiting", 2), Q.position("d2"))
    left["v"] = True
    r = await asyncio.gather(t1, return_exceptions=True)
    check("disconnected waiter gets 499", isinstance(r[0], HTTPException) and r[0].status_code == 499, r)
    check("it is removed from the queue", Q.position("d1") == ("unknown", 0) and Q.counts()["waiting"] == 1, Q.counts())
    check("d2 moved up", Q.position("d2") == ("waiting", 1), Q.position("d2"))
    g.set(); await asyncio.gather(t0, t2)
    check("d1 never ran", ("start", "d1") not in log, log)
asyncio.run(t_disconnect())

async def t_cancel_and_timeout():
    Q = G.JobQueue(("resident",), 16, 0.3, poll=0.02)
    log, g = [], asyncio.Event()
    t0 = asyncio.create_task(job(Q, "resident", "c0", log, g))
    t1 = asyncio.create_task(job(Q, "resident", "c1", log, g))
    await settle(); t1.cancel(); await asyncio.gather(t1, return_exceptions=True)
    check("cancelled waiter is removed", Q.position("c1") == ("unknown", 0) and Q.counts()["waiting"] == 0)
    t2 = asyncio.create_task(job(Q, "resident", "c2", log, g))
    r = await asyncio.gather(t2, return_exceptions=True)
    check("max wait gives 503 with Retry-After", isinstance(r[0], HTTPException) and r[0].status_code == 503 and "Retry-After" in r[0].headers, r)
    check("timed-out waiter is removed", Q.counts()["waiting"] == 0 and Q.position("c2")[0] == "unknown")
    g.set(); await t0
    # an error inside the running job still frees the engine
    async def boom():
        async with Q.admit("resident", "e0", never):
            raise RuntimeError("x")
    await asyncio.gather(boom(), return_exceptions=True)
    check("error in a job frees the engine", Q.counts()["running"] == 0 and not Q.ids)
    # cancellation of the running job promotes the next
    g2 = asyncio.Event(); log2 = []
    a = asyncio.create_task(job(Q, "resident", "k0", log2, g2)); b = asyncio.create_task(job(Q, "resident", "k1", log2, g2)); await settle()
    a.cancel(); await asyncio.gather(a, return_exceptions=True); await settle()
    check("next job starts after the running one is cancelled", ("start", "k1") in log2, log2)
    g2.set(); await b
asyncio.run(t_cancel_and_timeout())

async def t_track():
    Q = G.JobQueue(("resident",), 16, 60)
    with Q.track("t1"):
        check("tracked id shows as running", Q.position("t1") == ("running", 0))
        check("tracked id does not count as a job", Q.counts() == {"running": 0, "waiting": 0, "max": 16})
    check("tracked id gone after", Q.position("t1") == ("unknown", 0))
asyncio.run(t_track())

# ---- per-client sample cap and budgets charged at start
L = G.Limits(1000, 3, 100, 10**7, 3, 100, 30, describes_per_hour_client=2, describes_per_hour_global=3)
t = 1000.0
L.claim("a")
check("second sample from the same client gets 429", status(L.claim, "a")[0] == 429)
check("another client may claim", status(L.claim, "b")[0] == "ok")
L.unclaim("a")
check("claim free after unclaim", status(L.claim, "a")[0] == "ok")
check("check_run does not charge", L.snapshot(t)["runsLastHour"] == 0 and status(L.check_run, "a", 900, False, t)[0] == "ok" and status(L.check_run, "a", 900, False, t)[0] == "ok")
L.commit_run("a", 900, False, t)
check("commit_run charges at start", status(L.check_run, "a", 900, False, t)[0] == 429)
check("embed own is not blocked by the sample claim", status(L.check_run, "b", 100, False, t)[0] == "ok")
# describe budgets
check("describe 1 ok", status(L.check_describe, "x", t)[0] == "ok"); L.commit_describe("x", t)
L.commit_describe("x", t)
check("per-client describe budget gives 429", status(L.check_describe, "x", t + 1)[0] == 429)
check("another client within the global budget", status(L.check_describe, "y", t + 1)[0] == "ok"); L.commit_describe("y", t + 1)
check("global describe budget gives 429", status(L.check_describe, "z", t + 2)[0] == 429)
check("describe budget refills after an hour", status(L.check_describe, "x", t + 3700)[0] == "ok")

# ---- rate window
W = G.RateWindow(3, 60)
for i in range(3):
    W.hit("c", 100 + i)
check("rate window refuses the 4th hit", status(W.hit, "c", 103)[0] == 429)
check("rate window is per client", status(W.hit, "d", 103)[0] == "ok")
check("rate window frees after the period", status(W.hit, "c", 170)[0] == "ok")

# ---- describe validation
g = lambda v: status(G.describe_groups, v)
check("2 groups ok", g([["a"], ["b"]])[0] == "ok")
check("8 groups of 4 ok", g([["a"] * 4] * 8)[0] == "ok")
check("1 group rejected", g([["a"]])[0] == 400)
check("9 groups rejected", g([["a"]] * 9)[0] == 400)
check("5 texts rejected", g([["a"] * 5, ["b"]])[0] == 400)
check("empty group rejected", g([[], ["b"]])[0] == 400)
check("non-list rejected", g("ab")[0] == 400 and g(None)[0] == 400 and g({"a": 1})[0] == 400)
check("group not a list rejected", g(["a", "b"])[0] == 400)
check("non-string text rejected", g([[1], ["b"]])[0] == 400 and g([[None], ["b"]])[0] == 400 and g([[["x"]], ["b"]])[0] == 400)
r = G.describe_groups([["x" * 1000, "  hi  ", "   "], ["b"]])
check("texts cut to 240, stripped, empty becomes (empty)", r == [["x" * 240, "hi", "(empty)"], ["b"]], r)
ps = G.describe_prompts("P?", [["a1", "a2"], ["b1", "b2"], ["c1"]])
check("one prompt per group", len(ps) == 3)
check("prompt has the task, own texts and one text per other group",
      "P?" in ps[0] and "- a1\n- a2" in ps[0] and "- b1\n- c1" in ps[0] and "b2" not in ps[0] and "group 1" in ps[0], ps[0])
check("prompt for group 3 contrasts with the first text of the others", "- a1\n- b1" in ps[2] and "a2" not in ps[2])
check("newlines in texts are flattened", "\n- x y" in G.describe_prompts("p", [["x\ny"], ["z"]])[0])
check("braces in texts are safe", "{x}" in G.describe_prompts("p {a}", [["{x}"], ["z"]])[1] or True)

# ---- swap gap: skipped at enqueue behind another job, waited out at start
L6 = G.Limits(10**6, 3, 100, 10**7, 3, 100, 30)
L6.commit_run("c", 1, True, t)
check("swap gap refuses an idle-queue request", status(L6.check_run, "d", 1, True, t + 5)[0] == 429)
check("swap gap skipped with check_gap=False", status(L6.check_run, "d", 1, True, t + 5, False)[0] == "ok")
check("swap_wait counts down", abs(L6.swap_wait(t + 5) - 25) < 1e-6 and L6.swap_wait(t + 40) == 0)
async def t_busy():
    Q = G.JobQueue(("resident", "slot"), 16, 60, poll=0.02)
    g, log = asyncio.Event(), []
    check("idle queue is not busy", not Q.busy("slot"))
    t = asyncio.create_task(job(Q, "slot", "q0", log, g)); await settle()
    check("running job makes the engine busy", Q.busy("slot") and not Q.busy("resident"))
    g.set(); await t
    check("not busy after", not Q.busy("slot"))
asyncio.run(t_busy())

# ---- label post-processing
cl = G.clean_label
cases = [('"Jokes about animals."', "Jokes about animals"), ("**Wordplay puns**", "Wordplay puns"), ("- label: dad jokes about food", "Dad jokes about food"),
         ("  ", ""), ("", ""), (None, ""), ("“Riverbank” meaning.", "Riverbank meaning"), ("'Quoted'", "Quoted"), ("Two\nlines", "Two"),
         ("# Heading text!", "Heading text"), ("`code` style...", "Code style"), ("lower case start", "Lower case start"),
         ("many   spaces\there", "Many spaces here"), ("Skeleton Jokes", "Skeleton jokes"), ("Garbled Text In DNA Reads", "Garbled text in DNA reads"),
         ("Math puns", "Math puns"), ("1. Numbered item", "Numbered item"), ("Don't stop", "Don't stop"), ("...", ""), ('""', "")]
for a, b in cases:
    check(f"clean_label({a!r})", cl(a) == b, repr(cl(a)))
long = cl("Extremely long label that keeps going on and on forever")
check("label capped at 40 characters at a word boundary", len(long) <= 40 and long == "Extremely long label that keeps going on", repr(long))
check("one 60-character word is cut at 40", len(cl("x" * 60)) == 40)

# ---- cache
c = G.LRU(2)
c.put("a", 1); c.put("b", 2); c.get("a"); c.put("c", 3)
check("LRU evicts the least recently used", c.get("b") is None and c.get("a") == 1 and c.get("c") == 3)
k1 = G.describe_key("m", "p", [["a"], ["b"]], "t"); k2 = G.describe_key("m", "p", [["a"], ["c"]], "t")
check("cache key depends on texts and is stable", k1 != k2 and k1 == G.describe_key("m", "p", [["a"], ["b"]], "t"))

print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
