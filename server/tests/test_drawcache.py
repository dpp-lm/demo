"""Unit tests of the draw cache store (drawcache.py); no GPU: .venv/bin/python tests/test_drawcache.py"""
import os, stat, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import drawcache as D

ok = True
def check(name, cond, extra=""):
    global ok
    ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, extra)
mode = lambda p: stat.S_IMODE(os.stat(p).st_mode)

def draws(a, b, tag="x"):
    return [{"text": f"{tag}{i}", "logp": -float(i), "logq": -float(i) / 2, "nTokens": 3 + i % 5, "ended": i % 2 == 0} for i in range(a, b)]
META = dict(model_key="m", repo="r", revision="rev", tprime=1.0, max_new=16)
def fresh(**kw):
    errs = []
    return D.DrawCache(os.path.join(tempfile.mkdtemp(), "dc"), True, on_error=errs.append, **kw), errs

K = D.stream_key("p", "m", "r", "rev", 1.0, 16)
# key components: every one separates streams
base = ["p", "m", "r", "rev", 1.0, 16]
variants = [["p2", "m", "r", "rev", 1.0, 16], ["p", "m2", "r", "rev", 1.0, 16], ["p", "m", "r16", "rev", 1.0, 16], ["p", "m", "r", "rev2", 1.0, 16], ["p", "m", "r", "rev", 1.5, 16], ["p", "m", "r", "rev", 1.0, 32]]
check("stream key is a function of prompt, model, repo, revision, T', max tokens", len({D.stream_key(*v) for v in variants + [base]}) == 7 and D.stream_key(*base) == D.stream_key(*base))
check("4-bit and 16-bit weights (different repos) have different streams", D.stream_key("p", "m", "Qwen/Qwen3-14B-AWQ", "x", 1.0, 16) != D.stream_key("p", "m", "Qwen/Qwen3-14B", "x", 1.0, 16))
check("T' 1.0 and 1 are the same key", D.stream_key("p", "m", "r", "rev", 1, 16) == K)

c, errs = fresh()
check("empty stream has length 0", c.length(K) == 0)
check("directory 700 after the first use", (c.length(K), mode(c.dir))[1] == 0o700 and mode(c.path) == 0o600, (oct(mode(c.dir)), oct(mode(c.path))))
check("append returns the new length", c.append(K, META, 0, draws(0, 4), [None] * 4, 11, "r1") == 4 and c.length(K) == 4)
c.append(K, META, 4, draws(4, 8), [b"\x00\x00\x80?" * 2] * 4, 22, "r2")
a, b_ = c.read(K, 0, 4), c.read(K, 4, 8)
check("prefix property: [0, 4) then [4, 8) equals [0, 8)", a + b_ == c.read(K, 0, 8) and c.read(K, 2, 6) == (a + b_)[2:6])
check("draws come back exactly (text, logp, logq, nTokens, ended)", c.read(K, 5, 6) == draws(5, 6))
check("own states are stored per draw", c.read_own(K, 2, 6) == [None, None, b"\x00\x00\x80?" * 2, b"\x00\x00\x80?" * 2])
try:
    c.read(K, 0, 9); r = None
except ValueError as e:
    r = e
check("reading past the end raises", r is not None)
# append only at the end
for bad_len in (0, 3, 9):
    try:
        c.append(K, META, bad_len, draws(100, 102), [None] * 2, 1); r = None
    except D.Conflict as e:
        r = e
    check(f"append at position {bad_len} of a stream of 8 is refused", r is not None and c.length(K) == 8)
try:
    c.append(D.stream_key("new", "m", "r", "rev", 1.0, 16), META, 5, draws(0, 2), [None] * 2, 1); r = None
except D.Conflict as e:
    r = e
check("a new stream must start at position 0", r is not None)
# failure appends nothing
bad = draws(8, 12); del bad[2]["logq"]
try:
    c.append(K, META, 8, bad, [None] * 4, 5); r = None
except KeyError as e:
    r = e
check("a batch that fails half way appends nothing (rolled back)", r is not None and c.length(K) == 8 and len(c.read(K, 0, 8)) == 8 and c.batches(K) == [(0, 4, 11), (4, 4, 22)])
try:
    c.append(K, META, 8, [], [], 5); r = None
except ValueError as e:
    r = e
check("an empty batch is refused", r is not None)
try:
    c.append(K, META, 8, draws(8, 12), [None] * 3, 5); r = None
except ValueError as e:
    r = e
check("draws and own states must match in number", r is not None and c.length(K) == 8)
check("a batch is stored whole, in order, with its seed and run id", c.append(K, META, 8, draws(8, 12), [None] * 4, 33, "r3") == 12 and c.batches(K)[-1] == (8, 4, 33) and c.read(K, 8, 12) == draws(8, 12))
# cap of 1000
c2, _ = fresh()
K2 = D.stream_key("cap", "m", "r", "rev", 1.0, 16)
for s in range(0, 1000, 200):
    c2.append(K2, META, s, draws(s, s + 200), [None] * 200, s + 1)
check("a stream holds 1000 draws", c2.length(K2) == 1000)
try:
    c2.append(K2, META, 1000, draws(1000, 1002), [None] * 2, 9); r = None
except ValueError as e:
    r = e
check("a stream holds at most 1000 draws", r is not None and c2.length(K2) == 1000)
# seeds
used = [(s, s + n) for _, n, s in c.batches(K)]
seeds = [c.pick_seed(K, 200) for _ in range(300)]
check("picked seed ranges never overlap an earlier batch of the stream", all(all(s + 200 <= a or s >= b for a, b in used) for s in seeds))
# make the only free area tiny: ranges cover almost everything
c3, _ = fresh()
c3.rng = type("R", (), {"vals": [5, 5, 10**9 // 1000, 7_000_000], "randrange": lambda self, n: self.vals.pop(0)})()
c3.append(K, META, 0, draws(0, 10), [None] * 10, 4)
check("a seed that would overlap is redrawn", c3.pick_seed(K, 10) == 10**9 // 1000)
# LRU eviction of whole streams
t = [100.0]
c4 = D.DrawCache(os.path.join(tempfile.mkdtemp(), "dc"), True, max_bytes=30_000, clock=lambda: t[0])
keys = [D.stream_key(f"p{i}", "m", "r", "rev", 1.0, 16) for i in range(6)]
for i, k in enumerate(keys[:3]):
    t[0] += 1; c4.append(k, META, 0, [dict(d, text="y" * 100) for d in draws(0, 50)], [None] * 50, i)
t[0] += 1; c4.read(keys[0], 0, 50)    # the first stream is used again: the second is now the least recently used
before = [c4.length(k) for k in keys[:3]]
t[0] += 1; c4.append(keys[3], META, 0, [dict(d, text="y" * 100) for d in draws(0, 50)], [None] * 50, 9)
after = [c4.length(k) for k in keys[:4]]
check("over the cap, whole streams are evicted, least recently used first", before == [50, 50, 50] and after[1] == 0 and after[0] == 50 and after[3] == 50, after)
check("size stays within the cap", c4.size() <= 30_000, c4.size())
check("an evicted stream leaves no draws or batches", c4.read_own(keys[1], 0, 3) == [None] * 3 and c4.batches(keys[1]) == [])
# embeddings
c5, _ = fresh()
vec = lambda x: (4, bytes([x]) * 16)
check("embeddings: nothing cached at first", c5.get_embs("sigA", ["a", "b"]) == {})
st = c5.put_embs("sigA", {"a": vec(1), "b": vec(2)})
check("embeddings: stored and read back", c5.get_embs("sigA", ["a", "b", "c"]) == {"a": bytes([1]) * 16, "b": bytes([2]) * 16} and st["a"] == bytes([1]) * 16)
check("embeddings: the first vector of a text is kept", c5.put_embs("sigA", {"a": vec(9)})["a"] == bytes([1]) * 16 and c5.get_embs("sigA", ["a"])["a"] == bytes([1]) * 16)
check("embeddings: another embedder id, revision or dtype is another entry", c5.get_embs("sigB", ["a"]) == {} and c5.get_embs("sigA|rev2", ["a"]) == {})
c6 = D.DrawCache(os.path.join(tempfile.mkdtemp(), "dc"), True, max_bytes=20_000, clock=(lambda: (t.__setitem__(0, t[0] + 1), t[0])[1]))
for i in range(300):
    c6.put_embs("s", {f"text{i}": (64, bytes(256))})
check("embeddings: over the cap the least recently used go", c6.size() <= 20_000 and c6.get_embs("s", ["text299"]) != {} and c6.get_embs("s", ["text0"]) == {}, c6.size())
check("stats counts", c.stats()["streams"] == 1 and c.stats()["draws"] == 12)
# switches
check("DPP_DRAW_CACHE=0 turns it off", D.DrawCache.from_env({"DPP_DRAW_CACHE": "0"}, "/x").enabled is False and D.DrawCache.from_env({}, "/x").enabled is True)
check("size cap from DPP_DRAW_CACHE_MAX_GB and the directory from DPP_DRAW_CACHE_DIR", D.DrawCache.from_env({"DPP_DRAW_CACHE_MAX_GB": "0.5", "DPP_DRAW_CACHE_DIR": "/y"}, "/x").max_bytes == 2**29 and D.DrawCache.from_env({"DPP_DRAW_CACHE_DIR": "/y"}, "/x").dir == "/y")
off = D.DrawCache(os.path.join(tempfile.mkdtemp(), "dc"), False)
try:
    off.append(K, META, 0, draws(0, 2), [None] * 2, 1); r = None
except RuntimeError as e:
    r = e
check("a disabled cache refuses appends and creates nothing", r is not None and not os.path.exists(off.dir))

print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
