"""Live check of the draw cache against a running server: two identical stream requests must return identical samples and G, the second
with cached == n, no generation time and (almost) no wall time.
Usage: python3 tests/check_draw_cache.py [BASE_URL] [EMBEDDER]   (default http://localhost:8040, qwen3-emb). Uses a prompt of its own with the
current time in it, so the stream starts empty."""
import base64, json, sys, time, urllib.request
base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8040").rstrip("/")
emb = sys.argv[2] if len(sys.argv) > 2 else "qwen3-emb"
prompt = f"Name one fruit and one fact about it, in one sentence. (cache check {int(time.time())})"
body = {"prompt": prompt, "n": 8, "t_prime": 1.0, "max_new_tokens": 32, "embedder": emb, "offset": 0}

def post(b):
    r = urllib.request.Request(base + "/sample", json.dumps(b).encode(), {"Content-Type": "application/json", "X-Real-IP": "198.51.100.99"})
    t = time.time()
    out = json.load(urllib.request.urlopen(r, timeout=300))
    return out, time.time() - t

ok = True
def check(name, cond, extra=""):
    global ok
    ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, extra)

a, ta = post(body)
b, tb = post(body)
check("first request: generated (cached 0, streamLength 8, seed null)", a["cached"] == 0 and a["streamLength"] == 8 and a["seed"] is None and a["offset"] == 0, (a["cached"], a["streamLength"]))
check("second request: identical samples", a["samples"] == b["samples"])
check("second request: identical G (byte for byte)", a["G"] == b["G"])
check("second request: cached == n, generate_s 0", b["cached"] == 8 and b["timing"]["generate_s"] == 0, b["timing"])
check("first request used the GPU, the second did not", a["timing"]["generate_s"] > 0)
n = 8; G = __import__("array").array("f"); G.frombytes(base64.b64decode(b["G"]))
check("G is 8 x 8, diagonal exactly 1, symmetric", len(G) == 64 and all(G[i * n + i] == 1.0 for i in range(n)) and all(G[i * n + j] == G[j * n + i] for i in range(n) for j in range(n)))
c, tc = post(dict(body, n=4, offset=4))
check("prefix: [4, 8) is the second half of [0, 8)", c["samples"] == a["samples"][4:] and c["cached"] == 4 and c["timing"]["generate_s"] == 0)
d, td = post(dict(body, n=4, offset=8))
check("offset 8 extends the stream by 4 new draws", d["cached"] == 0 and d["streamLength"] == 12 and d["samples"] != a["samples"][:4])
print(f"wall: first {ta:.2f} s (generate_s {a['timing']['generate_s']}, embed_s {a['timing']['embed_s']}), second {tb:.3f} s (generate_s {b['timing']['generate_s']}, embed_s {b['timing']['embed_s']}), extension {td:.2f} s")
print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
