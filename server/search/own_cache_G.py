#!/usr/bin/env python
"""Own-state G files for the page's cached pools (data/cache/<id>_T<tp>.own.G.bin), from exact own states.

Usage: DPP_ROOT=<server dir> .venv/bin/python search/own_cache_G.py CACHE_DIR OUT_DIR [--mu own_emb/<key>/mu_full.npy]
For every buffer of CACHE_DIR/index.json that lists an `own:<repo>` G file, teacher-forces the pool's completions (text and `ended`
flags from CACHE_DIR/<pool>.json) through the running service (POST /embed_own, exact mode), centers the recovered mean hidden
states by mu_full (the mean over the model's demo pools, as the server does for live samples), normalizes, and writes the N x N
matrix of float64 inner products, diagonal exactly 1, as little-endian float32 row-major to OUT_DIR/<file name of index.json>.
Prints, per pool, how the new G compares with the file of the same name in CACHE_DIR (the 256-d projection it replaces).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from own_client import embed_own, ROOT, rank_avg

cache, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
idx = json.load(open(os.path.join(cache, "index.json")))
keys = {"Qwen/Qwen3-14B-AWQ": "qwen3-14b"}
mus = {}
rows = []
for b in idx["buffers"]:
    f = next((v for k, v in b["G"].items() if k.startswith("own:")), None)
    if f is None:
        continue
    key = keys[b["model"]]
    if key not in mus:
        mus[key] = np.load(f"{ROOT}/own_emb/{key}/mu_full.npy").astype(np.float64)
    p = json.load(open(os.path.join(cache, b["pool"])))
    S = p["samples"]
    assert len(S) == b["size"] == p["N"]
    d = embed_own(key, p["prompt"], [s["text"] for s in S], [s["ended"] for s in S])
    assert min(d["count"]) >= 1 and d["D"] == mus[key].shape[0], (d["D"], d["count"][:3])
    v = d["V"] - mus[key]
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    G = v @ v.T
    np.fill_diagonal(G, 1.0)
    G32 = np.ascontiguousarray(G, dtype="<f4")
    open(os.path.join(out, f), "wb").write(G32.tobytes())
    n = len(S)
    old = np.fromfile(os.path.join(cache, f), dtype="<f4").reshape(n, n).astype(np.float64)
    iu = np.triu_indices(n, 1)
    a, c = old[iu], G32.astype(np.float64)[iu]
    ra, rc = rank_avg(a) - rank_avg(a).mean(), rank_avg(c) - rank_avg(c).mean()
    rows.append(dict(file=f, N=n, forced_ok=int(sum(d["forcedOk"])), max_abs=float(np.abs(old - G32).max()), mean_abs=float(np.abs(a - c).mean()),
                     mean_old=float(a.mean()), mean_new=float(c.mean()), spearman=float(ra @ rc / np.sqrt((ra @ ra) * (rc @ rc)))))
    print(json.dumps(rows[-1]), flush=True)
print("SUMMARY", json.dumps(dict(files=len(rows), max_abs=max(r["max_abs"] for r in rows), mean_abs=float(np.mean([r["mean_abs"] for r in rows])),
                                  min_spearman=min(r["spearman"] for r in rows))), flush=True)
