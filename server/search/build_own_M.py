#!/usr/bin/env python
"""Build the exact recovery matrix for the LM's own states (CPU only), replacing the random projection P.

Usage: CUDA_VISIBLE_DEVICES= nice -n 19 build_own_M.py MODELKEY

The sampler sees logits l = W h + c·1 for the final hidden state h (c: any per-step shift). With A = [W | 1]
and M = (AᵀA)⁻¹, the first d entries of M Aᵀ l are h exactly. This is linear in l, so the mean hidden state
of a sequence is recovered from the mean of its logits: h̄ = [M Aᵀ l̄]_{1..d}. The plugin therefore only
sums logits per sequence (no per-step matmul) and the server recovers the full d-dim h̄ once per sequence.
No random projection: G is formed from the full states.

Writes own_emb/<key>/M.npy ((d+1) x (d+1), float64) and own_emb/<key>/meta_exact.json. When M.npy is
present the plugin and the server use exact mode (P.safetensors is then ignored). The centring mean is
own_emb/<key>/mu_full.npy (d,), built from uncentred h̄ (/sample with own_raw) like mu.npy was.
"""
import argparse, json, os, time
import numpy as np
from build_own_P import find_W, ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key"); ap.add_argument("--repo", default=None); ap.add_argument("--chunk", type=int, default=8192)
    a = ap.parse_args()
    ms = {m["key"]: m for m in json.load(open(ROOT + "/dpp-lm-demo/data/models.json"))["models"]}
    repo = a.repo or ms[a.key]["repo"]
    g, n, info = find_W(repo)
    sl = g.get_slice(n); V, d = sl.get_shape()
    t0 = time.time()
    G = np.zeros((d + 1, d + 1))
    for s in range(0, V, a.chunk):
        w = sl[s:min(s + a.chunk, V)]
        W = w.float().numpy().astype(np.float64) if hasattr(w, "float") else np.asarray(w, dtype=np.float64)
        A = np.concatenate([W, np.ones((W.shape[0], 1))], 1)
        G += A.T @ A
    ev = np.linalg.eigvalsh(G)
    M = np.linalg.inv(G)
    M = (M + M.T) / 2
    od = f"{ROOT}/own_emb/{a.key}"; os.makedirs(od, exist_ok=True)
    np.save(od + "/M.npy", M)
    meta = dict(key=a.key, repo=repo, mode="exact", d=int(d), V=int(V), cond_A=float(np.sqrt(ev[-1] / ev[0])), source=info,
                layout="M = (A^T A)^-1, A = [W | 1]; h = (M A^T l)[:d]", seconds=round(time.time() - t0, 1))
    json.dump(meta, open(od + "/meta_exact.json", "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
