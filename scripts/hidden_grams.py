"""Final-token hidden-state embeddings for a gallery pool.

For each completion y, runs the generating LM on (chat prompt + y) and takes the hidden state at
the last token of y, at several layers. Each layer's vectors are centred with a fixed mean (see
--mean) and scaled to unit length; the pool JSON gains, per layer, the N×N Gram matrix of those
unit vectors (the page needs only the Gram matrix: kernel, volumes and PCA all derive from it).

Usage:
  python scripts/hidden_grams.py stats  data/joke.json Qwen/Qwen3-1.7B           # diagnostics
  python scripts/hidden_grams.py write  data/joke.json Qwen/Qwen3-1.7B --mean data/means-qwen3-1.7b.npz
"""
import argparse
import base64
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def states(tok, model, prompt, texts, chat):
    """Hidden states (all layers) at the last completion token: array [L+1, N, D]."""
    head = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, **chat)
    head_ids = tok(head, add_special_tokens=False).input_ids
    out = []
    with torch.no_grad():
        for t in texts:
            ids = head_ids + tok(t or " ", add_special_tokens=False).input_ids
            hs = model(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states
            out.append(torch.stack([h[0, -1] for h in hs]).float().numpy())
    return np.stack(out, 1)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stats", "write", "means"])
    ap.add_argument("pools", nargs="+", help="pool JSON files, then the model repo last")
    ap.add_argument("--mean", help="npz of per-layer reference means (from the 'means' command)")
    ap.add_argument("--layers", default="", help="comma-separated layers to store (default: a spread + last)")
    a = ap.parse_args()
    *files, repo = a.pools
    chat = {"enable_thinking": False} if "Qwen3" in repo else {}
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).eval()
    nl = model.config.num_hidden_layers
    layers = [int(x) for x in a.layers.split(",")] if a.layers else sorted({nl // 4, nl // 2, 3 * nl // 4, nl - 4, nl})

    allH = {}
    for f in files:
        rec = json.load(open(f))
        allH[f] = states(tok, model, rec["prompt"], [s["text"] for s in rec["samples"]], chat)
        print(f"{f}: states {allH[f].shape}", flush=True)

    if a.cmd == "means":
        H = np.concatenate(list(allH.values()), 1)  # [L+1, sum N, D]
        np.savez_compressed(a.mean, mean=H.mean(1), model=repo)
        print(f"wrote reference means over {H.shape[1]} completions to {a.mean}")
        return

    ref = np.load(a.mean)["mean"] if a.mean else None
    for f, H in allH.items():
        rec = json.load(open(f))
        N = H.shape[1]
        off = ~np.eye(N, dtype=bool)
        if a.cmd == "stats":
            for l in layers:
                for name, X in (("raw", H[l]), ("pool-centred", H[l] - H[l].mean(0)),
                                *((("ref-centred", H[l] - ref[l]),) if ref is not None else ())):
                    U = unit(X)
                    G = U @ U.T
                    ev = np.clip(np.linalg.eigvalsh(G), 0, None)[::-1]
                    erank = np.exp(-(p := ev / ev.sum())[p > 0] @ np.log(p[p > 0]))
                    print(f"{f} layer {l:2d} {name:13s} mean cos {G[off].mean():.3f}  "
                          f"top eig {ev[0] / ev.sum():.2f}  eff. rank {erank:5.1f}")
            continue
        grams = {}
        for l in layers:
            U = unit(H[l] - (ref[l] if ref is not None else 0)).astype(np.float32)
            grams[str(l)] = base64.b64encode((U @ U.T).astype(np.float32).tobytes()).decode("ascii")
        rec["hidden"] = {"model": repo, "position": "last completion token", "layers": layers,
                         "numLayers": nl, "centred": "reference mean" if ref is not None else "none",
                         "grams": grams}
        json.dump(rec, open(f, "w"))
        print(f"{f}: stored Gram matrices for layers {layers}")


if __name__ == "__main__":
    main()
