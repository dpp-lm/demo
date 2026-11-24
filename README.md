# Diverse sampling from language models: demo

An interactive companion to *A Model of Diverse Sampling from Language Models*
([arXiv:2611.25003](https://arxiv.org/abs/2611.25003)). A static page (GitHub Pages) plus GPU
backends that draw completions.

## What it does

1. **Draw** (a backend, docs/BACKEND.md): N completions of the prompt, drawn i.i.d. from
   p°_{T′} by Qwen3 14B (4-bit AWQ), with the exact log p and log q of each.
2. **Embed** (the same backend): Qwen3-Embedding-0.6B, unit-normalised; the backend sends the
   exact N×N similarity matrix G.
3. **Select** (`js/dpp.js`, in the browser): K completions, either
   - *Most likely*: greedy MAP of ∏ κ_i · det G_S, or
   - *Sample*: an exact K-DPP sample with importance weights w = κ / p°_{T′}.

   κ = p^{1/T} (Global, the paper) or p^{1/T} · p°_{T′} (Global and local).
4. **Plot** (`js/main.js`, `js/plot3d.js`): for K ≤ 3, axes fitted to the selection, so the drawn
   box is the volume being scored; otherwise the top principal directions of G.

Precomputed pools for the example prompts are in `data/cache/` (completions plus G per embedder).

## Backends

- The GPU server (Manuel's), no sign-in.
- Modal (`modal/`, deployed by `.github/workflows/modal.yml`), Hugging Face sign-in.
- The Hugging Face Space (`space/`), Hugging Face sign-in.

`node scripts/conformance.mjs <url>` checks a backend against docs/BACKEND.md
(`--gradio` for the Space; `--no-sample` to skip the GPU run). What the page does while waiting
and when something fails: docs/run-flow.md.

## Develop

```sh
npm test                          # DPP routines vs brute-force enumeration
python3 -m http.server 8000       # then open http://localhost:8000
node tests/smoke.cjs http://localhost:8000/    # headless checks (Playwright)
node tests/overflow.cjs http://localhost:8000/ # nothing spills out of its box, at 11 widths
python3 scripts/dev-server.py   # serves the page with a mirror of the GPU server:
                                # open http://localhost:8770/?server=local to run on it from localhost
```
