# A Model of Diverse Sampling from Language Models: demo

An interactive companion to *A Model of Diverse Sampling from Language Models*
([arXiv:2611.25003](https://arxiv.org/abs/2611.25003)). It is a static site:
the language model, the embedder and the K-DPP all run in the visitor's browser,
so hosting is free (GitHub Pages) and there is no server to fall over.

## What it does

1. **Generate** (`js/lm.js`, in a Web Worker): N completions of your prompt drawn
   i.i.d. from the proposal q = p°_{T′} (local temperature T′), with log p(y) and
   log q(y) recorded token by token, end-of-string included, from the same logits
   that drew the sample. Our own logits processor does the sampling, so library
   defaults (top-k, top-p, repetition penalty) cannot silently change q.
2. **Embed**: MiniLM-L6, mean-pooled and ℓ2-normalized (φ).
3. **Visualize** (`js/main.js`, `js/plot3d.js`): uncentered PCA of φ in 1D/2D/3D
   (the eigenvalue bars pick the dimension), with each point scaled by its kernel
   column norm, so length = quality and direction = meaning.
4. **Select** (`js/dpp.js`): a size-K subset, either
   - *Sample*: an exact K-DPP sample with columns √(w/N) φ, w = p^{1/T}/q, or
   - *Most likely*: greedy MAP with columns √κ φ, κ = p^{1/T}.

   The global temperature T trades quality (small T) against diversity (large T).

## Develop

```sh
npm install --onnxruntime-node-install-cuda=skip
npm test                                       # DPP maths vs brute force
python3 -m http.server 8000                    # then open http://localhost:8000
NODE_USE_ENV_PROXY=1 node tests/check-lm.mjs 1.5 q4   # log p / log q vs teacher forcing (downloads the model)
NODE_USE_ENV_PROXY=1 node scripts/build-gallery.mjs 100 128 # rebuild data/ (a few minutes per prompt on CPU)
```

`NODE_USE_ENV_PROXY=1` is only needed behind an HTTPS proxy.

Use 4-bit (`q4`) or fp32 weights. The 8-bit (`q8`) export quantizes
activations dynamically, so step-by-step decoding and a full forward pass
disagree by tens of nats; the weights would still be self-consistent, but
`tests/check-lm.mjs` would fail.

## Deploy

`.github/workflows/pages.yml` runs the unit tests and publishes
`index.html`, `css/`, `js/` and `data/` to GitHub Pages on pushes to `main`
(or by hand via *Run workflow*). Enable it once under
*Settings → Pages → Source: GitHub Actions*.
