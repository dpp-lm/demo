# To do

## Next batch (agreed in the review of Manuel's demo)

Correctness and speed
1. Log-space DPP routines (Manuel's greedy MAP as pivoted Cholesky and staged exact K-DPP sampler),
   plus: log-space normalizer and inclusion probabilities; adversarial reference tests (huge weight
   spans, exact and near duplicates, rank < K, K = N, extreme T) checked against brute-force
   enumeration in high precision. Fixes Sample failing and Most likely returning fewer than K on the
   sorting pool.
2. Faster eigensolver (Householder + QL), compared against Jacobi in the tests; time a slider move
   before and after; skip eigendecompositions that are not needed.
3. Ship the exact N×N similarity matrix G (float64 from float32 vectors, sent as float32) from every
   backend and in the cache, instead of embeddings; the page derives the plot from G. No random
   projection anywhere. Embedders run their pooling/normalization in float32.

Configuration and data
4. Seed/draw semantics: the stored pool is a buffer of i.i.d. draws per (prompt, weights, where it
   ran, T′, max tokens). A pool of N is a slice; draw k = completions (k−1)N+1 … kN. "Draw new ones"
   takes the next unused slice; raising N samples only the shortfall; lowering N takes a prefix.
   Only draws from the same backend are combined.
5. Target quality setting: Global (paper), κ = p^{1/T}; Global and local, κ = p^{1/T}·p°_{T′}.
   Both modes and all readouts follow it. T slider's right end labelled ∞.
6. One T for both modes.
7. Limits on every backend: N ≤ 200; N × max tokens ≤ 51,200; buffer ≤ 1,000 draws. Say so next to the
   fields instead of capping silently.
8. /embed on every backend (monstro, Modal, the Space): changing the embedder re-embeds the same
   completions.
9. docs/BACKEND.md: our own normative contract (status, sample, embed, error codes, CORS, version)
   and a conformance check run against each backend.

Readouts and plot
10. "Pool fit to target" readout: θ̂ = K²ρ̂/(N−K+1) with ρ̂ = N·Σ_i(π_i/K)² from the pool's inclusion
    probabilities under the target's importance weights; in nats; shown in both modes (Sample: estimate
    of the paper's KL bound, optimistic; Most likely: mismatch signal). Replaces ESS. Warning with next
    steps when it is large. boundOnInstance stays out of the interface.
11. Axes fitted to the selection for K ≤ 3 (the drawn box is exactly the scored volume), with a smooth
    rotation when the selection changes. No toggle.
12. Inclusion probabilities on the cards, named "inclusion", in both modes; the ? gives
    Pr(S) = det(L_S)/Z and the inclusion probability of a completion (copies counted together).
13. Chip names the target; the status line names the draw shown.
14. Keep "N completions (M distinct)" and the K message; no near-duplicate threshold.
18. The PCA spectrum bars move out of the plot: hidden by default, opened from the strip under the
    plot together with the other health indicators (log P(S), quality, diversity, pool fit). The
    1-D/2-D/3-D switch moves into that panel with them.

Running, waiting and failing (docs/run-flow.md)
15. 5-minute time limit; no redirect to Hugging Face without asking (panel with [Sign in] and
    [Use the GPU server]); keep the settings after a refused sign-in; specific failure messages with
    next-step buttons; fix Modal's "on-device model" message; say when a backend caps a setting;
    explain slow runs (20 s / 90 s); keep runs that finished after a settings change; show the
    server's timing breakdown.
16. When the GPU server is unreachable at page load, mark it "not available right now" and make Modal
    the default for that visit (an explicit default, not a fallback mid-run).

Cleanup
17. Delete the unused in-browser code (js/lm.js, js/worker.js) and the gallery script that used it.

Cut (not doing): Manuel's Expert mode (paper presets, T′ picker, shared-T switch, weight switch,
vector-length lock, point size for quality, unit positions, open the cone, axes toggle), the
multi-model rail and morphing, his extra data files, in-browser models, Gemma, mpnet, owner-only
routes in the page, the drawn-vs-true volume readout, paper T ticks, the T = T′ marker.

## Run log and shared cache (for us, not a visitor feature)
- Store every run: full configuration (prompt, exact weights, embedder, N, T′, max tokens), seed,
  completions with exact log p and log q, token counts, timestamp. Nothing about the visitor.
- Cache keyed by (prompt, exact weights, T′, max tokens): a growing stream of i.i.d. draws.
  A request for N is served from the first N draws; only the shortfall is sampled.
  Embeddings cached per (text, embedder). 4-bit and 16-bit weights keep separate streams.
- Where: Manuel's server (needs his agreement: it currently does not log prompt text),
  then Modal (on hold). Export to a private place we can download from.
- One-line notice near the prompt: "Prompts and completions are saved and may be used for research."
- Open question: should ↑ on a cached setting add fresh draws, or show the cached ones?

## Waiting on Manuel
- CORS for https://timvieira.github.io on the server.
- Qwen3-Embedding-4B on the server; then add it to the page and re-embed the precomputed pools.
- The own-state files for Qwen3 14B (P, mean, meta), so other backends compute the same vectors.

## Later
- Modal and the Space use MiniLM-L6; give them the same embedders as the server.
- Check the LM's own states against Qwen3-Embedding (the picks on the joke and sorting presets)
  before relying on them.

## Later (from the review of Manuel's demo)
- Clustering: colour completions by cluster, and/or order completions by cluster (Manuel's demo has
  four methods, a group count and LLM-written group names). Decide whether any of it is faithful and
  useful enough; colour must follow the house style.
- Views of the similarity structure: show the embedding space's notion of similarity directly
  (the spectrum; the distribution of pairwise similarities as a whole, not thresholded; possibly
  a monochrome kernel matrix ordered without clustering).
- Kernel matrix visualizations: we want a version of these (Manuel's primal L = BBᵀ and dual C = BᵀB
  heatmaps are the starting point). Design it to fit the demo: house colours, compact, no hand-picking of the subset.
- When K is too large for the pool, explain it from the spectrum (effective dimension), not a threshold.

