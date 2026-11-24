# To do

## Done (the batch agreed in the review of Manuel's demo)

Log-space DPP routines with brute-force reference tests; Householder + QL eigensolver; exact G from
every backend and in the cache; buffers of i.i.d. draws with draw slices; Target quality (Global /
Global and local), one T, T = ∞; limits (N ≤ 200, max tokens ≤ 256, buffer ≤ 1000; the N × tokens
limit was dropped: 200 × 256 = 51,200, so it could never bind); /embed on Modal and the Space;
docs/BACKEND.md and scripts/conformance.mjs; pool fit θ̂; fitted axes for K ≤ 3 with rotation;
inclusion on the cards; chip and status line; run-flow fixes (docs/run-flow.md); the GPU server
marked unavailable at load; in-browser code deleted; spectrum panel in the strip.

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
- The server to follow docs/BACKEND.md (scripts/conformance.mjs): refuse n > 200 with 400
  bad_request instead of capping; send G instead of emb; version and limits in /status; /embed.
- Qwen3-Embedding-4B is now on the server (qwen3-emb-4b): decide whether to add it to the page and
  re-embed the precomputed pools (Modal and the Space would need it too).
- The own-state files for Qwen3 14B (P, mean, meta), so other backends compute the same vectors.

## Later
- Check the LM's own states against Qwen3-Embedding (the picks on the joke and sorting presets)
  before relying on them.

## Later: pool fit and the default T (see docs/figures/pool-fit-vs-T.png)
- Report the fit as an effective number of completions, N/ρ̂ = 1/Σ(π_i/K)² (between K and N by
  construction), e.g. "the target effectively uses 4 of these 200 completions". Keep θ̂ in nats in the
  popover. Drop the 1-nat warning threshold; the note always shows the count and which way raises it
  (move T toward T′). Optionally translate θ̂ into a total-variation bound,
  TV ≤ min(√(θ/2), √(1 − e^(−θ))).
- Default T = T′ for the Global target (the only point where pool and target match: θ̂ at its floor for
  T′ = 1); T = ∞ is the matched point for Global and local. Today's T = 20 sits on the poor-fit plateau.

## Later: the bouquet (all completions point one way from the origin)
- Completions of one prompt share a large common direction (cosines 0.5–0.9; on the story prompt the
  first direction carries 52% of the spectrum), so the cloud is a narrow bouquet away from the origin.
  Under the paper's cosine kernel this is honest: diversity is volume measured from the origin.
- Options: (A) keep it and add a "?" explaining it; (B) centre the kernel on the pool's mean: pool-
  dependent, leaves the paper's fixed kernel and Theorem 5.2, so avoid; (C) the RBF kernel
  S_β = exp∘(β(G − 1)): origin-free (small β: det ≈ β^(K−1)·det(EᵀE), E = φ_j − φ_1), still a fixed kernel
  over strings, centred picture. Decide together with whether option C ships (see the DPP/GP handoff).

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

