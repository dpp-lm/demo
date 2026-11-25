# To do

## Done: the full-page layout
The plot fills the page between the controls (left) and the selected subset and all completions as
cards (right rail); readouts, Spectrum and View in a bar at the bottom. The 3-D view turns about the
origin, opens with one 30° turn (no auto-rotation), starts with the drawing centred, pans with
Shift/right-drag or two fingers, and resets on double-click. Floor grid and x, y, z axes with arrows.

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

## Next: the matrix views (in discussion)
Full notes: docs/matrix-views.md (the views, the kernel / eigendecomposition / PCA relationships, open questions).
- A, high priority: the pool's similarity matrix, reordered so similar completions form blocks (VAT,
  Bezdek & Hathaway 2002; Manuel's "pairs of completions" panel is the start). Details under discussion.
  A dendrogram only if its ordering is coherent with the DPP's geometry (no arbitrary linkage rule).
- Themes: eigenvectors of the kernel as themes over the completions (the dual view is where they live);
  effective dimension Σ λ/(λ+1) (also explains "K too large").
- Conditioning (Schur complement): shared by greedy MAP and sampling; value as an exploration tool TBD.
- Pairwise repulsion: π_ij/(π_i π_j) (the marginal-kernel picture; a K-DPP needs π_ij computed directly).

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
- The exact own-states PR (see "The LM's own states" below).
- The server to follow docs/BACKEND.md (scripts/conformance.mjs): refuse n > 200 with 400
  bad_request instead of capping; send G instead of emb; version and limits in /status; /embed.

## The LM's own states as an embedding
- Decided: exact own states, no random projection. The sampler sums each sequence's raw logits; the
  server recovers the full 5120-d mean hidden state ((AᵀA)⁻¹Aᵀ l̄, A = [W | 1]), centres it, and sends G.
  Pooling is the mean over generated steps, including the step that samples EOS (from his code).
- PR opened on Manuel's fork (branch own-exact on timvieira/dpp-lm-demo, one commit on his main
  e769c13). Draft, note and numpy check in docs/manuel/ on the scratch branch (own-exact.patch, own-exact-PR.md,
  own_exact_check.py). It is not tested on a GPU.
- Waiting on Manuel: the GPU checks in the PR (does the hook see raw logits, with no −inf masking?
  recovery against HF hidden states), merge, then build_own_M.py and a precompute run (mu_full.npy and
  the G files).
- Then: refresh data/cache/*.own.G.bin from his new G files (today's come from the 256-d projection),
  and check the own states against Qwen3-Embedding (the picks on the joke and sorting presets).
- Modal and the Space: no own states for now (his P files are the projection we are dropping).

## Later: Qwen3-Embedding-4B on Modal
- Modal's L40S (48 GB) cannot hold Qwen3 14B (≈30 GB) plus the 4B embedder in float32 (16 GB) and the
  generation memory. Options: an 80 GB A100 (≈25–30% more per GPU-second; exact, like the GPU server), or
  the embedder in bfloat16 on the L40S (similarities off by ~1e-3). Until then the page lists 4B on Modal
  and explains, with a way forward, if it is chosen.

## Untested
- Hugging Face sign-in end to end, now that the page is back at the main address.
- Two-finger panning on a real touch screen.

## Later: pool fit and the default T (see docs/figures/pool-fit-vs-T.png)
- Report the fit as an effective number of completions, N/ρ̂ = 1/Σ(π_i/K)² (between K and N by
  construction), e.g. "the target effectively uses 4 of these 200 completions". Keep θ̂ in nats in the
  popover. (The 1-nat warning threshold is already gone.) The note could show the count and which way raises it
  (for the Global target, move T toward 1). Optionally translate θ̂ into a total-variation bound,
  TV ≤ min(√(θ/2), √(1 − e^(−θ))).
- The default T. T and T′ are unrelated (T′ tempers each token, T the whole completion), so T = T′ is
  not a matched point. Global matches the pool only at T = 1 with T′ = 1 (then κ = p = q); for T′ ≠ 1 it
  never matches. Global and local matches at T = ∞ for every T′ (κ = p°_{T′} = q). Today's default,
  Global at T = 20, sits on the poor-fit plateau. Options: Global at T = 1, or Global and local at large T.

## Later: a "follow the action" camera (decide later)
- Today the view is set once, when the points are first drawn (axes, zoom, framing), and then only the
  visitor moves it. An optional mode could re-frame or re-fit the axes as the selection changes.

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

