# To do

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
