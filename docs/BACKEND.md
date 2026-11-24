# Backend contract (version 1)

Every backend the page can run on (the GPU server, Modal, the Hugging Face Space) answers the same
requests with the same meaning. `scripts/conformance.mjs` checks a backend against this document.

## Model and embedder

- The model is Qwen3 14B with 4-bit AWQ weights (`Qwen/Qwen3-14B-AWQ`), chat template applied,
  thinking off. Model key: `qwen3-14b`.
- Completions are exact ancestral samples from p°_{T′}: at each step the next token is drawn from
  softmax(logits / T′). For each completion the backend returns
  - `logp` = log p(y | x), the sum of log-softmax(logits) over the chosen tokens;
  - `logq` = log p°_{T′}(y | x), the same sum with logits / T′;
  - `nTokens`, and `ended` (true if the completion stopped at an end-of-sequence token, false if it
    hit `max_new_tokens`).
- Embedder `qwen3-emb` is `Qwen/Qwen3-Embedding-0.6B`, run in float32, left padding, unit-normalised.
  An empty completion is embedded as `(empty)`.

## The similarity matrix G

`G` is the N×N matrix of inner products of the unit embeddings, computed in float64 from the float32
vectors, diagonal set to exactly 1, sent as little-endian float32, row-major, base64. The page builds
every view (plot, spectrum, DPP) from G alone, so no random projection or PCA happens on a backend.

(The GPU server currently returns the vectors as `emb` + `D` instead; the page computes G from them in
float64. It should move to `G`.)

## Limits

| | |
|---|---|
| `n` | 2 ≤ n ≤ 200 |
| `max_new_tokens` | 8 ≤ max_new_tokens ≤ 256 |
| `t_prime` | 0.2 ≤ t_prime ≤ 2 |
| prompt | 1 to 2000 characters |

A request outside the limits is refused with `bad_request`; a backend never caps silently. The answer
reports what was run (`N`, `maxNewTokens`, `Tprime`), and the page refuses an answer that differs
from the request (`mismatch`).

## Requests

HTTP backends (the GPU server, Modal):

- `GET /status` → `{version, model, modelKey, embedders: {id: repo}, defaultEmbedder, limits: {maxN, maxNewTokens}, auth}`.
- `POST /sample` with `{prompt, n, t_prime, max_new_tokens, seed, model?, embedder?}` →
  `{prompt, model, embedder, embedderId, Tprime, maxNewTokens, N, D, seed, samples: [{text, logp, logq, nTokens, ended}], G, timing}`.
  `timing` holds seconds per stage, e.g. `{generate_s, embed_s}`.
- `POST /embed` with `{texts, embedder}` → `{embedder, embedderId, N, D, G}`.

Errors are JSON `{code, detail}` with an HTTP status:

| code | status | meaning |
|---|---|---|
| `bad_request` | 400 | outside the limits, or malformed |
| `auth` | 401 | sign-in needed or the token was refused |
| `rate_limit` | 429 | this visitor's daily runs are used up |
| `budget` | 429 | the month's GPU budget is used up |
| `busy` | 429 | the backend is serving other requests; try again shortly |
| `server` | 500 | the run failed; `detail` says why |

CORS: the backend allows `https://timvieira.github.io` (and `http://localhost:*` for development),
methods GET and POST, header `Authorization` and `Content-Type`.

The Hugging Face Space speaks Gradio instead, through its queue protocol: `GET /config` gives the
function index of `sample` and `embed`; `POST /gradio_api/queue/join` with
`{data, fn_index, trigger_id, session_hash}`; then the event stream at
`/gradio_api/queue/data?session_hash=…` ends with `process_completed`. `sample` takes
`data: [prompt, model, n, t_prime, max_new_tokens, seed]`, `embed` takes `data: [texts, embedder]`; the
answers are the same JSON objects as above. Errors are Gradio errors whose message starts with the code,
e.g. `bad_request: Allowed: …`; ZeroGPU's own quota errors keep their text and have no code.
(Gradio's simpler `/gradio_api/call/…` endpoint sends `null` for every error, and its JS client sends
credentials, which the Space's CORS answer does not allow for other sites; so neither is used.)

## Version

`version` in `/status` is the contract version. A change to the meaning of any field above is a new
version.
