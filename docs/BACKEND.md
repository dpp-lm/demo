# Backend contract

The page produces pools of completions through backends. A backend is anything that can sample N completions from a model and return, for each one, its exact log probabilities and an embedding. Backends are listed in `data/backends.json` and implemented in `js/backends.js`.

## Configuration

`data/backends.json` is an array in priority order. Each entry has an `id`, a `kind`, and kind-specific fields.

| Field | Meaning |
|---|---|
| `id` | Short name, used in the page's state. |
| `kind` | `http`, `space`, or `browser`. |
| `url` | Base URL of an `http` backend (no trailing slash) or of a Space. `/dpp_api` is the same-origin demo server. |
| `auth` | `"hf"` if the backend needs a Hugging Face access token (sent as `Authorization: Bearer`). Default `none`. |
| `embedders` | Embedder ids the backend offers, used when `/status` does not list them. |
| `label` | Name shown in the status line. |

The query parameter `?backend=<url>` adds a temporary `http` backend at the front of the list. The value must be an `http` or `https` URL. The status line shows its origin while it is in use. Example: `?backend=http://localhost:8000` points the page at a local server.

Signed-in backends (`auth: "hf"`) are used only when the page is opened with `?cloud` or the visitor is already signed in.

## The `http` kind

The server needs CORS for the page's origin: `Access-Control-Allow-Origin`, and `authorization` and `content-type` in `Access-Control-Allow-Headers` for `GET` and `POST`.

### `GET {url}/status`

Returns JSON. Two shapes are accepted.

Multi-model server (the deployed one):

```json
{
  "model": "qwen3-14b", "modelName": "Qwen3 14B", "defaultModel": "qwen3-14b",
  "models": [{"key": "qwen3-14b", "name": "Qwen3 14B", "params": 14.8, "repo": "...", "slot": "resident", "state": "ready"}],
  "embedders": {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "qwen3-emb": "Qwen/Qwen3-Embedding-0.6B", "own": null},
  "ownModels": ["qwen3-14b"], "defaultEmbedder": "qwen3-emb",
  "maxN": 200, "maxNewTokens": 256, "maxTextChars": 2000, "embed": true, "continuePool": true,
  "describe": true, "maxDescribeGroups": 8, "maxDescribeTexts": 4,
  "queue": {"running": 1, "waiting": 2, "max": 16}, "generating": 1, "waiting": 2
}
```

`state` is `ready`, `cold` (loads on first use), or `unavailable`. `embedders` is an object from key to repo (`null` for `own`); an array of ids is also accepted. `own` (the language model's own states) is offered only for models in `ownModels`. Fields other than `models` are optional. The page bounds the run settings by `maxN` and `maxNewTokens`.

Single-model server (the Modal deploy):

```json
{"model": "qwen3-14b", "gpu_minutes_left_this_month": 41}
```

This means one model with key `model`, embedders `["minilm"]` unless the configuration says otherwise.

### `POST {url}/sample`

Request (JSON):

| Field | Type | Meaning |
|---|---|---|
| `prompt` | string | Prompt, chat-formatted by the server. |
| `n` | integer | Number of completions. |
| `t_prime` | number | Sampling temperature T′ of the proposal, in [0.2, 2]. |
| `max_new_tokens` | integer | Cap on generated tokens. |
| `seed` | integer | Random seed. A new seed gives new samples. |
| `model` | string | Optional. Sent only to a multi-model server. |
| `embedder` | string | Optional. Sent when the backend offers several embedders or the requested one is not its first. The deployed server accepts `minilm`, `qwen3-emb`, `mpnet`, and `own` (only for models in `ownModels`). |
| `request_id` | string | Optional, 1 to 64 characters of `A-Za-z0-9_-`. Lets the page ask `GET /queue?id=` where the request stands. 400 if malformed, 409 if another unfinished request uses it. |

The deployed server runs generating work through a FIFO queue, one running job per engine (the resident model and the swappable slot model run side by side). A request waits in the queue before it runs, so the response can take a long time to start; the server answers 503 if the wait exceeds 10 minutes and 429 when 16 jobs are already waiting. A client may have one `/sample` request queued or running at a time (a second gets 429). If the client disconnects while waiting, the job is dropped.
Response (JSON):

```json
{
  "prompt": "...", "model": "...", "modelName": "...", "modelKey": "...", "embedder": "sentence-transformers/all-MiniLM-L6-v2", "seed": 1, "timing": {},
  "Tprime": 1.0, "maxNewTokens": 64, "N": 30, "D": 384,
  "samples": [{"text": "...", "logp": -12.3, "logq": -11.9, "nTokens": 17, "ended": true}],
  "emb": "<base64 float32, N x D, row-major>"
}
```

- `logp` is the exact natural log of p(y), the probability of the whole completion under the model at temperature 1. `logq` is the exact natural log of q(y), the probability under the locally annealed proposal at temperature `t_prime`. Both include the end-of-string token when the completion ended (`ended: true`). A completion cut at `max_new_tokens` has `ended: false` and no end-of-string term.
- Weights in the page are `p^(1/T) / q`, so errors in `logp` or `logq` change the selected sets. Approximate or length-normalized values are not accepted.
- `emb` is little-endian float32, N rows of D values, unit length (the page renormalizes). `embedder` names the embedder actually used (the deployed server reports its repo id, or `own`; the page maps it back to its key). It may differ from the request if the backend lacks the requested one, and the page then says so.
- The page decodes `emb` with the check `len(emb) = N * D`.

### `POST {url}/embed` (optional)

Embeds completions that already exist, so switching the embedder on a live pool does not resample. Advertised by `"embed": true` in `/status`.

Request: `{"texts": ["..."], "embedder": "qwen3-emb", "model": "qwen3-14b", "prompt": "...", "ended": [true, false]}`. `model` and `prompt` are required for `own`. `ended` lists, per text, whether the completion ended with the end-of-string token. The page sends it so that completions cut at the token cap do not get a spurious end token.

Response: `{"N": 30, "embedder": "qwen3-emb", "D": 1024, "emb": "<base64 float32, N x D>", "approximate": false, "note": "..."}` with rows in the order of `texts`. For `own` the result is teacher-forced and flagged `approximate: true`; the page shows `note` in the status line.

Without this endpoint the page disables the embedder switch on a live pool and says why. `request_id` is accepted here too. Only `own` embeddings use the queue (they run the language model); the other embedders answer at once.

### `POST {url}/describe` (optional)

Names groups of completions with the resident language model. Advertised by `"describe": true` in `/status`, with `maxDescribeGroups` (8) and `maxDescribeTexts` (4).

Request: `{"prompt": "Tell me a joke.", "groups": [["text", "text"], ["text"]], "request_id": "abc123"}`. `groups` has 2 to `maxDescribeGroups` lists of 1 to `maxDescribeTexts` strings. The server cuts each text to 240 characters and the prompt to `maxTextChars`. The body is at most 32 KB. `request_id` is optional.

Response: `{"names": ["Puns about science", ""], "model": "qwen3-14b", "ms": 420}`. There is one name per group, in order. A name has at most 40 characters, starts with a capital letter, and has no quotes or trailing punctuation. A name is `""` when the model returned nothing usable, and the page then falls back to "Group i". `model` is the key of the model that wrote the names (always the resident model). A repeated identical request is answered from a cache. Limits: 120 calls per client per hour and 600 per hour overall (429 with `Retry-After`).

### `GET {url}/queue?id=<request_id>` (optional)

Returns where a request that carried `request_id` stands: `{"state": "waiting", "ahead": 2}`. `state` is `waiting`, `running` or `unknown` (finished, never queued, or id not seen). `ahead` counts the jobs in front of it on its engine, the running one included, and is 0 for `running` and `unknown`. The route needs no key, returns nothing about other requests, and is limited to 120 calls per minute per client (429).

### Errors

Errors use HTTP status codes and a JSON body `{"detail": "message"}` (an `error` field is also read).

| Code | Page behavior |
|---|---|
| 400 | Bad request, for example an empty prompt. The message is shown. |
| 401 | Sign-in needed. For `auth: "hf"` backends the stored token is dropped. |
| 429 | Usage limit reached or the server is busy. With a `Retry-After` header the page says how long to wait ("Try again in about 2 minutes."). The deployed server limits each client to one queued or running `/sample` request, 150k generated tokens per hour and 3 model swaps per hour, with a 30 s global gap between swaps. It also answers 429 when 16 jobs are already waiting. |
| 503 | Backend not available, an engine died, or the request waited more than 10 minutes in the queue. The page says "The model is restarting. Try again in a minute." Network failures are treated the same way. |
| 403, 413, 415, 499, 500 | The message is shown. 413 is a body over the size limit (64 KB for `/sample`, 1.5 MiB for `/embed`). 415 means the content type was not `application/json`. |

## The `space` kind

A Hugging Face Space with a Gradio endpoint `sample` taking `[prompt, model, n, t_prime, max_new_tokens, seed]` and returning the pool JSON above. The page posts to `{url}/gradio_api/call/sample` and reads the server-sent result. It needs the visitor's Hugging Face token.

## The `browser` kind

`js/worker.js` loads a small model with transformers.js in a Web Worker and returns the same pool. It uses MiniLM only. Models that fit depend on the device (WebGPU with float16, or WebAssembly for models up to a size limit).

## Running the existing backends

- **Demo server (`/dpp_api`)**: a reverse-proxied GPU server on the same origin. Run it where the page is served and proxy `/dpp_api` to it.
- **Modal** (`modal/app.py` in the upstream repository): `modal run modal/app.py::download` once to fill the weights volume, then `modal deploy modal/app.py`. It serves Qwen3-14B on one GPU with MiniLM embeddings and requires a Hugging Face bearer token. The deploy URL goes in `data/backends.json`.
- **Hugging Face Space** (`space/app.py` in the upstream repository): deploy as a ZeroGPU Space and put its `.hf.space` URL in `data/backends.json`.
- **Your own server**: implement `/status` and `/sample` as above and open the page with `?backend=<your URL>`.

## Pools with `kernels` and no `emb`

The upstream repository adds pools that carry `kernels`, N × N Gram matrices from the language model's hidden states at chosen layers. This page does not use `kernels`. A pool without `emb` is shown with a message and no plot.
