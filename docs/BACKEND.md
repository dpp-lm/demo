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

Multi-model server:

```json
{
  "models": [{"key": "qwen3-14b", "name": "Qwen3 14B", "state": "ready"}],
  "defaultModel": "qwen3-14b",
  "embedders": ["minilm", "qwen3-emb"],
  "maxN": 200,
  "maxTokens": 256,
  "embed": true,
  "continuePool": false
}
```

`state` is `ready`, `cold` (loads on first use), or `unavailable`. Fields other than `models` are optional.

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
| `embedder` | string | Optional. Sent when the backend offers several embedders or the requested one is not its first. |

Response (JSON):

```json
{
  "prompt": "...", "model": "...", "modelName": "...", "embedder": "minilm",
  "Tprime": 1.0, "maxNewTokens": 64, "N": 30, "D": 384,
  "samples": [{"text": "...", "logp": -12.3, "logq": -11.9, "nTokens": 17, "ended": true}],
  "emb": "<base64 float32, N x D, row-major>"
}
```

- `logp` is the exact natural log of p(y), the probability of the whole completion under the model at temperature 1. `logq` is the exact natural log of q(y), the probability under the locally annealed proposal at temperature `t_prime`. Both include the end-of-string token when the completion ended (`ended: true`). A completion cut at `max_new_tokens` has `ended: false` and no end-of-string term.
- Weights in the page are `p^(1/T) / q`, so errors in `logp` or `logq` change the selected sets. Approximate or length-normalized values are not accepted.
- `emb` is little-endian float32, N rows of D values, unit length (the page renormalizes). `embedder` names the embedder actually used. It may differ from the request if the backend lacks the requested one, and the page then says so.
- The page decodes `emb` with the check `len(emb) = N * D`.

### `POST {url}/embed` (optional)

Embeds completions that already exist, so switching the embedder on a live pool does not resample. Advertised by `"embed": true` in `/status`.

Request: `{"texts": ["..."], "embedder": "qwen3-emb", "model": "qwen3-14b", "prompt": "..."}`.

Response: `{"embedder": "qwen3-emb", "D": 1024, "emb": "<base64 float32, N x D>"}` with rows in the order of `texts`.

Without this endpoint the page disables the embedder switch on a live pool and says why.

### Errors

Errors use HTTP status codes and a JSON body `{"detail": "message"}` (an `error` field is also read).

| Code | Page behavior |
|---|---|
| 400 | Bad request, for example an empty prompt. The message is shown. |
| 401 | Sign-in needed. For `auth: "hf"` backends the stored token is dropped. |
| 429 | Usage limit reached. The message is shown. |
| 503 | Backend not available. Network failures are treated the same way. |

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
