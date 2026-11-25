# API of the sampling server

This server (`server/`) implements version 1 of the backend contract in `docs/BACKEND.md`, which `scripts/conformance.mjs` checks, and adds the routes below the contract (`/describe`, `/queue`, `/event`, `/stats*`, `/runlog`, `/own_stats`). Behind Caddy the base URL is `https://<host>/dpp_api`. All bodies are JSON (`Content-Type: application/json`). Numbers of seconds are floats, times are UTC.

## `GET /status`

```json
{
  "version": 1, "model": "Qwen/Qwen3-14B-AWQ", "modelKey": "qwen3-14b", "modelName": "Qwen3 14B", "defaultModel": "qwen3-14b",
  "models": [{"key": "qwen3-14b", "name": "Qwen3 14B", "params": 14.8, "repo": "Qwen/Qwen3-14B-AWQ", "slot": "resident", "state": "ready"}],
  "embedders": {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "qwen3-emb": "Qwen/Qwen3-Embedding-0.6B", "qwen3-emb-4b": "Qwen/Qwen3-Embedding-4B", "mpnet": "...", "own": null},
  "ownModels": ["qwen3-14b"], "defaultEmbedder": "minilm",
  "limits": {"maxN": 200, "maxNewTokens": 256}, "auth": false,
  "maxN": 200, "maxNewTokens": 256, "maxTextChars": 2000, "embed": true, "continuePool": true,
  "describe": true, "maxDescribeGroups": 8, "maxDescribeTexts": 4,
  "queue": {"running": 0, "waiting": 0, "max": 16}, "generating": 0, "waiting": 0, "runsLastHour": 12,
  "logsRuns": true, "usage": true, "drawCache": true
}
```

- Contract fields: `version`, `model`, `modelKey`, `embedders` (id to repo; `own` has `null`), `defaultEmbedder`, `limits`, `auth` (always `false`: no sign-in).
- `maxN` and `maxNewTokens` at the top level are the older spelling of `limits`, kept for the page.
- `state` of a model is `ready`, `loading`, `cold` or `unavailable`. `own` is offered for the models in `ownModels`.
- `logsRuns`: the server stores every successful `/sample` run with its prompt and completions (the run log, below). `drawCache`: `/sample` accepts `offset` (below). `usage`: the server stores sessions, UI events and one row per request, with the client address and the prompt of each request (see Usage below). The page sends `visitor`, `session` and `POST /event` only when `usage` is true, and shows a notice when `logsRuns` is true.

## `POST /sample`

Request:

| Field | Type | Meaning |
|---|---|---|
| `prompt` | string | 1 to 2000 characters after stripping. Chat-formatted by the server. |
| `n` | integer | 2 to 200 completions. |
| `t_prime` | number | Sampling temperature T′, 0.2 to 2. |
| `max_new_tokens` | integer | 8 to 256. |
| `seed` | integer | Optional. The completions use the seeds `seed + i`. |
| `model` | string | Optional key of `/status.models`, default `qwen3-14b`. |
| `embedder` | string | Optional: `minilm` (default), `qwen3-emb`, `qwen3-emb-4b`, `mpnet`, `own` (only for `ownModels`). |
| `emb` | boolean | Optional. `true` adds `emb`, the unit vectors (see below). Default: not sent. |
| `request_id` | string | Optional, 1 to 64 characters of `A-Za-z0-9_-`, for `GET /queue`. 409 while another unfinished request uses it. |
| `offset` | integer | Optional, at least 0. Serves draws `[offset, offset + n)` of the shared stream (see "Draw cache"). Without it the request behaves as before. |
| `visitor`, `session` | string | Optional ids from the page, 8 to 64 characters of `A-Za-z0-9_-`; see "Usage". A malformed id is ignored. |

A value outside the limits is refused with 400 `bad_request`. Nothing is capped. Owner-only options (`embedders`, `own_raw`, `return_ids`) need `X-Own-Secret`.

Response:

```json
{
  "prompt": "...", "model": "Qwen/Qwen3-14B-AWQ", "modelName": "Qwen3 14B", "modelKey": "qwen3-14b",
  "embedder": "Qwen/Qwen3-Embedding-0.6B", "embedderId": "qwen3-emb",
  "Tprime": 1.0, "maxNewTokens": 64, "N": 30, "D": 1024, "seed": 1,
  "samples": [{"text": "...", "logp": -12.3, "logq": -11.9, "nTokens": 17, "ended": true}],
  "G": "<base64>", "timing": {"generate_s": 2.4, "embed_s": 0.3}
}
```

- `logp` = log p(y | x), `logq` = log of the proposal p°_{T′}(y | x), both exact sums over the chosen tokens. `ended` is true when the completion stopped at an end-of-sequence token (counted in `nTokens`, `logp`, `logq`), false when it hit `max_new_tokens`.
- `G` is the N×N matrix of inner products of the unit embeddings, computed in float64 from the float32 vectors, symmetrized, clipped to [−1, 1], diagonal exactly 1, sent as little-endian float32, row-major, base64. It is sent for every embedder, including `own`. Identical texts can have rows that differ by about 1e-5 (batched embedding); the page replaces them by the first copy's row.
- `emb` (only with `"emb": true`): the N×D unit vectors as little-endian float32, row-major, base64. Without it the response omits `emb`, which is 200×2560 floats for `qwen3-emb-4b` and 200×5120 for `own` in exact mode.
- `embedder` is the repo of the embedder (`own` for the model's own states), `embedderId` its key.

## Draw cache (`offset`)

A stream is an ordered list of i.i.d. completions for one key: (prompt as run, i.e. stripped, model key, repo and weights revision of the model, T′, max new tokens). 4-bit and 16-bit weights, and any other revision, have separate streams. A request with an integer `offset` returns draws `[offset, offset + n)` of the stream and samples only what the stream does not hold yet. The embedder does not belong to the key: draws are embedded per request (below).

- **Opt in.** Only a body with `offset` uses a stream; `offset` must be an integer, at least 0 (400 `bad_request` otherwise, and when the cache is off, `/status.drawCache` false). A request without `offset` is unchanged: its `seed` is honored and its draws are never stored, because repeated seeds would add dependent copies. With `offset` the `seed` is ignored. `embedders`, `own_raw` and `return_ids` cannot be combined with `offset`.
- **Limits.** `offset + n` is at most 1000 (400 otherwise). A request beyond the end of the stream first fills the gap, so the shortfall `offset + n − streamLength` is at most 200; otherwise 400 `bad_request` whose detail names the largest allowed offset (`streamLength + 200 − n`).
- **i.i.d.** Extensions use seeds the server picks at random (a completion is a function of seed and index, so the seed range of a new batch never overlaps one used earlier in the same stream; the seed is recorded). A batch is appended whole, in generation order, in one transaction, at the current end of the stream only. Nothing is filtered, deduplicated or reordered. A failed or cancelled batch appends nothing. Concurrent requests for one stream are serialized: the second reads what the first appended.
- **Response** is the `/sample` response plus `offset`, `streamLength` (after the request), `cached` (how many of the n draws were already in the stream when sampling started), and `seed: null`. `timing.generate_s` is 0 for an answer served entirely from the stream; `timing.embed_s` is the time to build `G`.
- **Accounting.** An answer served entirely from the stream does not enter the GPU queue, does not count against the GPU budgets and does not need the one-request-per-client slot; it is limited to 120 per minute per client (429 `rate_limit`). A shortfall is sampled like any other run (queue, budgets for the shortfall only). The run log gets one record per generated batch, with `"stream": {"key", "start", "end"}` (the stream key and the positions) and the server's seed; cached reads are not logged. The usage store records `cached` and `offset` for every request.
- **Embeddings.** Unit vectors are cached per (text, embedder id, repo, revision, dtype). `G` is built from them (float64 inner products, diagonal exactly 1), so identical texts have identical rows and a repeated request returns the same `G` byte for byte. Missing vectors are embedded on the GPU (outside the queue, like `/embed`). For `own`, the uncentered mean state of each draw is stored when the draw is generated (for every stream extension, whenever `own` is available for the model), and the vector is that state minus the model's `mu_full`, normalized. Own states cannot be recomputed exactly later (teacher forcing differs from the generation states, about 5e-4 in G), so draws stored without them answer an `own` request with 409 `bad_request`.
- **Storage.** `drawcache/draws.sqlite3` (directory 700, file 600; `DPP_DRAW_CACHE=0` turns it off, `DPP_DRAW_CACHE_DIR` moves it, `DPP_DRAW_CACHE_MAX_GB`, default 4, caps it). Over the cap, whole streams are evicted least recently used first, then cached embeddings. A stream holds at most 1000 draws.

## `POST /embed`

Request `{texts, embedder, model?, prompt?, ended?, emb?, visitor?, session?, request_id?}`: `texts` is 1 to 200 strings of at most 2000 characters. For `own`, `model` and `prompt` are required, and `ended` (one boolean per text, default all true) says whether the completion ended with the end-of-sequence token. The own states are recomputed by teacher forcing the text, so they are close to the vectors `/sample` returns for the same completions (cosine about 0.9999 for Qwen3 and Gemma 3).

Response `{N, embedder, embedderId, D, G, approximate, timing, note?, model?, emb?}`. `G` is as above. `approximate` is true for `own`.

## Errors

Every error is JSON `{"code": "...", "detail": "message"}`. `detail` is the readable message FastAPI always sent; old readers that read `detail` keep working. Responses have `Retry-After` where a wait helps.

| `code` | Status | When |
|---|---|---|
| `bad_request` | 400 (also 404, 405, 409, 413, 415, 499) | Outside the limits, malformed JSON or ids, wrong content type, body too large (413: 64 KB for `/sample`, 1.5 MiB for `/embed`), unknown route, a request id in use (409). |
| `auth` | 401 | A route or option for the owner, or the wrong or missing stats password. |
| `rate_limit` | 429 | This client's hourly share is used (tokens, embeds, describes, model swaps), too many failed stats attempts, too many events. |
| `busy` | 429 | One `/sample` per client is already queued or running, 16 jobs are waiting, the server's hourly capacity is used, another client just swapped the model. Also 503 when a queued job waited more than 600 s. |
| `starting` | 503 | An engine died or is restarting. Not in the contract; the page maps 503 to it already. |
| `server` | 500 (also 503 for stats when no stats password is set) | The run failed. |

`budget` (the month's GPU budget) is not used by this server.

## CORS

Allowed origins: `https://timvieira.github.io`, `https://dpp-lm.github.io` (the project site), `https://zueri-monstro.rufous-hamlet.ts.net` (the server's own page), and `http://localhost` with any port. Methods `GET`, `POST`. Request headers `Authorization` and `Content-Type`. `Retry-After` is exposed. The owner and stats headers are not allowed cross-origin: the stats page runs on the server's own origin, where no CORS check applies. Error answers carry the same headers.

## `POST /describe`, `GET /queue`

Unchanged from the page that used them. `/describe` body `{prompt, groups: [[text, ...], ...], request_id?, visitor?, session?}` with 2 to 8 groups of 1 to 4 texts (each cut to 240 characters, at most 32 KB) returns `{names, model, ms}`: one label of at most 40 characters per group, `""` when nothing usable came back. Limits: 120 calls per client and 600 per hour overall (429). `GET /queue?id=<request_id>` returns `{"state": "waiting"|"running"|"unknown", "ahead": int}`; 120 calls per minute per client.

## Run log (research)

Every successful `/sample` run is appended as one line to `runlog/runs-YYYY-MM-DD.jsonl` (UTC day; directory mode 700, files 600; `DPP_RUNLOG=0` turns it off, `DPP_RUNLOG_DIR` moves it). A line holds the configuration, seed, completions and timing, and nothing about the visitor (no address, client id, user agent, `request_id`, `visitor`, `session`, token) and no embeddings or `G`. `/embed` and `/describe` are not logged.

```json
{"schema":1,"kind":"sample","run_id":"2b5e3c1c0f4a4d2f9d8f3f3b5a7c1e90","time":"2026-11-25T14:24:04Z",
 "prompt":"Name a colour.","modelKey":"qwen3-14b","model":"Qwen/Qwen3-14B-AWQ","revision":"31c69efc29464b6bb0aee1398b5a7b50a99340c3",
 "embedderId":"qwen3-emb","embedder":"Qwen/Qwen3-Embedding-0.6B","N":4,"Tprime":1.0,"maxNewTokens":16,"seed":1,
 "samples":[{"text":"Blue.","logp":-0.0530041455999708,"logq":-0.05300712585449219,"nTokens":3,"ended":true}],
 "timing":{"generate_s":0.31,"embed_s":0.05}}
```

A batch generated for a stream (`offset` requests) has the same fields, `seed` is the server's seed, and an extra `"stream": {"key": "<sha256>", "start": 0, "end": 30}`. `run_id` is random and links the line to a usage row (below). `revision` is the commit hash of the local weights snapshot. Writes happen under one lock as one full line; a failed write is logged to the service log and never fails the request.

`GET /runlog` (header `X-Own-Secret`, the owner key; 401 otherwise) returns `{"enabled": true, "days": [{"day": "2026-11-25", "lines": 3, "bytes": 2750}]}`. `GET /runlog?day=YYYY-MM-DD` streams that day's file as `application/x-ndjson` (404 when absent).

## Usage

The usage store is a SQLite database (`usage/usage.sqlite3`, directory mode 700, file 600; `DPP_USAGE=0` turns it off, `DPP_USAGE_DIR` moves it), separate from the run log. It stores sessions, UI events and one row per request to `/sample`, `/embed` and `/describe`: time, session, origin host, route, model, embedder, N, T′, max tokens, outcome (`ok` or the error code, `aborted` when the client left), queue wait, run time, `generate_s` and `embed_s`, tokens, completions, the `run_id` of a successful sample, and, for each request, the client address (`addr`) and the prompt text (`prompt`, cut at 2000 characters, the prompt limit of `/sample`). Sessions and events also store `addr`. Completions are not stored in this database: a generated sample points to its run log line by `run_id`, and a sample answered from the draw cache points to its stream by a `stream_key` (an internal hash of the prompt and settings, never returned) with `offset` and `N`. The run log still holds nothing about the visitor. An existing database is migrated when the server opens it: the new columns are added and old rows keep an empty `addr` and `prompt`.

**Identity.** The page sends optional `visitor` (a random id kept in localStorage) and `session` (a random id per page load) in the body of `/sample`, `/embed`, `/describe` and `/event`. A request without them (for example from `https://timvieira.github.io`) is grouped by day under `a-<16 hex>`, a keyed hash of the client address. The key is random, held in memory only, replaced every UTC day and never written anywhere, so the id changes daily (and at a restart). The hash only groups such requests: the address itself is stored in the `addr` columns. The session of such a request is `d-<16 hex>`. **Origin** is the host of the `Origin` header, else of the `Referer`, else empty. **Address** is the limit key of the request, `req_client` in `server.py`: the peer address, or behind the local Caddy proxy the address in `X-Real-IP`; an IPv6 address is reduced to its /64 prefix; `shared` when the proxy sends no valid address.

### `POST /event`

Cheap (one SQLite insert, no GPU queue). Body at most 4 KB, `Content-Type: application/json`. Always answers 204 with no body when the request is valid, also when the usage store is off or the event is dropped. 400 for a bad `session` or an event not on the allowlist, 429 after 120 events per minute per client. Unknown fields are ignored. At most 500 events are stored per session.

```json
{"visitor": "<id>", "session": "<id>", "type": "load", "referrer": "news.example.org", "expert": false}
{"visitor": "<id>", "session": "<id>", "type": "ui", "name": "mode", "value": "map"}
```

`load` is sent once per page load; `referrer` is a host or empty (a URL is reduced to its host) and `expert` a boolean. `ui` names: `expert_on`, `expert_off`, `mode` (value `map` or `sample`), `matrices_open`, `spectrum_open`, `groups_on`, `preset` (value a short preset id of at most 32 characters of `A-Za-z0-9_.:-`). The value is dropped for other names and for values that do not fit.

### Stats routes

All need the header `X-Stats-Password` (the stats password) or `X-Own-Secret` (the owner key). Without a configured stats password and without a valid owner key they answer 503 `server` with a detail that says so; a wrong or missing password is 401 `auth`; after 10 failed attempts within 5 minutes a client gets 429 `rate_limit` (also for the right password) until the window passes. Responses carry `Cache-Control: no-store` and `X-Robots-Tag: noindex`.

**`GET /stats?days=N`** (N a positive integer, default 30; `days=all`, or N above 100,000, returns the whole history, with `since` the first day with data and `days` the number of days from it to today; the window is otherwise the last N UTC days including today):

```json
{
  "generated": "2026-11-25T15:00:00Z", "days": 30, "since": "2026-10-27",
  "totals": {"visitors": 12, "sessions": 20, "sessionsWithRun": 9, "requests": 80, "runs": 41, "completions": 2900, "tokens": 120000,
             "gpuSeconds": 410.5, "errors": 3, "cachedRequests": 8, "cachedCompletions": 410, "medianSessionS": 95, "meanSessionS": 210},
  "today": {"day": "2026-11-25", "visitors": 3, "sessions": 4, "runs": 7, "completions": 400, "errors": 0, "requests": 11},
  "daily": [{"day": "2026-11-24", "visitors": 5, "sessions": 6, "runs": 10, "completions": 700, "errors": 1}],
  "breakdowns": {
    "byOrigin": [{"key": "zueri-monstro.rufous-hamlet.ts.net", "requests": 60, "runs": 30, "sessions": 14}],
    "byEmbedder": [{"key": "qwen3-emb", "count": 25}],
    "byErrorCode": [{"key": "rate_limit", "count": 2}],
    "n": [{"key": 30, "count": 10}], "tprime": [{"key": 1.0, "count": 20}], "maxNewTokens": [{"key": 64, "count": 12}],
    "medianQueueWaitS": 0.0, "medianGenerateS": 3.1, "medianEmbedS": 0.4,
    "busiestHours": [{"hour": 0, "requests": 2}, "... 24 entries, hour 0 to 23 UTC ..."]
  },
  "sessions": [{"id": "<session id>", "visitor": "<first 8 characters of the visitor id>", "start": "2026-11-25T14:20:11Z", "durationS": 95,
                "origin": "zueri-monstro.rufous-hamlet.ts.net", "referrer": "news.example.org", "runs": 2, "requests": 3, "events": 5, "addr": "203.0.113.5"}],
  "sessionsTruncated": 0
}
```

- A session belongs to the day of its first activity. `visitors` counts distinct visitor ids, so an anonymous visitor counts once per day. `cachedRequests` counts successful `/sample` requests served entirely from a stream, `cachedCompletions` the draws read from streams (including the cached part of a request that also sampled); `completions` and `tokens` count only what was generated. `runs` are successful `/sample` requests, `requests` all requests to the three routes, `errors` requests whose outcome is not `ok`. `gpuSeconds` is the run time (after the queue wait) of every request. Session length is last activity minus first activity.
- `daily` has one entry per UTC day from the first day with data (at most N entries) to today, so it is long for `days=all`; days without activity are zeros. `today` has the same fields as a `daily` entry.
- `byOrigin` is sorted by `requests`; a request without an origin has the key `""`, the same value `origin` has in the sessions. `byEmbedder` counts successful runs. `byErrorCode` counts failed requests. `n`, `tprime` and `maxNewTokens` count successful runs by value, sorted by key. The medians are over `/sample` queue waits and over successful runs' `generate_s` and `embed_s`.
- `sessions` has the 200 most recent sessions by last activity; `sessionsTruncated` is how many more the window holds. `addr` is the client address of the session's first request or event that carried one, `""` for sessions stored before addresses were kept.

**`GET /stats/session?id=<session id>`**: `{session, events, requests}` in time order (404 for an unknown id, 400 for a malformed one).

```json
{"session": {"id": "...", "visitor": "a1b2c3d4", "start": "2026-11-25T14:20:11Z", "durationS": 95, "origin": "...", "referrer": "", "addr": "203.0.113.5"},
 "events": [{"time": "2026-11-25T14:20:11Z", "type": "load", "name": "", "value": ""}, {"time": "...", "type": "ui", "name": "mode", "value": "map"}],
 "requests": [{"time": "...", "route": "/sample", "model": "qwen3-14b", "embedder": "qwen3-emb", "N": 30, "Tprime": 1.0, "maxNewTokens": 64,
               "outcome": "ok", "queueWaitS": 0.0, "runS": 4.2, "generateS": 3.1, "embedS": 0.4, "tokens": 900, "completions": 30,
               "runId": "<32 hex>", "prompt": "...", "addr": "203.0.113.5", "cached": 0, "offset": null, "drawTexts": null}]}
```

`prompt` is the stored prompt of the request, also for failed requests when the body was read far enough to contain it (`null` for old rows; for those with a `runId` it is read from the run log, `null` when the log has no such line). `addr` is the address of the request. `drawTexts` is for a successful `/sample` with `offset`: the draws `[offset, offset + N)` of the stream, read from the draw cache as `[{text, logp, logq, nTokens, ended}]`, whether or not the request also generated some of them; `null` when the stream has been evicted, for requests without `offset` (their completions are in the run log, by `runId`), and for other routes. Requests that are not successful samples have `runId` null.

**`GET /stats/run?id=<run_id>`**: the full run log line (above), with all completions, `logp` and `logq`. 404 when the run log is off or the id is unknown.

## `GET /own_stats` (owner)

Header `X-Own-Secret`. Returns `{"resident": {"repo", "mode": "exact"|"projected", "V", "r", "dtype", "slots_used", "steps", "allocated_mib", "P_mib"} | null}`; `null` while the plugin state is unloaded (it loads at the first own request and unloads after 60 s without one).

## Own states (`embedder: "own"`)

For a completion, the vector is the mean over the generation steps of the model's final-norm hidden state h_t, including the step that samples the end-of-sequence token. Two modes, chosen per model by `own_emb/<key>/M.npy`:

- **Exact** (M.npy present, built by `search/build_own_M.py`): the plugin sums the raw logits of each sequence (V floats per sequence); the server recovers h̄ = [M Aᵀ l̄]₁..d with A = [W | 1], M = (AᵀA)⁻¹, subtracts `mu_full.npy` (the mean of h̄ over the model's demo pools), and normalizes. `G` comes from the full 5120-d vectors for Qwen3 14B. The plugin has 448 sequence slots in this mode, so one own-state pass takes at most 400 texts.
- **Projected** (no M.npy): the 256-d random projection of the earlier version.

A change of mode needs a restart of the service, because the plugin chooses its mode when it loads. The server answers 500 with a message that says so when it sees sums of the wrong width.
