# PoolDPP demo sampler

A FastAPI server that draws pools of completions from a local language model with exact log p and log q per string,
and returns unit-norm embeddings of each completion. The static demo page (`../index.html`) talks to it through a reverse
proxy under `/dpp_api/`. It runs on one 24 GB GPU: a resident Qwen3 14B AWQ engine and one small model in a swappable slot.

## What it serves

| Route | Use |
|---|---|
| `GET /health` | `{"status":"ok"}`; 503 when an engine fails `check_health()`; `starting` during start-up |
| `GET /status` | contract fields (`version`, `modelKey`, `embedders`, `limits`, `auth`), models and their state (`ready`, `loading`, `cold`, `unavailable`), `logsRuns`, `usage` |
| `POST /sample` | draw `n` completions from the annealed proposal q = p_{T'} and return log p, log q, embeddings |
| `POST /embed` | embed existing completions after the fact (`minilm`, `qwen3-emb`, `qwen3-emb-4b`, `mpnet`, or the LM's own states `own`) |
| `POST /embed_own` | owner only: teacher-force completions and return uncentered own states |
| `POST /describe` | name 2 to 8 groups of completions in a few words each, with the resident model |
| `GET /queue?id=<request_id>` | `{"state": "waiting"\|"running"\|"unknown", "ahead": int}` for a request that carried a `request_id` |
| `POST /event` | the page's load and UI events for the usage store (204; never touches the GPU queue) |
| `GET /stats?days=N`, `/stats/session?id=`, `/stats/run?id=` | usage dashboard data; need `X-Stats-Password` (or the owner key) |
| `GET /runlog[?day=YYYY-MM-DD]` | owner only: list the run log's days, or stream one day's file |
| `GET /own_stats` | owner only: mode and memory of the own-state plugin |

Requests are JSON (`Content-Type: application/json`). The exact shapes are in `API.md`, which follows `docs/BACKEND.md` (contract version 1).

`POST /sample` body: `prompt` (1 to 2000 characters), `n` (2 to 200, default 30), `t_prime` (0.2 to 2, default 1),
`max_new_tokens` (8 to 256, default 128), `seed` (integer, default random), `model` (a key of `/status`), `embedder`, optional `emb`,
`visitor`, `session`. A value outside these limits is refused with 400 `bad_request`; nothing is capped.
The completions of one request use the seeds `seed + i`: choose seeds at least `n` apart to append independent draws to a pool.
Response: `samples[{text, logp, logq, nTokens, ended}]`, `G`, `D`, `N`, `embedder`, `embedderId`, `Tprime`, `maxNewTokens`, `seed`, `timing`.
`G` is the N x N matrix of inner products of the unit embeddings (float64 products of the float32 vectors, symmetric, diagonal exactly 1, little-endian
float32 row-major, base64) for every embedder. The vectors themselves (`emb`, N x D float32 base64) are sent only when the request has `"emb": true`.

**Errors** are JSON `{"code", "detail"}` with the codes `bad_request`, `auth`, `rate_limit`, `busy`, `server` of the contract (and `starting` for a 503 while an engine restarts); `detail` is the readable message.

`POST /embed` body: `texts` (1 to 200 strings of at most 2000 characters), `embedder`, and for `own` also `model`, `prompt` and optionally
`ended` (one boolean per text). The response has `G` (and `emb` only with `"emb": true`). The own states are recomputed by teacher forcing the text, so they are close to, not equal to, the
vectors `/sample` returns from the generation itself (cosine about 0.9999 for Gemma 3 and Qwen3, about 0.99 for Gemma 2 9B, whose stored
text loses trailing whitespace tokens).

Options `embedders`, `own_raw` and `return_ids` of `/sample` need the owner key.

**Queue.** Generating work (`/sample`, `/embed` with `own`, `/describe`, `/embed_own`) is admitted through a FIFO queue with one running job
per engine (the resident 14B engine and the slot engine run concurrently). A slot model swap happens when its job reaches the head of the
slot queue. A client may hold one `/sample` request at a time, queued or running (a second one gets 429); `/describe` and `/embed` are not
counted against that cap. At most 16 jobs wait over both engines (429 with `Retry-After` beyond that). A waiting job whose client
disconnects leaves the queue (499), and one that waits longer than 10 minutes gets 503. Hourly budgets are checked when a job is queued
(429 if already exhausted) and charged when it starts running. Plain `/embed` embedders (MiniLM, Qwen3-Embedding, MPNet) bypass the queue.

`/sample`, `/embed`, `/embed_own` and `/describe` accept an optional `request_id` (1 to 64 characters of `A-Za-z0-9_-`; 400 otherwise, 409 while
another unfinished request uses it). `GET /queue?id=<request_id>` answers `waiting` (with `ahead`: the running job plus the waiting jobs in
front on the same engine), `running` (`ahead` 0) or `unknown` (`ahead` 0). It needs no key and returns nothing else; it is limited to
120 calls per minute per client. `/status` has `"queue": {"running", "waiting", "max": 16}` and, as before, `generating` and `waiting`.

`POST /describe` body: `prompt`, `groups` (2 to 8 lists of 1 to 4 strings; each text is cut to 240 characters), optional `request_id`
(at most 32 KB). One batched greedy generation on the resident model (temperature 0, thinking off, 12 tokens, stop at a newline) labels each group
from the prompt, that group's texts and one text of each other group. Response `{"names": [...], "model": "qwen3-14b", "ms": int}`, one
name per group in order, at most 40 characters, sentence case, quotes and trailing punctuation removed; `""` when nothing usable came back.
Identical requests are answered from a 256-entry in-memory cache without cost. The prompt is the built-in `DESCRIBE_TEMPLATE` of
`api_guard.py`, or the file `describe_prompt.txt` next to `server.py` when present (placeholders `{prompt}`, `{k}`, `{i}`, `{group}`, `{others}`).
`/status` advertises `"describe": true, "maxDescribeGroups": 8, "maxDescribeTexts": 4`.

## The LM's own states ("own" embedder)

For a completion, the vector is the mean over generation steps of the model's final-norm hidden state, including the step that samples the end-of-sequence
token. The plugin (`plugin/dpp_vllm_plugin/own.py`, `DPP_OWN_EMB=1`) works in one of two modes per model, chosen by the files in `own_emb/<key>/`:

- **Exact** (`M.npy` present). The plugin sums the raw logits of each sequence (V floats, 448 slots, 260 MiB for Qwen3). The sampler's logits are l = W h + c·1
  (W: the unembedding matrix, c: a per-step shift). With A = [W | 1] and M = (AᵀA)⁻¹, the first d entries of M Aᵀ l are h, and the map is linear, so
  the server recovers the mean state from the mean logits, h̄ = [M Aᵀ l̄]₁..d, subtracts `mu_full.npy` (the mean over the model's demo pools) and normalizes.
  `G` comes from the full d-dimensional vectors (d = 5120 for Qwen3 14B). Build: `DPP_ROOT=. python search/build_own_M.py <key>` (CPU, about 35 s),
  then `search/precompute_own.py` writes `mu_full.npy` and the pools' `<id>.G.bin`. A pass takes at most 400 texts.
- **Projected** (no `M.npy`). The 256-d random projection of `search/build_own_P.py`.

The plugin chooses its mode when it loads (at the first own request, and again after 60 s without one), so changing the files needs a restart of the service.
Measured on Qwen3 14B AWQ (`tests/check_own_exact.py`): the logits the hook sees are float16 with no NaN or inf, the least-squares residual of l on A is
2.1e-4 of |l| (float16 rounding of the logits), the recovered h differs from the hidden state of the same forward pass by 1.7e-4 per step (relative) and 5e-5 for the mean,
and G differs by at most 7.7e-6.
Models whose final logits are soft-capped (Gemma 2) are handled by inverting the cap in the plugin.

## Draw cache

`POST /sample` with an integer `offset` returns draws `[offset, offset + n)` of a stream of i.i.d. draws for (prompt, model, repo and weights revision, T′, max tokens), sampling only the shortfall. Details in `API.md`, "Draw cache". `drawcache.py` is the store (SQLite in `drawcache/`, directory 700; `DPP_DRAW_CACHE=0` off, `DPP_DRAW_CACHE_DIR`, `DPP_DRAW_CACHE_MAX_GB`, default 4), `server.py` (`stream_sample`, `extend_stream`) the route. Rules that keep the stream i.i.d.: only server-chosen seeds (ranges never reused within a stream), a batch is appended whole and in order at the current end or not at all, requests with their own `seed` never touch it, one writer per stream. Cached text embeddings are keyed by (text, embedder, revision); the own state of every draw is stored at generation time (uncentered, float32), because it cannot be recomputed exactly afterwards. A request served entirely from a stream skips the queue and the GPU budgets (120 per minute per client). The page sends `offset` only when `/status.drawCache` is true (`js/draws-lib.js`).

## Run log

Every successful `/sample` run is appended to `runlog/runs-YYYY-MM-DD.jsonl` (one UTC day per file; directory mode 700, files 600; under the server directory, never under the
static site). A line has the configuration (prompt, model key, repo and weights revision, embedder id, N, T′, max new tokens, seed), the completions with
exact `logp`, `logq`, `nTokens`, `ended`, the timing, the UTC time and a random `run_id`. It has no visitor field (no address, client id, user agent, `request_id`,
`visitor`, `session`) and no embeddings or `G`. One writer lock, one full line per write; a failed write is logged to `logs/server.log` and the request still succeeds.
`DPP_RUNLOG=0` turns it off (`/status.logsRuns` is then false), `DPP_RUNLOG_DIR` moves it. `GET /runlog` (owner key) lists days and streams a day. Copy the files over SSH otherwise.
`/embed` and `/describe` are not logged.

## Usage stats

A second store, `usage/usage.sqlite3` (SQLite, directory mode 700, file 600, `DPP_USAGE=0` to turn off, `DPP_USAGE_DIR` to move), holds sessions, UI events and one row per request to
`/sample`, `/embed`, `/describe` (time, session, origin host, route, settings, outcome, queue wait, run time, tokens, `run_id`). It also holds the client address (the limit key of `req_client`: the `X-Real-IP` behind Caddy, IPv6 as its /64) in sessions, events and requests, and the prompt text of every request (cut at 2000 characters). It holds no completions: a generated sample points to its run log line by `run_id`, and a sample answered from the draw cache points to its stream (an internal `stream_key`, plus `offset` and N), from which `/stats/session` reads the texts as long as the stream is cached. The run log stays free of visitor data. An older database file is migrated when the server opens it (columns added, old rows keep empty values). The page sends `visitor` (random, kept in its localStorage) and `session` (random per page load),
and `POST /event` for the page load and a few UI events. A request without ids is grouped per UTC day under a keyed hash of the client address; the key is random, kept in
memory and replaced every day, and the address is stored next to it. `GET /stats`, `/stats/session`, `/stats/run` serve the dashboard and need `X-Stats-Password`.

`/stats?days=all` returns the whole history. `totals.cachedRequests` and `cachedCompletions` count reads from the draw cache.

**Setting the stats password (owner).** The server never creates one. Choose a password and write it to `~/dpp-sampler/.stats_password` as the only line
(`umask 077; printf '%s\n' 'your password' > ~/dpp-sampler/.stats_password`; mode 600), or set `DPP_STATS_PASSWORD_FILE` to another path or `DPP_STATS_PASSWORD` in the unit.
The file is read on every request, so no restart is needed. The owner key (`X-Own-Secret`) also works for the stats routes. While neither is configured, the routes answer 503.
After 10 failed attempts within 5 minutes a client gets 429. The stats page runs on the server's own origin (no CORS involved); `X-Stats-Password` is not in the CORS allow list.

## Install and run

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # versions installed on the demo server; requirements-freeze.txt has the full list
pip install -e plugin/                     # registers the vLLM plugin that returns log q next to log p
DPP_LOGQ=1 VLLM_USE_V2_MODEL_RUNNER=0 python -m uvicorn server:app --host 127.0.0.1 --port 8040
```

The model weights come from the Hugging Face hub (`HF_HUB_OFFLINE=1` after the first download). `ninja` must be on `PATH`
(torch.compile and FlashInfer JIT).

Example systemd user unit (`~/.config/systemd/user/dpp-sampler.service`; `%h` is the home directory):

```
[Unit]
Description=PoolDPP demo sampler
After=network-online.target
StartLimitIntervalSec=900
StartLimitBurst=5

[Service]
WorkingDirectory=%h/dpp-sampler
Environment=DPP_LOGQ=1
Environment=VLLM_USE_V2_MODEL_RUNNER=0
Environment=DPP_RESIDENT_UTIL=0.50
Environment=DPP_SLOT_UTIL=0.36
Environment=DPP_OWN_EMB=1
Environment=HF_HUB_OFFLINE=1
Environment=TOKENIZERS_PARALLELISM=false
Environment=PATH=%h/dpp-sampler/.venv/bin:<directory with ninja>:/usr/local/bin:/usr/bin:/bin
ExecStart=%h/dpp-sampler/.venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8040 --timeout-graceful-shutdown 20
Restart=always
RestartSec=15
TimeoutStopSec=40
StandardOutput=append:%h/dpp-sampler/logs/server.log
StandardError=append:%h/dpp-sampler/logs/server.log

[Install]
WantedBy=default.target
```

## Cross-origin use

`server.py` adds `CORSMiddleware` for the origins `https://timvieira.github.io`, the server's own page and `http://localhost` with any port (methods GET and POST, headers `Authorization` and `Content-Type`; `Retry-After` is exposed). Error answers carry the headers too. To allow another origin, change `CORS_ORIGINS` or `CORS_ORIGIN_REGEX` in `server.py` and restart.

## Embedders

`minilm` (default), `qwen3-emb` (Qwen3-Embedding-0.6B, D = 1024), `qwen3-emb-4b` (Qwen3-Embedding-4B, last-token pooling, float16, batch 8, D = 2560), `mpnet`, and `own`. Preload it with `DPP_PRELOAD=qwen3-emb,qwen3-emb-4b` only together with `DPP_MODELS=qwen3-14b`, because the slot models need that memory. With the 14B resident and both Qwen embedders preloaded (used + free = 24125 MiB), idle free memory was 2,059 to 2,279 MiB. A 200 x 256 `/sample` with `qwen3-emb-4b` left at least 1,483 MiB free (generate 43.0 s, embed 8.6 s), and `/embed` of 200 texts of 1,990 characters left at least 919 MiB (14.7 s).

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `DPP_LOGQ` | required `1` | the plugin returns log q |
| `VLLM_USE_V2_MODEL_RUNNER` | required `0` | the plugin patches the V1 sampler |
| `DPP_RESIDENT_UTIL` / `DPP_SLOT_UTIL` | 0.50 / 0.36 | GPU fraction of the resident and the default slot engine |
| `DPP_OWN_EMB` | off | `1` enables own states (plugin, `own_emb/<key>/`, `/embed` with `own`) |
| `DPP_OWN_DIR` | `./own_emb` | where the plugin reads `P.safetensors`, `meta.json`, `mu.npy` per model |
| `DPP_RUNLOG` / `DPP_RUNLOG_DIR` | on / `./runlog` | `0` turns the run log off; the directory of the daily files |
| `DPP_USAGE` / `DPP_USAGE_DIR` | on / `./usage` | `0` turns the usage store off; the directory of `usage.sqlite3` |
| `DPP_STATS_PASSWORD_FILE` / `DPP_STATS_PASSWORD` | `./.stats_password` / unset | where the owner sets the stats password (file first, then the variable) |
| `DPP_PRELOAD` | `qwen3-emb` | embedders loaded at start-up (comma list, empty for none) |
| `DPP_MODELS` | all | comma list of model keys to serve. It must include the resident `qwen3-14b` and only known keys, or start-up fails. Other keys get 400 "Unknown model" and no slot engine loads. Unset serves every model. |
| `DPP_EMBEDDER` | `minilm` | embedder used when a request names none |
| `DPP_RUNS_PER_HOUR` / `DPP_EMBEDS_PER_HOUR` | 300 / 1200 | global ceilings |
| `DPP_LOGQ_CHECK` | off | `1` makes the plugin cross-check log q (slow, for tests) |

`model_util.json` (see `model_util.example.json`) maps a model key to its slot GPU fraction and overrides the table in `server.py`
at the next engine start.


## GPU budget

One 24 GB card (24,125 MiB usable) holds the resident 14B engine (util 0.50, about 12.3 GiB plus cached sampler buffers), one slot engine,
and the main process (MiniLM, Qwen3-Embedding preloaded, the P matrices). The slot fractions in `MODELS` leave at least 300 MiB free
when both engines run a 200 x 256 request with the Qwen3-Embedding embedder (measured warmed up; minimum free over each run):

| Slot model | util | extra engine arguments | KV cache | min free, qwen3-emb | min free, own embedder |
|---|---|---|---|---|---|
| Gemma 2 9B (AWQ int4) | 0.30 | | 1,902 tokens | 1144 to 1298 MiB | 698 to 1613 MiB |
| Gemma 3 4B (QAT int4) | 0.30 | `language_model_only` | 21,202 tokens | 702 MiB | |
| Qwen3 4B (bf16) | 0.35 | `max_num_seqs=32` | 5,024 tokens | 371 MiB | |
| Phi-3.5 mini (bf16) | 0.34 | `max_num_seqs=32` | 1,757 tokens | 776 MiB | |

Lower fractions do not start (Qwen3 4B needs more than 0.34 even with 32 sequences). The owner-only option `embedders` with three
embedders on a 4B slot model leaves less (about 120 MiB), so use it with a smaller `n` there.

## Owner key

A file `.own_secret` next to `server.py` (mode 600, random string; never commit it) enables the owner routes. A request with the header
`X-Own-Secret: <contents>` skips the per-client limits and may use the owner-only options. Without the file `/embed_own` answers 404.

## Limits

Defined at the top of `server.py`; the client key is the visitor address (IPv6 on its /64), taken from `X-Real-IP` when the peer is
the local reverse proxy. Without a real address all traffic shares one key, so the per-client limits act as global ones.

| Limit | Value |
|---|---|
| request body | 64 KB (`/sample`), 1.5 MiB (`/embed`), 4 MiB (owner) |
| `/sample` requests per client | 1 queued or running |
| `/describe` calls per client | 120 per hour (cache hits are free), 600 per hour global |
| generated tokens per client | 150,000 per hour (n x max_new_tokens, token bucket) |
| slot model swaps | 3 per client per hour, 30 s between any two swaps |
| `/embed` calls per client | 240 per hour |
| global | 300 runs, 1,000,000 tokens, 1,200 embeds per hour |
| queue | 16 waiting jobs over both engines, 600 s maximum wait (503) |
| request body | 32 KB (`/describe`) |

Violations answer 429 with `Retry-After`. `/docs`, `/redoc` and `/openapi.json` are disabled.

## Health and recovery

`/health` checks both engines. The watchdog checks them every 30 s: two failures of the resident engine, or an `EngineDeadError`
in a request on it, exit the process with code 3 so systemd restarts it. A dead slot engine is reset and rebuilt on the next request;
three slot deaths within 10 minutes also exit.

## Tests

```
python tests/test_api_guard.py     # validation, limits, rotation (no GPU)
python tests/test_queue.py         # FIFO queue, request ids, /describe validation and labels (no GPU)
python tests/test_exit_paths.py    # exit and recovery paths with fake engines (no GPU)
python tests/http_checks.py basic  # against a running server: malformed input, docs, owner options
python tests/http_checks.py limits
python tests/http_checks.py swaps
python tests/check_plugin_exact.py 1.5 <hf repo>   # plugin log q against vLLM's own, stop the service first (needs a free GPU)
```
`http_checks.py` reads the owner key from `DPP_OWN_SECRET_FILE` (default `.own_secret`).

## Rebuilding the page data

All scripts take the layout root from `DPP_ROOT` (default: the current directory) and the page data directory from `DPP_DATA`
(default `dpp-lm-demo/data`).

1. `search/precompute_models.py` samples the 14 pools of each served model through the running server and runs `search/build_model_data.py`,
   which writes the pool files and refits the shared display bases on all models.
2. `search/embed_pools.py [--only <model>]` writes the Qwen3-Embedding vectors and their bases.
3. `search/build_own_P.py <model>`, then `search/precompute_own.py` (teacher forcing) or `search/regen_pools_own.py` plus
   `search/own_from_pools.py` (own states from the generation itself), then `search/own_sanity.py`.

## Tests

No GPU: `tests/test_api_guard.py`, `test_queue.py`, `test_exit_paths.py`, `test_usage.py` (run log and usage store), `test_drawcache.py` (the store), `test_draws.py` (`/sample` with `offset`, fake generation), and `test_routes.py` (imports `server.py`, starts no engine, calls the cheap routes
through starlette's TestClient; needs vllm and torch to import). Run each as `.venv/bin/python tests/<file>`; they print one `PASS` or `FAIL` line per check.
Against the running service: `tests/http_checks.py basic|limits`, and `node scripts/conformance.mjs <base url>` from the page's repository. GPU, with the service stopped:
`tests/check_plugin_exact.py` (log q) and `tests/check_own_exact.py` (what the sampler hook sees, and the recovery of the hidden states).
