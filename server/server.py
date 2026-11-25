"""PoolDPP demo sampler on this box's GPU, served same-origin at /dpp_api/* by Caddy.

Same contract as Tim's Modal backend (dpp-lm-demo/modal/app.py): POST /sample draws N
completions i.i.d. from the locally annealed proposal q = p°_{T'} and returns, for each,
log p(y) and log q(y) (end-of-string included), recorded token by token from the logits
that drew the sample (dpp_vllm_plugin), plus unit-norm embeddings. The page does the DPP.

Twelve models: the resident engine (Qwen3 14B AWQ) is always loaded; one small bf16 model at
a time sits in the slot engine and is swapped on demand (the resident engine keeps serving).

Run: DPP_LOGQ=1 VLLM_USE_V2_MODEL_RUNNER=0 .venv/bin/python -m uvicorn server:app --port 8040
(dpp-sampler.service sets these). Configuration is read from the environment below.
"""
import asyncio
import base64
import collections
import contextlib
import functools
import datetime
import gc
import hashlib
import hmac
import json
import logging
import math
import os
import random
import re
import threading
import time
import uuid

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer, GenerationConfig
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.exceptions import EngineDeadError
from fastapi.responses import FileResponse, JSONResponse, Response

import api_guard as G
import usage as U
import drawcache as D

RESIDENT_KEY = "qwen3-14b"
RESIDENT_UTIL = float(os.environ.get("DPP_RESIDENT_UTIL", "0.50"))  # fraction of the card for the 14B engine
SLOT_UTIL = float(os.environ.get("DPP_SLOT_UTIL", "0.36"))          # fraction of the card for the slot engine
NOTHINK = {"enable_thinking": False}
# key: repo, display name, params (B), chat-template kwargs, slot. Keys match js/lm.js in the page.
MODELS = {
    "qwen3-14b": dict(repo="Qwen/Qwen3-14B-AWQ", name="Qwen3 14B", params=14.8, chat=NOTHINK, slot="resident"),
    "qwen2.5-0.5b": dict(repo="Qwen/Qwen2.5-0.5B-Instruct", name="Qwen2.5 0.5B", params=0.49),
    "qwen2.5-coder-0.5b": dict(repo="Qwen/Qwen2.5-Coder-0.5B-Instruct", name="Qwen2.5 Coder 0.5B", params=0.49),
    "qwen3-0.6b": dict(repo="Qwen/Qwen3-0.6B", name="Qwen3 0.6B", params=0.6, chat=NOTHINK),
    "gemma-3-1b": dict(repo="google/gemma-3-1b-it", name="Gemma 3 1B", params=1.0),
    "qwen2.5-1.5b": dict(repo="Qwen/Qwen2.5-1.5B-Instruct", name="Qwen2.5 1.5B", params=1.5),
    "qwen2.5-coder-1.5b": dict(repo="Qwen/Qwen2.5-Coder-1.5B-Instruct", name="Qwen2.5 Coder 1.5B", params=1.5),
    "qwen3-1.7b": dict(repo="Qwen/Qwen3-1.7B", name="Qwen3 1.7B", params=1.7, chat=NOTHINK),
    "smollm2-1.7b": dict(repo="HuggingFaceTB/SmolLM2-1.7B-Instruct", name="SmolLM2 1.7B", params=1.7),
    "qwen2.5-coder-3b": dict(repo="Qwen/Qwen2.5-Coder-3B-Instruct", name="Qwen2.5 Coder 3B", params=3.1),
    # Tight slot models: fewer concurrent sequences means smaller sampler buffers, so the 14B and the slot can both run n = 200 requests.
    "phi-3.5-mini": dict(repo="microsoft/Phi-3.5-mini-instruct", name="Phi-3.5 mini", params=3.8, util=0.34, engine=dict(max_num_seqs=32)),
    "qwen3-4b": dict(repo="Qwen/Qwen3-4B", name="Qwen3 4B", params=4.0, chat=NOTHINK, util=0.35, engine=dict(max_num_seqs=32)),
    # text only: language_model_only skips the vision tower (Gemma 3 4B has one)
    # QAT int4 checkpoint of google/gemma-3-4b-it (bf16 needs util 0.375 and leaves no GPU headroom); util 0.30 is 7.1 GiB.
    # Gemma 2: generation_config lists only <eos> (1), but turns end with <end_of_turn> (107), so 107 is an extra stop id.
    "gemma-2-9b": dict(repo="hugging-quants/gemma-2-9b-it-AWQ-INT4", name="Gemma 2 9B", params=9.2, util=0.30, extra_eos=[107]),
    "gemma-3-4b": dict(repo="gaunernst/gemma-3-4b-it-int4-awq", name="Gemma 3 4B", params=4.3, util=0.30,
                       engine=dict(language_model_only=True)),
}
MODELS["gemma-3-1b"]["eos_id"] = 106   # Gemma ends turns with <end_of_turn>; generation_config lists <eos> (1) first
MODELS["gemma-3-4b"]["eos_id"] = 106
MODELS["gemma-2-9b"]["eos_id"] = 107
MODELS["gemma-2-9b"]["ended_tail"] = " \n"   # /embed: stored Gemma 2 text lost its trailing whitespace tokens (closest single tail, cosine 0.9965 vs 0.992 without)
for _m in MODELS.values():
    _m.setdefault("chat", {})
    _m.setdefault("slot", "small")
# DPP_MODELS=key1,key2 serves only these models (must include the resident key; unset = all). Others get a 400 and no slot engine loads.
MODELS = G.filter_models(MODELS, os.environ.get("DPP_MODELS"), RESIDENT_KEY)
DEFAULT_MODEL = RESIDENT_KEY
EMBEDDERS = {
    "minilm": {"repo": "sentence-transformers/all-MiniLM-L6-v2", "pool": "mean", "dtype": torch.float32, "batch": 32},
    "qwen3-emb": {"repo": "Qwen/Qwen3-Embedding-0.6B", "pool": "last", "dtype": torch.float16, "batch": 8},
    "qwen3-emb-4b": {"repo": "Qwen/Qwen3-Embedding-4B", "pool": "last", "dtype": torch.float16, "batch": 8},
    "mpnet": {"repo": "sentence-transformers/all-mpnet-base-v2", "pool": "mean", "dtype": torch.float16, "batch": 8},
}
DEFAULT_EMBEDDER = os.environ.get("DPP_EMBEDDER", "minilm")
# "own": the LM's own states, computed inside the vLLM sampler (dpp_vllm_plugin.own) and returned with the pool.
OWN_KEY = "own"
OWN_ON = os.environ.get("DPP_OWN_EMB") == "1"
OWN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "own_emb")
OWN_SECRET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".own_secret")
OWN_EXACT_MAX_SEQS = 400  # plugin CAP_EXACT is 448 slots; leaves room for slots of a just-cancelled request
OWN_LMAX = 258  # longest teacher-forced token list (plugin table has 260 entries)
_OWN_MU = {}


def own_exact(key):
    """Exact mode: own_emb/<key>/M.npy exists (search/build_own_M.py). The plugin then sums logits and the server recovers
    the full mean hidden state; there is no random projection."""
    return OWN_ON and os.path.exists(os.path.join(OWN_DIR, key, "M.npy"))


def own_has_P(key):
    return OWN_ON and (own_exact(key) or os.path.exists(os.path.join(OWN_DIR, key, "P.safetensors")))


_OWN_REC = {}


def own_recover(key, lbar):
    """Mean hidden states (n x d, float64) from mean logits (n x V): h = [M Aᵀ l]_{1..d}, A = [W | 1], M = (AᵀA)⁻¹.
    Exact up to rounding (the per-step shift c in l = W h + c·1 is absorbed by the 1 column). W is lm_head (V x d), read
    once from the model snapshot as in build_own_P.find_W and kept as float32; AᵀA is badly conditioned, so the products
    are done in float64, a chunk of W at a time."""
    if key not in _OWN_REC:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "search"))
        from build_own_P import find_W
        g, n, _ = find_W(MODELS[key]["repo"])
        w = g.get_tensor(n)
        W = w.float().numpy() if hasattr(w, "float") else np.asarray(w, dtype=np.float32)
        _OWN_REC[key] = (np.ascontiguousarray(W, dtype=np.float32), np.load(os.path.join(OWN_DIR, key, "M.npy")))
    W, M = _OWN_REC[key]
    V, d = W.shape
    lbar = np.asarray(lbar, dtype=np.float64)
    if lbar.shape[1] != V:   # the plugin of this engine still holds P (projected sums): it switches to exact mode only when it reloads
        raise RuntimeError(f"own states: the plugin returned {lbar.shape[1]}-d sums, expected {V}-d logit sums (M.npy exists): restart the service")
    lbar = lbar[:, :V]
    ya = np.zeros((lbar.shape[0], d + 1))
    for a in range(0, V, 16384):
        ya[:, :d] += lbar[:, a:a + 16384] @ W[a:a + 16384].astype(np.float64)
    ya[:, d] = lbar.sum(1)
    return ya @ M[:, :d]


def own_mean(sums, counts, key):
    """Per-sequence mean of what the plugin accumulated: projected states (projected mode) or full hidden states (exact)."""
    n = len(counts)
    mean = np.frombuffer(sums, dtype="<f4").reshape(n, -1) / np.maximum(np.asarray(counts, np.float32), 1)[:, None]
    if own_exact(key):
        return own_recover(key, mean)
    return mean


def own_mu(key):
    """Per-model mean of the uncentered vectors (own_emb/<key>/mu.npy; mu_full.npy, d long, in exact mode), or None."""
    p = os.path.join(OWN_DIR, key, "mu_full.npy" if own_exact(key) else "mu.npy")
    try:
        mt = os.path.getmtime(p)
    except OSError:
        _OWN_MU.pop(key, None)
        return None
    if key not in _OWN_MU or _OWN_MU[key][0] != mt:   # reload when the file changes
        _OWN_MU[key] = (mt, np.load(p).astype(np.float64))
    return _OWN_MU[key][1]


def own_available(key):
    return own_has_P(key) and own_mu(key) is not None


def own_unit(sums, counts, key):
    """Centered, unit-normalized own vectors from the plugin's per-sequence sums and step counts (float64)."""
    v = own_mean(sums, counts, key) - own_mu(key)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def model_revision(key):
    """Commit hash of the local Hugging Face snapshot of the model's weights ('' when unknown)."""
    if key not in _REVISION:
        try:
            _REVISION[key] = os.path.basename(snapshot_download(MODELS[key]["repo"], local_files_only=True).rstrip("/"))
        except Exception:
            _REVISION[key] = ""
    return _REVISION[key]


async def own_take(engine, items):
    res = await engine.collective_rpc("dpp_own_take", args=(items,))
    if not res or res[0] is None:
        raise RuntimeError("the sampler returned no own states; is P loaded (DPP_OWN_EMB=1)?")
    return res[0]


def own_vectors(sums, counts, key, raw=False):
    """(unit own vectors in float64, uncentered mean states in float32 or None); blocking, run it in a thread."""
    mean = own_mean(sums, counts, key)
    v = mean - own_mu(key)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12), (mean.astype("<f4") if raw else None)


def eos_first(key):
    """First EOS id of generation_config.json (the end-of-turn id for Qwen and Gemma), as in research/own_emb/extract.py."""
    repo = MODELS[key]["repo"]
    e = None
    try:
        e = GenerationConfig.from_pretrained(repo).eos_token_id
    except Exception:
        pass
    if e is None:
        e = model_tok(key)[0].eos_token_id
    return int(e[0] if isinstance(e, list) else e)
# ---- limits (documented in CLAUDE.md, "Hardening")
MAX_N, MAX_TOKENS, MAX_PROMPT_CHARS = 200, 256, 2000
MAX_TEXT_CHARS = 2000                    # /embed: longest text
PRELOAD = [k for k in os.environ.get("DPP_PRELOAD", "qwen3-emb").split(",") if k]  # embedders loaded at startup
MAX_QUEUE_WAITING = 16  # jobs waiting over both engines (one job runs per engine); more get a 429
QUEUE_MAX_WAIT = 600    # seconds a job waits in the queue before it gets a 503 (the proxy chain has no response timeout)
QUEUE_POLLS_PER_MINUTE = 120             # per client, GET /queue
SWAP_WAIT_TIMEOUT = 300  # seconds a pool waits for a model swap
MAX_BODY_SAMPLE = 64 * 1024              # bytes, POST /sample
MAX_BODY_EMBED = 1536 * 1024             # bytes, POST /embed: 200 texts x 2000 characters at up to 3.9 bytes per character
MAX_BODY_OWNER = 4 * 1024 * 1024         # bytes, owner-only routes and requests carrying the owner key
MAX_BODY_DESCRIBE = 32 * 1024            # bytes, POST /describe
DESCRIBE_TOKENS = 12                     # generated tokens per group name
DESCRIBE_CACHE_SIZE = 256
DESCRIBE_TEMPLATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "describe_prompt.txt")  # optional override, read per request
CLIENT_DESCRIBES_PER_HOUR = 120          # per client, POST /describe calls that generate (cache hits are free)
DESCRIBES_PER_HOUR = 600                 # all clients
MAX_EMBEDDERS = 3                        # distinct entries of `embedders`
CLIENT_TOKENS_PER_HOUR = 150_000         # per client, n x max_new_tokens summed (token bucket, refilled evenly)
CLIENT_SWAPS_PER_HOUR = 3                # per client, slot model swaps it triggers
SWAP_MIN_INTERVAL = 30                   # seconds between two slot swaps, all clients
CLIENT_EMBEDS_PER_HOUR = 240             # per client, POST /embed calls
EMBEDS_PER_HOUR = int(os.environ.get("DPP_EMBEDS_PER_HOUR", "1200"))   # all clients
GLOBAL_RUNS_PER_HOUR = int(os.environ.get("DPP_RUNS_PER_HOUR", "300"))  # all clients, /sample runs
GLOBAL_TOKENS_PER_HOUR = 1_000_000       # all clients, n x max_new_tokens (about 30 percent of one busy engine-hour)
EMBED_OWN_TOKEN_COST = 100               # tokens charged per text of an own-state /embed
WATCHDOG_INTERVAL = 30                   # seconds between engine health checks
WATCHDOG_FAILS = 2                       # consecutive failed checks of the resident engine before the process exits
SLOT_DEATHS_MAX = 3                      # slot engine deaths within SLOT_DEATH_WINDOW seconds before the process exits
SLOT_DEATH_WINDOW = 600
LOG_MAX_BYTES = 20 * 1024 * 1024         # logs/server.log is rotated (copy-truncate, 4 old files) above this size
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
GIB = 2**30
CONTRACT_VERSION = 1                     # docs/BACKEND.md
CORS_ORIGINS = ["https://timvieira.github.io", "https://dpp-lm.github.io", "https://zueri-monstro.rufous-hamlet.ts.net"]   # Tim's copy, the project site and this server's own page
CORS_ORIGIN_REGEX = r"^http://localhost(:\d+)?$"   # development pages

log = logging.getLogger("dpp-sampler")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# Run log (DPP_RUNLOG=0 turns it off, DPP_RUNLOG_DIR moves it): one JSON line per successful /sample, see README.md "Run log".
RUNLOG = G.RunLog.from_env(os.environ, os.path.join(os.path.dirname(os.path.abspath(__file__)), "runlog"), on_error=log.error)
_REVISION = {}
# Usage store (sessions, events, one row per request; no text): DPP_USAGE=0 turns it off, DPP_USAGE_DIR moves it. See README.md "Usage stats".
USAGE = U.UsageStore.from_env(os.environ, os.path.join(os.path.dirname(os.path.abspath(__file__)), "usage"), on_error=log.error)
SALT = U.DailySalt()   # random per UTC day, in memory only
STATS_PASSWORD_FILE = os.environ.get("DPP_STATS_PASSWORD_FILE") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".stats_password")
# Draw cache (DPP_DRAW_CACHE=0 turns it off, DPP_DRAW_CACHE_DIR moves it, DPP_DRAW_CACHE_MAX_GB caps it): streams of i.i.d. draws, see README.md.
DRAWS = D.DrawCache.from_env(os.environ, os.path.join(os.path.dirname(os.path.abspath(__file__)), "drawcache"), on_error=log.error)
CACHED_READS_PER_MINUTE = 120   # per client, /sample requests served entirely from a stream
_KEYLOCKS = {}
MAX_BODY_EVENT = 4096
EVENTS_PER_MINUTE = 120      # per client, POST /event
STATS_FAILS = 10             # failed /stats password attempts per client in STATS_FAIL_WINDOW seconds before 429
STATS_FAIL_WINDOW = 300
S = {}  # resident engine, embedders, queue counters
SLOT = {"key": None, "engine": None, "inflight": 0, "loading": None, "pending": None, "cond": None}
TOKS = {}  # key -> (tokenizer, eos ids)


class Embedder:
    """Unit-norm sentence embeddings φ(y): mean pooling (MiniLM) or last-token pooling (Qwen3).

    Texts are embedded in order of increasing length in small batches, so peak activation memory is bounded
    (the GPU is shared with two vLLM engines). A batch that runs out of memory is retried with half the size."""

    def __init__(self, repo, pool, dtype, batch=8):
        self.repo, self.pool, self.batch = repo, pool, batch
        self.tok = AutoTokenizer.from_pretrained(repo, padding_side="left" if pool == "last" else "right")
        self.model = AutoModel.from_pretrained(repo, dtype=dtype).to("cuda").eval()

    def _embed_batch(self, texts):
        enc = self.tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        h = self.model(**enc).last_hidden_state
        if self.pool == "mean":
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            v = h[:, -1]  # left-padded, so the last position is the last real token
        r = torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy()
        del enc, h, v
        return r

    @torch.inference_mode()
    def __call__(self, texts, batch_size=None):
        texts = [t if t.strip() else "(empty)" for t in texts]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        bs = batch_size or self.batch
        i = 0
        while i < len(order):
            idx = order[i:i + bs]
            try:
                out[idx] = self._embed_batch([texts[j] for j in idx])
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                log.warning("embedder %s: out of memory, retrying with batch size %d", self.repo, bs)
                continue
            i += len(idx)
            if self.pool == "last":
                torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        return out


def get_embedder(k):
    """Load an embedder on first use (MiniLM is loaded at startup)."""
    if k not in S["emb"]:
        t0 = time.time()
        S["emb"][k] = Embedder(**EMBEDDERS[k])
        log.info("loaded embedder %s in %.1fs", k, time.time() - t0)
    return S["emb"][k]


def is_cached(key):
    """The model's files are in the local HF cache (the service runs with HF_HUB_OFFLINE=1)."""
    m = MODELS[key]
    if "cached" not in m:
        try:
            snapshot_download(m["repo"], local_files_only=True)
            m["cached"] = True
        except Exception:
            return False
    return True


def model_tok(key):
    """Tokenizer and EOS ids: generation_config.json eos_token_id (list or int) when present, else the tokenizer's."""
    if key not in TOKS:
        repo = MODELS[key]["repo"]
        tok = AutoTokenizer.from_pretrained(repo)
        eos = None
        try:
            eos = GenerationConfig.from_pretrained(repo).eos_token_id
        except Exception:
            pass
        if eos is None:
            eos = tok.eos_token_id
        TOKS[key] = (tok, set(eos if isinstance(eos, list) else [eos]) | set(MODELS[key].get("extra_eos", [])))
    return TOKS[key]


def model_tuning(key, util):
    """(GPU fraction, extra engine arguments) of the slot engine of `key`: the table's `util` and `engine`, else the default util;
    model_util.json overrides it at the next engine start, either `{"key": 0.33}` or `{"key": {"util": 0.33, "max_num_seqs": 32}}`."""
    try:
        o = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_util.json")))
    except (OSError, ValueError):
        o = {}
    v, eng = o.get(key), dict(MODELS[key].get("engine", {}))
    u = MODELS[key].get("util", util)
    if isinstance(v, dict):
        u = v.get("util", u)
        eng.update({k: x for k, x in v.items() if k != "util"})
    elif v is not None:
        u = v
    return float(u), eng


def model_util(key, util):
    return model_tuning(key, util)[0]


def engine_args(key, util):
    u, eng = model_tuning(key, util)
    kw = dict(model=MODELS[key]["repo"], gpu_memory_utilization=u, max_model_len=1536, max_num_seqs=128,
              enable_prefix_caching=True, logprobs_mode="raw_logprobs", generation_config="vllm", seed=0,
              **({} if key == RESIDENT_KEY else {"enforce_eager": True, "dtype": "bfloat16"}))
    kw.update(eng)   # per-model overrides (language_model_only, max_num_seqs, ...)
    return AsyncEngineArgs(**kw)


def free_gib():
    return torch.cuda.mem_get_info()[0] / GIB


async def swap_slot(key):
    """Shut the slot engine down, wait until its GPU memory is released, start `key`. Called with
    SLOT["loading"] == key and no slot request in flight. Always leaves the slot state consistent."""
    old, t0 = SLOT["key"], time.time()
    try:
        if SLOT["engine"] is not None:
            eng = SLOT["engine"]
            SLOT["engine"] = SLOT["key"] = None
            await asyncio.to_thread(eng.shutdown)
            del eng
            gc.collect()
            target = SLOT["free_before"] - (torch.cuda.memory_reserved() - SLOT["reserved_before"]) / GIB - 1.0  # 1.0: the other engine may cache sampler buffers or load P (own states) meanwhile
            for _ in range(60):
                if free_gib() >= target:
                    break
                await asyncio.sleep(0.5)
            else:
                raise RuntimeError(f"slot GPU memory not released: free {free_gib():.2f} GiB, expected {target:.2f}")
            log.info("slot %s shut down, GPU free %.2f GiB", old, free_gib())
        SLOT["free_before"], SLOT["reserved_before"] = free_gib(), torch.cuda.memory_reserved()
        model_tok(key)
        eng = await asyncio.to_thread(AsyncLLM.from_engine_args, engine_args(key, SLOT_UTIL))
        SLOT["engine"], SLOT["key"] = eng, key
        log.info("slot %s ready in %.1fs (swap from %s), GPU free %.2f GiB", key, time.time() - t0, old, free_gib())
    except BaseException:
        log.exception("slot swap to %s failed", key)
        raise
    finally:
        SLOT["loading"] = None
        async with SLOT["cond"]:
            SLOT["cond"].notify_all()


async def slot_acquire(key):
    """Return the slot engine holding `key`, counted in flight. A request for another model stops
    new admissions to the current one, waits for the in-flight ones, then swaps (serialized)."""
    cond, deadline = SLOT["cond"], time.time() + SWAP_WAIT_TIMEOUT
    while True:
        async with cond:
            if SLOT["loading"] is None and SLOT["key"] == key and (SLOT["pending"] in (None, key) or SLOT["inflight"] == 0):
                SLOT["pending"] = None
                SLOT["inflight"] += 1
                return SLOT["engine"]
            if SLOT["loading"] is None and SLOT["inflight"] == 0:
                SLOT["loading"], SLOT["pending"] = key, None
                swap = asyncio.create_task(swap_slot(key))
            else:
                if SLOT["key"] != key and SLOT["loading"] is None:
                    SLOT["pending"] = key  # stop admitting more of the current model
                swap = None
                left = deadline - time.time()
                if left <= 0:
                    raise HTTPException(429, "The model is busy being swapped. Please try again in a minute.")
                try:
                    await asyncio.wait_for(cond.wait(), min(left, 5))
                except asyncio.TimeoutError:
                    pass
        if swap is not None:
            await asyncio.shield(swap)  # raises if the swap failed


async def slot_release():
    async with SLOT["cond"]:
        SLOT["inflight"] -= 1
        SLOT["cond"].notify_all()


@contextlib.asynccontextmanager
async def lifespan(app):
    assert os.environ.get("DPP_LOGQ") == "1", "set DPP_LOGQ=1 so the vLLM plugin records log q"
    SLOT["cond"] = asyncio.Condition()
    model_tok(RESIDENT_KEY)
    S["engine"] = AsyncLLM.from_engine_args(engine_args(RESIDENT_KEY, RESIDENT_UTIL))
    S["emb"] = {"minilm": Embedder(**EMBEDDERS["minilm"])}
    S["emb_lock"] = asyncio.Lock()
    S["queue"] = G.JobQueue(("resident", "slot"), MAX_QUEUE_WAITING, QUEUE_MAX_WAIT)   # one running job per engine, FIFO
    S["qpoll"] = G.RateWindow(QUEUE_POLLS_PER_MINUTE, 60)
    S["evrate"] = G.RateWindow(EVENTS_PER_MINUTE, 60)
    S["cacherate"] = G.RateWindow(CACHED_READS_PER_MINUTE, 60)
    S["authfail"] = G.RateWindow(STATS_FAILS, STATS_FAIL_WINDOW)
    S["dcache"] = G.LRU(DESCRIBE_CACHE_SIZE)
    S["limits"] = G.Limits(CLIENT_TOKENS_PER_HOUR, CLIENT_SWAPS_PER_HOUR, GLOBAL_RUNS_PER_HOUR, GLOBAL_TOKENS_PER_HOUR,
                           CLIENT_EMBEDS_PER_HOUR, EMBEDS_PER_HOUR, SWAP_MIN_INTERVAL,
                           describes_per_hour_client=CLIENT_DESCRIBES_PER_HOUR, describes_per_hour_global=DESCRIBES_PER_HOUR)
    S["slot_deaths"] = collections.deque()
    for k in PRELOAD:
        await asyncio.to_thread(get_embedder, k)
    os.makedirs(LOG_DIR, exist_ok=True)
    await sample_pool(RESIDENT_KEY, S["engine"], "Name a color in one word.", 4, 1.0, 8, 0, DEFAULT_EMBEDDER)  # warm up
    S["watchdog"] = asyncio.create_task(watchdog())
    log.info("ready: resident %s (util %.2f, free %.2f GiB), slot util %.2f, available %s", RESIDENT_KEY,
             RESIDENT_UTIL, free_gib(), SLOT_UTIL, [k for k in MODELS if is_cached(k)])
    yield
    S["watchdog"].cancel()
    for e in (S["engine"], SLOT["engine"]):
        if e is not None:
            e.shutdown()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
# Browser pages on GitHub Pages call this API cross-origin. Same-origin requests carry no Origin header and are untouched;
# preflights (OPTIONS with Access-Control-Request-Method) are answered here, before routing, limits and the 415 check.
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_origin_regex=CORS_ORIGIN_REGEX, allow_methods=["GET", "POST"],
                   allow_headers=["Authorization", "Content-Type"], expose_headers=["Retry-After"])


def die(reason):
    """Exit with a nonzero code so systemd (Restart=always) starts a fresh process; the engines die with it."""
    log.critical("exiting: %s", reason)
    logging.shutdown()
    os._exit(3)


def engines_to_check():
    return [("resident", S.get("engine")), ("slot", SLOT["engine"] if SLOT["loading"] is None else None)]


def slot_engine_failed(engine, why):
    """A slot engine died (it is rebuilt on the next request). Too many deaths in a short time: restart the process."""
    if SLOT["engine"] is engine and engine is not None:
        log.error("slot engine %s died: %s", SLOT["key"], why)
        SLOT["engine"] = SLOT["key"] = None
        threading.Thread(target=lambda: _quiet(engine.shutdown), daemon=True).start()
        now = time.time()
        d = S["slot_deaths"]
        d.append(now)
        while d and now - d[0] > SLOT_DEATH_WINDOW:
            d.popleft()
        if len(d) >= SLOT_DEATHS_MAX:
            die(f"slot engine died {len(d)} times in {SLOT_DEATH_WINDOW} s")


def _quiet(f):
    try:
        f()
    except BaseException:
        pass


def engine_error(engine, err):
    """An exception from a request on `engine`: restart the process when the resident engine is dead, reset the slot otherwise."""
    if isinstance(err, EngineDeadError):
        if engine is S.get("engine"):
            die("EngineDeadError on the resident engine")
        slot_engine_failed(engine, "EngineDeadError")
        return True
    return False


async def watchdog():
    fails = 0
    last_rot = 0.0
    while True:
        await asyncio.sleep(WATCHDOG_INTERVAL)
        try:
            bad = await G.first_unhealthy(engines_to_check())
            if bad is None:
                fails = 0
            elif bad[0] == "slot":
                slot_engine_failed(SLOT["engine"], bad[1])
            else:
                fails += 1
                log.error("resident engine unhealthy (%d/%d): %s", fails, WATCHDOG_FAILS, bad[1])
                if fails >= WATCHDOG_FAILS:
                    die("resident engine failed its health check")
            if time.time() - last_rot > 3600:
                last_rot = time.time()
                if G.rotate_log(os.path.join(LOG_DIR, "server.log"), LOG_MAX_BYTES):
                    log.info("rotated logs/server.log")
        except Exception:
            log.exception("watchdog step failed")


def req_client(req):
    return G.client_key(req.client.host if req.client else "", req.headers.get("x-real-ip"))


def engine_name(key):
    return "resident" if MODELS[key]["slot"] == "resident" else "slot"


async def wait_swap_gap(req, key):
    """A job that needs a slot swap waits out the global gap since the last swap (at most SWAP_MIN_INTERVAL seconds)."""
    w = S["limits"].swap_wait() if needs_swap(key) else 0
    if w > 0:
        try:
            await or_disconnect(req, asyncio.sleep(w))
        except _Gone:
            raise HTTPException(499, "Client closed the request.")


def needs_swap(key):
    return MODELS[key]["slot"] != "resident" and SLOT["key"] != key and SLOT["loading"] != key


class _Gone(Exception):
    """The client disconnected while waiting; `.late` is the result the wait produced anyway."""
    def __init__(self, late=None):
        self.late = late


async def or_disconnect(req, coro, poll=1.0):
    """Await `coro`, cancel it and raise _Gone when the client disconnects meanwhile."""
    task = asyncio.ensure_future(coro)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=poll)
            if done:
                return task.result()
            if await req.is_disconnected():
                task.cancel()
                late = None
                try:
                    late = await task
                except BaseException:
                    pass
                raise _Gone(late)
    except asyncio.CancelledError:
        task.cancel()
        raise


async def acquire_slot(req, key):
    """slot_acquire that gives up when the client leaves; returns the engine."""
    try:
        return await or_disconnect(req, slot_acquire(key))
    except _Gone as g:
        if g.late is not None:
            await slot_release()
        raise HTTPException(499, "Client closed the request.")


async def sample_pool(key, engine, prompt, n, t_prime, max_new_tokens, seed, embedder, extra=(), own_raw=False, return_ids=False, with_emb=False):
    """Exact ancestral sampling from p°_{T'} with per-string log p and log q."""
    cfg = MODELS[key]
    tok, eos = model_tok(key)
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, **cfg["chat"])
    ids = tok(text, add_special_tokens=False).input_ids
    # Everything that could change the proposal is set explicitly: no top-k, top-p, min-p,
    # penalties, or model-default sampling settings (generation_config="vllm").
    want_own = embedder == OWN_KEY or OWN_KEY in extra
    own_tag = uuid.uuid4().hex if want_own else None
    ea = None
    if want_own:
        ea = {"dpp_own": own_tag, "dpp_max": max_new_tokens,
              "dpp_stop": sorted(eos | ({tok.eos_token_id} if tok.eos_token_id is not None else set()))}
    sp = SamplingParams(n=n, temperature=t_prime, top_p=1.0, top_k=0, min_p=0.0, repetition_penalty=1.0,
                        presence_penalty=0.0, frequency_penalty=0.0, max_tokens=max_new_tokens, logprobs=1,
                        seed=seed, stop_token_ids=sorted(eos), detokenize=False,
                        output_kind=RequestOutputKind.FINAL_ONLY, extra_args=ea)
    t0 = time.time()
    final = None
    async for out in engine.generate(TokensPrompt(prompt_token_ids=ids), sp, uuid.uuid4().hex):
        final = out
    gen_s = time.time() - t0
    samples = []
    for c in sorted(final.outputs, key=lambda c: c.index):
        toks = list(c.token_ids)
        logp = logq = 0.0
        for t, d in zip(toks, c.logprobs):
            if len(d) != 2:
                raise RuntimeError("the sampler returned no log q; is dpp_vllm_plugin loaded (DPP_LOGQ=1)?")
            logp += d[t].logprob
            logq += next(v.logprob for k, v in d.items() if k != t)
        ended = c.finish_reason == "stop" and bool(toks) and toks[-1] in eos
        samples.append({"text": tok.decode([t for t in toks if t not in eos], skip_special_tokens=True).strip(),
                        "logp": logp, "logq": logq, "nTokens": len(toks), "ended": ended, **({"ids": toks} if return_ids else {})})
    t1 = time.time()
    texts = [s["text"] for s in samples]
    own = None
    if want_own:
        r = (await own_take(engine, [[own_tag, len(samples)]]))[0]
        if min(r["count"]) < 1:
            raise RuntimeError("own states missing for some samples")
        # the recovery is a 200 x 152k x 5k float64 product: off the event loop, so /health and /status keep answering
        own64, own_means = await asyncio.to_thread(own_vectors, r["sum"], r["count"], key, own_raw)
        own = own64.astype(np.float32)
    async with S["emb_lock"]:
        def embed_all():
            e = own if embedder == OWN_KEY else get_embedder(embedder)(texts)
            es = {k: base64.b64encode((own if k == OWN_KEY else get_embedder(k)(texts)).tobytes()).decode("ascii")
                  for k in extra if k != embedder}
            torch.cuda.empty_cache()  # the caching allocator would otherwise keep ~1.3 GiB of embedding activations
            return e, es
        emb, embs = await asyncio.to_thread(embed_all)
    timing = {"generate_s": round(gen_s, 2), "embed_s": round(time.time() - t1, 2)}
    pool = {
        "prompt": prompt, "model": cfg["repo"], "modelName": cfg["name"], "modelKey": key,
        "embedder": OWN_KEY if embedder == OWN_KEY else EMBEDDERS[embedder]["repo"], "embedderId": embedder,
        "Tprime": t_prime, "maxNewTokens": max_new_tokens, "N": len(samples), "D": int(emb.shape[1]),
        "seed": seed, "timing": timing, "samples": samples,
        "G": G.gram_b64(emb),   # docs/BACKEND.md: float64 inner products of the float32 unit vectors
    }
    if with_emb:
        pool["emb"] = base64.b64encode(np.ascontiguousarray(emb, dtype="<f4").tobytes()).decode("ascii")
    if embs:  # the same pool under other embedders (for comparing them; not used by the page)
        pool["embs"] = embs
    if own_raw and want_own:  # uncentered mean states, projected or full (for building mu / mu_full from live generations)
        pool["ownRaw"] = base64.b64encode(own_means.tobytes()).decode("ascii")
    return pool


def _log_run(rec):
    """One JSON line per run: settings, sizes, timing. Prompt text is not logged."""
    day = datetime.datetime.now().strftime("%Y-%m")
    with open(os.path.join(LOG_DIR, f"runs-{day}.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


@app.exception_handler(StarletteHTTPException)
async def http_error(request, exc):
    """Every error is JSON {code, detail} (docs/BACKEND.md); `detail` stays the readable message FastAPI always sent."""
    return JSONResponse({"code": G.error_code(exc), "detail": exc.detail}, status_code=exc.status_code, headers=getattr(exc, "headers", None))


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    return JSONResponse({"code": "bad_request", "detail": "The request is malformed."}, status_code=400)


@app.exception_handler(Exception)
async def unhandled(request, exc):
    log.error("unhandled %s on %s", type(exc).__name__, request.url.path)
    headers = {}
    origin = request.headers.get("origin", "")
    if origin in CORS_ORIGINS or re.fullmatch(CORS_ORIGIN_REGEX, origin):   # ServerErrorMiddleware answers outside CORSMiddleware
        headers = {"access-control-allow-origin": origin, "vary": "Origin"}
    return JSONResponse({"code": "server", "detail": "Internal error."}, status_code=500, headers=headers)


# ---------------------------------------------------------------- draw cache: /sample with `offset`

def embedder_sig(k):
    """Identity of an embedder's vectors: id, repo, local snapshot revision, dtype."""
    cfg = EMBEDDERS[k]
    try:
        rev = os.path.basename(snapshot_download(cfg["repo"], local_files_only=True).rstrip("/"))
    except Exception:
        rev = ""
    return f"{k}|{cfg['repo']}|{rev}|{str(cfg['dtype']).replace('torch.', '')}"


@contextlib.asynccontextmanager
async def key_lock(skey):
    """One writer per stream: a second request for the same key waits, then reads what the first appended."""
    lock = _KEYLOCKS.setdefault(skey, [asyncio.Lock(), 0])
    lock[1] += 1
    try:
        async with lock[0]:
            yield
    finally:
        lock[1] -= 1
        if lock[1] == 0:
            _KEYLOCKS.pop(skey, None)


def _unit32(rows):
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)


async def stream_vectors(skey, key, embedder, offset, draws, pool=None):
    """Unit vectors (float32, N x D) of draws [offset, offset + N) of a stream under `embedder`. Text embedders: one cached vector per
    (text, embedder id, revision), computed on the GPU only for texts not cached yet (the first vector of a text is its vector, so
    identical texts have identical rows). `own`: the states stored at generation time, centered like live samples; HTTP 409 when those
    draws have none (or were stored in another mode)."""
    n = len(draws)
    if embedder == OWN_KEY:
        raws = await asyncio.to_thread(DRAWS.read_own, skey, offset, offset + n)
        mu = own_mu(key)
        if mu is None or any(r is None or len(r) != 4 * mu.shape[0] for r in raws):
            raise HTTPException(409, "These draws have no stored own states (stored before own states were available, or in another mode). Use another embedder or other positions.")
        v = np.frombuffer(b"".join(raws), dtype="<f4").reshape(n, -1).astype(np.float64) - mu
        return _unit32(v).astype(np.float32)
    sig = embedder_sig(embedder)
    texts = [d["text"] if d["text"].strip() else "(empty)" for d in draws]
    have = await asyncio.to_thread(DRAWS.get_embs, sig, texts)
    missing = sorted({t for t in texts if t not in have})
    if missing:
        async with S["emb_lock"]:
            def run():
                e = get_embedder(embedder)(missing)
                torch.cuda.empty_cache()
                return e
            e = await asyncio.to_thread(run)
        stored = await asyncio.to_thread(DRAWS.put_embs, sig, {t: (int(e.shape[1]), np.ascontiguousarray(e[i], dtype="<f4").tobytes()) for i, t in enumerate(missing)})
        have.update(stored)
    return np.stack([np.frombuffer(have[t], dtype="<f4") for t in texts])


async def stream_answer(skey, key, embedder, offset, n, t_prime, max_new, prompt, cached, stream_len, gen_s, t0, with_emb):
    draws = await asyncio.to_thread(DRAWS.read, skey, offset, offset + n)
    t1 = time.time()
    emb = await stream_vectors(skey, key, embedder, offset, draws)
    cfg = MODELS[key]
    pool = {"prompt": prompt, "model": cfg["repo"], "modelName": cfg["name"], "modelKey": key,
            "embedder": OWN_KEY if embedder == OWN_KEY else EMBEDDERS[embedder]["repo"], "embedderId": embedder,
            "Tprime": t_prime, "maxNewTokens": max_new, "N": n, "D": int(emb.shape[1]), "seed": None,
            "timing": {"generate_s": gen_s, "embed_s": round(time.time() - t1, 2)}, "samples": draws,
            "G": G.gram_b64(emb, same=[d["text"] for d in draws]), "offset": offset, "streamLength": stream_len, "cached": cached}
    if with_emb:
        pool["emb"] = base64.b64encode(np.ascontiguousarray(emb, dtype="<f4").tobytes()).decode("ascii")
    return pool


async def extend_stream(skey, key, engine, prompt, t_prime, max_new, embedder, length, need):
    """Generate `need` draws with a server-chosen seed and append them whole at position `length`. Returns (new length, pool, seed, run_id).
    Nothing is appended unless generation, own-state recovery and embedding all succeeded."""
    seed = await asyncio.to_thread(DRAWS.pick_seed, skey, need)
    extra = (OWN_KEY,) if own_available(key) and embedder != OWN_KEY else ()
    pool = await sample_pool(key, engine, prompt, need, t_prime, max_new, seed=seed, embedder=embedder, extra=extra, own_raw=True, with_emb=True)
    n = len(pool["samples"])
    own = [None] * n
    if "ownRaw" in pool:
        raw = base64.b64decode(pool["ownRaw"])
        w = len(raw) // n
        own = [raw[i * w:(i + 1) * w] for i in range(n)]
    draws = [{k: s[k] for k in ("text", "logp", "logq", "nTokens", "ended")} for s in pool["samples"]]
    run_id = uuid.uuid4().hex
    meta = {"model_key": key, "repo": MODELS[key]["repo"], "revision": model_revision(key), "tprime": t_prime, "max_new": max_new}
    new_len = await asyncio.to_thread(DRAWS.append, skey, meta, length, draws, own, seed, run_id)
    if embedder != OWN_KEY:   # the vector of a text is the first one stored; later copies and requests use it
        e = np.frombuffer(base64.b64decode(pool["emb"]), dtype="<f4").reshape(n, -1)
        first = {}
        for i, d in enumerate(draws):
            first.setdefault(d["text"] if d["text"].strip() else "(empty)", i)
        try:
            await asyncio.to_thread(DRAWS.put_embs, embedder_sig(embedder), {t: (int(e.shape[1]), np.ascontiguousarray(e[i], dtype="<f4").tobytes()) for t, i in first.items()})
        except Exception:
            log.exception("draw cache: embeddings of the new draws were not stored")
    return new_len, pool, seed, run_id


async def stream_sample(req, body, owner, prompt, n, t_prime, max_new, key, embedder, offset, rid):
    """POST /sample with an integer `offset`: draws [offset, offset + n) of the stream for (prompt, model, revision, T', max tokens)."""
    if body.get("embedders") or body.get("own_raw") or body.get("return_ids"):
        raise HTTPException(400, "The options embedders, own_raw and return_ids cannot be combined with offset.")
    end = offset + n
    if end > D.MAX_STREAM:
        raise HTTPException(400, f"A stream holds at most {D.MAX_STREAM} draws: offset + n must be at most {D.MAX_STREAM}.")
    repo, revision = MODELS[key]["repo"], model_revision(key)
    skey = D.stream_key(prompt, key, repo, revision, t_prime, max_new)
    length = await asyncio.to_thread(DRAWS.length, skey)
    if end - length > MAX_N:
        raise HTTPException(400, f"offset must be at most {length + MAX_N - n}: the stream has {length} draws and one request adds at most {MAX_N}.")
    client, t0, with_emb = req_client(req), time.time(), bool(body.get("emb"))
    rec = {"t": round(t0, 1), "model": key, "n": n, "t_prime": t_prime, "max_new": max_new, "embedder": embedder, "prompt_chars": len(prompt),
           "client": G.short(client), "owner": owner, "offset": offset}
    req.state.u.update(offset=offset, stream_key=skey)
    try:
        if end <= length:   # entirely from the stream: no queue, no GPU budget, a cheap rate limit
            if not owner:
                S["cacherate"].hit(client)
            req.state.u.update(cached=n, completions=0)   # `completions` of the usage store counts draws generated, not read
            with S["queue"].track(rid):
                pool = await stream_answer(skey, key, embedder, offset, n, t_prime, max_new, prompt, n, length, 0.0, t0, with_emb)
            rec.update(status="ok", cached=n, **pool["timing"])
            return pool
        # a shortfall: sampled like any other run (queue, one request per client, budgets for the shortfall only)
        cost = (end - length) * max_new
        now = t0
        L, Q = S["limits"], S["queue"]
        resident = MODELS[key]["slot"] == "resident"
        rec["cost"] = cost
        if not owner:
            L.claim(client)
        committed = started = False
        try:
            if not owner:
                L.check_run(client, cost, needs_swap(key), now, check_gap=resident or not Q.busy("slot"))
            async with Q.admit("resident" if resident else "slot", rid, req.is_disconnected):
                rec["wait_s"] = round(time.time() - now, 2)
                req.state.u["wait_s"] = rec["wait_s"]
                if not owner:
                    await wait_swap_gap(req, key)
                    L.commit_run(client, cost, needs_swap(key), time.time())
                    committed = True
                engine, held = S["engine"], False
                try:
                    if not resident:
                        try:
                            engine = await acquire_slot(req, key)
                            held = True
                        except (HTTPException, asyncio.CancelledError):
                            raise
                        except BaseException:
                            log.exception("model %s could not be started", key)
                            raise HTTPException(500, "The model could not be started on the server. Please try again.")
                    async with key_lock(skey):
                        cur = await asyncio.to_thread(DRAWS.length, skey)   # another request may have extended the stream while this one waited
                        need = end - cur
                        gen_s, new_pool = 0.0, None
                        if need > MAX_N:
                            raise HTTPException(400, f"offset must be at most {cur + MAX_N - n}: the stream has {cur} draws and one request adds at most {MAX_N}.")
                        if need > 0:
                            started = True
                            task = asyncio.create_task(extend_stream(skey, key, engine, prompt, t_prime, max_new, embedder, cur, need))
                            try:   # whatever ends this request (disconnect, cancellation, an error) also ends the task: a batch is appended whole or not at all
                                while not task.done():
                                    await asyncio.wait({task}, timeout=1.0)
                                    if not task.done() and await req.is_disconnected():
                                        task.cancel()   # nothing is appended: the append is the last step of the task
                                        rec["status"] = "aborted"
                                        raise HTTPException(499, "Client closed the request.")
                                new_len, new_pool, seed, run_id = task.result()
                            except asyncio.CancelledError:
                                task.cancel()
                                raise
                            except HTTPException:
                                raise
                            except D.Conflict:
                                log.exception("draw cache conflict")
                                raise HTTPException(503, "The draw cache changed while sampling. Please try again.", headers={"Retry-After": "5"})
                            except Exception as e:
                                log.exception("stream sample failed")
                                rec["status"] = f"error: {type(e).__name__}"
                                if engine_error(engine, e):
                                    raise HTTPException(503, "The model server had a problem and is restarting. Please try again in a minute.", headers={"Retry-After": "60"})
                                raise HTTPException(500, "Sampling failed on the server. Please try again.")
                            gen_s = new_pool["timing"]["generate_s"]
                            await asyncio.to_thread(RUNLOG.write, G.sample_record(
                                new_pool, key, repo, revision, embedder, time.time(), run_id, stream={"key": skey, "start": cur, "end": new_len}))
                            req.state.u.update(run_id=run_id, tokens=sum(x["nTokens"] for x in new_pool["samples"]))
                        cached = max(0, min(n, cur - offset))
                        req.state.u.update(cached=cached, completions=len(new_pool["samples"]) if new_pool else 0, gen_s=gen_s)
                        pool = await stream_answer(skey, key, embedder, offset, n, t_prime, max_new, prompt, cached, max(cur, end), gen_s, t0, with_emb)
                        if new_pool is not None:
                            pool["timing"]["embed_s"] = round(pool["timing"]["embed_s"] + new_pool["timing"]["embed_s"], 2)
                        req.state.u.update(embed_s=pool["timing"]["embed_s"])
                        rec.update(status="ok", cached=cached, **pool["timing"])
                        return pool
                finally:
                    if held:
                        await slot_release()
        finally:
            if not owner:
                L.unclaim(client)
                if committed:
                    L.end_run(client, 0 if started else cost)
    finally:
        rec["s"] = round(time.time() - t0, 2)
        try:
            _log_run(rec)
        except OSError:
            pass


def note_ids(req, body):
    """Remember the page's optional `visitor` and `session` ids (malformed ones are ignored) and the body's `prompt` (a string) for the usage record of this request."""
    req.state.u["visitor"], req.state.u["session"] = U.valid_id(body.get("visitor")), U.valid_id(body.get("session"))
    if isinstance(body.get("prompt"), str):   # stored in the usage record, also when the request fails later (capped by usage_logged)
        req.state.u["prompt"] = body["prompt"]


def identity(req, visitor, session, t):
    """(session, visitor) to store: the page's ids, else the keyed hash of the client address of the day (never the address)."""
    anon, day = SALT.anon_id(req_client(req))
    visitor = visitor or anon
    session = session or "d-" + hashlib.sha256(f"{visitor}|{day}".encode()).hexdigest()[:16]
    return session, visitor


def usage_logged(route):
    """Record one usage row per call of the route: time, session, origin, settings (filled in by the handler through req.state.u),
    outcome ('ok' or the error code) and timing. A failure of the store never touches the request."""
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(req: Request):
            t0 = time.time()
            u = req.state.u = {}
            outcome = "ok"
            try:
                return await fn(req)
            except HTTPException as e:
                outcome = "aborted" if e.status_code == 499 else G.error_code(e)
                raise
            except asyncio.CancelledError:
                outcome = "aborted"
                raise
            except BaseException:
                outcome = "server"
                raise
            finally:
                try:
                    if USAGE.enabled:
                        session, visitor = identity(req, u.get("visitor"), u.get("session"), t0)
                        wait = u.get("wait_s")
                        USAGE.record_request(t0, session, visitor, U.request_origin(req.headers), route, outcome, model=u.get("model"),
                                             embedder=u.get("embedder"), n=u.get("n"), tprime=u.get("tprime"), max_new=u.get("max_new"),
                                             wait_s=wait, run_s=round(time.time() - t0 - (wait or 0), 2), gen_s=u.get("gen_s"),
                                             embed_s=u.get("embed_s"), tokens=u.get("tokens"), completions=u.get("completions"),
                                             run_id=u.get("run_id"), cached=u.get("cached"), offset=u.get("offset"),
                                             addr=req_client(req), prompt=(u.get("prompt") or "")[:MAX_PROMPT_CHARS], stream_key=u.get("stream_key"))
                except Exception:
                    log.exception("usage record failed")
        return wrapper
    return deco


@app.get("/health")
async def health():
    if "engine" not in S:
        return {"status": "starting"}
    bad = await G.first_unhealthy(engines_to_check(), timeout=5)
    if bad is not None:
        return JSONResponse({"status": "unhealthy", "engine": bad[0]}, status_code=503)
    return {"status": "ok"}


def model_state(key):
    if MODELS[key]["slot"] == "resident":
        return "ready" if "engine" in S else "loading"
    if not is_cached(key):
        return "unavailable"
    if SLOT["loading"] == key:
        return "loading"
    return "ready" if SLOT["key"] == key and SLOT["engine"] is not None else "cold"


@app.get("/status")
async def status():
    res = MODELS[RESIDENT_KEY]
    q = S["queue"].counts() if "queue" in S else {"running": 0, "waiting": 0, "max": MAX_QUEUE_WAITING}
    models = [{"key": k, "name": m["name"], "params": m["params"], "repo": m["repo"], "slot": m["slot"],
               "state": model_state(k)} for k, m in sorted(MODELS.items(), key=lambda kv: kv[1]["params"])]
    return {"version": CONTRACT_VERSION, "model": res["repo"], "modelKey": RESIDENT_KEY, "modelName": res["name"],
            "defaultModel": DEFAULT_MODEL, "models": models,
            "embedders": {**{k: v["repo"] for k, v in EMBEDDERS.items()}, **({OWN_KEY: None} if OWN_ON else {})},
            "ownModels": [k for k in MODELS if own_available(k)],
            "defaultEmbedder": DEFAULT_EMBEDDER, "limits": {"maxN": MAX_N, "maxNewTokens": MAX_TOKENS}, "auth": False,
            "maxN": MAX_N, "maxNewTokens": MAX_TOKENS, "maxTextChars": MAX_TEXT_CHARS, "logsRuns": RUNLOG.enabled, "usage": USAGE.enabled, "drawCache": DRAWS.enabled,
            "embed": True, "continuePool": True,
            "describe": True, "maxDescribeGroups": G.DESCRIBE_MAX_GROUPS, "maxDescribeTexts": G.DESCRIBE_MAX_TEXTS,
            "queue": q, "generating": q["running"], "waiting": q["waiting"],
            "runsLastHour": S["limits"].snapshot()["runsLastHour"] if "limits" in S else 0}


@app.post("/sample")
@usage_logged("/sample")
async def sample_endpoint(req: Request):
    owner = G.is_owner(req.headers, OWN_SECRET_FILE)
    body = await G.read_json(req, MAX_BODY_OWNER if owner else MAX_BODY_SAMPLE)
    note_ids(req, body)
    # docs/BACKEND.md: a value outside the limits is refused with 400 bad_request, never capped
    prompt = G.text(body.get("prompt"), "prompt", MAX_PROMPT_CHARS, truncate=False)
    n = G.number(body.get("n", 30), "n", True, 2, MAX_N, clamp=False)
    t_prime = G.number(body.get("t_prime", 1.0), "t_prime", False, 0.2, 2.0, clamp=False)
    max_new = G.number(body.get("max_new_tokens", 128), "max_new_tokens", True, 8, MAX_TOKENS, clamp=False)
    seed = G.number(body.get("seed", random.randrange(2**31)), "seed", True) % 2**31
    key = body.get("model") or DEFAULT_MODEL
    if not isinstance(key, str) or key not in MODELS:
        raise HTTPException(400, f"Unknown model; use one of {sorted(MODELS)}.")
    if model_state(key) == "unavailable":
        raise HTTPException(400, f"Model {key} is not available on this server.")
    ok_emb = set(EMBEDDERS) | ({OWN_KEY} if OWN_ON else set())
    embedder = body.get("embedder", DEFAULT_EMBEDDER)
    if not isinstance(embedder, str) or embedder not in ok_emb:
        raise HTTPException(400, f"Unknown embedder; use one of {sorted(ok_emb)}.")
    extra = G.embedder_list(body.get("embedders"), ok_emb, MAX_EMBEDDERS)
    own_raw, return_ids = bool(body.get("own_raw")), bool(body.get("return_ids"))
    if (extra or own_raw or return_ids) and not owner:
        raise HTTPException(401, "The options embedders, own_raw and return_ids are for the owner's scripts.")
    if (embedder == OWN_KEY or OWN_KEY in extra) and not own_available(key):
        raise HTTPException(400, f"The embedder 'own' (the LM's own states) is not available for model {key}.")
    rid = G.request_id(body.get("request_id"))
    req.state.u.update(model=key, embedder=embedder, n=n, tprime=t_prime, max_new=max_new)
    if body.get("offset") is not None:   # opt in to the shared draw cache; without `offset` nothing below changes
        off = body["offset"]
        if isinstance(off, bool) or not isinstance(off, int) or off < 0:
            raise HTTPException(400, "offset must be an integer, at least 0.")
        if not DRAWS.enabled:
            raise HTTPException(400, "The draw cache is off on this server; send the request without offset.")
        return await stream_sample(req, body, owner, prompt, n, t_prime, max_new, key, embedder, off, rid)
    now = time.time()
    L, Q = S["limits"], S["queue"]
    client, cost = req_client(req), n * max_new
    resident = MODELS[key]["slot"] == "resident"
    rec = {"t": round(now, 1), "model": key, "n": n, "t_prime": t_prime, "max_new": max_new, "embedder": embedder,
           "prompt_chars": len(prompt), "client": G.short(client), "cost": cost, "owner": owner}
    if not owner:
        L.claim(client)    # 1 sample request per client, queued or running (429)
    committed = started = False   # committed: the hourly budgets were charged; started: generation began, the tokens are spent
    try:
        if not owner:
            # refuse now when a budget is already exhausted (the swap gap only when nobody is ahead); charge at start
            L.check_run(client, cost, needs_swap(key), now, check_gap=resident or not Q.busy("slot"))
        async with Q.admit("resident" if resident else "slot", rid, req.is_disconnected):
            rec["wait_s"] = round(time.time() - now, 2)
            req.state.u["wait_s"] = rec["wait_s"]
            if not owner:
                await wait_swap_gap(req, key)
                L.commit_run(client, cost, needs_swap(key), time.time())
                committed = True
            engine, held = S["engine"], False
            try:
                if not resident:
                    try:
                        engine = await acquire_slot(req, key)
                        held = True
                    except HTTPException:
                        raise
                    except asyncio.CancelledError:
                        raise
                    except BaseException:
                        log.exception("model %s could not be started", key)
                        raise HTTPException(500, "The model could not be started on the server. Please try again.")
                started = True
                task = asyncio.create_task(sample_pool(key, engine, prompt, n, t_prime, max_new_tokens=max_new, seed=seed, embedder=embedder,
                                                       extra=extra, own_raw=own_raw, return_ids=return_ids,
                                                       with_emb=bool(body.get("emb"))))
                try:
                    # Stop generating if the visitor leaves (Stop button, closed tab).
                    while not task.done():
                        await asyncio.wait({task}, timeout=1.0)
                        if not task.done() and await req.is_disconnected():
                            task.cancel()
                            rec["status"] = "aborted"
                            raise HTTPException(499, "Client closed the request.")
                    pool = task.result()
                    run_id = uuid.uuid4().hex   # random; the run log record and the usage row share it, nothing else does
                    await asyncio.to_thread(RUNLOG.write, G.sample_record(pool, key, MODELS[key]["repo"], model_revision(key), embedder, time.time(), run_id))
                    req.state.u.update(run_id=run_id, tokens=sum(s["nTokens"] for s in pool["samples"]), completions=len(pool["samples"]),
                                       gen_s=pool["timing"]["generate_s"], embed_s=pool["timing"]["embed_s"])
                    rec.update(status="ok", **pool["timing"], ended=sum(s["ended"] for s in pool["samples"]),
                               tokens=sum(s["nTokens"] for s in pool["samples"]))
                    return pool
                except HTTPException:
                    raise
                except Exception as e:
                    log.exception("sample failed")
                    rec["status"] = f"error: {type(e).__name__}"
                    if engine_error(engine, e):
                        raise HTTPException(503, "The model server had a problem and is restarting. Please try again in a minute.", headers={"Retry-After": "60"})
                    raise HTTPException(500, "Sampling failed on the server. Please try again.")
            finally:
                if held:
                    await slot_release()
    finally:
        if not owner:
            L.unclaim(client)
            if committed:
                L.end_run(client, 0 if started else cost)
        rec["s"] = round(time.time() - now, 2)
        try:
            _log_run(rec)
        except OSError:
            pass


async def forced_own(key, prompt, texts, ended, eos_id=None, tail="", req=None):
    """Teacher-force completions through the engine of `key` (slot swapped in when needed) and return the plugin's
    per-text mean projected states: {"sum": bytes (N x r float32), "count": [...], "logp": [...], "forcedOk": [...],
    "nTokens": [...], "empty": [...], "promptTokens": int, "eos0": int}. `tail` is appended to the text of ended samples
    before tokenizing (for models whose stored text lost trailing whitespace)."""
    tok, eos = model_tok(key)
    cfg = MODELS[key]
    chat = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                                   **cfg["chat"])
    pids = tok(chat, add_special_tokens=False).input_ids
    eos0 = int(eos_id) if eos_id is not None else MODELS[key].get("eos_id", eos_first(key))  # only affects log p, not the states
    stop = sorted(eos | ({tok.eos_token_id} if tok.eos_token_id is not None else set()))
    base = "own" + uuid.uuid4().hex[:12]
    seqs, empty = [], []
    for i, (t, e) in enumerate(zip(texts, ended)):
        seq = tok(str(t) + (tail if e else ""), add_special_tokens=False).input_ids + ([eos0] if e else [])
        if not seq:
            seq = [eos0]
            empty.append(i)
        if len(seq) > OWN_LMAX:
            raise HTTPException(400, f"Completion {i} has {len(seq)} tokens (limit {OWN_LMAX}).")
        seqs.append(seq)
    if own_exact(key) and len(seqs) > OWN_EXACT_MAX_SEQS:   # the plugin's exact mode has 448 slots (a V-long logit sum each); the rest would silently get no states
        raise HTTPException(400, f"At most {OWN_EXACT_MAX_SEQS} texts per own-state pass.")
    resident = cfg["slot"] == "resident"
    engine, held = S["engine"], False
    try:
        if not resident:
            engine = await (acquire_slot(req, key) if req is not None else slot_acquire(key))
            held = True

        async def one(i):
            seq = seqs[i]
            sp = SamplingParams(n=1, temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, repetition_penalty=1.0,
                                presence_penalty=0.0, frequency_penalty=0.0, max_tokens=len(seq), logprobs=1,
                                seed=0, stop_token_ids=stop, detokenize=False, output_kind=RequestOutputKind.FINAL_ONLY,
                                extra_args={"dpp_own": f"{base}.{i}", "dpp_force": seq, "dpp_force_eos": eos0,
                                            "dpp_stop": stop, "dpp_max": len(seq)})
            final = None
            async for out in engine.generate(TokensPrompt(prompt_token_ids=pids), sp, f"{base}x{i}"):
                final = out
            c = final.outputs[0]
            toks = list(c.token_ids)
            lp = 0.0
            for t, d in zip(toks, c.logprobs):
                lp += d[t].logprob
            return toks == seq, lp, len(toks)

        res = await asyncio.gather(*[one(i) for i in range(len(seqs))])
        taken = await own_take(engine, [[f"{base}.{i}", 1] for i in range(len(seqs))])
    except HTTPException:
        raise
    except Exception as e:
        log.exception("forced own pass failed")
        if engine_error(engine, e):
            raise HTTPException(503, "The model server had a problem and is restarting. Please try again in a minute.", headers={"Retry-After": "60"})
        raise HTTPException(500, "The embedding pass failed on the server. Please try again.")
    finally:
        if held:
            await slot_release()
    return {"sum": b"".join(t["sum"] for t in taken), "count": [t["count"][0] for t in taken], "logp": [r[1] for r in res],
            "forcedOk": [bool(r[0]) for r in res], "nTokens": [r[2] for r in res], "empty": empty,
            "promptTokens": len(pids), "eos0": eos0}


@app.post("/embed_own")
async def embed_own(req: Request):
    """Internal: teacher-force given completions through the engine and return their own-state vectors.

    Protected by the shared secret in the file `.own_secret` next to server.py (header X-Own-Secret); without that file the route is 404.
    Body: {model, prompt, texts: [...], ended: [...]}. Returns uncentered mean projected states (float32 base64, N x r),
    step counts, per-text sum of log p, and whether the engine sampled exactly the forced tokens."""
    if not os.path.exists(OWN_SECRET_FILE):
        raise HTTPException(404, "Not found.")
    if not G.is_owner(req.headers, OWN_SECRET_FILE):
        raise HTTPException(401, "Forbidden.")
    body = await G.read_json(req, MAX_BODY_OWNER)
    key = body.get("model") or DEFAULT_MODEL
    texts, ended = body.get("texts"), body.get("ended")
    prompt = body.get("prompt")
    if (not isinstance(key, str) or key not in MODELS or not own_has_P(key) or not isinstance(texts, list) or not isinstance(ended, list)
            or len(texts) != len(ended) or not texts or len(texts) > 4 * MAX_N or not isinstance(prompt, str) or not prompt
            or not all(isinstance(t, str) for t in texts)):
        raise HTTPException(400, "Bad request or no P for this model.")
    eos_id = G.number(body["eos_id"], "eos_id", True, 0, 10**7, clamp=False) if body.get("eos_id") is not None else None
    rid = G.request_id(body.get("request_id"))
    async with S["queue"].admit(engine_name(key), rid, req.is_disconnected):
        r = await forced_own(key, prompt, texts, [bool(e) for e in ended], eos_id, req=req)
    counts = r["count"]
    mean = await asyncio.to_thread(own_mean, r["sum"], counts, key)
    return {"D": int(mean.shape[1]), "vec": base64.b64encode(mean.astype("<f4").tobytes()).decode("ascii"), "count": counts,
            "logp": r["logp"], "forcedOk": r["forcedOk"], "nTokens": r["nTokens"],
            "empty": r["empty"], "promptTokens": r["promptTokens"], "eos0": r["eos0"]}


@app.post("/embed")
@usage_logged("/embed")
async def embed_endpoint(req: Request):
    """Public: embed existing completions after the fact, in the encoding /sample uses (float32 little endian, base64, N x D).

    Body (JSON, at most MAX_BODY_EMBED bytes): {texts: [str, ...] (1 to 200, each up to MAX_TEXT_CHARS characters),
    embedder: "minilm" | "qwen3-emb" | "mpnet" | "own", model, prompt, ended}. For "own", model (a key with own states) and
    prompt (the prompt the texts answer) are required, and ended (one bool per text, default all true: the completion ended
    with the model's end-of-turn token) tells whether to append that token. The own states are recomputed by teacher forcing
    the text through the model's engine (the slot is swapped when needed), so they approximate the vectors /sample returns
    from the generation itself."""
    owner = G.is_owner(req.headers, OWN_SECRET_FILE)
    body = await G.read_json(req, MAX_BODY_OWNER if owner else MAX_BODY_EMBED)
    note_ids(req, body)
    texts = body.get("texts")
    if not isinstance(texts, list) or not 1 <= len(texts) <= MAX_N or not all(isinstance(t, str) for t in texts):
        raise HTTPException(400, f"texts must be a list of 1 to {MAX_N} strings.")
    if max(len(t) for t in texts) > MAX_TEXT_CHARS:
        raise HTTPException(400, f"Each text is limited to {MAX_TEXT_CHARS} characters.")
    embedder = body.get("embedder", DEFAULT_EMBEDDER)
    ok_emb = set(EMBEDDERS) | ({OWN_KEY} if OWN_ON else set())
    if not isinstance(embedder, str) or embedder not in ok_emb:
        raise HTTPException(400, f"Unknown embedder; use one of {sorted(ok_emb)}.")
    rid = G.request_id(body.get("request_id"))
    req.state.u.update(embedder=embedder, n=len(texts), model=body.get("model") if isinstance(body.get("model"), str) and embedder == OWN_KEY else None)
    L, Q = S["limits"], S["queue"]
    client, now = req_client(req), time.time()
    if not owner:
        L.check_embed(client, now)
    t0 = time.time()
    out = {"N": len(texts), "embedder": embedder if embedder == OWN_KEY else EMBEDDERS[embedder]["repo"], "embedderId": embedder,
           "approximate": False}
    if embedder != OWN_KEY:
        if not owner:
            L.commit_embed(client, now)
        with Q.track(rid):   # a plain embedder takes well under a second and bypasses the queue
            async with S["emb_lock"]:
                def run():
                    e = get_embedder(embedder)(texts)
                    torch.cuda.empty_cache()
                    return e
                emb = await asyncio.to_thread(run)
    else:
        key = body.get("model") or ""
        prompt = G.text(body.get("prompt"), "prompt", MAX_PROMPT_CHARS, required=False)
        if not isinstance(key, str) or key not in MODELS or not own_available(key):
            raise HTTPException(400, f"The embedder 'own' needs a model with own states; use one of {[k for k in MODELS if own_available(k)]}.")
        if not prompt:
            raise HTTPException(400, "The embedder 'own' needs the prompt the texts answer.")
        if model_state(key) == "unavailable":
            raise HTTPException(400, f"Model {key} is not available on this server.")
        ended = body.get("ended")
        if ended is None:
            ended = [True] * len(texts)
        if not isinstance(ended, list) or len(ended) != len(texts):
            raise HTTPException(400, "ended must be a list with one boolean per text.")
        cost = len(texts) * EMBED_OWN_TOKEN_COST
        if not owner:
            L.check_run(client, cost, needs_swap(key), now, check_gap=engine_name(key) == "resident" or not Q.busy("slot"))
            L.check_embed(client, now)
        async with Q.admit(engine_name(key), rid, req.is_disconnected):
            req.state.u["wait_s"] = round(time.time() - now, 2)
            if not owner:
                await wait_swap_gap(req, key)
                L.commit_run(client, cost, needs_swap(key), time.time())
                L.commit_embed(client, time.time())
            r = await forced_own(key, prompt, texts, [bool(e) for e in ended], tail=MODELS[key].get("ended_tail", ""), req=req)
        if min(r["count"]) < 1:
            raise HTTPException(500, "The embedding pass returned no states for some texts.")
        emb64, _ = await asyncio.to_thread(own_vectors, r["sum"], r["count"], key)
        emb = emb64.astype(np.float32)
        out.update(model=key, approximate=True,
                   note="Teacher-forced from the text, so close to but not identical to the vectors /sample returns for the same completions "
                        "(cosine about 0.9999 for Gemma 3 and Qwen3, about 0.99 for Gemma 2 9B, whose stored text loses trailing whitespace tokens).")
    out.update(D=int(emb.shape[1]), G=G.gram_b64(emb), timing={"embed_s": round(time.time() - t0, 2)})
    if body.get("emb"):   # the raw unit vectors (N x D float32) only on request; G is what the page uses
        out["emb"] = base64.b64encode(np.ascontiguousarray(emb, dtype="<f4").tobytes()).decode("ascii")
    return out


@app.get("/queue")
async def queue_position(req: Request, id: str = ""):
    """State of the request with this `request_id`: waiting (ahead = jobs in front on its engine, the running one included), running, or unknown."""
    S["qpoll"].hit(req_client(req))
    rid = G.request_id(id or None)
    if rid is None:
        raise HTTPException(400, "Pass the request_id as ?id=.")
    state, ahead = S["queue"].position(rid)
    return JSONResponse({"state": state, "ahead": ahead}, headers={"Cache-Control": "no-store"})


def describe_template():
    """The /describe prompt: describe_prompt.txt next to this file when present (for tuning without a restart), else the built-in one."""
    try:
        return open(DESCRIBE_TEMPLATE_FILE).read().strip()
    except OSError:
        return G.DESCRIBE_TEMPLATE


async def run_describe(prompts):
    """One greedy, short, newline-terminated generation per prompt on the resident engine, batched by vLLM."""
    key, engine = RESIDENT_KEY, S["engine"]
    tok, eos = model_tok(key)
    sp = SamplingParams(n=1, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0,
                        frequency_penalty=0.0, max_tokens=DESCRIBE_TOKENS, stop=["\n"], stop_token_ids=sorted(eos),
                        output_kind=RequestOutputKind.FINAL_ONLY)

    async def one(p):
        text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, **MODELS[key]["chat"])
        ids = tok(text, add_special_tokens=False).input_ids
        final = None
        async for out in engine.generate(TokensPrompt(prompt_token_ids=ids), sp, uuid.uuid4().hex):
            final = out
        return final.outputs[0].text

    return await asyncio.gather(*[one(p) for p in prompts])


@app.post("/describe")
@usage_logged("/describe")
async def describe_endpoint(req: Request):
    """Public: name each group of completions in a few words with the resident model.

    Body (JSON, at most MAX_BODY_DESCRIBE bytes): {prompt, groups: [[text, ...], ...] (2 to 8 groups, 1 to 4 texts each, texts cut to
    240 characters), request_id?}. Returns {names: [str, ...], model, ms}; a name is "" when the model gave nothing usable."""
    owner = G.is_owner(req.headers, OWN_SECRET_FILE)
    body = await G.read_json(req, MAX_BODY_DESCRIBE)
    note_ids(req, body)
    rid = G.request_id(body.get("request_id"))
    prompt = G.text(body.get("prompt"), "prompt", MAX_TEXT_CHARS)
    groups = G.describe_groups(body.get("groups"))
    req.state.u.update(model=RESIDENT_KEY, n=len(groups))
    template = describe_template()
    try:
        prompts = G.describe_prompts(prompt, groups, template)
    except (KeyError, IndexError, ValueError):
        log.error("describe_prompt.txt has a bad placeholder; using the built-in prompt")
        template = G.DESCRIBE_TEMPLATE
        prompts = G.describe_prompts(prompt, groups, template)
    t0 = time.time()
    ck = G.describe_key(RESIDENT_KEY, prompt, groups, template)
    hit = S["dcache"].get(ck)
    if hit is not None:
        return {"names": list(hit), "model": RESIDENT_KEY, "ms": int((time.time() - t0) * 1000)}
    L, Q = S["limits"], S["queue"]
    client = req_client(req)
    if not owner:
        L.check_describe(client, t0)
    async with Q.admit("resident", rid, req.is_disconnected):
        req.state.u["wait_s"] = round(time.time() - t0, 2)
        if not owner:
            L.commit_describe(client, time.time())
        try:
            raw = await run_describe(prompts)
        except Exception as e:
            log.exception("describe failed")
            if engine_error(S["engine"], e):
                raise HTTPException(503, "The model server had a problem and is restarting. Please try again in a minute.", headers={"Retry-After": "60"})
            raise HTTPException(500, "Naming the groups failed on the server. Please try again.")
    names = [G.clean_label(r) for r in raw]
    S["dcache"].put(ck, names)
    return {"names": names, "model": RESIDENT_KEY, "ms": int((time.time() - t0) * 1000)}


# ---------------------------------------------------------------- usage events, stats, run log export

@app.post("/event")
async def event_endpoint(req: Request):
    """The page's own events: `load` once per page load ({referrer, expert}) and `ui` events named from U.UI_NAMES. Cheap: one small
    SQLite insert, never the GPU queue. Answers 204. Body (at most 4 KB): {visitor?, session, type, ...}; unknown fields are ignored."""
    body = await G.read_json(req, MAX_BODY_EVENT)
    S["evrate"].hit(req_client(req))
    sid = U.valid_id(body.get("session"))
    if sid is None:
        raise G.bad("session must be 8 to 64 characters of A-Za-z0-9_-.")
    ev = U.clean_event(body)
    if ev is None:
        raise G.bad(f'type must be "load" or "ui" (name one of {list(U.UI_NAMES)}).')
    if USAGE.enabled:
        now = time.time()
        _, visitor = identity(req, U.valid_id(body.get("visitor")), sid, now)
        USAGE.record_event(now, sid, visitor, U.request_origin(req.headers), ev[0], ev[1], ev[2], ev[3], addr=req_client(req))
    return Response(status_code=204)


def stats_gate(req):
    """401 without the stats password (or the owner key), 503 when the owner has not set one, 429 after repeated failures."""
    client = req_client(req)
    if S["authfail"].count(client) >= STATS_FAILS:
        raise G.bad("Too many failed attempts. Try again in a few minutes.", 429, {"Retry-After": str(STATS_FAIL_WINDOW)}, "rate_limit")
    r = U.check_password(req.headers, OWN_SECRET_FILE, STATS_PASSWORD_FILE, os.environ)
    if r == "unconfigured":
        raise G.bad("Usage stats are off: the owner has not set a stats password (see README.md, \"Usage stats\").", 503, None, "server")
    if r == "bad":
        S["authfail"].hit(client)
        raise G.bad("Wrong or missing stats password.", 401, None, "auth")


NO_STORE = {"Cache-Control": "no-store", "X-Robots-Tag": "noindex"}


@app.get("/stats")
async def stats_endpoint(req: Request, days: str = "30"):
    """Aggregated usage of the last `days` UTC days, or `days=all` for the whole history. Header X-Stats-Password (or X-Own-Secret). Shapes: server/API.md."""
    stats_gate(req)
    if not USAGE.enabled:
        raise G.bad("The usage store is off (DPP_USAGE=0).", 503, None, "server")
    if days != "all" and not (days.isdigit() and int(days) >= 1):
        raise G.bad('days must be a positive integer or "all".')
    return JSONResponse(await asyncio.to_thread(USAGE.stats, days if days == "all" else int(days)), headers=NO_STORE)


@app.get("/stats/session")
async def stats_session(req: Request, id: str = ""):
    stats_gate(req)
    if U.valid_id(id) is None:
        raise G.bad("id must be a session id.")
    out = await asyncio.to_thread(USAGE.session, id)
    if out is None:
        raise G.bad("Unknown session.", 404)
    for q in out["requests"]:
        skey = q.pop("_stream")
        if q["runId"] and not q["prompt"]:   # rows from before the usage store kept prompts: the run log has it, by run_id
            rec = await asyncio.to_thread(RUNLOG.find, q["runId"], await asyncio.to_thread(USAGE.run_day, q["runId"]))
            q["prompt"] = rec["prompt"] if rec else None
        q["drawTexts"] = None   # the completions of a stream request: draws [offset, offset + N) read from the draw cache, None when evicted
        if skey and q["outcome"] == "ok" and q["offset"] is not None and q["N"] and DRAWS.enabled:
            try:
                q["drawTexts"] = await asyncio.to_thread(DRAWS.read, skey, q["offset"], q["offset"] + q["N"], False)
            except Exception:   # the stream was evicted or is shorter than the request
                pass
    return JSONResponse(out, headers=NO_STORE)


@app.get("/stats/run")
async def stats_run(req: Request, id: str = ""):
    """The full run-log record of a run (completions, logp, logq)."""
    stats_gate(req)
    if not RUNLOG.enabled:
        raise G.bad("The run log is off (DPP_RUNLOG=0).", 404)
    rec = await asyncio.to_thread(RUNLOG.find, id, await asyncio.to_thread(USAGE.run_day, id))
    if rec is None:
        raise G.bad("Unknown run.", 404)
    return JSONResponse(rec, headers=NO_STORE)


@app.get("/runlog")
async def runlog_export(req: Request, day: str = ""):
    """Owner only (header X-Own-Secret, as /embed_own): without `day` the list of days with line counts, else that day's file streamed."""
    if not G.is_owner(req.headers, OWN_SECRET_FILE):
        raise HTTPException(401, "The run log is for the owner.")
    if not day:
        return JSONResponse({"enabled": RUNLOG.enabled, "days": await asyncio.to_thread(RUNLOG.days)}, headers=NO_STORE)
    path = RUNLOG.path(day)
    if not os.path.isfile(path):
        raise HTTPException(404, "No run log for this day.")
    return FileResponse(path, media_type="application/x-ndjson", headers=NO_STORE)


@app.get("/own_stats")
async def own_stats(req: Request):
    """Owner only: the plugin's own-state statistics in the resident engine (mode exact or projected, slots in use, allocated MiB), or null
    while the state is unloaded (it loads on the first own request and unloads after 60 s without one)."""
    if not G.is_owner(req.headers, OWN_SECRET_FILE):
        raise HTTPException(401, "For the owner.")
    res = await S["engine"].collective_rpc("dpp_own_stats")
    return JSONResponse({"resident": res[0] if res else None}, headers=NO_STORE)
