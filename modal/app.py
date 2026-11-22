"""Sampling backend for the diverse-sampling demo, on Modal.

Same job as the Hugging Face Space (space/app.py): draw N completions i.i.d. from the locally
annealed proposal q = p°_{T'} and return, for each one, the exact log p(y) and log q(y)
(end-of-string included), plus unit-norm MiniLM embeddings. The page does the DPP part.

Spending guards (the app must stay inside Modal's free monthly credit):
  * only visitors signed in with Hugging Face may call it (their token is checked with HF);
  * each visitor gets RUNS_PER_USER_PER_DAY runs a day;
  * GPU time is metered and the endpoint refuses work once GPU_SECONDS_PER_MONTH is used;
  * one GPU container at most, shut down after SCALEDOWN seconds idle; inputs are capped.

Deploy:   modal deploy modal/app.py
Weights:  modal run modal/app.py::download   (once; fills the volume on a CPU machine)
"""
import datetime
import os
import time

import modal

MODEL_KEY = "qwen3-14b"
MODEL_REPO = "Qwen/Qwen3-14B"
CHAT_KW = {"enable_thinking": False}
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
GPU = "L40S"                    # 48 GB: Qwen3-14B in bf16 plus the KV cache for a batch of 100
GPU_DOLLARS_PER_SEC = 0.000542  # modal.com/pricing, L40S
# $30/month free credit. Budget ~$21 of GPU time and leave the rest for CPU, memory and slack.
GPU_SECONDS_PER_MONTH = int(os.environ.get("GPU_SECONDS_PER_MONTH", 21 / GPU_DOLLARS_PER_SEC))
RUNS_PER_USER_PER_DAY = 20
SCALEDOWN = 120                 # seconds an idle GPU container stays up
MAX_N, MAX_TOKENS = 100, 256
ORIGINS = ["https://timvieira.github.io", "http://localhost:8765"]

app = modal.App("dpp-lm-sampler")
cache = modal.Volume.from_name("dpp-lm-hf-cache", create_if_missing=True)
usage = modal.Dict.from_name("dpp-lm-usage", create_if_missing=True)

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.13.0", "transformers==5.24.0", "sentence-transformers==6.3.0", "numpy")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
)
web_image = modal.Image.debian_slim(python_version="3.12").pip_install("fastapi[standard]")


def _month():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m")


def _charge(seconds):
    """Add GPU seconds to this month's meter."""
    key = f"gpu-seconds:{_month()}"
    usage[key] = usage.get(key, 0) + float(seconds)


@app.function(image=gpu_image, volumes={"/cache": cache}, timeout=3600)
def download():
    """Fetch the weights into the volume (CPU only, so it costs almost nothing)."""
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_REPO, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    snapshot_download(EMBEDDER)
    cache.commit()
    print("weights cached")


@app.cls(image=gpu_image, gpu=GPU, volumes={"/cache": cache}, max_containers=1,
         scaledown_window=SCALEDOWN, timeout=600)
class Sampler:
    @modal.enter()
    def load(self):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.started = time.time()
        # Charge the start-up and the final idle window up front; exit() settles the rest.
        self.charged = 60 + SCALEDOWN
        _charge(self.charged)
        self.tok = AutoTokenizer.from_pretrained(MODEL_REPO)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_REPO, dtype=torch.bfloat16).to("cuda").eval()
        self.embedder = SentenceTransformer(EMBEDDER, device="cuda")

    @modal.exit()
    def settle(self):
        _charge(max(0.0, time.time() - self.started - self.charged))

    @modal.method()
    def sample_pool(self, prompt, n, t_prime, max_new_tokens, seed):
        import base64
        import numpy as np
        t0 = time.time()
        try:
            samples = sample(self.tok, self.model, CHAT_KW, prompt, n, t_prime, max_new_tokens, 100, seed)
            emb = self.embedder.encode([s["text"] or "(empty)" for s in samples], normalize_embeddings=True)
            emb = np.asarray(emb, dtype=np.float32)
        finally:
            dt = time.time() - t0
            self.charged += dt
            _charge(dt)
        return {
            "prompt": prompt, "model": MODEL_REPO, "modelName": MODEL_REPO.split("/")[-1], "embedder": EMBEDDER,
            "Tprime": t_prime, "maxNewTokens": max_new_tokens, "N": len(samples), "D": int(emb.shape[1]),
            "samples": samples, "emb": base64.b64encode(emb.tobytes()).decode("ascii"),
        }


def sample(tok, model, chat, prompt, n, t_prime, max_new_tokens, batch_size, seed):
    """Exact ancestral sampling from p°_{T'} with per-string log p and log q (as in space/app.py)."""
    import inspect
    import torch
    with torch.no_grad():
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, **chat)
        dev = next(model.parameters()).device
        prompt_ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
        ids = model.generation_config.eos_token_id
        ids = ids if isinstance(ids, (list, tuple)) else [ids if ids is not None else tok.eos_token_id]
        eos = {int(i) for i in ids if i is not None}
        eos_t = torch.tensor(sorted(eos), device=dev)
        gen = torch.Generator(device=dev).manual_seed(int(seed))
        cache_kw = "cache" if "cache" in inspect.signature(type(model).forward).parameters else "past_key_values"
        out = []
        while len(out) < n:
            B = min(batch_size, n - len(out))
            logp = torch.zeros(B, dtype=torch.float64, device=dev)
            logq = torch.zeros(B, dtype=torch.float64, device=dev)
            done = torch.zeros(B, dtype=torch.bool, device=dev)
            ntok = torch.zeros(B, dtype=torch.long, device=dev)
            steps, past, step_in = [], None, prompt_ids.repeat(B, 1)
            for _ in range(max_new_tokens):
                res = model(input_ids=step_in, use_cache=True, **{cache_kw: past})
                past = getattr(res, "cache", None) or getattr(res, "past_key_values", None)
                logits = res.logits[:, -1, :].float()
                lp1 = torch.log_softmax(logits, dim=-1)            # log p(. | prefix)
                lpT = torch.log_softmax(logits / t_prime, dim=-1)  # log p°_{T'}(. | prefix)
                nxt = torch.multinomial(lpT.exp(), 1, generator=gen).squeeze(1)
                active = ~done
                nxt = torch.where(active, nxt, eos_t[0])
                logp += torch.where(active, lp1.gather(1, nxt[:, None]).squeeze(1).double(), 0.0)
                logq += torch.where(active, lpT.gather(1, nxt[:, None]).squeeze(1).double(), 0.0)
                ntok += active.long()
                steps.append(nxt)
                done |= active & torch.isin(nxt, eos_t)
                if bool(done.all()):
                    break
                step_in = nxt[:, None]
            toks = torch.stack(steps, 1).cpu().tolist()
            ntok, done, logp, logq = ntok.cpu().tolist(), done.cpu().tolist(), logp.cpu().tolist(), logq.cpu().tolist()
            for b in range(B):
                seq = toks[b][: ntok[b]]
                out.append({
                    "text": tok.decode([t for t in seq if t not in eos], skip_special_tokens=True).strip(),
                    "logp": logp[b], "logq": logq[b], "nTokens": ntok[b], "ended": bool(done[b]),
                })
        return out


def hf_user(token):
    """The Hugging Face username a token belongs to, or None."""
    import json
    import urllib.request
    for url in ("https://huggingface.co/oauth/userinfo", "https://huggingface.co/api/whoami-v2"):
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
            with urllib.request.urlopen(req, timeout=10) as r:
                info = json.load(r)
            name = info.get("preferred_username") or info.get("name")
            if name:
                return name
        except Exception:
            continue
    return None


@app.function(image=web_image, scaledown_window=60, timeout=900)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    import asyncio
    import random
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware

    api = FastAPI()
    api.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"],
                       allow_headers=["authorization", "content-type"])

    async def budget_left():
        return GPU_SECONDS_PER_MONTH - await usage.get.aio(f"gpu-seconds:{_month()}", 0)

    @api.get("/status")
    async def status():
        return {"model": MODEL_KEY, "gpu_minutes_left_this_month": round(max(0, await budget_left()) / 60)}

    @api.post("/sample")
    async def sample_endpoint(req: Request):
        auth = req.headers.get("authorization", "")
        token = auth[7:] if auth.lower().startswith("bearer ") else ""
        user = await asyncio.to_thread(hf_user, token) if token else None
        if not user:
            raise HTTPException(401, "Sign in with Hugging Face to use this model.")
        if await budget_left() <= 0:
            raise HTTPException(429, "This month's GPU budget for the demo is used up. Please use an on-device model.")
        day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        key = f"runs:{day}:{user}"
        runs = await usage.get.aio(key, 0)
        if runs >= RUNS_PER_USER_PER_DAY:
            raise HTTPException(429, f"You have used today's {RUNS_PER_USER_PER_DAY} runs. Please come back tomorrow.")
        await usage.put.aio(key, runs + 1)
        body = await req.json()
        prompt = str(body.get("prompt", "")).strip()[:2000]
        if not prompt:
            raise HTTPException(400, "The prompt is empty.")
        n = max(2, min(int(body.get("n", 30)), MAX_N))
        t_prime = min(max(float(body.get("t_prime", 1.0)), 0.2), 2.0)
        max_new = max(8, min(int(body.get("max_new_tokens", 128)), MAX_TOKENS))
        seed = int(body.get("seed", random.randrange(2**31)))
        print(f"run by {user}: n={n} T'={t_prime} max={max_new}")
        return await Sampler().sample_pool.remote.aio(prompt, n, t_prime, max_new, seed)

    return api
