"""Sampling backend for the diverse-sampling demo (a Hugging Face Space).

Draws N completions i.i.d. from the locally annealed proposal q = p°_{T'} and returns,
for each one, the exact log p(y) and log q(y) (end-of-string included), plus unit-norm
embeddings. The demo page does the DPP part in the browser.

Runs on free CPU hardware as is; on ZeroGPU the @spaces.GPU decorator attaches a GPU.
"""
# On ZeroGPU, `spaces` must be imported before torch initializes CUDA.
try:
    import spaces
except Exception:  # running locally
    spaces = None

import base64
import inspect
import os

import gradio as gr
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

def _duration(prompt, model_key, n, t_prime, max_new_tokens, seed=0):
    # GPU seconds to reserve: a little for setup plus the decoding steps.
    return int(min(120, 15 + 0.25 * int(max_new_tokens) * (1 + int(n) // 100)))


if spaces is not None:
    gpu = spaces.GPU(duration=_duration)  # a no-op outside ZeroGPU
else:
    def gpu(f):
        return f

try:  # the spaces package knows whether this Space runs on ZeroGPU
    from spaces.config import Config as _SpacesConfig
    ON_ZEROGPU = bool(_SpacesConfig.zero_gpu)
except Exception:
    ON_ZEROGPU = os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "true")
# On ZeroGPU the GPU only exists inside @spaces.GPU calls; don't probe CUDA at import time there.
DEVICE = "cuda" if ON_ZEROGPU or torch.cuda.is_available() else "cpu"

# key -> (repo, chat-template options). On ZeroGPU the Space serves one large model,
# loaded onto the GPU at startup (ZeroGPU wants models placed on cuda at module level).
SMALL = {
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", {"enable_thinking": False}),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", {"enable_thinking": False}),
}
LARGE = {
    "qwen3-14b": ("Qwen/Qwen3-14B-AWQ", {"enable_thinking": False}),   # 4-bit: the demo server's weights
}
MODELS = LARGE if ON_ZEROGPU else SMALL
DEFAULT = os.environ.get("MODEL_KEY", next(iter(MODELS)))
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"

_cache = {}


def unpack_awq(qweight, qzeros, scales, group_size, bits=4):
    """AWQ "gemm" packing (AutoAWQ): 8 4-bit values per int32, in the order 0 2 4 6 1 3 5 7.
    Returns the linear layer's weight [out, in] = ((q - zero) * scale)^T, in the scales' dtype."""
    import torch
    shifts = torch.arange(0, 32, bits, device=qweight.device)
    def unpack(x):
        v = torch.bitwise_right_shift(x[:, :, None], shifts[None, None, :]).view(x.shape[0], -1)
        order = torch.arange(v.shape[1], device=x.device).view(-1, 32 // bits)[:, [0, 4, 1, 5, 2, 6, 3, 7]].reshape(-1)
        return torch.bitwise_and(v[:, order], 2**bits - 1)
    q, z = unpack(qweight), unpack(qzeros)
    w = (q - z.repeat_interleave(group_size, 0)).to(scales.dtype) * scales.repeat_interleave(group_size, 0)
    return w.t().contiguous()


def load_awq_unpacked(repo, device, path=None):
    """The AWQ checkpoint as a plain fp16 model: every 4-bit weight unpacked to (q - zero) * scale.
    These are exactly the 4-bit model's weights; only the matrix kernels differ from an int4 engine.
    The model is built on `device` without random initialisation, then filled tensor by tensor."""
    import glob, json, os
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.initialization import no_init_weights
    if path is None:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo)
    qc = json.load(open(os.path.join(path, "config.json")))["quantization_config"]
    assert qc["quant_method"] == "awq" and qc["version"] == "gemm" and qc["bits"] == 4 and qc["zero_point"]
    cfg = AutoConfig.from_pretrained(path)
    cfg.__dict__.pop("quantization_config", None)
    with no_init_weights(), torch.device(device):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float16)
    params = dict(model.named_parameters())
    done, packed = set(), {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, framework="pt", device=str(device)) as st:
            for k in st.keys():
                if k.endswith((".qweight", ".qzeros", ".scales")):
                    base, part = k.rsplit(".", 1)
                    packed.setdefault(base, {})[part] = st.get_tensor(k)
                    t = packed[base]
                    if len(t) == 3:
                        w = unpack_awq(t["qweight"], t["qzeros"], t["scales"].to(torch.float16), qc["group_size"])
                        params[base + ".weight"].data.copy_(w)
                        done.add(base + ".weight")
                        del packed[base], w
                else:
                    params[k].data.copy_(st.get_tensor(k))
                    done.add(k)
    if getattr(cfg, "tie_word_embeddings", False):
        model.tie_weights()
        done |= {k for k in params if k.startswith("lm_head")}
    missing = set(params) - done
    assert not missing and not packed, (sorted(missing)[:5], list(packed)[:5])
    return model.eval()


def load(key):
    if key not in _cache:
        repo, chat = MODELS[key]
        tok = AutoTokenizer.from_pretrained(repo)
        on_gpu = ON_ZEROGPU or DEVICE == "cuda"
        if repo.endswith("-AWQ"):   # unpacked on the CPU; ZeroGPU moves it to the GPU below
            model = load_awq_unpacked(repo, "cpu")
        else:
            model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16 if on_gpu else torch.float32)
        model.eval()
        if on_gpu:
            model.to("cuda")
        _cache[key] = (tok, model, chat)
    return _cache[key]


if ON_ZEROGPU:
    load(DEFAULT)


_embedder = SentenceTransformer(EMBEDDER, device="cpu")


def eos_ids(tok, model):
    ids = model.generation_config.eos_token_id
    if ids is None:
        ids = tok.eos_token_id
    ids = ids if isinstance(ids, (list, tuple)) else [ids]
    return {int(i) for i in ids if i is not None}


@torch.no_grad()
def sample(tok, model, chat, prompt, n, t_prime, max_new_tokens, batch_size, seed, keep_ids=False):
    """Exact ancestral sampling from p°_{T'} with per-string log p and log q.

    Everything stays on the model's device; only the chosen tokens are copied back.
    """
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, **chat)
    dev = next(model.parameters()).device
    prompt_ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    eos = eos_ids(tok, model)
    eos_t = torch.tensor(sorted(eos), device=dev)
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    # transformers >= 5 renamed past_key_values to cache.
    cache_kw = "cache" if "cache" in inspect.signature(type(model).forward).parameters else "past_key_values"
    out = []
    while len(out) < n:
        B = min(batch_size, n - len(out))
        logp = torch.zeros(B, dtype=torch.float64, device=dev)
        logq = torch.zeros(B, dtype=torch.float64, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        ntok = torch.zeros(B, dtype=torch.long, device=dev)
        steps = []
        past, step_in = None, prompt_ids.repeat(B, 1)
        for _ in range(max_new_tokens):
            res = model(input_ids=step_in, use_cache=True, **{cache_kw: past})
            past = getattr(res, "cache", None) or getattr(res, "past_key_values", None)
            assert past is not None, "the model returned no KV cache"
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
            ids = toks[b][: ntok[b]]
            out.append({
                "text": tok.decode([t for t in ids if t not in eos], skip_special_tokens=True).strip(),
                "logp": logp[b], "logq": logq[b], "nTokens": ntok[b], "ended": bool(done[b]),
                **({"ids": ids} if keep_ids else {}),
            })
    return out


@gpu
def run(prompt, model_key, n, t_prime, max_new_tokens, seed=0):
    tok, model, chat = load(model_key)
    batch = 100 if ON_ZEROGPU or DEVICE == "cuda" else 5
    return sample(tok, model, chat, prompt, int(n), float(t_prime), int(max_new_tokens), batch, seed)


def sample_pool(prompt: str, model: str = "", n: int = 20, t_prime: float = 1.0,
                max_new_tokens: int = 64, seed: int = 0):
    """Draw a pool of completions with exact log p / log q and MiniLM embeddings."""
    model = model or DEFAULT
    if model not in MODELS:
        raise gr.Error(f"Unknown model {model!r}. Available: {', '.join(MODELS)}")
    if not prompt.strip():
        raise gr.Error("The prompt is empty.")
    n = max(2, min(int(n), 200 if ON_ZEROGPU else 40))
    max_new_tokens = max(8, min(int(max_new_tokens), 256))
    t_prime = min(max(float(t_prime), 0.2), 2.0)
    try:
        samples = run(prompt, model, n, t_prime, max_new_tokens, seed)
    except gr.Error:
        raise
    except Exception as e:  # pass the reason on to API callers instead of a bare "error"
        print(f"sample_pool failed: {type(e).__name__}: {e}", flush=True)
        raise gr.Error(f"{type(e).__name__}: {e}")
    emb = _embedder.encode([s["text"] or "(empty)" for s in samples], normalize_embeddings=True)
    emb = np.asarray(emb, dtype=np.float32)
    repo = MODELS[model][0]
    return {
        "prompt": prompt, "model": repo, "modelName": repo.split("/")[-1], "embedder": EMBEDDER,
        "Tprime": t_prime, "maxNewTokens": max_new_tokens, "N": len(samples), "D": int(emb.shape[1]),
        "samples": samples, "emb": base64.b64encode(emb.tobytes()).decode("ascii"),
    }


def models():
    """The models this Space offers."""
    return {k: v[0] for k, v in MODELS.items()}


with gr.Blocks(title="Diverse sampling backend") as demo:
    gr.Markdown("Backend for the [diverse-sampling demo](https://timvieira.github.io/dpp-lm-demo/). "
                "Returns completions with exact log-probabilities and embeddings as JSON.")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", value="Give one creative use for an old bicycle, in one sentence.")
        model = gr.Dropdown(choices=list(MODELS), value=DEFAULT, label="Model", allow_custom_value=True)
    with gr.Row():
        n = gr.Slider(2, 200, value=10, step=1, label="Samples")
        t_prime = gr.Slider(0.2, 2.0, value=1.0, step=0.1, label="Temperature T′")
        max_new = gr.Slider(8, 256, value=48, step=8, label="Max tokens")
        seed = gr.Number(value=0, precision=0, label="Seed")
    out = gr.JSON(label="Pool")
    gr.Button("Sample").click(sample_pool, [prompt, model, n, t_prime, max_new, seed], out, api_name="sample")
    gr.Button("List models", visible=False).click(models, None, out, api_name="models")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(show_error=True)
