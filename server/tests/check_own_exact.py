"""GPU check of the exact own states (the PR's recovery), in one vLLM process with the plugin loaded (stop the service first).

Usage: DPP_OWN_DIR=<dir with qwen3-14b/{meta.json,M.npy}> .venv/bin/python tests/check_own_exact.py [UTIL] [OUT.json]
Run it with the repo's search/ on the path (build_own_P.find_W reads W) and DPP_ROOT set as for build_own_M.py.

1. What the sampler hook sees. `Own.step_pre` is wrapped: for every step of a single-sequence request it keeps the logits row it
   receives (dtype, shape, NaN / inf / -inf counts over the whole tensor) and the final-norm hidden state h that produced it (a
   forward pre-hook on the model's logits processor). Then, for each step row l, with A = [W | 1] and M = (A^T A)^-1:
   coef = M A^T l, fit = A coef, residual l - fit. Reported relative to |l|. At T' != 1 the hook logits are checked not to be
   divided by T' by reproducing the reported log p (raw) and log q (annealed) of the chosen tokens from log_softmax of the hook logits.
2. Recovery against the real hidden states: the first d entries of coef (per step) and of the mean over steps (what the server
   computes from the plugin's sums) against h and its mean over the same steps, taken inside the same forward pass.
"""
import json, os, sys, time
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # the engine runs in this process, so the hooks below see its tensors
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
os.environ.setdefault("DPP_LOGQ", "1")
os.environ.setdefault("DPP_OWN_EMB", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "search"))
import numpy as np
import torch

UTIL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.60
OUT = sys.argv[2] if len(sys.argv) > 2 else "own_exact_check.json"
REPO = "Qwen/Qwen3-14B-AWQ"
from build_own_P import find_W
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer, GenerationConfig
import dpp_vllm_plugin.own as own_mod

own_dir = os.environ["DPP_OWN_DIR"]
M = np.load(os.path.join(own_dir, "qwen3-14b", "M.npy"))                       # (d+1) x (d+1) float64
g, name, info = find_W(REPO)
W = g.get_tensor(name)
W = (W.float().numpy() if hasattr(W, "float") else np.asarray(W)).astype(np.float64)   # V x d
V, d = W.shape
print("W", W.shape, "M", M.shape, flush=True)

CAP = {"steps": [], "dtype": None, "shape": None, "nonfinite": 0, "neginf": 0, "nan": 0, "posinf": 0, "calls": 0}
H = []


def wrap_step_pre():
    orig = own_mod.Own.step_pre

    def step_pre(self, logits, runner):
        CAP["calls"] += 1
        CAP["dtype"], CAP["shape"] = str(logits.dtype), list(logits.shape)
        z = logits.float()
        CAP["nan"] += int(torch.isnan(z).sum()); CAP["neginf"] += int(torch.isneginf(z).sum()); CAP["posinf"] += int(torch.isposinf(z).sum())
        CAP["steps"].append(z[0].cpu().double().numpy())          # single-sequence batches: row 0 is the own request
        return orig(self, logits, runner)
    own_mod.Own.step_pre = step_pre


def install_hidden_hook(model):
    def hook(mod, args):
        H.append(args[1].detach().float().cpu().double().numpy())    # (rows, d): hidden states that go into lm_head
    model.logits_processor.register_forward_pre_hook(hook)
    return True


tok = AutoTokenizer.from_pretrained(REPO)
eos = GenerationConfig.from_pretrained(REPO).eos_token_id
eos = eos if isinstance(eos, list) else [eos]
prompts = ["Give one creative use for an old bicycle, in one sentence.", "Name a color and say why.", "Write a two-line poem about rain.",
           "What is 17 times 23? Answer briefly.", "Suggest a name for a pet turtle.", "Tell me a short joke about computers."]
llm = LLM(REPO, gpu_memory_utilization=UTIL, max_model_len=1024, max_num_seqs=8, logprobs_mode="raw_logprobs", generation_config="vllm",
          enable_prefix_caching=False, enforce_eager=True)
wrap_step_pre()
llm.apply_model(install_hidden_hook)

runs = []
for ti, Tp in enumerate((1.0, 1.0, 1.5, 0.6, 1.5, 1.0)):
    text = tok.apply_chat_template([{"role": "user", "content": prompts[ti]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pids = tok(text, add_special_tokens=False).input_ids
    tag = f"chk{ti}"
    CAP["steps"].clear(); H.clear()
    sp = SamplingParams(n=1, temperature=Tp, top_p=1.0, top_k=0, min_p=0.0, max_tokens=60, logprobs=1, seed=ti, stop_token_ids=sorted(eos),
                        detokenize=False, extra_args={"dpp_own": tag, "dpp_max": 60, "dpp_stop": sorted(set(eos) | {tok.eos_token_id})})
    out = llm.generate(TokensPrompt(prompt_token_ids=pids), sp, use_tqdm=False)[0].outputs[0]
    take = llm.collective_rpc("dpp_own_take", args=([[tag, 1]],))[0][0]
    cnt = take["count"][0]
    s = np.frombuffer(take["sum"], dtype="<f4").astype(np.float64)
    toks = list(out.token_ids)
    L = np.stack(CAP["steps"][:cnt]); Hs = np.concatenate(H, 0)[:cnt]
    lp_reported = np.array([dd[t].logprob for t, dd in zip(toks, out.logprobs)])
    lq_reported = np.array([next(v.logprob for k, v in dd.items() if k != t) for t, dd in zip(toks, out.logprobs)])
    n = min(len(toks), cnt)
    Lt = L[:n]
    lsm = lambda x: x - (np.log(np.exp(x - x.max(1, keepdims=True)).sum(1, keepdims=True)) + x.max(1, keepdims=True))
    lp_hook = lsm(Lt)[np.arange(n), toks[:n]]
    lq_hook = lsm(Lt / Tp)[np.arange(n), toks[:n]]
    # least squares per step row
    Y = np.concatenate([Lt @ W, Lt.sum(1, keepdims=True)], 1)                    # rows of (A^T l)^T
    coef = Y @ M                                                                 # (M A^T l)^T, M symmetric
    fit = coef[:, :d] @ W.T + coef[:, d:d + 1]
    res = Lt - fit
    rel_res = np.linalg.norm(res, axis=1) / np.linalg.norm(Lt, axis=1)
    rel_res_c = np.linalg.norm(res, axis=1) / np.linalg.norm(Lt - Lt.mean(1, keepdims=True), axis=1)
    h_ref = Hs[:n]
    h_err = np.linalg.norm(coef[:, :d] - h_ref, axis=1) / np.linalg.norm(h_ref, axis=1)
    # check that W h_ref reproduces l up to the logits dtype, and what the per-step constant is
    wh = h_ref @ W.T
    c_est = (Lt - wh).mean(1)
    rel_wh = np.linalg.norm(Lt - wh - c_est[:, None], axis=1) / np.linalg.norm(Lt, axis=1)
    # the mean over steps, as the server computes it from the plugin sums
    lbar = s / cnt
    ybar = np.concatenate([lbar @ W, [lbar.sum()]])
    hbar = ybar @ M[:, :d]
    href = h_ref.mean(0)
    runs.append(dict(prompt=prompts[ti], Tprime=Tp, steps=int(cnt), tokens=len(toks), ended=bool(toks and toks[-1] in eos),
                     plugin_sum_vs_hook_mean_rel=float(np.linalg.norm(lbar - Lt.mean(0)) / np.linalg.norm(Lt.mean(0))),
                     max_abs_logp_vs_hook=float(np.abs(lp_hook - lp_reported[:n]).max()), max_abs_logq_vs_hook=float(np.abs(lq_hook - lq_reported[:n]).max()),
                     max_abs_logp_vs_hook_if_divided=float(np.abs(lsm(Lt / Tp)[np.arange(n), toks[:n]] - lp_reported[:n]).max()),
                     ls_residual_rel_max=float(rel_res.max()), ls_residual_rel_mean=float(rel_res.mean()), ls_residual_rel_centered_max=float(rel_res_c.max()),
                     wh_vs_l_rel_max=float(rel_wh.max()), c_mean_abs_max=float(np.abs(c_est).max()),
                     step_h_rel_err_max=float(h_err.max()), step_h_rel_err_mean=float(h_err.mean()),
                     mean_h_rel_err=float(np.linalg.norm(hbar - href) / np.linalg.norm(href)), hbar=hbar, href=href))
    print(json.dumps({k: v for k, v in runs[-1].items() if k not in ("hbar", "href")}), flush=True)

hb = np.stack([r["hbar"] for r in runs]); hr = np.stack([r["href"] for r in runs])
mu = hr.mean(0)


def gram(X):
    X = X - mu
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X @ X.T
dG = np.abs(gram(hb) - gram(hr))
summary = dict(logits_dtype=CAP["dtype"], logits_shape=CAP["shape"], hook_calls=CAP["calls"], nan=CAP["nan"], neginf=CAP["neginf"], posinf=CAP["posinf"],
               d=d, V=V, runs=len(runs), max_G_diff_recovered_vs_reference=float(dG.max()),
               max_over_runs={k: max(r[k] for r in runs) for k in ("ls_residual_rel_max", "ls_residual_rel_centered_max", "wh_vs_l_rel_max", "step_h_rel_err_max",
                                                                    "mean_h_rel_err", "max_abs_logp_vs_hook", "max_abs_logq_vs_hook", "plugin_sum_vs_hook_mean_rel", "c_mean_abs_max")},
               min_over_runs_logp_if_divided=min(r["max_abs_logp_vs_hook_if_divided"] for r in runs if r["Tprime"] != 1.0))
print("SUMMARY", json.dumps(summary), flush=True)
json.dump(dict(summary=summary, runs=[{k: v for k, v in r.items() if k not in ("hbar", "href")} for r in runs]), open(OUT, "w"), indent=1)
print("DONE", flush=True)
