"""Check the algebra of exact own states on a small random "model" (numpy only, no GPU).

The sampler sees, at each generated step t, logits l_t = W h_t + c_t·1 (W: V x d output matrix, h_t the
final hidden state, c_t any per-step constant such as a log-softmax shift). With A = [W | 1] and
M = (AᵀA)⁻¹, the first d entries of M Aᵀ l are h. Because this is linear in l, the mean hidden state of a
sequence is recovered from the mean of its logits:  h̄ = [M Aᵀ l̄]_{1..d}.  So the sampler only has to sum
logits; the recovery happens once per sequence at the end, with no projection.

Run: python3 own_exact_check.py
"""
import numpy as np

rng = np.random.default_rng(0)
V, d, r, n, T = 3000, 160, 16, 40, 25          # vocab, hidden size, projection rank (today: 256 of 5120), sequences, steps
W = rng.standard_normal((V, d)) / np.sqrt(d)
A = np.concatenate([W, np.ones((V, 1))], 1)
M = np.linalg.inv(A.T @ A)

# per-sequence hidden states: a shared direction plus a sequence-specific one plus per-step noise
base = rng.standard_normal(d)
H = base + 0.6 * rng.standard_normal((n, d))[:, None, :] + 0.3 * rng.standard_normal((n, T, d))
L = H @ W.T
L = L - np.log(np.exp(L - L.max(-1, keepdims=True)).sum(-1, keepdims=True)) - L.max(-1, keepdims=True)   # log-softmax: a per-step shift

hbar_true = H.mean(1)
lbar = L.mean(1)                                    # what the sampler accumulates (sum / count)
hbar = (lbar @ A @ M)[:, :d]                         # M symmetric: (M Aᵀ l̄)ᵀ = l̄ A M
print(f"recovery of the mean hidden state: max rel error {np.abs(hbar - hbar_true).max() / np.abs(hbar_true).max():.1e}")

def gram(X, mu):
    X = X - mu
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X @ X.T

mu = hbar_true.mean(0)
G_exact = gram(hbar, mu)
Q, _ = np.linalg.qr(rng.standard_normal((d, r)))
G_proj = gram(hbar @ Q, mu @ Q)                       # today's method: a random r-dim projection first
off = ~np.eye(n, dtype=bool)
print(f"G from exact states vs G from the true states: max |diff| {np.abs(G_exact - gram(hbar_true, mu)).max():.1e}")
print(f"G from a random {r}-of-{d} projection vs exact: max |diff| {np.abs(G_proj - G_exact)[off].max():.2f}, mean {np.abs(G_proj - G_exact)[off].mean():.2f}")
