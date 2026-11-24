// Pure helpers for the expert tools (no DOM, so Node can test them). Moved from the old page's
// js/shared.js and js/backends.js; each is used by a tool on the port list (see server/API.md for the
// server side). Nothing here is imported by the default page.

import { columnLogScale, ess, eigSym } from './dpp.js';

/**
 * Unit vectors whose Gram matrix is G (N×N, flat, symmetric, unit diagonal): row i is V·√Λ at i from the
 * eigendecomposition G = VΛVᵀ, negative eigenvalues set to 0, then renormalized. A kernel stored only as a
 * Gram matrix (the LM's activations) gets the vectors the PCA, clustering and matrices need.
 * Returns N Float32Array rows of equal length (the number of positive eigenvalues, at least 1).
 */
export function gramToVectors(G, N) {
  const { values, vectors } = eigSym(G, N);
  const top = Math.max(values[0] ?? 0, 0), tol = 1e-9 * Math.max(top, 1e-300);
  const keep = [];
  for (let k = 0; k < N; k++) if (values[k] > tol) keep.push(k);
  if (!keep.length) keep.push(0);
  return Array.from({ length: N }, (_, i) => {
    const r = new Float32Array(keep.length);
    let n = 0;
    keep.forEach((k, j) => { const x = Math.sqrt(Math.max(values[k], 0)) * vectors[k][i]; r[j] = x; n += x * x; });
    n = Math.sqrt(n);
    if (n > 0) for (let j = 0; j < r.length; j++) r[j] /= n;
    return r;
  });
}

/** The paper's DPP weight for a picking mode: p^(1/T) for 'map' (quality), p^(1/T)/q for 'sample' (importance). */
export const paperWeight = (mode) => (mode === 'map' ? 'quality' : 'importance');

/**
 * The weight family the DPP kernel uses: 'quality' (w = p^(1/T)), 'importance' (w = p^(1/T)/q) and 'unit' (w = 1) are themselves;
 * anything else (unknown, missing) behaves as the picking mode's paper weight.
 */
export const resolveWeight = (weight, mode) => (['quality', 'importance', 'unit'].includes(weight) ? weight : paperWeight(mode));


/**
 * Readout notes: low effective sample size (Kish ESS of the weight in use below K) and few distinct completions (fewer than max(K, N/10)).
 * With `near` ({ size }: the largest group at a tight cosine threshold) it adds a note when half the pool is one answer.
 * Returns { kind: 'ess' | 'distinct' | 'near' | 'one', text }[].
 */
export function readoutNotes({ ess, K, N, distinct, near = null }) {
  const out = [];
  if (Number.isFinite(ess) && ess < K) out.push({ kind: 'ess', text: `Weights are concentrated on about ${Math.max(1, Math.round(ess))} completions, so sets barely vary.` });
  if (distinct < Math.max(K, N / 10)) out.push({ kind: 'distinct', text: `Only ${distinct} distinct completions in this pool. Try a higher sampling temperature T′.` });
  if (near && !out.some((n) => n.kind === 'distinct') && near.size >= 0.5 * N && near.size >= 3) out.push({ kind: 'near', text: `Most completions (${near.size} of ${N}) are near-duplicates of one answer. Try a higher sampling temperature T′.` });
  if (distinct === K && distinct >= 1) out.push({ kind: 'one', text: `Only one set of ${K} distinct completions exists, so Resample and T change nothing.` });
  return out;
}

/** A completion's text with runs of whitespace collapsed and the ends trimmed, so strings that differ only in spacing compare equal. */
export const normText = (t) => String(t).replace(/\s+/g, ' ').trim();

/** Number of distinct completions, whitespace-insensitive. */
export const distinctCount = (samples) => new Set(samples.map((x) => normText(x.text))).size;

/** The K the picker can use: at most the number of distinct completions and N - 1 (at least 1). */
export const effectiveK = (K, distinct, N) => Math.max(1, Math.min(K, distinct, N - 1));

/**
 * log w_i for the kernel L_ij = √w_i φ_i·φ_j √w_j, what columnLogScale returns for the resolved family:
 * quality → the 'map' column scale, importance → the 'sample' column scale, unit → zeros. With the mode's paper weight it equals
 * columnLogScale(mode, ...).
 */
export function weightLogScale(weight, mode, logp, logq, T) {
  const fam = resolveWeight(weight, mode);
  if (fam === 'unit') return logp.map(() => 0);
  return columnLogScale(fam === 'quality' ? 'map' : 'sample', logp, logq, T);
}

/**
 * Kish effective sample size (Σw)²/Σw² of the weights the DPP uses. Every weight is target / q for a pool drawn from q:
 * importance p^(1/T)/q (target p^(1/T)), quality p^(1/T) q/q (target q·p^(1/T)), unit q/q = 1 (target q, so the ESS is N).
 */
export const weightEss = (weight, mode, logp, logq, T) => ess(weightLogScale(weight, mode, logp, logq, T));

/**
 * The paper's Ambrosia operating points (K = 3, N = 200). `Ts` is the sweep of T; `T` is the value a click on the preset sets. `weight` and `lock` are applied on click: the paper's DPP weight (MAP: quality, p^(1/T); Samp: importance, p^(1/T)/q) with the display tied to it. The match requires the resolved DPP weight to be the paper's for the preset's mode.
 * PoolDPP-MAP: pools drawn at T′ = 1.5. PoolDPP-Samp: pools drawn at T′ = 1.0.
 */
export const PRESETS = {
  map: { label: 'MAP, paper', mode: 'map', Tprime: 1.5, K: 3, T: 3, Ts: [1, 3, 7.5, 10], weight: 'quality', lock: true },
  samp: { label: 'Samp, paper', mode: 'sample', Tprime: 1.0, K: 3, T: 0.82, Ts: [0.43, 0.54, 0.67, 0.82, 1], weight: 'importance', lock: true },
};

/**
 * Key of the preset whose picking mode, T′ and K match and whose grid contains T (relative tolerance 1e-6), or null.
 * `weight` (optional) is the DPP weight in use; it must equal the preset's weight.
 */
export function matchPreset({ mode, Tprime, K, T, weight }) {
  for (const [key, p] of Object.entries(PRESETS)) {
    if (weight !== undefined && weight !== p.weight) continue;
    if (p.mode === mode && Number(Tprime) === p.Tprime && K === p.K && p.Ts.some((t) => Math.abs(t - T) <= 1e-6 * t)) return key;
  }
  return null;
}

/**
 * Which settings a preset (or a Paper T chip) changes, in reading order: 'mode' (picking mode), 'T', 'K', 'tp' (the pool's T′),
 * 'weight' (DPP weight choice), 'len' (vector length shown), 'lock'. before and after: { mode, T, K, tp, weight, len, lock }.
 * T compares with relative tolerance 1e-9, tp as numbers.
 */
export function presetDiff(before, after) {
  const same = {
    mode: (a, b) => a === b,
    T: (a, b) => Math.abs(a - b) <= 1e-9 * Math.max(Math.abs(a), Math.abs(b)),
    K: (a, b) => a === b,
    tp: (a, b) => Number(a) === Number(b),
    weight: (a, b) => a === b,
    len: (a, b) => a === b,
    lock: (a, b) => a === b,
  };
  return Object.keys(same).filter((k) => !same[k](before[k], after[k]));
}

/**
 * Display lengths c_i ∈ (0, 1] of the points, a display choice independent of the picking mode.
 * 'unit': all 1. 'quality': √κ_i / max √κ_j with κ = p^(1/T), in every mode. 'importance': √(κ_i/q_i) / max, in every mode.
 * Unknown choices resolve as the picking mode's paper weight. Same convention as dpp.js kernel().
 */
export function displayScale(choice, mode, logp, logq, T) {
  if (choice === 'unit') return logp.map(() => 1);
  const ls = weightLogScale(choice, mode, logp, logq, T);
  const m = Math.max(...ls);
  return ls.map((s) => Math.exp(0.5 * (s - m)));
}


// ---------- server requests ----------

/** A random request id for /sample, /embed and /describe: 32 hex characters (the server accepts 1 to 64 of [A-Za-z0-9_-]). */
export function newRequestId(cryptoObj = globalThis.crypto) {
  const bytes = cryptoObj.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, (x) => x.toString(16).padStart(2, '0')).join('');
}

/** Status line for a /queue answer while a request waits: null when it is not waiting (the page shows its running text). */
export function queueText(q) {
  if (q?.state !== 'waiting') return null;
  const n = Math.max(0, Math.round(Number(q.ahead) || 0));
  return n > 0 ? `Waiting in line, ${n} ahead.` : 'Waiting in line, next up.';
}

/** Response of POST /describe -> { names: string[] (one per group, '' when the model gave none), model, ms }. */
export function decodeDescribe(body, nGroups) {
  if (!body || !Array.isArray(body.names) || body.names.length !== nGroups) throw new Error('The server returned the wrong number of group names.');
  return { names: body.names.map((x) => (typeof x === 'string' ? x.trim() : '')), model: body.model ?? null, ms: body.ms ?? null };
}
