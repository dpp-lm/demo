// Pure helpers for the expert tools (no DOM, so Node can test them). Moved from the old page's
// js/shared.js and js/backends.js; each is used by a tool on the port list (see server/API.md for the
// server side). Nothing here is imported by the default page.

import { columnLogScale, ess, eigSym } from './dpp.js';
import { clusterLinkage, cutClusters, clusterPool, clusterKeywords, distFromGram } from './layout.js';
import { seriation, medoidOf } from './matrices.js';

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

// ---------- what the GPU server offers beyond the default page ----------

/** "Qwen/Qwen3-Embedding-4B" -> "Qwen3-Embedding-4B": the repository's short name, used as the plain label. */
export const repoLabel = (repo) => String(repo).split('/').pop();

/**
 * Embedders and models that a GET /status body lists and the default page does not offer.
 * `known`: embedder ids the page already has. `defKey`: the model key the page already has.
 * Returns { embedders: [{ id, repo, label }], models: [{ key, name, repo, state, embedders: [id] }] }.
 * `embedders` (the extra ones) lists only entries with a repository (the server's `own: null` is not one).
 * A model is listed when the server reports more than one and its state is not 'unavailable'; its `embedders`
 * are the ids it can be used with: every embedder of the server, and `own` only if `ownModels` has the model.
 */
export function serverExtras(status, { known = ['qwen3-emb', 'own'], defKey = 'qwen3-14b' } = {}) {
  const emb = status?.embedders && typeof status.embedders === 'object' && !Array.isArray(status.embedders) ? status.embedders : {};
  const embedders = Object.entries(emb)
    .filter(([id, repo]) => typeof repo === 'string' && repo && !known.includes(id))
    .map(([id, repo]) => ({ id, repo, label: repoLabel(repo) }));
  const list = Array.isArray(status?.models) ? status.models : [];
  const own = Array.isArray(status?.ownModels) ? status.ownModels : [];
  const models = list.length < 2 ? [] : list
    .filter((m) => m && typeof m.key === 'string' && m.key !== defKey && m.state !== 'unavailable')
    .map((m) => ({
      key: m.key, name: m.name ?? m.key, repo: m.repo ?? m.key, state: m.state ?? 'ready',
      embedders: [...known.filter((id) => id !== 'own' && id in emb), ...embedders.map((e) => e.id), ...(known.includes('own') && own.includes(m.key) ? ['own'] : [])],
    }));
  return { embedders, models };
}


// ---------- groups: coloring the completions by cluster (a reading aid, never seen by the DPP) ----------

/** The grouping methods, in menu order. 'tree' cuts the average-linkage tree on 1 - cos that also orders the similarity matrix; 'far' is complete linkage and 'avg' average linkage on sqrt(2 - 2 cos); 'kmeans' needs vectors. */
export const GROUP_METHODS = [
  { id: 'tree', label: 'Tree' }, { id: 'far', label: 'Far apart' }, { id: 'avg', label: 'Average' }, { id: 'kmeans', label: 'k-means' },
];

/**
 * Group the N completions from their unit Gram matrix G alone. Returns { label: Int32Array (group 0.. by decreasing size), k }.
 * `tree` (optional) is a precomputed seriation(G, N) and `vectors` rows with Gram G (k-means only; computed from G when missing).
 * With `distinct` <= 3 distinct strings the pool is one group.
 */
export function groupLabels(G, N, { method = 'tree', k = 6, tree = null, vectors = null, distinct = N } = {}) {
  if (N < 2 || distinct <= 3) return { label: new Int32Array(N), k: N ? 1 : 0 };
  if (method === 'kmeans') {
    const r = clusterPool(vectors ?? gramToVectors(G, N), k, 1);
    return { label: r.label, k: r.k };
  }
  const merges = method === 'tree' ? (tree ?? seriation(G, N)).merges
    : clusterLinkage(distFromGram(G, N), N, { method: method === 'avg' ? 'average' : 'complete' });
  const c = cutClusters(merges, N, { k, minSize: 1 });
  return { label: c.label, k: c.k };
}

/** The first five words of a text, with an ellipsis when it is longer. */
export function shortText(t) {
  const w = String(t).replace(/\s+/g, ' ').trim().split(' ');
  return w.length > 5 ? w.slice(0, 5).join(' ') + '…' : w.join(' ');
}

/** One name per group 0..k-1: 'num' "Group 1", 'kw' the group's keyword, 'medoid' the start of its most typical completion. */
export function groupNames(kind, { label, k, texts, logp = null, G, N }) {
  if (kind === 'kw') return clusterKeywords(texts, label, k, logp).map((x, c) => x || `Group ${c + 1}`);
  if (kind === 'medoid') {
    return Array.from({ length: k }, (_, c) => {
      const m = []; for (let i = 0; i < N; i++) if (label[i] === c) m.push(i);
      return m.length ? shortText(texts[medoidOf(G, N, m)]) : `Group ${c + 1}`;
    });
  }
  return Array.from({ length: k }, (_, c) => `Group ${c + 1}`);
}

/** Members of each group: counts[c] for c in 0..k-1 (label < 0 is not counted). */
export function groupCounts(label, k) {
  const n = new Array(k).fill(0);
  for (const c of label) if (c >= 0 && c < k) n[c]++;
  return n;
}

/** Palette slot (1..8) of group c, cycling; the page's --c1 .. --c8. */
export const groupSlot = (c) => (c % 8) + 1;

/** The log of the kernel's diagonal weight used to pick the subset: Most likely uses kappa, Sample uses kappa / q. */
export const selectionLogScale = (mode, logKappa, logq) => (mode === 'map' ? Array.from(logKappa) : logKappa.map((x, i) => x - logq[i]));

const TREES = new WeakMap();
/** The average-linkage seriation of the pool (see matrices.seriation), computed once per Gram matrix object G. */
export function treeOf(G, N) {
  let t = TREES.get(G);
  if (!t) { t = seriation(G, N); TREES.set(G, t); }
  return t;
}


// ---------- "Not super principled": options the default page leaves out ----------

/** The note under the section title and at the end of each option's "?" text. */
export const NP_NOTE = 'Options the default page leaves out because they have no principled justification. The paper does not use them.';

/** Defaults of the section. `weight` and `len` are 'follow' or a family; `axes` is 'fit' (his default: fitted once to the first selection) or 'fixed' (principal axes). */
export const NP_DEFAULTS = { sharedT: true, weight: 'follow', len: 'follow', sizeQ: false, ghost: false, cone: false, axes: 'fit' };
export const WEIGHT_FAMILIES = ['quality', 'importance', 'unit'];
export const FAMILY_NAME = { quality: 'Quality', importance: 'Importance', unit: 'Unit' };

/** True when every option of the section is at its default (a switch-off then needs no recompute). */
export const npIsDefault = (st) => Object.keys(NP_DEFAULTS).every((k) => st[k] === NP_DEFAULTS[k]);

/** The weight family the kernel uses to pick the subset: the override when one is set, else the mode's weight (Most likely κ, Sample κ/q). `xo` is the section's state or null. */
export const kernelFamily = (xo, mode) => (xo && WEIGHT_FAMILIES.includes(xo.weight) ? xo.weight : mode === 'map' ? 'quality' : 'importance');

/**
 * The family the arrow lengths are drawn from. A Vector length choice wins; 'follow' follows the weight override,
 * and without one is √κ, the default page's length in both modes.
 */
export function drawnFamily(xo, mode) {
  if (!xo) return 'quality';
  if (WEIGHT_FAMILIES.includes(xo.len)) return xo.len;
  return WEIGHT_FAMILIES.includes(xo.weight) ? xo.weight : 'quality';
}

/** log w for a weight family from log κ (`lk`) and log κ/q (`lw`): Quality κ, Importance κ/q, Unit 1. */
export const weightScale = (fam, lk, lw) => (fam === 'unit' ? lk.map(() => 0) : fam === 'importance' ? lw : lk);

/** Drawn lengths in (0, 1] from a family's log weights: √w relative to the largest; Unit is all 1. */
export function lengthsFrom(fam, lk, lw) {
  if (fam === 'unit') return lk.map(() => 1);
  const ls = weightScale(fam, lk, lw), m = Math.max(...ls);
  return ls.map((s) => Math.exp(0.5 * (s - m)));
}

/** Point radius factors 0.35 + 0.65 √κ/max √κ, so the least likely point stays visible. */
export function pointRadii(lk) {
  const m = Math.max(...lk);
  return lk.map((s) => 0.35 + 0.65 * Math.exp(0.5 * (s - m)));
}

const ARROWS = { quality: '√κ arrows', importance: '√(κ/q) arrows', unit: 'unit-length arrows' };
/** What the drawn volume shows when the arrows are not the kernel's: '' when the drawn length is the kernel's weight, else " of the … arrows" (with the muted span). */
export function volumeSuffix(kernelFam, drawnFam) {
  return kernelFam === drawnFam ? '' : ` <span class="muted">of the ${ARROWS[drawnFam]}</span>`;
}

/**
 * T of the picking mode `to` after leaving `from`. With one T for both modes (`shared`) it stays. Otherwise each mode keeps its own:
 * `Tby` is { map?, sample? } (mutated), and a mode not visited yet starts at the T in use.
 */
export function modeT({ shared, Tby, from, to, T }) {
  if (shared || !Tby) return T;
  Tby[from] = T;
  return Tby[to] ?? T;
}

const DIFF_NAME = { mode: 'picking mode', T: 'T', K: 'K', tp: 'T′', weight: 'DPP weight', len: 'vector length' };
/**
 * The note under the paper-settings buttons. `now` and `preset` carry { mode, T, K, tp, weight, len, lock }. With a match: "Matches MAP, paper".
 * Without one, after a button was pressed (`last`): "Differs from MAP, paper in T and K". Else ''.
 */
export function presetNote({ match, last, now }) {
  const name = (k) => PRESETS[k].label;
  if (match) return `Matches ${name(match)}.`;
  if (!last) return '';
  const p = PRESETS[last];
  const want = { mode: p.mode, T: p.T, K: p.K, tp: p.Tprime, weight: p.weight, len: now.len, lock: now.lock };   // the page's default draws √κ in both modes, so a preset does not dictate the length
  if (p.mode === now.mode && p.Ts.some((t) => Math.abs(t - now.T) <= 1e-6 * t)) want.T = now.T;   // any T of the paper's grid counts
  const diff = presetDiff(now, want).filter((k) => k in DIFF_NAME).map((k) => DIFF_NAME[k]);
  return diff.length ? `Differs from ${name(last)} in ${diff.length > 1 ? `${diff.slice(0, -1).join(', ')} and ${diff[diff.length - 1]}` : diff[0]}.` : `Matches ${name(last)}.`;
}

/** Largest absolute difference between two lists of coordinate rows (null when the shapes differ). */
export function maxShift(A, B) {
  if (!A || !B || A.length !== B.length) return null;
  let m = 0;
  for (let i = 0; i < A.length; i++) for (let d = 0; d < A[i].length; d++) m = Math.max(m, Math.abs(A[i][d] - B[i][d]));
  return m;
}
