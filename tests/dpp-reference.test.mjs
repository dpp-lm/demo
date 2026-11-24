// Adversarial checks of the DPP routines against brute force: enumerate every K-subset of a small
// pool and compute log P(S) = Σ_{i∈S} s_i + log det G_S − log Z directly (log space throughout, so the
// reference itself cannot underflow). Run: node --test tests/dpp-reference.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { gram, kdpp, greedyMAPLog, eigSym, symEig, mulberry32 } from '../js/dpp.js';

// ---- reference ----
function logdetChol(G, n, S) {
  const k = S.length, C = new Float64Array(k * k);
  let ld = 0;
  for (let i = 0; i < k; i++) for (let j = 0; j <= i; j++) {
    let s = G[S[i] * n + S[j]];
    for (let m = 0; m < j; m++) s -= C[i * k + m] * C[j * k + m];
    if (i === j) { if (!(s > 1e-13)) return -Infinity; C[i * k + i] = Math.sqrt(s); ld += Math.log(s); } else C[i * k + j] = s / C[j * k + j];
  }
  return ld;
}
function subsets(n, k, start = 0, cur = [], out = []) {
  if (cur.length === k) { out.push(cur.slice()); return out; }
  for (let i = start; i < n; i++) { cur.push(i); subsets(n, k, i + 1, cur, out); cur.pop(); }
  return out;
}
const lse = (xs) => { const m = Math.max(...xs); return m === -Infinity ? -Infinity : m + Math.log(xs.reduce((a, x) => a + Math.exp(x - m), 0)); };
function reference(G, n, s, K) {
  const all = subsets(n, K).map((S) => ({ S, lw: S.reduce((a, i) => a + s[i], 0) + logdetChol(G, n, S) }));
  const logZ = lse(all.map((x) => x.lw));
  const pi = new Float64Array(n);
  for (const x of all) { const p = Math.exp(x.lw - logZ); for (const i of x.S) pi[i] += p; }
  const best = all.reduce((a, b) => (b.lw > a.lw ? b : a));
  return { all, logZ, pi, best, prob: (S) => Math.exp(all.find((x) => x.S.join() === S.join()).lw - logZ) };
}
// The same greedy rule as greedyMAPLog, computed the slow, obvious way.
function greedyRef(G, n, s, K) {
  const S = [];
  for (let t = 0; t < K; t++) {
    let best = -1, bv = -Infinity;
    for (let j = 0; j < n; j++) {
      if (S.includes(j)) continue;
      const v = s[j] + logdetChol(G, n, [...S, j].sort((a, b) => a - b)) - (S.length ? logdetChol(G, n, S.slice().sort((a, b) => a - b)) : 0);
      if (v > bv) { bv = v; best = j; }
    }
    if (bv === -Infinity) break;
    S.push(best);
  }
  return S;
}

// ---- pools ----
function pool({ n, D, seed, dup = [], near = [], rank }) {
  const rng = mulberry32(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(rng() + 1e-300)) * Math.cos(2 * Math.PI * rng());
  const basis = rank ? Array.from({ length: rank }, () => Array.from({ length: D }, gauss)) : null;
  const phi = Array.from({ length: n }, () => {
    const v = basis ? basis.reduce((acc, b) => { const c = gauss(); return acc.map((x, d) => x + c * b[d]); }, new Array(D).fill(0)) : Array.from({ length: D }, gauss);
    const r = Math.hypot(...v); return v.map((x) => x / r);
  });
  for (const [a, b] of dup) phi[b] = phi[a].slice();
  for (const [a, b] of near) { const v = phi[a].map((x, d) => x + 1e-4 * (d === 0 ? 1 : 0)); const r = Math.hypot(...v); phi[b] = v.map((x) => x / r); }
  return gram(phi);
}
function scales(n, span, seed) {
  const rng = mulberry32(seed + 99);
  return Array.from({ length: n }, () => -span * rng());
}

const CASES = [
  { name: 'friendly', n: 10, D: 6, K: 3, span: 2 },
  { name: 'weights span 50 nats', n: 10, D: 6, K: 3, span: 50 },
  { name: 'weights span 300 nats', n: 10, D: 6, K: 4, span: 300 },
  { name: 'weights span 1000 nats', n: 11, D: 7, K: 5, span: 1000 },
  { name: 'exact duplicates', n: 10, D: 6, K: 3, span: 20, dup: [[0, 1], [2, 3], [2, 4]] },
  { name: 'near duplicates', n: 10, D: 6, K: 3, span: 20, near: [[0, 1], [5, 6]] },
  { name: 'duplicates and 500 nats', n: 10, D: 6, K: 4, span: 500, dup: [[0, 1], [0, 2]] },
  { name: 'K = N', n: 6, D: 8, K: 6, span: 30 },
  { name: 'K = rank', n: 10, D: 6, K: 4, span: 40, rank: 4 },
  { name: 'K = 1', n: 9, D: 5, K: 1, span: 200 },
];

for (const [ci, c] of CASES.entries()) {
  test(`kdpp inclusion probabilities and set probabilities: ${c.name}`, () => {
    const G = pool({ ...c, seed: 10 + ci }), s = scales(c.n, c.span, ci);
    const ref = reference(G, c.n, s, c.K);
    const d = kdpp(G, c.n, s, c.K);
    // When the spectrum spans more than ~1e10, the staged sampler splits it at the largest eigenvalue gap
    // and is exact only up to the ratio at that gap. Constructed cases (K = N here) reach ~5e-6; on our
    // real pools the error is below 1e-10 (checked by exhaustive enumeration, K = 3). Display is in %.
    const tol = c.name === 'K = N' ? 1e-5 : 1e-8;
    for (let i = 0; i < c.n; i++) assert.ok(Math.abs(d.pi[i] - ref.pi[i]) < tol, `π_${i}: ${d.pi[i]} vs ${ref.pi[i]}`);
    assert.ok(Math.abs(d.pi.reduce((a, b) => a + b, 0) - c.K) < tol, 'inclusion probabilities sum to K');
    // log P(S) for the ten most probable sets
    const top = ref.all.filter((x) => x.lw > -Infinity).sort((a, b) => b.lw - a.lw).slice(0, 10);
    for (const x of top) {
      const want = x.lw - ref.logZ, got = d.logProb(x.S);
      if (want > -30) assert.ok(Math.abs(got - want) < 1e-6, `log P(${x.S}): ${got} vs ${want}`);
    }
  });

  test(`greedyMAPLog: ${c.name}`, () => {
    const G = pool({ ...c, seed: 10 + ci }), s = scales(c.n, c.span, ci);
    const ref = reference(G, c.n, s, c.K);
    const g = greedyMAPLog(G, c.n, s, c.K);
    assert.deepEqual(g.set.slice().sort((a, b) => a - b), greedyRef(G, c.n, s, c.K).sort((a, b) => a - b), 'same set as the slow greedy');
    assert.equal(g.set.length, c.K, 'K items when a non-degenerate K-set exists');
    const lw = g.set.reduce((a, i) => a + s[i], 0) + logdetChol(G, c.n, g.set.slice().sort((a, b) => a - b));
    let fact = 1; for (let k = 2; k <= c.K; k++) fact *= k;
    assert.ok(lw >= ref.best.lw - 2 * Math.log(fact) - 1e-9, 'within (K!)² of the best set');
  });
}

test('kdpp sampling frequencies match the exact distribution (weights span 300 nats)', () => {
  const c = CASES[2], G = pool({ ...c, seed: 12 }), s = scales(c.n, c.span, 2);
  const ref = reference(G, c.n, s, c.K), d = kdpp(G, c.n, s, c.K), rng = mulberry32(7);
  const T = 40000, count = new Map();
  for (let t = 0; t < T; t++) { const S = d.sample(rng).slice().sort((a, b) => a - b).join(); count.set(S, (count.get(S) ?? 0) + 1); }
  let tv = 0;
  for (const x of ref.all) tv += Math.abs((count.get(x.S.join()) ?? 0) / T - Math.exp(x.lw - ref.logZ));
  assert.ok(tv / 2 < 0.02, `total variation ${(tv / 2).toFixed(4)}`);
});

test('kdpp sampling frequencies match the exact distribution (friendly)', () => {
  const c = CASES[0], G = pool({ ...c, seed: 10 }), s = scales(c.n, c.span, 0);
  const ref = reference(G, c.n, s, c.K), d = kdpp(G, c.n, s, c.K), rng = mulberry32(8);
  const T = 40000, count = new Map();
  for (let t = 0; t < T; t++) { const S = d.sample(rng).slice().sort((a, b) => a - b).join(); count.set(S, (count.get(S) ?? 0) + 1); }
  let tv = 0;
  for (const x of ref.all) tv += Math.abs((count.get(x.S.join()) ?? 0) / T - Math.exp(x.lw - ref.logZ));
  assert.ok(tv / 2 < 0.02, `total variation ${(tv / 2).toFixed(4)}`);
});

test('rank < K: no set can be sampled; greedy stops at the rank', () => {
  const G = pool({ n: 10, D: 6, seed: 3, rank: 3 }), s = scales(10, 10, 3);
  assert.equal(kdpp(G, 10, s, 4).sample(mulberry32(1)), null);
  assert.equal(greedyMAPLog(G, 10, s, 4).set.length, 3);
});

test('eigSym agrees with Jacobi (symEig) on a 200×200 Gram matrix', () => {
  const G = pool({ n: 200, D: 64, seed: 5 });
  const a = eigSym(G, 200), b = symEig(G, 200);
  for (let k = 0; k < 20; k++) assert.ok(Math.abs(a.values[k] - b.values[k]) < 1e-10 * a.values[0], `eigenvalue ${k}`);
  // reconstruction error of the fast solver
  let err = 0;
  for (let i = 0; i < 200; i += 7) for (let j = 0; j < 200; j += 7) {
    let x = 0; for (let k = 0; k < 200; k++) x += a.values[k] * a.vectors[k][i] * a.vectors[k][j];
    err = Math.max(err, Math.abs(x - G[i * 200 + j]));
  }
  assert.ok(err < 1e-10, `reconstruction ${err}`);
});
