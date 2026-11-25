// The matrix-orderings lab (lab/orderings.js), checked on precomputed pools.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { flatPivotOrder, nestedPivotOrder, averageLinkageOrder, themeOrder, pivotedCholesky } from '../lab/orderings.js';
import { condition } from '../lab/condition.js';
import { themeProfiles, THEME_COLORS } from '../lab/profiles.js';
import { greedyMAPLog, logDetSub, kdpp, mulberry32 } from '../js/dpp.js';

const dir = new URL('../data/cache/', import.meta.url);
const index = JSON.parse(readFileSync(new URL('index.json', dir)));
const EMB = 'Qwen/Qwen3-Embedding-4B';

// G with identical rows for identical strings (as the page builds it), and L = diag(√w) G diag(√w), w = p^{1/T}.
function load(pool, T = 20) {
  const b = index.buffers.find((x) => x.pool === pool);
  const s = JSON.parse(readFileSync(new URL(b.pool, dir))).samples, N = s.length;
  const raw = new Float32Array(readFileSync(new URL(b.G[EMB], dir)).buffer.slice(0));
  const first = new Map(), c = s.map((x, i) => (first.has(x.text) ? first.get(x.text) : (first.set(x.text, i), i)));
  const G = new Float64Array(N * N);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) G[i * N + j] = i === j ? 1 : raw[c[i] * N + c[j]];
  const lk = s.map((x) => x.logp / T), m = Math.max(...lk), w = lk.map((x) => Math.exp(x - m));
  const L = new Float64Array(N * N);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) L[i * N + j] = Math.sqrt(w[i] * w[j]) * G[i * N + j];
  return { N, G, L, lk };
}
const isPermutation = (o, N) => o.length === N && new Set(o).size === N && o.every((i) => i >= 0 && i < N);

for (const pool of ['story_T1.0.json', 'joke_T1.5.json']) {
  const { N, G, L, lk } = load(pool);

  test(`${pool}: every order is a permutation`, () => {
    for (const r of [flatPivotOrder(L, N, 3), nestedPivotOrder(L, N), averageLinkageOrder(L, N), themeOrder(L, N, 3)]) assert.ok(isPermutation(r.order, N));
  });

  test(`${pool}: the first K pivots of L are the greedy (Most likely) picks`, () => {
    for (const K of [1, 3, 6]) {
      const piv = pivotedCholesky(L, N).piv.slice(0, K).sort((a, b) => a - b);
      assert.deepEqual(piv, greedyMAPLog(G, N, lk, K).set.slice().sort((a, b) => a - b));
    }
  });

  test(`${pool}: tree heights are volume factors and never rise from parent to child`, () => {
    const { piv } = pivotedCholesky(G, N), { parent, height } = nestedPivotOrder(G, N);
    for (let t = 1; t < 6; t++) {
      const ratio = Math.exp(logDetSub(G, N, piv.slice(0, t + 1)) - logDetSub(G, N, piv.slice(0, t)));
      assert.ok(Math.abs(ratio - height[piv[t]] ** 2) < 1e-8);
    }
    for (let i = 0; i < N; i++) if (parent[i] >= 0) assert.ok(height[i] <= height[parent[i]] + 1e-12);
  });

  test(`${pool}: theme profiles: parts sum to π_i, each colour sums to its p, rest = K − coloured p; colours stable under a small change of T`, () => {
    const pr = themeProfiles(L, N, 3), ref = kdpp(G, N, lk, 3).pi;
    for (let i = 0; i < N; i++) {
      assert.ok(Math.abs(pr.pi[i] - ref[i]) < 1e-10);
      assert.ok(Math.abs(pr.groups.reduce((s, g) => s + pr.part(i, g), 0) + pr.restPart(i) - pr.pi[i]) < 1e-12);
    }
    for (const g of pr.groups) assert.ok(Math.abs([...Array(N).keys()].reduce((s, i) => s + pr.part(i, g), 0) - g.p) < 1e-9);
    assert.ok(Math.abs(pr.restP - (3 - pr.groups.reduce((s, g) => s + g.p, 0))) < 1e-12);
    assert.ok(pr.groups.length <= THEME_COLORS.length && pr.groups.every((g) => THEME_COLORS.includes(g.color)));
    const L2 = load(pool, 21).L, pr2 = themeProfiles(L2, N, 3, pr.prev);
    assert.deepEqual(pr2.groups.map((g) => g.color), pr.groups.map((g) => g.color));
  });

  test(`${pool}: inclusion splits exactly over the themes`, () => {
    for (const K of [2, 5]) {
      const { pi, p } = themeOrder(L, N, K), ref = kdpp(G, N, lk, K).pi;
      for (let i = 0; i < N; i++) assert.ok(Math.abs(pi[i] - ref[i]) < 1e-10);
      assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - K) < 1e-8);
    }
  });
}

// Conditioning (lab/condition.js): P(i in S | A in S) against brute force over every K-set of a small kernel.
test('conditioning: π(i | A) matches brute-force enumeration; volume-zero selections are refused', () => {
  const rng = mulberry32(7), n = 9, D = 5, K = 4;
  const X = Array.from({ length: n }, () => Array.from({ length: D }, () => rng() - 0.5));
  const dot = (a, b) => a.reduce((s, x, d) => s + x * b[d], 0);
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) L[i * n + j] = dot(X[i], X[j]) + (i === j ? 0.05 : 0);
  const sets = [];
  (function rec(s, c) { if (c.length === K) { sets.push(c.slice()); return; } for (let i = s; i < n; i++) { c.push(i); rec(i + 1, c); c.pop(); } })(0, []);
  const w = sets.map((S) => Math.exp(logDetSub(L, n, S))), has = (S, xs) => xs.every((x) => S.includes(x));
  for (const A of [[], [2], [2, 5], [0, 3, 7]]) {
    const c = condition(L, n, A, K), PA = sets.reduce((s, S, k) => s + (has(S, A) ? w[k] : 0), 0);
    for (let i = 0; i < n; i++) assert.ok(Math.abs(sets.reduce((s, S, k) => s + (has(S, [...A, i]) ? w[k] : 0), 0) / PA - c.pi[i]) < 1e-12);
  }
  // An exact copy of a selected item has leftover zero: not addable, and selecting both is refused.
  const m = n + 1, src = (i) => (i === n ? 2 : i), L2 = new Float64Array(m * m);
  for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) L2[i * m + j] = dot(X[src(i)], X[src(j)]);
  assert.equal(condition(L2, m, [2], K).addable[n], 0);
  assert.equal(condition(L2, m, [2, n], K).ok, false);
});

test('theme profiles: equal eigenvalues are one theme (their eigenspace is determined, the individual vectors are not)', () => {
  const d = [3, 3, 1, 0.5, 0.2, 0.1], n = d.length, L = new Float64Array(n * n);
  d.forEach((x, i) => (L[i * n + i] = x));
  const pr = themeProfiles(L, n, 2);
  assert.equal(pr.groups[0].eigs.length, 2);
  assert.match(pr.groups[0].label, /^themes 1\+2$/);
  assert.ok(Math.abs(pr.part(0, pr.groups[0]) - pr.part(1, pr.groups[0])) < 1e-12);   // symmetric items get equal parts
});
