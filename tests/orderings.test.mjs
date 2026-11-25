// The matrix-orderings lab (lab/orderings.js), checked on precomputed pools.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { flatPivotOrder, nestedPivotOrder, averageLinkageOrder, themeOrder, pivotedCholesky } from '../lab/orderings.js';
import { greedyMAPLog, logDetSub, kdpp } from '../js/dpp.js';

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

  test(`${pool}: inclusion splits exactly over the themes`, () => {
    for (const K of [2, 5]) {
      const { pi, p } = themeOrder(L, N, K), ref = kdpp(G, N, lk, K).pi;
      for (let i = 0; i < N; i++) assert.ok(Math.abs(pi[i] - ref[i]) < 1e-10);
      assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - K) < 1e-8);
    }
  });
}
