import test from 'node:test';
import assert from 'node:assert/strict';
import {
  gram, symEig, logDetSub, kernel, spectrum, sampleKDPP, greedyMAP, logZ,
  inclusionProbs, mulberry32, kmeansSelect, boundOnInstance, ess,
} from '../js/dpp.js';

function randomPool(N, D, seed) {
  const rng = mulberry32(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(rng() + 1e-12)) * Math.cos(2 * Math.PI * rng());
  const phi = Array.from({ length: N }, () => {
    const v = Float32Array.from({ length: D }, gauss);
    const n = Math.hypot(...v);
    return v.map((x) => x / n);
  });
  const logp = Array.from({ length: N }, () => -10 * rng() - 2);
  return { phi, logp, rng };
}

function subsets(N, K, start = 0, cur = [], out = []) {
  if (cur.length === K) { out.push(cur.slice()); return out; }
  for (let i = start; i < N; i++) { cur.push(i); subsets(N, K, i + 1, cur, out); cur.pop(); }
  return out;
}

function bruteForce(L, N, K) {
  const all = subsets(N, K);
  const dets = all.map((S) => Math.exp(logDetSub(L, N, S)));
  const Z = dets.reduce((a, b) => a + b, 0);
  return { all, probs: dets.map((d) => d / Z), Z };
}

test('symEig reconstructs A', () => {
  const { phi } = randomPool(9, 5, 1);
  const G = gram(phi);
  const { values, vectors } = symEig(G, 9);
  for (let i = 0; i < 9; i++) for (let j = 0; j < 9; j++) {
    let s = 0;
    for (let k = 0; k < 9; k++) s += values[k] * vectors[k][i] * vectors[k][j];
    assert.ok(Math.abs(s - G[i * 9 + j]) < 1e-10);
  }
  for (let k = 1; k < 9; k++) assert.ok(values[k - 1] >= values[k]);
});

test('logZ and inclusion probabilities match brute force', () => {
  const N = 8, K = 3;
  const { phi, logp } = randomPool(N, 6, 2);
  const G = gram(phi);
  const { L } = kernel(G, N, logp.map((x) => x / 2));
  const spec = spectrum(L, N);
  const bf = bruteForce(L, N, K);
  assert.ok(Math.abs(logZ(spec, K) - Math.log(bf.Z)) < 1e-9);
  const pi = inclusionProbs(spec, K);
  const piBF = new Float64Array(N);
  bf.all.forEach((S, s) => S.forEach((i) => (piBF[i] += bf.probs[s])));
  for (let i = 0; i < N; i++) assert.ok(Math.abs(pi[i] - piBF[i]) < 1e-9, `pi[${i}]`);
  assert.ok(Math.abs(pi.reduce((a, b) => a + b, 0) - K) < 1e-9);
});

test('sampleKDPP matches the exact K-DPP distribution', () => {
  const N = 7, K = 3;
  const { phi, logp } = randomPool(N, 5, 3);
  const G = gram(phi);
  const { L } = kernel(G, N, logp.map((x) => x / 3));
  const spec = spectrum(L, N);
  const bf = bruteForce(L, N, K);
  const key = (S) => S.join(',');
  const pos = new Map(bf.all.map((S, i) => [key(S), i]));
  const counts = new Float64Array(bf.all.length);
  const rng = mulberry32(7);
  const M = 60000;
  for (let m = 0; m < M; m++) counts[pos.get(key(sampleKDPP(spec, K, rng)))]++;
  let tv = 0;
  for (let s = 0; s < counts.length; s++) tv += Math.abs(counts[s] / M - bf.probs[s]);
  tv /= 2;
  assert.ok(tv < 0.015, `total variation ${tv}`);
});

test('sampleKDPP returns null when rank < K (Ẑ = 0)', () => {
  const N = 6;
  const { phi } = randomPool(N, 2, 4); // rank ≤ 2
  const G = gram(phi);
  const spec = spectrum(G, N);
  assert.equal(sampleKDPP(spec, 3), null);
});

test('greedyMAP logdet is consistent and within (K!)² of the optimum', () => {
  const N = 9, K = 3;
  const { phi, logp } = randomPool(N, 6, 5);
  const G = gram(phi);
  const { L } = kernel(G, N, logp.map((x) => x / 5));
  const { set, logdet } = greedyMAP(L, N, K);
  assert.equal(set.length, K);
  assert.ok(Math.abs(logdet - logDetSub(L, N, set)) < 1e-9);
  const best = Math.max(...subsets(N, K).map((S) => logDetSub(L, N, S)));
  assert.ok(logdet <= best + 1e-12);
  assert.ok(logdet >= best - 2 * Math.log(6));
});

test('duplicate strings get zero joint probability', () => {
  const { phi, logp } = randomPool(5, 6, 6);
  phi[4] = phi[1];
  const G = gram(phi);
  const { L } = kernel(G, 5, logp);
  assert.equal(logDetSub(L, 5, [1, 4]), -Infinity);
  const { set } = greedyMAP(L, 5, 4);
  assert.ok(!(set.includes(1) && set.includes(4)));
});

test('kmeansSelect returns ≤ K distinct indices', () => {
  const { phi, logp } = randomPool(30, 8, 8);
  const S = kmeansSelect(phi, logp, 5);
  assert.ok(S.length <= 5 && new Set(S).size === S.length);
});

test('ess bounds', () => {
  assert.ok(Math.abs(ess([0, 0, 0, 0]) - 4) < 1e-12);
  assert.ok(Math.abs(ess([0, -1000, -1000]) - 1) < 1e-9);
});

test('bound on instance: rho >= 1 and theta decays as 1/N', () => {
  const { phi, logp } = randomPool(12, 6, 9);
  const texts = phi.map((_, i) => 's' + (i % 10)); // two duplicates
  const b = boundOnInstance({ phi, logp, texts, T: 2, K: 3 });
  assert.equal(b.distinct, 10);
  assert.ok(b.rho >= 1 - 1e-9);
  assert.ok(Math.abs(b.theta(100) / b.theta(199) - (199 - 3 + 1) / (100 - 3 + 1)) < 1e-9);
  assert.ok(b.tight(100) <= b.theta(100) + 1e-12);
});
