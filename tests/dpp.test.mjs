import test from 'node:test';
import assert from 'node:assert/strict';
import {
  gram, symEig, eigSym, logDetSub, kernel, spectrum, sampleKDPP, greedyMAP, logZ,
  inclusionProbs, greedyMAPLog, kdpp, mulberry32, kmeansSelect, boundOnInstance, ess,
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

test('spectrum with eigSym gives the same inclusion probabilities as with symEig', () => {
  const { phi, logp } = randomPool(40, 12, 5);
  const N = 40, G = gram(phi);
  const { L } = kernel(G, N, logp.map((x) => x / 3));
  const fast = spectrum(L, N);
  const { values, vectors } = symEig(L, N);
  const vmax = Math.max(values[0], 1e-300);
  const slow = { lambda: values.map((x) => Math.max(0, x / vmax)), vectors, logScale: Math.log(vmax) };
  for (const K of [2, 3, 5]) {
    const a = inclusionProbs(fast, K), b = inclusionProbs(slow, K);
    for (let i = 0; i < N; i++) assert.ok(Math.abs(a[i] - b[i]) < 1e-9, `K=${K} i=${i}: ${a[i]} vs ${b[i]}`);
  }
  assert.ok(Math.abs(fast.logScale - slow.logScale) < 1e-9);
  assert.ok(eigSym);
});

// ---------- scale-free greedy MAP and staged K-DPP ----------

/** Brute-force log P(S) = Σ s_i + log det G_S − logsumexp, over all K-subsets, in log space. */
function bruteLog(G, N, s, K) {
  const all = subsets(N, K);
  const raw = all.map((S) => S.reduce((a, i) => a + s[i], 0) + logDetSub(G, N, S));
  const m = Math.max(...raw);
  const lz = m + Math.log(raw.reduce((a, x) => a + Math.exp(x - m), 0));
  const logp = raw.map((x) => x - lz);
  const probs = logp.map(Math.exp);
  const pi = new Float64Array(N);
  all.forEach((S, k) => S.forEach((i) => (pi[i] += probs[k])));
  return { all, raw, logp, probs, pi };
}

test('greedyMAPLog and kdpp agree with the old code on a non-degenerate pool', () => {
  const N = 9, K = 3;
  const { phi, logp } = randomPool(N, 6, 11);
  const G = gram(phi);
  const s = logp.map((x) => x / 3);
  const { L } = kernel(G, N, s);
  const old = greedyMAP(L, N, K), neu = greedyMAPLog(G, N, s, K);
  assert.deepEqual(neu.set, old.set);
  assert.ok(Math.abs(neu.logdet - old.logdet) < 1e-9);
  neu.gains.forEach((g, t) => assert.ok(Math.abs(g - old.gains[t]) < 1e-9));
  const spec = spectrum(L, N);
  const d = kdpp(G, N, s, K);
  assert.equal(d.stages, 1);
  const piOld = inclusionProbs(spec, K);
  for (let i = 0; i < N; i++) assert.ok(Math.abs(d.pi[i] - piOld[i]) < 1e-12);
  const r1 = mulberry32(5), r2 = mulberry32(5);
  for (let m = 0; m < 200; m++) {
    const a = d.sample(r1), b = sampleKDPP(spec, K, r2);
    assert.deepEqual(a, b);
    assert.ok(Math.abs(d.logProb(a) - (logDetSub(L, N, a) - logZ(spec, K))) < 1e-9);
  }
  const bf = bruteLog(G, N, s, K);
  bf.all.forEach((S, k) => assert.ok(Math.abs(d.logProb(S) - bf.logp[k]) < 1e-9));
  for (let i = 0; i < N; i++) assert.ok(Math.abs(d.pi[i] - bf.pi[i]) < 1e-9);
});

// N = 8, K = 3, log-scales spanning 60-150 nats with gaps of tens of nats between groups, shuffled.
const DEGENERATE = [
  { name: 'two heavy, then a pair, then light', s: [150, 148, 95, 92, 70, 66, 60, 62], dup: [] },
  { name: 'one heavy, two close, rest light', s: [150, 100, 98, 80, 70, 66, 60, 62], dup: [] },
  { name: 'duplicate of the top item', s: [150, 148, 95, 92, 70, 66, 60, 62], dup: [[0, 1]] },
  { name: 'duplicates among light items', s: [150, 120, 95, 92, 70, 66, 60, 62], dup: [[4, 5], [2, 3]] },
  { name: 'duplicate heavy and light', s: [150, 110, 108, 85, 70, 66, 60, 62], dup: [[0, 3], [5, 6]] },
];

DEGENERATE.forEach(({ name, s: s0, dup }, c) => {
  test(`degenerate pool: ${name}`, () => {
    const N = 8, K = 3;
    const { phi, rng } = randomPool(N, 5, 20 + c);
    for (const [a, b] of dup) phi[b] = phi[a];
    const perm = [...Array(N).keys()].sort(() => rng() - 0.5);
    const s = perm.map((i) => s0[i]);
    const G = gram(phi);
    const bf = bruteLog(G, N, s, K);
    // Greedy: the old absolute 1e-12 rule stops early here; the new one picks the log-space argmax.
    const g = greedyMAPLog(G, N, s, K);
    assert.equal(g.set.length, K);
    const best = bf.raw.indexOf(Math.max(...bf.raw));
    const gs = g.set.slice().sort((a, b) => a - b);
    assert.deepEqual(gs, bf.all[best]);
    const logdetShift = g.set.reduce((a, i) => a + s[i], 0) - K * Math.max(...s) + logDetSub(G, N, g.set);
    assert.ok(Math.abs(g.logdet - logdetShift) < 1e-8);
    if (bf.all[best].join() === gs.join()) assert.deepEqual(gs, bf.all[best]);
    // K-DPP.
    const d = kdpp(G, N, s, K);
    assert.ok(d.stages > 1, 'staging is exercised');
    assert.ok(Math.abs(d.pi.reduce((a, b) => a + b, 0) - K) < 1e-9);
    for (let i = 0; i < N; i++) assert.ok(Math.abs(d.pi[i] - bf.pi[i]) < 1e-6, `pi[${i}] ${d.pi[i]} vs ${bf.pi[i]}`);
    bf.all.forEach((S, k) => {
      if (bf.probs[k] > 1e-6) assert.ok(Math.abs(Math.exp(d.logProb(S)) - bf.probs[k]) < 1e-6, `logProb ${S} ${Math.exp(d.logProb(S))} vs ${bf.probs[k]}`);
    });
    const pos = new Map(bf.all.map((S, k) => [S.join(), k]));
    const counts = new Float64Array(bf.all.length);
    const r = mulberry32(99), M = 20000;
    for (let m = 0; m < M; m++) {
      const S = d.sample(r);
      assert.ok(S && S.length === K && new Set(S).size === K, `bad sample ${S}`);
      counts[pos.get(S.join())]++;
    }
    bf.probs.forEach((p, k) => {
      const se = Math.sqrt((p * (1 - p)) / M);
      assert.ok(Math.abs(counts[k] / M - p) <= 4 * se + 1e-9, `freq ${bf.all[k]} ${counts[k] / M} vs ${p}`);
    });
  });
});

test('greedyMAPLog returns K items under sharp weights where the old greedy stops early', () => {
  const N = 8, K = 3;
  const { phi } = randomPool(N, 5, 31);
  const s = [150, 100, 60, 40, 30, 20, 10, 0];
  const G = gram(phi);
  const old = greedyMAP(kernel(G, N, s).L, N, K);
  assert.ok(old.set.length < K);
  assert.deepEqual(greedyMAPLog(G, N, s, K).set, [0, 1, 2]);
});

test('rank-deficient pool: sample is null, greedy stops at the rank', () => {
  const { phi } = randomPool(6, 2, 4); // rank 2
  const G = gram(phi);
  const d = kdpp(G, 6, [3, 2, 1, 0, -1, -2], 3);
  assert.equal(d.sample(mulberry32(1)), null);
  assert.equal(greedyMAPLog(G, 6, [3, 2, 1, 0, -1, -2], 3).set.length, 2);
});
