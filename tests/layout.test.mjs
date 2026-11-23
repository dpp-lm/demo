import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { gram, symEig, eigSym, mulberry32 } from '../js/dpp.js';
import {
  gnomonic, bulkView, pcaCoords, simplexVolume, displayedFactor, trueFactor, drawSubsets, fitLayout, spanView, openCone, clusterPool, clusterKeywords, fitView, alignTo, pcaRotate, hull2D, polyArea, tipSheet, clusterLinkage, cutClusters, distFromGram, distFromCoords, clusterDiameters,
} from '../js/layout.js';

/** Unit vectors in a cone of half-angle ~spread around a random axis. */
function conePool(N, D, spread, seed) {
  const rng = mulberry32(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(rng() + 1e-12)) * Math.cos(2 * Math.PI * rng());
  const axis = Float64Array.from({ length: D }, gauss);
  const an = Math.hypot(...axis);
  const phi = Array.from({ length: N }, () => {
    const v = Float64Array.from({ length: D }, (_, k) => axis[k] / an + spread * gauss() / Math.sqrt(D));
    const n = Math.hypot(...v);
    return Float32Array.from(v, (x) => x / n);
  });
  const logp = Array.from({ length: N }, () => -10 * rng() - 2);
  const logq = logp.map((x) => x + 0.5 * (rng() - 0.5));
  return { phi, logp, logq, rng };
}

function fact(n) { return n <= 1 ? 1 : n * fact(n - 1); }

test('eigSym matches symEig', () => {
  const { phi } = conePool(30, 12, 0.5, 1);
  const G = gram(phi);
  const A = eigSym(G, 30), B = symEig(G, 30);
  for (let k = 0; k < 12; k++) assert.ok(Math.abs(A.values[k] - B.values[k]) < 1e-9);
  const v = A.vectors[0];
  let r = 0;
  for (let i = 0; i < 30; i++) { let s = 0; for (let j = 0; j < 30; j++) s += G[i * 30 + j] * v[j]; r = Math.max(r, Math.abs(s - A.values[0] * v[i])); }
  assert.ok(r < 1e-9);
});

test('gnomonic identity and bounds on h_S', () => {
  const N = 40, D = 20;
  const { phi, rng } = conePool(N, D, 0.6, 7);
  const G = gram(phi);
  const { a, Gg, ok } = gnomonic(G, N);
  assert.ok(ok);
  // Explicit g_i vectors reproduce Gg.
  const Y = pcaCoords(Gg, N, N - 1, true);
  for (const K of [2, 3, 4]) {
    for (let n = 0; n < 40; n++) {
      const S = [];
      while (S.length < K) { const i = Math.floor(rng() * N); if (!S.includes(i)) S.push(i); }
      S.sort((x, y) => x - y);
      const vol = simplexVolume(Y, S); // centering does not change volumes; Y spans all of span(g)
      let pa = 1;
      for (const i of S) pa *= a[i];
      const tr = trueFactor(G, N, S);
      const h = tr / (fact(K - 1) * pa * vol);
      assert.ok(h >= 1 - 1e-8, `h=${h}`);
      assert.ok(h <= 1 / Math.max(...S.map((i) => a[i])) + 1e-8);
    }
  }
});

test('identity: displayed factor equals √det G_S up to h_S for K = 2, 3, 4 (full-dimensional view)', () => {
  const N = 30, D = 15;
  const { phi, rng } = conePool(N, D, 0.5, 3);
  const G = gram(phi);
  const { a, Gg } = gnomonic(G, N);
  const Y = pcaCoords(Gg, N, N - 1, true);
  for (const K of [2, 3, 4]) {
    const S = [];
    while (S.length < K) { const i = Math.floor(rng() * N); if (!S.includes(i)) S.push(i); }
    const disp = displayedFactor(Y, S, a), tr = trueFactor(G, N, S);
    // distance from 0 to the affine hull of g_S from first principles, via Gg-based least squares
    const m = K - 1;
    const Gs = (p, q) => Gg[S[p] * N + S[q]];
    const M = Array.from({ length: m }, (_, p) => Array.from({ length: m }, (_, q) => Gs(p + 1, q + 1) - Gs(p + 1, 0) - Gs(0, q + 1) + Gs(0, 0)));
    const bvec = Array.from({ length: m }, (_, p) => Gs(0, 0) - Gs(p + 1, 0));
    // solve M t = bvec (coefficients of the closest point g0 + Σ t_p (g_p − g0))
    const A = M.map((r, i) => [...r, bvec[i]]);
    for (let c = 0; c < m; c++) {
      let p = c; for (let r = c + 1; r < m; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
      [A[c], A[p]] = [A[p], A[c]];
      for (let r = 0; r < m; r++) if (r !== c) { const f = A[r][c] / A[c][c]; for (let j = c; j <= m; j++) A[r][j] -= f * A[c][j]; }
    }
    const t = A.map((r, i) => r[m] / r[i]);
    const w = [1 - t.reduce((s, x) => s + x, 0), ...t];
    let d2 = 0;
    for (let p = 0; p < K; p++) for (let q = 0; q < K; q++) d2 += w[p] * w[q] * Gs(p, q);
    const hS = Math.sqrt(1 + d2);
    assert.ok(Math.abs(tr - disp * hS) < 1e-9 * Math.max(1, tr), `K=${K}: ${tr} vs ${disp * hS}`);
  }
});

test('fitting lowers the loss and raises held-out Spearman on a synthetic pool', () => {
  const N = 60, D = 30;
  const { phi, logp, logq } = conePool(N, D, 0.7, 11);
  const G = gram(phi);
  const { a, Gg, ok } = gnomonic(G, N);
  assert.ok(ok);
  const init = pcaCoords(Gg, N, 3, true);
  const fit = fitLayout({ G, N, a, init, logp, logq, d: 3, seed: 1, iters: 200 });
  assert.ok(fit.loss1 < fit.loss0, `${fit.loss1} !< ${fit.loss0}`);
  const held = drawSubsets({ G, N, logp, logq, Ks: [3], seed: 99, nUniform: 200, nPerT: 0, kinds: ['uniform'] });
  const rankOf = (xs) => { const idx = xs.map((x, i) => [x, i]).sort((p, q) => p[0] - q[0]); const r = new Float64Array(xs.length); idx.forEach(([, i], k) => { r[i] = k; }); return r; };
  const rho = (Y) => {
    const x = rankOf(held.map((t) => displayedFactor(Y, t.S, a)));
    const y = rankOf(held.map((t) => trueFactor(G, N, t.S)));
    const mx = (x.length - 1) / 2;
    let sxy = 0, sxx = 0, syy = 0;
    for (let i = 0; i < x.length; i++) { sxy += (x[i] - mx) * (y[i] - mx); sxx += (x[i] - mx) ** 2; syy += (y[i] - mx) ** 2; }
    return sxy / Math.sqrt(sxx * syy);
  };
  const r0 = rho(init), r1 = rho(fit.Y);
  assert.ok(r1 > r0, `rho ${r1} !> ${r0}`);
});

test('bulkView flags an off-topic point and keeps the basis on the bulk', () => {
  const N = 40, D = 20;
  const { phi } = conePool(N, D, 0.3, 5);
  // Replace the last row by a vector nearly orthogonal to the cone axis.
  const ax = phi[0].map((_, k) => phi.slice(0, N - 1).reduce((s, r) => s + r[k], 0));
  const an = Math.hypot(...ax);
  const v0 = Float32Array.from({ length: D }, (_, k) => Math.sin(k + 1));
  const dot = v0.reduce((s, x, k) => s + x * ax[k] / an, 0);
  const orth = v0.map((x, k) => x - dot * ax[k] / an);
  const on = Math.hypot(...orth);
  phi[N - 1] = Float32Array.from(orth, (x, k) => 0.2 * ax[k] / an + Math.sqrt(1 - 0.04) * x / on);
  const G = gram(phi);
  const { Gg, ok } = gnomonic(G, N);
  assert.ok(ok);
  const bv = bulkView(Gg, N, 3);
  assert.equal(bv.bulk[N - 1], 0);
  assert.ok(bv.bulk.slice(0, N - 1).every((x) => x === 1));
  assert.ok(bv.coords.every((c) => c.every(Number.isFinite)));
  assert.ok(bv.frac[0] >= bv.frac[1] && bv.frac[1] >= bv.frac[2]);
  // Bulk points: the projection equals the centered PCA of the bulk-only Gram (up to axis signs).
  const Gb = new Float64Array((N - 1) ** 2);
  for (let i = 0; i < N - 1; i++) for (let j = 0; j < N - 1; j++) Gb[i * (N - 1) + j] = Gg[i * N + j];
  const ref = pcaCoords(Gb, N - 1, 3, true);
  for (let k = 0; k < 3; k++) {
    const s = Math.sign(ref[0][k] * bv.coords[0][k]) || 1;
    for (let i = 0; i < N - 1; i++) assert.ok(Math.abs(ref[i][k] - s * bv.coords[i][k]) < 1e-6);
  }
});

// ---------- spanView and openCone ----------

const detSmall = (M, k) => {
  const A = M.map((r) => r.slice());
  let d = 1;
  for (let c = 0; c < k; c++) {
    let p = c;
    for (let r = c + 1; r < k; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    if (p !== c) { [A[c], A[p]] = [A[p], A[c]]; d = -d; }
    d *= A[c][c];
    for (let r = c + 1; r < k; r++) { const f = A[r][c] / A[c][c]; for (let j = c; j < k; j++) A[r][j] -= f * A[c][j]; }
  }
  return d;
};
const dotN = (a, b) => { let s = 0; for (let k = 0; k < a.length; k++) s += a[k] * b[k]; return s; };
const sqrtDetG = (phi, S) => Math.sqrt(Math.max(detSmall(S.map((i) => S.map((j) => dotN(phi[i], phi[j]))), S.length), 0));

function checkExact(phi, sel, dims, ref) {
  const v = spanView(phi, sel, ref, dims);
  assert.ok(v.exact);
  const K = sel.length;
  for (const c of v.coords) { assert.equal(c.length, dims); assert.ok(Math.sqrt(dotN(c, c)) <= 1 + 1e-6); }
  for (const i of sel) for (const j of sel) assert.ok(Math.abs(dotN(v.coords[i], v.coords[j]) - dotN(phi[i], phi[j])) < 1e-6);
  const vol = Math.sqrt(Math.max(detSmall(sel.map((i) => sel.map((j) => dotN(v.coords[i], v.coords[j]))), K), 0));
  assert.ok(Math.abs(vol - sqrtDetG(phi, sel)) < 1e-6, `vol ${vol} vs ${sqrtDetG(phi, sel)}`);
  if (K === dims) assert.ok(Math.abs(Math.abs(detSmall(sel.map((i) => Array.from(v.coords[i])), K)) - sqrtDetG(phi, sel)) < 1e-6);
  return v;
}

test('spanView is exact for K <= dims on random data', () => {
  const N = 40, D = 24;
  const { phi, rng } = conePool(N, D, 1.5, 21);
  const pca = pcaCoords(gram(phi), N, 3, false);
  for (const dims of [1, 2, 3]) for (let K = 1; K <= dims; K++) {
    const sel = [];
    while (sel.length < K) { const i = Math.floor(rng() * N); if (!sel.includes(i)) sel.push(i); }
    const v = checkExact(phi, sel, dims, pca);
    // Orthogonal Procrustes keeps the view close to ref: no worse than a random rotation, and exactly ref if ref is already such a view.
    assert.equal(v.frac.length, dims);
  }
  // Continuity: if ref is itself a rotated exact view of the same set, spanView returns it (up to rounding).
  const sel = [3, 11, 25];
  const a = spanView(phi, sel, pca, 3);
  const b = spanView(phi, sel, a.coords, 3);
  for (let i = 0; i < N; i++) for (let k = 0; k < 3; k++) assert.ok(Math.abs(a.coords[i][k] - b.coords[i][k]) < 1e-6);
});

test('spanView falls back for K > dims, K = 0 and dependent vectors', () => {
  const N = 20, D = 10;
  const { phi } = conePool(N, D, 1, 4);
  const pca = pcaCoords(gram(phi), N, 3, false);
  assert.equal(spanView(phi, [0, 1, 2, 3], pca, 3, [0.5]).exact, false);
  assert.equal(spanView(phi, [0, 1, 2, 3], pca, 3, [0.5]).coords, pca);
  assert.deepEqual(spanView(phi, [0, 1, 2, 3], pca, 3, [0.5]).frac, [0.5]);
  assert.equal(spanView(phi, [], pca, 3).exact, false);
  const dup = phi.slice(); dup[5] = dup[2];
  assert.equal(spanView(dup, [2, 5], pca, 3).exact, false);
});

const galleryPool = 'data/pools/qwen3-14b/story_T1.5.json';
test('spanView on a gallery pool reproduces √det G_S', { skip: !existsSync(new URL('../' + galleryPool, import.meta.url)) }, () => {
  const pool = JSON.parse(readFileSync(new URL('../' + galleryPool, import.meta.url)));
  const f = new Float32Array(Buffer.from(pool.emb, 'base64').buffer.slice(0));
  const phi = Array.from({ length: pool.N }, (_, i) => f.subarray(i * pool.D, (i + 1) * pool.D));
  const pca = pcaCoords(gram(phi), pool.N, 3, false);
  for (const sel of [[0, 1, 2], [5, 17, 30], [1, 2], [7]]) for (const dims of [2, 3]) if (sel.length <= dims) checkExact(phi, sel, dims, pca);
});

test('openCone is linear and keeps volume ratios', () => {
  const N = 50, D = 8;
  const { phi, rng } = conePool(N, D, 0.4, 9);
  const X = pcaCoords(gram(phi), N, 3, false);
  const { coords, s } = openCone(X);
  assert.ok(s >= 1 && s <= 4);
  assert.ok(s > 1.0001, `s=${s}`);
  const vol = (C, S) => Math.abs(detSmall(S.map((i) => Array.from(C[i])), 3));
  const trip = () => { const S = []; while (S.length < 3) { const i = Math.floor(rng() * N); if (!S.includes(i)) S.push(i); } return S; };
  const [A, B] = [trip(), trip()];
  assert.ok(Math.abs(vol(coords, A) / vol(X, A) - s * s) < 1e-9 * s * s);
  assert.ok(Math.abs(vol(coords, A) / vol(coords, B) - vol(X, A) / vol(X, B)) < 1e-9 * vol(X, A) / vol(X, B));
  // Mean direction unchanged, perpendicular widths multiplied by s.
  const one = openCone([[1, 0, 0]]);
  assert.equal(one.s, 1);
});

test('clusterPool recovers well-separated groups, is deterministic and sorted by size', () => {
  const D = 16, rng = mulberry32(5);
  const gauss = () => Math.sqrt(-2 * Math.log(rng() + 1e-12)) * Math.cos(2 * Math.PI * rng());
  const axes = [0, 1, 2].map((a) => Float64Array.from({ length: D }, (_, d) => (d === a ? 1 : 0)));
  const sizes = [5, 12, 8];
  const phi = [], truth = [];
  sizes.forEach((n, g) => { for (let j = 0; j < n; j++) { const v = Float64Array.from(axes[g], (x) => x + 0.05 * gauss()); const nn = Math.hypot(...v); phi.push(Float32Array.from(v, (x) => x / nn)); truth.push(g); } });
  const a = clusterPool(phi, 3, 1), b = clusterPool(phi, 3, 1);
  assert.equal(a.k, 3);
  assert.deepEqual(Array.from(a.label), Array.from(b.label));
  const counts = [0, 0, 0];
  a.label.forEach((c) => counts[c]++);
  assert.deepEqual(counts, [12, 8, 5]);
  // same truth group -> same label
  const map = new Map();
  truth.forEach((g, i) => { if (!map.has(g)) map.set(g, a.label[i]); assert.equal(map.get(g), a.label[i]); });
  // k is capped by the number of distinct rows
  assert.equal(clusterPool([phi[0], phi[0], phi[1]], 6, 1).k, 2);
});

test('clusterKeywords picks the contrastive word and its usual capitalization', () => {
  const texts = ['Rain fell on Kyoto', 'The rain in Kyoto stopped', 'kyoto temples in the rain', 'A storm over the harbor', 'The harbor storm grew', 'Storm at the harbor again'];
  const label = [0, 0, 0, 1, 1, 1];
  const kw = clusterKeywords(texts, label, 2);
  assert.ok(['Kyoto', 'rain'].includes(kw[0]), kw[0]);
  assert.ok(['storm', 'harbor'].includes(kw[1]), kw[1]);
  // capitalization follows the most frequent form
  assert.equal(clusterKeywords(['Kyoto Kyoto', 'kyoto', 'Kyoto x', 'zebra yak', 'zebra', 'yak zebra'], [0, 0, 0, 1, 1, 1], 2)[0], 'Kyoto');
  // fallback: two words of the most likely text when nothing separates the clusters
  const fb = clusterKeywords(['aa bb cc', 'aa bb cc'], [0, 1], 2, [-5, -1]);
  assert.deepEqual(fb, ['aa bb', 'aa bb']);
});

test('fitView reproduces a rank-3 Gram from a perturbed start and lowers the loss on a general one', () => {
  const N = 30, rng = mulberry32(3);
  const gauss = () => Math.sqrt(-2 * Math.log(rng() + 1e-12)) * Math.cos(2 * Math.PI * rng());
  const T = Array.from({ length: N }, () => { const v = [gauss(), gauss(), gauss()]; const n = Math.hypot(...v); return v.map((x) => x / n); });
  const G = new Float64Array(N * N);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) G[i * N + j] = T[i][0] * T[j][0] + T[i][1] * T[j][1] + T[i][2] * T[j][2];
  const sets = [[0, 1, 2], [3, 4, 5], [6, 7, 8]];
  const init = T.map((r) => r.map((x) => x + 0.15 * gauss()));
  const fit = fitView(G, N, sets, init, { iters: 2500, lr: 0.02 });
  assert.ok(fit.loss < fit.loss0);
  assert.ok(fit.loss < 1e-6, `loss ${fit.loss}`);
  // determinism
  const again = fitView(G, N, sets, init, { iters: 50 }), again2 = fitView(G, N, sets, init, { iters: 50 });
  assert.deepEqual(Array.from(again.coords[0]), Array.from(again2.coords[0]));
  // general (rank > 3) Gram: the loss goes down
  const { phi } = conePool(40, 20, 1.5, 8);
  const G2 = gram(phi);
  const pc = pcaCoords(G2, 40, 3, false);
  const f2 = fitView(G2, 40, [[0, 1, 2], [3, 4, 5]], pc);
  assert.ok(f2.loss < f2.loss0);
  // alignTo undoes a rotation; pcaRotate sorts axes by decreasing share
  const rot = alignTo(f2.coords, pc, 3);
  assert.equal(rot.length, 40);
  const pr = pcaRotate(f2.coords, 3);
  assert.ok(pr.frac[0] >= pr.frac[1] && pr.frac[1] >= pr.frac[2]);
});

test('hull2D and polyArea', () => {
  const pts = [[0, 0], [2, 0], [2, 2], [0, 2], [1, 1], [1, 0], [0.5, 1.5]];
  const h = hull2D(pts);
  assert.equal(h.length, 4);
  assert.equal(polyArea(h), 4);
  assert.equal(hull2D([[0, 0], [1, 1]]).length, 2);
  assert.equal(polyArea(hull2D([[0, 0], [1, 1], [2, 2]])), 0);
});

test('tipSheet: open triangulated sheet over arrow tips', () => {
  // A flat sheet: square corners and a centre at height 1, so u = (0, 0, 1), 4 triangles, area 4.
  const flat = [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1], [0, 0, 1]];
  const s = tipSheet(flat);
  assert.equal(s.tris.length, 4);
  assert.ok(Math.abs(s.area - 4) < 1e-9);
  // Every point is used and each triangle is a distinct triple of indices.
  assert.deepEqual([...new Set(s.tris.flat())].sort(), [0, 1, 2, 3, 4]);
  // Lifting the centre makes the sheet larger than its projection.
  const bump = tipSheet([[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1], [0, 0, 1.5]]);
  assert.ok(bump.area > 4);
  // Points on a cone around the mean direction: triangles cover the hull of the projections, no overlap (sum of projected areas).
  const ring = Array.from({ length: 6 }, (_, k) => [Math.cos(k), Math.sin(k), 3].map((x, d) => (d < 2 ? 0.5 * x : x)));
  const r = tipSheet(ring);
  assert.ok(r.tris.length >= 4 && r.area > 0);
  // Degenerate input.
  assert.equal(tipSheet([[1, 0, 1], [2, 0, 1], [3, 0, 1], [4, 0, 1]]).tris.length, 0);
  assert.equal(tipSheet([[1, 0, 1], [1, 0, 1], [0, 1, 1], [0, 0, 2]]).area, 0);
  assert.equal(tipSheet([[0, 0, 1], [1, 0, 1]]).tris.length, 0);
});

// ---------- linkage clustering ----------

/** Three tight groups of 10 on separate axes plus two far singletons, in D dimensions. */
function groupsPool(seed = 4, D = 16) {
  const rng = mulberry32(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(rng() + 1e-12)) * Math.cos(2 * Math.PI * rng());
  const unit = (v) => { const n = Math.hypot(...v); return Float32Array.from(v, (x) => x / n); };
  const axis = (a) => Float64Array.from({ length: D }, (_, d) => (d === a ? 1 : 0));
  const phi = [];
  for (let g = 0; g < 3; g++) for (let j = 0; j < 10; j++) phi.push(unit(axis(g).map((x) => x + 0.04 * gauss())));
  // Two singletons on the far side of all three groups (farther from every group than the groups are from each other).
  phi.push(unit(Float64Array.from({ length: D }, (_, d) => (d < 3 ? -1 : 0))));
  phi.push(unit(Float64Array.from({ length: D }, (_, d) => (d < 3 ? -1 : d === 5 ? 1.5 : 0))));
  return phi;
}

test('complete linkage: three groups and two isolated points, diameters below the gap, deterministic', () => {
  const phi = groupsPool(), N = phi.length, D = distFromGram(gram(phi), N);
  const merges = clusterLinkage(D, N, { method: 'complete' });
  assert.equal(merges.length, N - 1);
  for (let i = 1; i < merges.length; i++) assert.ok(merges[i].d >= merges[i - 1].d - 1e-6);
  const cut = cutClusters(merges, N, { k: 3, minSize: 3, D });
  assert.equal(cut.k, 3);
  assert.equal(cut.nIsolated, 2);
  assert.equal(cut.label[30], -1); assert.equal(cut.label[31], -1);
  for (let g = 0; g < 3; g++) { const l = cut.label[g * 10]; for (let j = 0; j < 10; j++) assert.equal(cut.label[g * 10 + j], l); }
  assert.equal(new Set([cut.label[0], cut.label[10], cut.label[20]]).size, 3);
  // Between-group distance is about sqrt(2); every cluster diameter is far below it.
  for (const dm of cut.diam) assert.ok(dm < 0.5, `diam ${dm}`);
  const again = cutClusters(clusterLinkage(D, N), N, { k: 3, minSize: 3, D });
  assert.deepEqual(Array.from(again.label), Array.from(cut.label));
  // Average linkage finds the same groups here.
  const av = cutClusters(clusterLinkage(D, N, { method: 'average' }), N, { k: 3, minSize: 3, D });
  assert.equal(av.k, 3); assert.equal(av.nIsolated, 2);
  // Without isolation every point is colored.
  const all = cutClusters(merges, N, { k: 5, minSize: 1, D });
  assert.equal(all.nIsolated, 0); assert.equal(all.k, 5);
});

test('cutClusters edge cases', () => {
  const phi = groupsPool(), N = phi.length, D = distFromGram(gram(phi), N), merges = clusterLinkage(D, N);
  // k larger than available colored clusters: no crash, labels valid.
  const big = cutClusters(merges, N, { k: 500, minSize: 3, D });
  assert.ok(big.k >= 3);
  assert.ok(big.label.every((l) => l >= -1 && l < big.k));
  // Identical points: distances 0.
  const same = Array.from({ length: 12 }, () => Float32Array.from([1, 0, 0]));
  const Ds = distFromGram(gram(same), 12);
  const c = cutClusters(clusterLinkage(Ds, 12), 12, { k: 3, minSize: 1, D: Ds });
  assert.equal(c.label.length, 12);
  assert.ok(c.diam.every((x) => x === 0));
  assert.equal(cutClusters(clusterLinkage(new Float32Array([0]), 1), 1, { k: 3 }).k, 1);
});

test('linkage timing on random unit vectors', () => {
  for (const N of [200, 400]) {
    const { phi } = conePool(N, 24, 3, 2);
    const D = distFromGram(gram(phi), N);
    const t0 = performance.now();
    const m = clusterLinkage(D, N);
    cutClusters(m, N, { k: 6, minSize: 3, D });
    const ms = performance.now() - t0;
    assert.ok(ms < 2000, `N=${N}: ${ms} ms`);
  }
});

test('k-means isolation rule flags two planted outliers only', () => {
  const D = 16, rng = mulberry32(8);
  const gauss = () => Math.sqrt(-2 * Math.log(rng() + 1e-12)) * Math.cos(2 * Math.PI * rng());
  const unit = (v) => { const n = Math.hypot(...v); return Float32Array.from(v, (x) => x / n); };
  const phi = [];
  for (let g = 0; g < 2; g++) for (let j = 0; j < 25; j++) phi.push(unit(Float64Array.from({ length: D }, (_, d) => (d === g ? 1 : 0) + 0.05 * gauss())));
  // Two outliers between the groups' directions and well off both (they join a group when k = 2).
  phi.push(unit(Float64Array.from({ length: D }, (_, d) => (d === 0 ? 0.9 : d === 4 ? 1.1 : 0))));
  phi.push(unit(Float64Array.from({ length: D }, (_, d) => (d === 1 ? 0.9 : d === 9 ? -1.1 : 0))));
  const plain = clusterPool(phi, 2, 1), iso = clusterPool(phi, 2, 1, { isolate: true });
  assert.equal(plain.nIsolated, 0);
  assert.equal(iso.label[50], -1); assert.equal(iso.label[51], -1);
  assert.equal(iso.nIsolated, 2);
  assert.ok(iso.k === 2 && iso.label.slice(0, 50).every((l) => l >= 0));
});

test('clusterKeywords ignores isolated points', () => {
  const kw = clusterKeywords(['rain over Kyoto', 'Kyoto rain again', 'storm harbor', 'harbor storm now', 'odd thing'], [0, 0, 1, 1, -1], 2);
  assert.equal(kw.length, 2);
  assert.ok(['Kyoto', 'rain'].includes(kw[0]));
});
