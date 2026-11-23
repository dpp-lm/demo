// Display geometry for the K-DPP picture: spanView and openCone (used by the page), plus the older gnomonic layout (tests only). Depends only on ./dpp.js.
//
// All φ_i lie in a narrow cone around the mean direction m, so the plain PCA picture is a needle.
// Project each φ_i from the origin onto the tangent plane at m: g_i = φ_i/a_i − m, a_i = φ_i·m.
// Exact identity for S with |S| = K:
//   √det G_S = (K−1)! · ∏_{i∈S} a_i · Vol_{K−1}(conv g_S) · h_S,   h_S = √(1 + d_S²) ∈ [1, 1/max a_i],
// d_S = distance from 0 to the affine hull of g_S. So a set's DPP score is (∏ s_i · area)² up to h_S²,
// with s_i = c_i a_i the point size. The only loss left is the projection to d ∈ {2,3} drawn dimensions.

import {
  eigSym, spectrum, logDetSub, kernel, columnLogScale, sampleKDPP, greedyMAP, mulberry32,
} from './dpp.js';

export { eigSym };

const FACT = [1, 1, 2, 6, 24, 120, 720];

// ---------- gnomonic view ----------

/**
 * Point sizes' geometric part and the tangent-plane Gram, from G alone.
 * a_i = φ_i·m = √λ₁ v₁[i] (sign of v₁ chosen so Σa > 0); Gg_ij = G_ij/(a_i a_j) − 1 is the Gram of g_i = φ_i/a_i − m.
 * ok is false when some a_i ≤ 1e-6 (a point at or beyond the horizon of the tangent plane).
 */
export function gnomonic(G, N, eig = eigSym(G, N)) {
  const { values, vectors } = eig;
  const v = vectors[0], s = Math.sqrt(Math.max(values[0], 0));
  let sum = 0;
  for (let i = 0; i < N; i++) sum += v[i];
  const sg = sum < 0 ? -1 : 1;
  const a = new Float64Array(N);
  let ok = true;
  for (let i = 0; i < N; i++) { a[i] = sg * s * v[i]; if (!(a[i] > 1e-6)) ok = false; }
  const Gg = new Float64Array(N * N);
  if (ok) for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) Gg[i * N + j] = G[i * N + j] / (a[i] * a[j]) - 1;
  return { a, Gg, ok };
}

/**
 * Coordinates sqrt(λ_k) v_k[i] for the top d eigenpairs of M, as N rows of Float64Array(d).
 * center = true double-centers M first (PCA of the points with Gram M); false is the uncentered PCA of the arrows.
 */
export function pcaCoords(M, N, d, center = true) {
  return pcaView(M, N, d, center).coords;
}

/** pcaCoords plus frac: the top 8 eigenvalues as shares of the total (negative ones counted as 0). */
export function pcaView(M, N, d, center = true) {
  let A = M;
  if (center) {
    const rm = new Float64Array(N);
    let tot = 0;
    for (let i = 0; i < N; i++) { let s = 0; for (let j = 0; j < N; j++) s += M[i * N + j]; rm[i] = s / N; tot += s; }
    tot /= N * N;
    A = new Float64Array(N * N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) A[i * N + j] = M[i * N + j] - rm[i] - rm[j] + tot;
  }
  const { values, vectors } = eigSym(A, N);
  const total = values.reduce((s, v) => s + Math.max(0, v), 0) || 1;
  const coords = Array.from({ length: N }, (_, i) => {
    const y = new Float64Array(d);
    for (let k = 0; k < d && k < N; k++) y[k] = Math.sqrt(Math.max(values[k], 0)) * vectors[k][i];
    return y;
  });
  return { coords, frac: values.slice(0, 8).map((v) => Math.max(0, v) / total) };
}

/**
 * Outlier-robust PCA of the tangent-plane points with Gram Gg. r_i = |g_i| = √Gg_ii; the bulk is r_i ≤ max(3·median r, 95th percentile r).
 * The basis is fitted on the bulk only (double-centered Gg restricted to bulk rows and columns); every point, outliers included,
 * is then placed by the out-of-sample kernel-PCA projection y_j = Λ^{-1/2} Uᵀ k̃_j, a linear projection of g_j.
 * Returns { coords (N rows of Float64Array(d)), frac (top 8 bulk eigenvalue shares), bulk (Uint8Array), r, origin (image of g = 0) }.
 */
export function bulkView(Gg, N, d = 3) {
  const r = Float64Array.from({ length: N }, (_, i) => Math.sqrt(Math.max(Gg[i * N + i], 0)));
  const sorted = Array.from(r).sort((x, y) => x - y);
  const q = (p) => sorted[Math.min(N - 1, Math.floor(p * (N - 1)))];
  const thr = Math.max(3 * q(0.5), q(0.95));
  const B = [];
  for (let i = 0; i < N; i++) if (r[i] <= thr) B.push(i);
  const nb = B.length;
  const bulk = new Uint8Array(N);
  for (const i of B) bulk[i] = 1;
  const rm = new Float64Array(nb);
  let tot = 0;
  for (let p = 0; p < nb; p++) { let s = 0; for (let c = 0; c < nb; c++) s += Gg[B[p] * N + B[c]]; rm[p] = s / nb; tot += s; }
  tot /= nb * nb;
  const A = new Float64Array(nb * nb);
  for (let p = 0; p < nb; p++) for (let c = 0; c < nb; c++) A[p * nb + c] = Gg[B[p] * N + B[c]] - rm[p] - rm[c] + tot;
  const { values, vectors } = eigSym(A, nb);
  const total = values.reduce((s, v) => s + Math.max(0, v), 0) || 1;
  const coords = Array.from({ length: N }, () => new Float64Array(d));
  for (let j = 0; j < N; j++) {
    let cm = 0;
    for (let p = 0; p < nb; p++) cm += Gg[B[p] * N + j];
    cm /= nb;
    for (let k = 0; k < d && k < nb; k++) {
      if (!(values[k] > 1e-12)) continue;
      const U = vectors[k];
      let s = 0;
      for (let p = 0; p < nb; p++) s += U[p] * (Gg[B[p] * N + j] - rm[p] - cm + tot);
      coords[j][k] = s / Math.sqrt(values[k]);
    }
  }
  // Image of g = 0 (the mean direction m, the tangent point) under the same projection.
  const origin = new Float64Array(d);
  for (let k = 0; k < d && k < nb; k++) {
    if (!(values[k] > 1e-12)) continue;
    let s = 0;
    for (let p = 0; p < nb; p++) s += vectors[k][p] * (tot - rm[p]);
    origin[k] = s / Math.sqrt(values[k]);
  }
  return { coords, frac: values.slice(0, 8).map((v) => Math.max(0, v) / total), bulk, r, origin };
}

// ---------- exact view of the selected set ----------

const dotv = (a, b) => { let s = 0; for (let k = 0; k < a.length; k++) s += a[k] * b[k]; return s; };

/** Remove from v its components along the orthonormal rows of B (twice, for numerical stability); returns the residual norm before normalizing. */
function orthoNormalize(v, B) {
  for (let pass = 0; pass < 2; pass++) for (const q of B) { const c = dotv(v, q); for (let k = 0; k < v.length; k++) v[k] -= c * q[k]; }
  const n = Math.sqrt(dotv(v, v));
  if (n > 0) for (let k = 0; k < v.length; k++) v[k] /= n;
  return n;
}

/**
 * Rotate (or reflect) the rows of X by the orthogonal matrix that best matches ref (orthogonal Procrustes):
 * maximize Σ ref_i · (R X_i); M = Σ ref_i X_iᵀ = U S Vᵀ, R = U Vᵀ. Identity if M is rank deficient. Returns N rows of Float64Array(dims).
 */
export function alignTo(X, ref, dims = 3) {
  const N = X.length;
  const M = new Float64Array(dims * dims);
  for (let i = 0; i < N; i++) for (let a = 0; a < dims; a++) for (let b = 0; b < dims; b++) M[a * dims + b] += ref[i][a] * X[i][b];
  let Rm = null;
  if (dims === 1) Rm = [M[0] < 0 ? -1 : 1];
  else {
    const MtM = new Float64Array(dims * dims);
    for (let a = 0; a < dims; a++) for (let b = 0; b < dims; b++) { let s = 0; for (let c = 0; c < dims; c++) s += M[c * dims + a] * M[c * dims + b]; MtM[a * dims + b] = s; }
    const { values, vectors } = eigSym(MtM, dims);
    const sv = values.map((v) => Math.sqrt(Math.max(v, 0)));
    if (sv[0] > 1e-12 && sv[dims - 1] > 1e-9 * sv[0]) {
      Rm = new Array(dims * dims).fill(0);
      for (let k = 0; k < dims; k++) {
        const v = vectors[k], u = new Float64Array(dims);
        for (let a = 0; a < dims; a++) { let s = 0; for (let b = 0; b < dims; b++) s += M[a * dims + b] * v[b]; u[a] = s / sv[k]; }
        for (let a = 0; a < dims; a++) for (let b = 0; b < dims; b++) Rm[a * dims + b] += u[a] * v[b];
      }
    }
  }
  return X.map((x) => {
    const y = new Float64Array(dims);
    if (!Rm) y.set(Float64Array.from(x).subarray(0, dims));
    else if (dims === 1) y[0] = Rm[0] * x[0];
    else for (let a = 0; a < dims; a++) { let s = 0; for (let b = 0; b < dims; b++) s += Rm[a * dims + b] * x[b]; y[a] = s; }
    return y;
  });
}

/** Rotate the rows of X onto their own principal axes (uncentered), so axis 1 carries the most. Returns { coords, frac } with frac_k = Σ_i y_ik² / N. */
export function pcaRotate(X, dims = 3) {
  const N = X.length, M = new Float64Array(dims * dims);
  for (const x of X) for (let a = 0; a < dims; a++) for (let b = 0; b < dims; b++) M[a * dims + b] += x[a] * x[b];
  const { vectors } = eigSym(M, dims);
  const coords = X.map((x) => Float64Array.from(vectors, (v) => { let s = 0; for (let a = 0; a < dims; a++) s += v[a] * x[a]; return s; }));
  return { coords, frac: Array.from({ length: dims }, (_, k) => coords.reduce((s, c) => s + c[k] * c[k], 0) / N) };
}

/**
 * Fixed display coordinates fitted to the sets the DPP picks: minimize Σ_{i≤j} w_ij (x_i·x_j − G_ij)² over X (N×dims),
 * w_ij = eps + (number of sets in `sets` containing both i and j; i = j: containing i), divided by Σ_{i≤j} w_ij.
 * Adam from `init` (rows), cosine-decayed step, deterministic. opts: { eps = 0.02, iters = 400, lr = 0.02, dims = 3 }.
 * Returns { coords (N rows of Float64Array(dims)), loss0, loss }.
 */
export function fitView(G, N, sets, init, opts = {}) {
  const { eps = 0.02, iters = 400, lr = 0.02, dims = 3 } = opts;
  const W = new Float64Array(N * N).fill(eps);
  for (const S of sets) for (const i of S) for (const j of S) W[i * N + j] += 1;
  let wsum = 0;
  for (let i = 0; i < N; i++) for (let j = i; j < N; j++) wsum += W[i * N + j];
  const X = new Float64Array(N * dims), g = new Float64Array(N * dims), m1 = new Float64Array(N * dims), m2 = new Float64Array(N * dims);
  for (let i = 0; i < N; i++) for (let d = 0; d < dims; d++) X[i * dims + d] = init[i][d];
  const lossGrad = () => {
    let loss = 0;
    g.fill(0);
    for (let i = 0; i < N; i++) for (let j = i; j < N; j++) {
      let dp = 0;
      for (let d = 0; d < dims; d++) dp += X[i * dims + d] * X[j * dims + d];
      const r = dp - G[i * N + j], w = W[i * N + j];
      loss += w * r * r;
      if (i === j) { for (let d = 0; d < dims; d++) g[i * dims + d] += 4 * w * r * X[i * dims + d]; }
      else for (let d = 0; d < dims; d++) { g[i * dims + d] += 2 * w * r * X[j * dims + d]; g[j * dims + d] += 2 * w * r * X[i * dims + d]; }
    }
    for (let k = 0; k < g.length; k++) g[k] /= wsum;
    return loss / wsum;
  };
  const loss0 = lossGrad();
  let loss = loss0;
  const b1 = 0.9, b2 = 0.999;
  for (let it = 0; it < iters; it++) {
    const step = lr * (0.05 + 0.95 * 0.5 * (1 + Math.cos((Math.PI * it) / iters)));
    for (let k = 0; k < X.length; k++) {
      m1[k] = b1 * m1[k] + (1 - b1) * g[k];
      m2[k] = b2 * m2[k] + (1 - b2) * g[k] * g[k];
      X[k] -= (step * (m1[k] / (1 - b1 ** (it + 1)))) / (Math.sqrt(m2[k] / (1 - b2 ** (it + 1))) + 1e-12);
    }
    loss = lossGrad();
  }
  return { coords: Array.from({ length: N }, (_, i) => Float64Array.from(X.subarray(i * dims, (i + 1) * dims))), loss0, loss };
}

/**
 * Display coordinates in which the selected vectors are drawn exactly.
 * phi: N unit vectors (length D). sel: selected indices. ref: N rows (≥ dims numbers), the fixed coordinates used for continuity.
 * If 1 ≤ K = |sel| ≤ dims and the selected vectors are independent, the axes are an orthonormal basis Q of span{φ_i : i ∈ sel}
 * completed with the top principal directions of the pool outside that span, then rotated (orthogonal Procrustes, reflections allowed)
 * to match ref as closely as possible. Every selected-set quantity (Gram, volume) is then exact in the drawing.
 * Otherwise returns { coords: ref, exact: false, frac: refFrac }.
 * frac_k = Σ_i coords[i][k]² / N (the share of the pool's spread on axis k).
 */
export function spanView(phi, sel, ref, dims = 3, refFrac = null) {
  const fallback = { coords: ref, exact: false, frac: refFrac };
  const N = phi.length, K = sel.length;
  if (K < 1 || K > dims || !N) return fallback;
  const D = phi[0].length;
  const Q = [];
  for (const i of sel) {
    const v = Float64Array.from(phi[i]);
    if (!(orthoNormalize(v, Q) > 1e-6)) return fallback;
    Q.push(v);
  }
  // Complete the basis with the top principal directions of the residuals (power iteration, started along ref).
  if (K < dims) {
    const R = phi.map((p) => { const r = Float64Array.from(p); for (const q of Q) { const c = dotv(r, q); for (let k = 0; k < D; k++) r[k] -= c * q[k]; } return r; });
    let big = 0;
    R.forEach((r, i) => { if (dotv(r, r) > dotv(R[big], R[big])) big = i; });
    for (let e = K; e < dims; e++) {
      let v = new Float64Array(D);
      R.forEach((r, i) => { const w = ref[i][e]; for (let k = 0; k < D; k++) v[k] += w * r[k]; });
      if (!(orthoNormalize(v, Q) > 1e-9)) { v = Float64Array.from(R[big]); orthoNormalize(v, Q); }
      for (let it = 0; it < 60; it++) {
        const w = new Float64Array(D);
        for (const r of R) { const c = dotv(r, v); if (c !== 0) for (let k = 0; k < D; k++) w[k] += c * r[k]; }
        if (!(orthoNormalize(w, Q) > 1e-12)) break;
        v = w;
      }
      if (!(orthoNormalize(v, Q) > 1e-9)) {
        // No residual left: take the first coordinate axis that is not in the span so far.
        for (let j = 0; j < D; j++) { v = new Float64Array(D); v[j] = 1; if (orthoNormalize(v, Q) > 1e-6) break; }
      }
      Q.push(v);
    }
  }
  const X = phi.map((p) => Float64Array.from(Q, (q) => dotv(p, q)));
  const coords = alignTo(X, ref, dims);
  const frac = Array.from({ length: dims }, (_, k) => coords.reduce((s, c) => s + c[k] * c[k], 0) / N);
  return { coords, exact: true, frac };
}

/**
 * Open the cone: one linear map A = m̂m̂ᵀ + s(I − m̂m̂ᵀ), m̂ the normalized mean of the rows, s = clamp(rmsAxial / rmsPerp, 1, 4).
 * Parallelepipeds map to parallelepipeds and every d-dimensional volume is multiplied by s^(d−1) (d = 1: no change), so volume ratios are unchanged.
 * Returns { coords, s }.
 */
export function openCone(coords) {
  const N = coords.length, d = coords[0]?.length ?? 0;
  const mean = new Float64Array(d);
  for (const c of coords) for (let k = 0; k < d; k++) mean[k] += c[k] / N;
  const mn = Math.sqrt(dotv(mean, mean));
  if (d < 2 || !(mn > 1e-12)) return { coords, s: 1 };
  const m = mean.map((x) => x / mn);
  let ax = 0, pp = 0;
  for (const c of coords) { const a = dotv(c, m); ax += a * a; pp += dotv(c, c) - a * a; }
  const rmsAx = Math.sqrt(ax / N), rmsPerp = Math.sqrt(Math.max(pp, 0) / N);
  const s = rmsPerp > 1e-12 ? Math.min(4, Math.max(1, rmsAx / rmsPerp)) : 1;
  const out = coords.map((c) => {
    const a = dotv(c, m), y = new Float64Array(d);
    for (let k = 0; k < d; k++) y[k] = a * m[k] + s * (c[k] - a * m[k]);
    return y;
  });
  return { coords: out, s };
}

// ---------- clusters ----------

/** Distances between unit vectors from their Gram matrix: sqrt(2 − 2 cos). Float32Array N×N. */
export function distFromGram(G, N) {
  const D = new Float32Array(N * N);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) D[i * N + j] = i === j ? 0 : Math.sqrt(Math.max(0, 2 - 2 * G[i * N + j]));
  return D;
}

/** Euclidean distances between rows (arrays of numbers). Float32Array N×N. */
export function distFromCoords(rows) {
  const N = rows.length, D = new Float32Array(N * N);
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
    let s = 0;
    for (let d = 0; d < rows[i].length; d++) s += (rows[i][d] - rows[j][d]) ** 2;
    D[i * N + j] = D[j * N + i] = Math.sqrt(s);
  }
  return D;
}

/**
 * Agglomerative clustering on a distance matrix D (N×N), method 'complete' (merge the two clusters whose farthest pair is closest;
 * every cluster's diameter is the height at which it formed) or 'average'. Lance–Williams updates with a cached nearest neighbour per
 * cluster; ties go to the smallest index, so the result is deterministic. Returns the merge sequence
 * [{ a, b, d, size }] (N − 1 entries; leaves are ids 0..N−1, merge m forms id N + m), heights non-decreasing.
 */
export function clusterLinkage(D, N, { method = 'complete' } = {}) {
  const avg = method === 'average';
  const d = Float64Array.from(D);
  const id = Int32Array.from({ length: N }, (_, i) => i), size = new Int32Array(N).fill(1), active = new Uint8Array(N).fill(1);
  const nn = new Int32Array(N), nnd = new Float64Array(N);
  const recompute = (i) => {
    let b = -1, bd = Infinity;
    for (let j = 0; j < N; j++) if (j !== i && active[j] && d[i * N + j] < bd) { bd = d[i * N + j]; b = j; }
    nn[i] = b; nnd[i] = bd;
  };
  for (let i = 0; i < N; i++) recompute(i);
  const merges = [];
  for (let step = 0; step < N - 1; step++) {
    let a = -1, best = Infinity;
    for (let i = 0; i < N; i++) if (active[i] && nnd[i] < best) { best = nnd[i]; a = i; }
    const lo = Math.min(a, nn[a]), hi = Math.max(a, nn[a]);
    for (let k = 0; k < N; k++) {
      if (!active[k] || k === lo || k === hi) continue;
      const v = avg ? (size[lo] * d[lo * N + k] + size[hi] * d[hi * N + k]) / (size[lo] + size[hi]) : Math.max(d[lo * N + k], d[hi * N + k]);
      d[lo * N + k] = d[k * N + lo] = v;
    }
    merges.push({ a: id[lo], b: id[hi], d: best, size: size[lo] + size[hi] });
    id[lo] = N + step; size[lo] += size[hi]; active[hi] = 0;
    for (let k = 0; k < N; k++) {
      if (!active[k] || k === lo) continue;
      if (nn[k] === lo || nn[k] === hi) recompute(k);
      else if (d[k * N + lo] < nnd[k]) { nn[k] = lo; nnd[k] = d[k * N + lo]; }
    }
    recompute(lo);
  }
  return merges;
}

/** Largest pairwise distance within each cluster 0..k−1 (Float32Array); label −1 is ignored. */
export function clusterDiameters(label, k, D, N) {
  const diam = new Float32Array(k);
  for (let i = 0; i < N; i++) if (label[i] >= 0) for (let j = i + 1; j < N; j++) if (label[j] === label[i] && D[i * N + j] > diam[label[i]]) diam[label[i]] = D[i * N + j];
  return diam;
}

/**
 * Cut a dendrogram from the top: repeatedly split the piece formed at the greatest height until k pieces have at least minSize
 * members, or the number of pieces reaches max(k, N/3), or nothing can be split; the state with the most such pieces is kept. Pieces smaller than minSize are isolated (label −1).
 * Colored pieces are numbered 0.. by decreasing size (ties: smallest member index). D (optional) gives the diameters.
 * Returns { label: Int32Array, k (colored clusters), diam: Float32Array, nIsolated }.
 */
export function cutClusters(merges, N, { k = 6, minSize = 1, D = null } = {}) {
  const root = N === 1 ? 0 : N + merges.length - 1;
  const memb = new Array(N + merges.length);
  for (let i = 0; i < N; i++) memb[i] = [i];
  merges.forEach((m, t) => { memb[N + t] = memb[m.a].concat(memb[m.b]); });
  const minIdx = (p) => Math.min(...memb[p]);
  let pieces = [root];
  const maxPieces = Math.max(k, Math.ceil(N / 3));
  const good = () => pieces.filter((p) => memb[p].length >= minSize).length;
  let bestPieces = pieces.slice(), bestGood = good();
  while (good() < k && pieces.length < maxPieces) {
    let best = -1;
    for (let q = 0; q < pieces.length; q++) {
      const p = pieces[q];
      if (p < N) continue;
      if (best < 0 || merges[p - N].d > merges[pieces[best] - N].d || (merges[p - N].d === merges[pieces[best] - N].d && minIdx(p) < minIdx(pieces[best]))) best = q;
    }
    if (best < 0) break;
    const m = merges[pieces[best] - N];
    pieces.splice(best, 1, m.a, m.b);
    if (good() > bestGood) { bestGood = good(); bestPieces = pieces.slice(); }
  }
  pieces = bestPieces; // the state with the most pieces of at least minSize (the first such state)
  const colored = pieces.filter((p) => memb[p].length >= minSize).sort((x, y) => memb[y].length - memb[x].length || minIdx(x) - minIdx(y));
  const label = new Int32Array(N).fill(-1);
  colored.forEach((p, c) => memb[p].forEach((i) => { label[i] = c; }));
  const kk = colored.length;
  return { label, k: kk, diam: D ? clusterDiameters(label, kk, D, N) : new Float32Array(kk), nIsolated: label.reduce((a, l) => a + (l < 0 ? 1 : 0), 0) };
}

/**
 * Spherical k-means on unit vectors: k-means++ start (cosine distance, seeded mulberry32), cosine assignment, at most 30 iterations.
 * k = min(k, number of distinct rows). Clusters are renumbered by decreasing size (ties: the cluster with the earliest member first).
 * With isolate, a point in a cluster of at least 4 members is set to label −1 when its distance to the centroid exceeds the cluster's
 * median + 3·MAD and its nearest other member is farther than 3× the cluster's median nearest-member distance.
 * Returns { label: Int32Array(N) (−1 isolated), k (clusters), nIsolated }. Deterministic for a given seed.
 */
export function clusterPool(phi, k = 6, seed = 1, { isolate = false } = {}) {
  const N = phi.length;
  if (!N) return { label: new Int32Array(0), k: 0 };
  const D = phi[0].length;
  const dot = (a, b) => { let s = 0; for (let d = 0; d < D; d++) s += a[d] * b[d]; return s; };
  const seen = new Map();
  phi.forEach((p, i) => { const key = Array.prototype.join.call(p, ','); if (!seen.has(key)) seen.set(key, i); });
  const distinct = [...seen.values()];
  k = Math.max(1, Math.min(k, distinct.length));
  const rng = mulberry32(seed);
  const C = [Float64Array.from(phi[distinct[Math.floor(rng() * distinct.length)]])];
  const best = new Float64Array(N).fill(-Infinity);
  while (C.length < k) {
    const last = C[C.length - 1];
    let tot = 0;
    const w = new Float64Array(N);
    for (let i = 0; i < N; i++) { best[i] = Math.max(best[i], dot(phi[i], last)); w[i] = Math.max(0, 1 - best[i]); tot += w[i]; }
    let pick = -1;
    if (tot > 1e-12) { let r = rng() * tot; for (let i = 0; i < N; i++) { r -= w[i]; if (r <= 0 && w[i] > 0) { pick = i; break; } } if (pick < 0) pick = w.indexOf(Math.max(...w)); }
    else pick = distinct.find((i) => !C.some((c) => dot(phi[i], c) > 1 - 1e-9)) ?? distinct[0];
    C.push(Float64Array.from(phi[pick]));
  }
  const label = new Int32Array(N).fill(-1);
  for (let it = 0; it < 30; it++) {
    let changed = false;
    for (let i = 0; i < N; i++) {
      let b = 0, bs = -Infinity;
      for (let c = 0; c < k; c++) { const v = dot(phi[i], C[c]); if (v > bs) { bs = v; b = c; } }
      if (label[i] !== b) { label[i] = b; changed = true; }
    }
    if (!changed) break;
    for (let c = 0; c < k; c++) {
      const m = new Float64Array(D);
      let n = 0;
      for (let i = 0; i < N; i++) if (label[i] === c) { n++; for (let d = 0; d < D; d++) m[d] += phi[i][d]; }
      if (!n) { // empty: restart from the point farthest from its center
        let far = 0, fs = Infinity;
        for (let i = 0; i < N; i++) { const v = dot(phi[i], C[label[i]]); if (v < fs) { fs = v; far = i; } }
        C[c] = Float64Array.from(phi[far]);
        continue;
      }
      const nm = Math.sqrt(dot(m, m)) || 1;
      for (let d = 0; d < D; d++) m[d] /= nm;
      C[c] = m;
    }
  }
  if (isolate) {
    const med = (x) => { const y = x.slice().sort((p, q) => p - q); return y[Math.floor(y.length / 2)]; };
    const drop = [];
    for (let c = 0; c < k; c++) {
      const M = [];
      for (let i = 0; i < N; i++) if (label[i] === c) M.push(i);
      if (M.length < 4) continue;
      const dc = M.map((i) => Math.sqrt(Math.max(0, 2 - 2 * dot(phi[i], C[c]))));
      const nnd = M.map((i) => Math.min(...M.filter((j) => j !== i).map((j) => Math.sqrt(Math.max(0, 2 - 2 * dot(phi[i], phi[j]))))));
      const m1 = med(dc), mad = med(dc.map((x) => Math.abs(x - m1))), nm = med(nnd);
      M.forEach((i, t) => { if (dc[t] > m1 + 3 * mad && nnd[t] > 3 * nm) drop.push(i); });
    }
    for (const i of drop) label[i] = -1;
  }
  const size = new Array(k).fill(0), first = new Array(k).fill(N);
  for (let i = 0; i < N; i++) if (label[i] >= 0) { size[label[i]]++; first[label[i]] = Math.min(first[label[i]], i); }
  const order = [...Array(k).keys()].filter((c) => size[c] > 0).sort((a, b) => size[b] - size[a] || first[a] - first[b]);
  const rank = new Int32Array(k).fill(-1);
  order.forEach((c, r) => { rank[c] = r; });
  for (let i = 0; i < N; i++) label[i] = label[i] < 0 ? -1 : rank[label[i]];
  return { label, k: order.length, nIsolated: label.reduce((a, l) => a + (l < 0 ? 1 : 0), 0) };
}

const STOP = new Set(('the and for with that this are was were has had have not but you your from they their them there then than into out about over under its who whom what when where which while will would could should can may might just also very more most some any all each one two how why too our his her she him use used using like said says get got make made').split(' '));

/**
 * One short keyword per cluster. Words: lowercase Unicode letter runs of length ≥ 3, minus a small stopword list.
 * Score = share of the cluster's texts containing the word − share of the other texts containing it; best positive score wins
 * (ties: more cluster texts, then alphabetical). The word is written the way it most often appears in the texts.
 * Fallback: the first two words of the cluster's most likely text (logp given) or first text.
 */
export function clusterKeywords(texts, label, k, logp = null) {
  const N = texts.length;
  const toks = texts.map((t) => (String(t).match(/\p{L}+/gu) ?? []));
  const uniq = toks.map((w) => new Set(w.map((x) => x.toLowerCase()).filter((x) => x.length >= 3 && !STOP.has(x))));
  const n = new Array(k).fill(0);
  for (let i = 0; i < N; i++) if (label[i] >= 0) n[label[i]]++;
  const df = Array.from({ length: k }, () => new Map());
  const all = new Map();
  for (let i = 0; i < N; i++) for (const w of uniq[i]) { if (label[i] >= 0) df[label[i]].set(w, (df[label[i]].get(w) ?? 0) + 1); all.set(w, (all.get(w) ?? 0) + 1); }
  // Proper nouns (capitalized away from the sentence start in most uses) make better names, so their score is doubled.
  const cap = new Map(), tot = new Map();
  for (const w of toks) w.forEach((x, j) => { const l = x.toLowerCase(); tot.set(l, (tot.get(l) ?? 0) + 1); if (j > 0 && x[0] !== l[0]) cap.set(l, (cap.get(l) ?? 0) + 1); });
  const proper = (w) => (cap.get(w) ?? 0) >= 0.6 * tot.get(w);
  const out = [], used = new Set();
  for (let c = 0; c < k; c++) {
    if (!n[c]) { out.push(''); continue; }
    let bw = null, bs = 0, bn = 0;
    for (const [w, cnt] of df[c]) {
      const rest = N - n[c];
      if (used.has(w)) continue; // one name per cluster
      const sc = (cnt / n[c] - (rest ? (all.get(w) - cnt) / rest : 0)) * (proper(w) ? 2 : 1);
      if (sc <= 1e-12) continue;
      const tie = Math.abs(sc - bs) <= 1e-12 && (cnt > bn || (cnt === bn && w < bw));
      if (bw === null || sc > bs + 1e-12 || tie) { bw = w; bs = sc; bn = cnt; }
    }
    if (bw) {
      used.add(bw);
      const forms = new Map();
      for (let i = 0; i < N; i++) if (label[i] === c) for (const x of toks[i]) if (x.toLowerCase() === bw) forms.set(x, (forms.get(x) ?? 0) + 1);
      out.push([...forms].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))[0][0]);
    } else {
      let top = -1;
      for (let i = 0; i < N; i++) if (label[i] === c && (top < 0 || (logp && logp[i] > logp[top]))) top = i;
      out.push(toks[top].slice(0, 2).join(' '));
    }
  }
  return out;
}

// ---------- open sheet through the tips ----------

/** Delaunay triangulation of 2D points (Bowyer–Watson). Returns index triples. No duplicate points allowed. */
function delaunay2D(P) {
  const n = P.length;
  let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
  for (const p of P) for (let d = 0; d < 2; d++) { lo[d] = Math.min(lo[d], p[d]); hi[d] = Math.max(hi[d], p[d]); }
  const span = Math.max(hi[0] - lo[0], hi[1] - lo[1]) || 1, cx = (lo[0] + hi[0]) / 2, cy = (lo[1] + hi[1]) / 2, M = 20 * span;
  const V = P.concat([[cx - M, cy - M], [cx + M, cy - M], [cx, cy + M]]);
  const orient = (a, b, c) => (V[b][0] - V[a][0]) * (V[c][1] - V[a][1]) - (V[b][1] - V[a][1]) * (V[c][0] - V[a][0]);
  const inCircle = (t, p) => {
    let [a, b, c] = t;
    if (orient(a, b, c) < 0) [b, c] = [c, b];
    const ax = V[a][0] - p[0], ay = V[a][1] - p[1], bx = V[b][0] - p[0], by = V[b][1] - p[1], cx2 = V[c][0] - p[0], cy2 = V[c][1] - p[1];
    return (ax * ax + ay * ay) * (bx * cy2 - cx2 * by) - (bx * bx + by * by) * (ax * cy2 - cx2 * ay) + (cx2 * cx2 + cy2 * cy2) * (ax * by - bx * ay) > 1e-12 * span ** 4;
  };
  let tris = [[n, n + 1, n + 2]];
  for (let i = 0; i < n; i++) {
    const bad = tris.filter((t) => inCircle(t, V[i]));
    const edges = new Map();
    for (const t of bad) for (let k = 0; k < 3; k++) {
      const a = t[k], b = t[(k + 1) % 3], key = a < b ? `${a},${b}` : `${b},${a}`;
      edges.set(key, edges.has(key) ? null : [a, b]);
    }
    tris = tris.filter((t) => !bad.includes(t));
    for (const e of edges.values()) if (e) tris.push([e[0], e[1], i]);
  }
  return tris.filter((t) => t.every((v) => v < n));
}

/**
 * An open triangulated sheet through K 3D points (a cloth over the tips of K arrows). The points are projected onto the plane
 * perpendicular to u, the normalized mean of the points as vectors from the origin (if that is near zero, the smallest-variance
 * principal axis), and the projections are Delaunay-triangulated. Returns { tris: index triples, area: summed 3D triangle area, u }.
 * Duplicate or collinear projections, or fewer than 3 points: { tris: [], area: 0 }.
 */
export function tipSheet(points) {
  const K = points.length, none = { tris: [], area: 0, u: null };
  if (K < 3) return none;
  const m = [0, 1, 2].map((d) => points.reduce((a, p) => a + p[d], 0) / K);
  const scale = Math.max(...points.map((p) => Math.hypot(p[0], p[1], p[2]))) || 1;
  let u = m.slice(), un = Math.hypot(...u);
  if (!(un > 1e-6 * scale)) {
    const C = new Float64Array(9);
    for (const p of points) for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) C[a * 3 + b] += (p[a] - m[a]) * (p[b] - m[b]);
    const { vectors } = eigSym(C, 3);
    u = Array.from(vectors[2]); un = Math.hypot(...u);
  }
  if (!(un > 0)) return none;
  u = u.map((x) => x / un);
  // Any orthonormal basis (e1, e2) of the plane perpendicular to u.
  const ax = u.map(Math.abs).indexOf(Math.min(...u.map(Math.abs)));
  const a0 = [0, 0, 0]; a0[ax] = 1;
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  let e1 = cross(u, a0); const n1 = Math.hypot(...e1); e1 = e1.map((x) => x / n1);
  const e2 = cross(u, e1);
  const P = points.map((p) => [p[0] * e1[0] + p[1] * e1[1] + p[2] * e1[2], p[0] * e2[0] + p[1] * e2[1] + p[2] * e2[2]]);
  for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) if (Math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]) < 1e-9 * scale) return none;
  const A2 = (t) => Math.abs((P[t[1]][0] - P[t[0]][0]) * (P[t[2]][1] - P[t[0]][1]) - (P[t[1]][1] - P[t[0]][1]) * (P[t[2]][0] - P[t[0]][0])) / 2;
  const tris = delaunay2D(P).filter((t) => A2(t) > 1e-12 * scale * scale);
  if (!tris.length) return none;
  let area = 0;
  for (const t of tris) {
    const [p, q, r] = t.map((i) => points[i]);
    const c = cross([q[0] - p[0], q[1] - p[1], q[2] - p[2]], [r[0] - p[0], r[1] - p[1], r[2] - p[2]]);
    area += Math.hypot(...c) / 2;
  }
  return { tris, area, u };
}

// ---------- hull of the tips ----------

/** Convex hull of 2D points (monotone chain), counter-clockwise, without repeated points. Fewer than 3 distinct points: returned as is. */
export function hull2D(pts) {
  const P = pts.map((p) => [p[0], p[1]]).sort((a, b) => a[0] - b[0] || a[1] - b[1]).filter((p, i, a) => !i || p[0] !== a[i - 1][0] || p[1] !== a[i - 1][1]);
  if (P.length < 3) return P;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (arr) => { const h = []; for (const p of arr) { while (h.length >= 2 && cross(h[h.length - 2], h[h.length - 1], p) <= 0) h.pop(); h.push(p); } h.pop(); return h; };
  return half(P).concat(half(P.slice().reverse()));
}

/** Area of a simple polygon (shoelace). */
export function polyArea(poly) {
  let a = 0;
  for (let i = 0; i < poly.length; i++) { const p = poly[i], q = poly[(i + 1) % poly.length]; a += p[0] * q[1] - q[0] * p[1]; }
  return Math.abs(a) / 2;
}

// ---------- geometry ----------

/** det of a small symmetric positive matrix (k ≤ 3 closed form, else Gaussian elimination). */
function det(M, k) {
  if (k === 1) return M[0];
  if (k === 2) return M[0] * M[3] - M[1] * M[2];
  if (k === 3) return M[0] * (M[4] * M[8] - M[5] * M[7]) - M[1] * (M[3] * M[8] - M[5] * M[6]) + M[2] * (M[3] * M[7] - M[4] * M[6]);
  const A = Array.from(M);
  let dt = 1;
  for (let c = 0; c < k; c++) {
    let p = c;
    for (let r = c + 1; r < k; r++) if (Math.abs(A[r * k + c]) > Math.abs(A[p * k + c])) p = r;
    if (A[p * k + c] === 0) return 0;
    if (p !== c) { for (let j = 0; j < k; j++) [A[c * k + j], A[p * k + j]] = [A[p * k + j], A[c * k + j]]; dt = -dt; }
    dt *= A[c * k + c];
    for (let r = c + 1; r < k; r++) { const f = A[r * k + c] / A[c * k + c]; for (let j = c; j < k; j++) A[r * k + j] -= f * A[c * k + j]; }
  }
  return dt;
}

/** Vol_{K−1}(conv y_S) = √det(EᵀE)/(K−1)! with E = [y_j − y_first]. Y: array of rows (any dimension ≥ K−1). */
export function simplexVolume(Y, S) {
  const K = S.length, m = K - 1, dim = Y[S[0]].length;
  if (m < 1) return 1;
  const y0 = Y[S[0]];
  const M = new Float64Array(m * m);
  for (let p = 0; p < m; p++) for (let q = p; q < m; q++) {
    let s = 0;
    const yp = Y[S[p + 1]], yq = Y[S[q + 1]];
    for (let c = 0; c < dim; c++) s += (yp[c] - y0[c]) * (yq[c] - y0[c]);
    M[p * m + q] = M[q * m + p] = s;
  }
  return Math.sqrt(Math.max(det(M, m), 0)) / FACT[m];
}

/**
 * Geometric factor shown for S. With a: (K−1)! ∏a_i Vol_{K−1}(conv y_S) (gnomonic layouts).
 * Without a: √det(Y_SᵀY_S), the parallelepiped volume of the arrows (needs K ≤ d).
 */
export function displayedFactor(Y, S, a = null) {
  const K = S.length;
  if (a) {
    let p = FACT[K - 1];
    for (const i of S) p *= a[i];
    return p * simplexVolume(Y, S);
  }
  const dim = Y[S[0]].length;
  if (K > dim) return NaN;
  const M = new Float64Array(K * K);
  for (let p = 0; p < K; p++) for (let q = p; q < K; q++) {
    let s = 0;
    for (let c = 0; c < dim; c++) s += Y[S[p]][c] * Y[S[q]][c];
    M[p * K + q] = M[q * K + p] = s;
  }
  return Math.sqrt(Math.max(det(M, K), 0));
}

/** √det G_S, the target of the geometric factor. */
export const trueFactor = (G, N, S) => Math.exp(0.5 * logDetSub(G, N, S));

// ---------- training and evaluation subsets ----------

function uniformSubset(N, K, rng) {
  const S = new Set();
  while (S.size < K) S.add(Math.floor(rng() * N));
  return [...S].sort((x, y) => x - y);
}

/**
 * Subsets drawn the way the page draws them: uniform, K-DPP samples at each T (sampling kernel), greedy MAP at each T.
 * Returns [{ S, K, T, kind }] with kind ∈ 'uniform' | 'dpp' | 'map'. Subsets with det G_S ≤ 1e-12 are dropped.
 * Spectra are shared across K. Pass `specs` (Map T → spectrum) to reuse them.
 */
export function drawSubsets({ G, N, logp, logq, Ks = [2, 3], Ts = [0.3, 1, 3, 10], nUniform = 200, nPerT = 100, seed = 0, kinds = ['uniform', 'dpp', 'map'], specs = new Map() }) {
  const rng = mulberry32(seed);
  const out = [];
  const keep = (S, K, T, kind) => { if (S && S.length === K && logDetSub(G, N, S) > Math.log(1e-12)) out.push({ S, K, T, kind }); };
  if (kinds.includes('uniform')) for (const K of Ks) for (let n = 0; n < nUniform; n++) keep(uniformSubset(N, K, rng), K, null, 'uniform');
  for (const T of Ts) {
    if (!kinds.includes('dpp') && !kinds.includes('map')) break;
    const ls = columnLogScale('sample', logp, logq, T);
    const { L } = kernel(G, N, ls);
    if (kinds.includes('dpp')) {
      if (!specs.has(T)) specs.set(T, spectrum(L, N));
      for (const K of Ks) for (let n = 0; n < nPerT; n++) keep(sampleKDPP(specs.get(T), K, rng), K, T, 'dpp');
    }
    if (kinds.includes('map')) {
      const Lm = kernel(G, N, columnLogScale('map', logp, logq, T)).L;
      for (const K of Ks) keep(greedyMAP(Lm, N, K).set.slice().sort((x, y) => x - y), K, T, 'map');
    }
  }
  return out;
}

// ---------- fitted layout ----------

function targetLog(G, N, a, t) {
  return 0.5 * logDetSub(G, N, t.S) - Math.log(FACT[t.K - 1]) - t.S.reduce((s, i) => s + Math.log(a[i]), 0);
}

/**
 * Fit y (N×d) to minimize Σ_S (log Vol_{K−1}(conv y_S) − log τ_S)², τ_S = √det G_S / ((K−1)! ∏a_i), by Adam.
 * K = 2 uses log distance, K = 3 uses ½ log(|u|²|v|² − (u·v)²) − log 2. Fits all K in `Ks` jointly.
 * init: N rows (e.g. gnomonic-PCA coordinates). Returns { Y, loss0, loss1, subsets, ms }.
 * `reg` is a ridge pull toward the start (λ·mean‖y−y₀‖²/rms²) against overfitting. Defaults: iters 150, nUniform 600, nPerT 150, lr 0.01, reg 1, MAP sets weighted ×`mapWeight`.
 */
export function fitLayout({ G, N, a, init, logp, logq, d = init[0].length, seed = 1, Ks = [2, 3], Ts = [0.3, 1, 3, 10], iters = 150, nUniform = 600, nPerT = 150, mapWeight = 5, lr = 0.01, reg = 1, subsets = null, specs }) {
  const t0 = Date.now();
  const train = (subsets || drawSubsets({ G, N, logp, logq, Ks, Ts, nUniform, nPerT, seed, specs })).filter((t) => Ks.includes(t.K) && t.K <= 3);
  for (const t of train) t.logTau = targetLog(G, N, a, t);
  const y = new Float64Array(N * d);
  for (let i = 0; i < N; i++) for (let c = 0; c < d; c++) y[i * d + c] = init[i][c];
  const view = (arr) => Array.from({ length: N }, (_, i) => arr.subarray(i * d, (i + 1) * d));
  // Global scale: volume of a K-set scales as s^(K−1); least squares on log s.
  {
    const Y = view(y);
    let num = 0, den = 0;
    for (const t of train) { const r = Math.log(Math.max(simplexVolume(Y, t.S), 1e-300)) - t.logTau; num += (t.K - 1) * r; den += (t.K - 1) ** 2; }
    if (den > 0) { const s = Math.exp(-num / den); for (let i = 0; i < y.length; i++) y[i] *= s; }
  }
  let rms = 0;
  for (let i = 0; i < y.length; i++) rms += y[i] * y[i];
  rms = Math.sqrt(rms / N) || 1;
  const y0 = Float64Array.from(y); // scaled start; reg pulls toward it
  const w = (t) => (t.kind === 'map' ? mapWeight : 1);
  let wsum = 0;
  for (const t of train) wsum += w(t);
  const g = new Float64Array(N * d), m1 = new Float64Array(N * d), m2 = new Float64Array(N * d);
  const b1 = 0.9, b2 = 0.999;
  let loss0 = NaN, loss = NaN;
  const u = new Float64Array(d), v = new Float64Array(d);
  for (let it = 0; it < iters; it++) {
    g.fill(0);
    loss = 0;
    for (const t of train) {
      const S = t.S, wt = w(t) / wsum;
      if (t.K === 2) {
        let n2 = 0;
        for (let c = 0; c < d; c++) { u[c] = y[S[1] * d + c] - y[S[0] * d + c]; n2 += u[c] * u[c]; }
        n2 = Math.max(n2, 1e-24);
        const r = 0.5 * Math.log(n2) - t.logTau;
        loss += wt * r * r;
        const f = (2 * r * wt) / n2; // d(½ log n2)/du = u/n2
        for (let c = 0; c < d; c++) { g[S[1] * d + c] += f * u[c]; g[S[0] * d + c] -= f * u[c]; }
      } else {
        let uu = 0, vv = 0, uv = 0;
        for (let c = 0; c < d; c++) {
          u[c] = y[S[1] * d + c] - y[S[0] * d + c]; v[c] = y[S[2] * d + c] - y[S[0] * d + c];
          uu += u[c] * u[c]; vv += v[c] * v[c]; uv += u[c] * v[c];
        }
        const Dt = Math.max(uu * vv - uv * uv, 1e-24);
        const r = 0.5 * Math.log(Dt) - Math.log(2) - t.logTau;
        loss += wt * r * r;
        const f = (r * wt) / Dt; // 2r · ½ · (dDt/Dt)
        for (let c = 0; c < d; c++) {
          const du = f * (2 * vv * u[c] - 2 * uv * v[c]), dv = f * (2 * uu * v[c] - 2 * uv * u[c]);
          g[S[1] * d + c] += du; g[S[2] * d + c] += dv; g[S[0] * d + c] -= du + dv;
        }
      }
    }
    if (reg > 0) for (let i = 0; i < y.length; i++) { const dy = y[i] - y0[i]; g[i] += (2 * reg * dy) / (rms * rms * N); }
    if (it === 0) loss0 = loss;
    const step = lr * rms * (0.5 * (1 + Math.cos((Math.PI * it) / iters)) * 0.95 + 0.05);
    for (let i = 0; i < y.length; i++) {
      m1[i] = b1 * m1[i] + (1 - b1) * g[i];
      m2[i] = b2 * m2[i] + (1 - b2) * g[i] * g[i];
      y[i] -= (step * (m1[i] / (1 - b1 ** (it + 1)))) / (Math.sqrt(m2[i] / (1 - b2 ** (it + 1))) + 1e-12);
    }
  }
  return { Y: view(y).map((r) => Float64Array.from(r)), loss0, loss1: loss, subsets: train, ms: Date.now() - t0 };
}
