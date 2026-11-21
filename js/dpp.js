// K-DPP machinery for the paper's pool-based sampler (§3–§5). Plain JS, no dependencies.
//
// Notation follows the paper: φ unit-norm embeddings, κ(y) = p(y)^{1/T},
// pool kernel L̂ = B̂ᵀB̂ with columns √(w/N) φ (sampling) or √κ φ (MAP).
// Everything is kept in log space and rescaled by the max before exponentiating;
// a K-DPP is invariant to rescaling its kernel (paper footnote 8).

// ---------- small dense linear algebra ----------

/** Gram matrix G_ij = φ_i·φ_j as a Float64Array (row-major N×N). */
export function gram(phi) {
  const N = phi.length, D = phi[0].length;
  const G = new Float64Array(N * N);
  for (let i = 0; i < N; i++) {
    const a = phi[i];
    for (let j = i; j < N; j++) {
      const b = phi[j];
      let s = 0;
      for (let d = 0; d < D; d++) s += a[d] * b[d];
      G[i * N + j] = G[j * N + i] = s;
    }
  }
  return G;
}

/** Symmetric eigendecomposition by cyclic Jacobi. Returns values descending, vectors[k] = k-th eigenvector. */
export function symEig(A, n) {
  const a = Float64Array.from(A);
  const v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;
  let off0 = 0;
  for (let i = 0; i < n * n; i++) off0 += a[i] * a[i];
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p * n + q] ** 2;
    if (off <= 1e-30 * off0 || off === 0) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q];
        if (Math.abs(apq) < 1e-300) continue;
        const app = a[p * n + p], aqq = a[q * n + q];
        const theta = (aqq - app) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k * n + p], akq = a[k * n + q];
          a[k * n + p] = c * akp - s * akq;
          a[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p * n + k], aqk = a[q * n + k];
          a[p * n + k] = c * apk - s * aqk;
          a[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k * n + p], vkq = v[k * n + q];
          v[k * n + p] = c * vkp - s * vkq;
          v[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const order = [...Array(n).keys()].sort((i, j) => a[j * n + j] - a[i * n + i]);
  const values = order.map((i) => a[i * n + i]);
  const vectors = order.map((i) => {
    const col = new Float64Array(n);
    for (let k = 0; k < n; k++) col[k] = v[k * n + i];
    return col;
  });
  return { values, vectors };
}

/** log det of the principal submatrix L_S via Cholesky; -Infinity if singular. */
export function logDetSub(L, N, S) {
  const k = S.length;
  const c = new Float64Array(k * k);
  let ld = 0;
  for (let i = 0; i < k; i++) {
    for (let j = 0; j <= i; j++) {
      let s = L[S[i] * N + S[j]];
      for (let m = 0; m < j; m++) s -= c[i * k + m] * c[j * k + m];
      if (i === j) {
        if (!(s > 1e-12 * Math.max(1e-300, L[S[i] * N + S[i]]))) return -Infinity;
        c[i * k + i] = Math.sqrt(s);
        ld += Math.log(s);
      } else c[i * k + j] = s / c[j * k + j];
    }
  }
  return ld;
}

// ---------- kernels ----------

/**
 * Column scales for the pool kernel, in log space: logScale[n] = log of the squared
 * column norm (log κ for MAP, log(w/N) for sampling). Returns { L, logShift } with
 * L = diag(c) G diag(c), c = exp((logScale - max)/2), and logShift = max (per-item).
 */
export function kernel(G, N, logScale) {
  let m = -Infinity;
  for (const s of logScale) if (s > m) m = s;
  const c = logScale.map((s) => Math.exp(0.5 * (s - m)));
  const L = new Float64Array(N * N);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) L[i * N + j] = c[i] * G[i * N + j] * c[j];
  return { L, logShift: m, c };
}

/** log κ(y) = log p(y)/T (Eq. 6). */
export const logKappa = (logp, T) => logp.map((lp) => lp / T);

/** log w(y) = log κ(y) − log q(y) (Eq. 8). */
export const logWeights = (logp, logq, T) => logp.map((lp, i) => lp / T - logq[i]);

/** Effective sample size of importance weights given in log space. */
export function ess(logw) {
  const m = Math.max(...logw);
  let s1 = 0, s2 = 0;
  for (const lw of logw) { const w = Math.exp(lw - m); s1 += w; s2 += w * w; }
  return (s1 * s1) / s2;
}

// ---------- elementary symmetric polynomials ----------

/** E[l][n] = e_l(λ_1..λ_n), l = 0..K, n = 0..N (Kulesza & Taskar Alg. 7). */
export function elemSym(lambda, K) {
  const N = lambda.length;
  const E = Array.from({ length: K + 1 }, () => new Float64Array(N + 1));
  for (let n = 0; n <= N; n++) E[0][n] = 1;
  for (let l = 1; l <= K; l++) {
    for (let n = 1; n <= N; n++) E[l][n] = E[l][n - 1] + lambda[n - 1] * E[l - 1][n - 1];
  }
  return E;
}

// ---------- K-DPP inference ----------

/** Eigendecomposition with eigenvalues clamped at 0 and rescaled so the max is 1. */
export function spectrum(L, N) {
  const { values, vectors } = symEig(L, N);
  const vmax = Math.max(values[0], 1e-300);
  const lambda = values.map((x) => Math.max(0, x / vmax));
  return { lambda, vectors, logScale: Math.log(vmax) };
}

/** Rank (number of numerically nonzero eigenvalues). */
export const rank = (lambda, tol = 1e-10) => lambda.filter((x) => x > tol).length;

/**
 * Exact K-DPP sample (Kulesza & Taskar 2012, Alg. 1 + Alg. 8) from a spectrum.
 * Returns indices, or null when rank < K (Ẑ = 0; the paper redraws the pool).
 */
export function sampleKDPP(spec, K, rng = Math.random) {
  const { lambda, vectors } = spec;
  const N = lambda.length;
  if (rank(lambda) < K) return null;
  const E = elemSym(lambda, K);
  // Phase 1: choose K eigenvectors.
  const chosen = [];
  let l = K;
  for (let n = N; n >= 1 && l > 0; n--) {
    if (n === l) { for (let m = n; m >= 1; m--) chosen.push(m - 1); break; }
    const pr = (lambda[n - 1] * E[l - 1][n - 1]) / E[l][n];
    if (rng() < pr) { chosen.push(n - 1); l--; }
  }
  // Phase 2: sample from the projection DPP spanned by the chosen eigenvectors.
  let V = chosen.map((k) => Float64Array.from(vectors[k]));
  const Y = [];
  while (V.length > 0) {
    const P = new Float64Array(N);
    let tot = 0;
    for (const v of V) for (let i = 0; i < N; i++) P[i] += v[i] * v[i];
    for (let i = 0; i < N; i++) tot += P[i];
    let u = rng() * tot, i = 0;
    for (; i < N - 1; i++) { u -= P[i]; if (u <= 0) break; }
    Y.push(i);
    // Eliminate: pick the column with largest |v_i|, project the rest onto e_i^⊥.
    let j = 0;
    for (let m = 1; m < V.length; m++) if (Math.abs(V[m][i]) > Math.abs(V[j][i])) j = m;
    const vj = V[j];
    V = V.filter((_, m) => m !== j).map((v) => {
      const f = v[i] / vj[i];
      const w = new Float64Array(N);
      for (let k = 0; k < N; k++) w[k] = v[k] - f * vj[k];
      return w;
    });
    // Gram–Schmidt.
    for (let a = 0; a < V.length; a++) {
      for (let b = 0; b < a; b++) {
        let d = 0;
        for (let k = 0; k < N; k++) d += V[a][k] * V[b][k];
        for (let k = 0; k < N; k++) V[a][k] -= d * V[b][k];
      }
      let nrm = 0;
      for (let k = 0; k < N; k++) nrm += V[a][k] ** 2;
      nrm = Math.sqrt(nrm);
      for (let k = 0; k < N; k++) V[a][k] /= nrm;
    }
  }
  return Y.sort((a, b) => a - b);
}

/**
 * Greedy MAP (Alg. 2; fast incremental-Cholesky form of Chen et al. 2018).
 * Returns { set, logdet, gains } where gains[t] = log of the t-th Schur complement.
 */
export function greedyMAP(L, N, K) {
  const d2 = new Float64Array(N);
  for (let i = 0; i < N; i++) d2[i] = L[i * N + i];
  const C = Array.from({ length: N }, () => []);
  const set = [], gains = [];
  const used = new Uint8Array(N);
  for (let t = 0; t < K; t++) {
    let j = -1;
    for (let i = 0; i < N; i++) if (!used[i] && (j < 0 || d2[i] > d2[j])) j = i;
    if (j < 0 || !(d2[j] > 1e-12)) break;
    used[j] = 1;
    set.push(j);
    gains.push(Math.log(d2[j]));
    const dj = Math.sqrt(d2[j]);
    for (let i = 0; i < N; i++) {
      if (used[i]) continue;
      let dot = 0;
      for (let m = 0; m < t; m++) dot += C[j][m] * C[i][m];
      const e = (L[j * N + i] - dot) / dj;
      C[i].push(e);
      d2[i] -= e * e;
    }
    C[j].push(dj);
  }
  return { set, logdet: gains.reduce((a, b) => a + b, 0), gains };
}

/** log Z of the K-DPP for kernel with given spectrum (excluding the kernel's logShift). */
export function logZ(spec, K) {
  const E = elemSym(spec.lambda, K);
  return Math.log(E[K][spec.lambda.length]) + K * spec.logScale;
}

/**
 * K-DPP inclusion probabilities P(i ∈ S) = Σ_n λ_n e_{K-1}(λ_{-n}) / e_K(λ) · v_n(i)².
 */
export function inclusionProbs(spec, K) {
  const { lambda, vectors } = spec;
  const N = lambda.length;
  const eK = elemSym(lambda, K)[K][N];
  const pi = new Float64Array(N);
  for (let n = 0; n < N; n++) {
    if (lambda[n] <= 0) continue;
    const rest = lambda.filter((_, m) => m !== n);
    const eKm1 = elemSym(rest, K - 1)[K - 1][N - 1];
    const coef = (lambda[n] * eKm1) / eK;
    const v = vectors[n];
    for (let i = 0; i < N; i++) pi[i] += coef * v[i] * v[i];
  }
  return pi;
}

// ---------- the two inference modes ----------

/**
 * Per-item log column scale for each mode.
 *  'sample': log(w/N) with w = κ/q  (Eq. 8) — the importance-weighted pool kernel.
 *  'map'   : log κ                 — the MAP mode needs no importance correction (§4).
 *  'naive' : log p                 — "naive DPP": proposal likelihood, no global T.
 */
export function columnLogScale(mode, logp, logq, T) {
  if (mode === 'sample') return logWeights(logp, logq, T);
  if (mode === 'map') return logKappa(logp, T);
  if (mode === 'naive') return logp.slice();
  throw new Error('unknown mode ' + mode);
}

// ---------- metrics (Eq. 16) ----------

export function quality(logp, S) {
  return S.reduce((a, i) => a + logp[i], 0) / S.length;
}

export function diversity(G, N, S) {
  return logDetSub(G, N, S);
}

// ---------- baselines ----------

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** K-means reranking: k-means++ on φ, then the highest-likelihood item per cluster. */
export function kmeansSelect(phi, logp, K, rng = mulberry32(0), iters = 50) {
  const N = phi.length, D = phi[0].length;
  const dist2 = (a, b) => { let s = 0; for (let d = 0; d < D; d++) s += (a[d] - b[d]) ** 2; return s; };
  const centers = [Float64Array.from(phi[Math.floor(rng() * N)])];
  while (centers.length < K) {
    const dd = phi.map((x) => Math.min(...centers.map((c) => dist2(x, c))));
    const tot = dd.reduce((a, b) => a + b, 0);
    if (tot <= 0) break;
    let u = rng() * tot, i = 0;
    for (; i < N - 1; i++) { u -= dd[i]; if (u <= 0) break; }
    centers.push(Float64Array.from(phi[i]));
  }
  let assign = new Int32Array(N);
  for (let it = 0; it < iters; it++) {
    let changed = false;
    for (let i = 0; i < N; i++) {
      let best = 0, bd = Infinity;
      centers.forEach((c, k) => { const d = dist2(phi[i], c); if (d < bd) { bd = d; best = k; } });
      if (assign[i] !== best) { assign[i] = best; changed = true; }
    }
    centers.forEach((c, k) => {
      const mem = [];
      for (let i = 0; i < N; i++) if (assign[i] === k) mem.push(i);
      if (!mem.length) return;
      c.fill(0);
      for (const i of mem) for (let d = 0; d < D; d++) c[d] += phi[i][d] / mem.length;
    });
    if (!changed && it > 0) break;
  }
  const best = new Map();
  for (let i = 0; i < N; i++) {
    const k = assign[i];
    if (!best.has(k) || logp[i] > logp[best.get(k)]) best.set(k, i);
  }
  return [...best.values()].sort((a, b) => a - b);
}

// ---------- the finite-sample bound (Thm 5.2), evaluated on the pool instance ----------

function logChoose(n, k) {
  if (k < 0 || k > n) return -Infinity;
  let s = 0;
  for (let i = 0; i < k; i++) s += Math.log(n - i) - Math.log(i + 1);
  return s;
}

/**
 * As in App. B: build a finite instance from the pool — distinct strings, with their
 * observed frequencies as the proposal q̃ — compute the target K-DPP's inclusion
 * probabilities there, and ρ = E_q̃[u₁²], u₁(y) = P(y∈S) / (K q̃(y)).
 * Returns { rho, theta(Np), tight(Np) } where theta = K²ρ/(Np−K+1) and
 * tight = log(e^θ − 1 + a₀) (Eq. 15).
 */
export function boundOnInstance({ phi, logp, texts, T, K }) {
  const idx = new Map();
  texts.forEach((t, i) => { if (!idx.has(t)) idx.set(t, []); idx.get(t).push(i); });
  const reps = [...idx.values()];
  const M = reps.length;
  const Ntot = texts.length;
  if (M < K) return null;
  const qt = reps.map((r) => r.length / Ntot);
  const sub = reps.map((r) => r[0]);
  const G = gram(sub.map((i) => phi[i]));
  const { L } = kernel(G, M, sub.map((i) => logp[i] / T));
  const spec = spectrum(L, M);
  if (rank(spec.lambda) < K) return null;
  const pi = inclusionProbs(spec, K);
  let rho = 0;
  for (let m = 0; m < M; m++) { const u = pi[m] / (K * qt[m]); rho += qt[m] * u * u; }
  const theta = (Np) => (K * K * rho) / (Np - K + 1);
  const a0 = (Np) => Math.exp(logChoose(Np - K, K) - logChoose(Np, K));
  const tight = (Np) => { const th = theta(Np); return th > 30 ? th : Math.log(Math.expm1(th) + a0(Np)); };
  return { rho, theta, tight, distinct: M };
}
