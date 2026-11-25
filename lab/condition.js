// Conditioning a K-DPP on a selected set A (docs/matrix-views.md, section 2B).
// Given that every item of A is in the sample, the rest is a (K - |A|)-DPP over the other items whose kernel is the
// Schur complement L^A = L_rest - L_{rest,A} L_A^{-1} L_{A,rest} (Kulesza & Taskar 2012). In GP terms, L^A is the
// posterior covariance after observing A; its diagonal is each item's leftover quality.
import { spectrum, inclusionProbs } from '../js/dpp.js';

// Rounding tolerance for "the leftover is exactly zero" (an item in the span of A): relative to the item's own
// diagonal, at the level of float64 rounding after a few hundred products. Not a modelling threshold.
const ZERO = 1e-10;

/**
 * Condition the kernel L (n x n, row-major) on including the items A, in order (one Gram-Schmidt step each).
 * Returns { ok, LA, leftover, addable, pi, reason }:
 *  LA        the conditioned kernel, n x n, with the rows and columns of A set to 0;
 *  leftover  LA's diagonal (each item's leftover quality given A);
 *  addable   addable[i]: i is not in A and has a nonzero leftover (adding it keeps the volume positive);
 *  pi        P(i in sample | A in sample) for the K-DPP (1 for items of A);
 *  ok=false  with a reason when A has volume zero, or no K-set contains A.
 */
export function condition(L, n, A, K) {
  const LA = Float64Array.from(L), inA = new Uint8Array(n);
  for (const a of A) {
    const d = LA[a * n + a];
    if (inA[a] || !(d > ZERO * L[a * n + a]) || !(L[a * n + a] > 0)) return { ok: false, reason: 'volume zero: an item is in the span of the others' };
    const col = new Float64Array(n);
    for (let i = 0; i < n; i++) col[i] = LA[i * n + a] / Math.sqrt(d);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) LA[i * n + j] -= col[i] * col[j];
    for (let i = 0; i < n; i++) LA[i * n + a] = LA[a * n + i] = 0;   // exactly 0 (they are, up to rounding)
    inA[a] = 1;
  }
  const leftover = new Float64Array(n), addable = new Uint8Array(n);
  for (let i = 0; i < n; i++) { leftover[i] = Math.max(0, LA[i * n + i]); addable[i] = !inA[i] && leftover[i] > ZERO * L[i * n + i] ? 1 : 0; }
  // The rest is a (K - |A|)-DPP over the items not in A.
  const k = K - A.length, rest = [...Array(n).keys()].filter((i) => !inA[i]), r = rest.length;
  const pi = new Float64Array(n);
  for (const a of A) pi[a] = 1;
  if (k > 0) {
    const sub = new Float64Array(r * r);
    for (let x = 0; x < r; x++) for (let y = 0; y < r; y++) sub[x * r + y] = LA[rest[x] * n + rest[y]];
    const p = inclusionProbs(spectrum(sub, r), k);
    if (!p.every(Number.isFinite)) return { ok: false, reason: `no set of ${K} contains this selection` };
    rest.forEach((i, x) => (pi[i] = p[x]));
  }
  return { ok: true, LA, leftover, addable, pi };
}
