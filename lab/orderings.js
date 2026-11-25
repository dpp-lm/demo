// Row orders for a kernel matrix M (n x n, row-major Float64Array, positive semidefinite).
// See docs/matrix-views.md, section 2A. Every function returns { order, ... } with order a permutation
// of 0..n-1 (order[r] = the item drawn in row r).
import { eigSym, elemSym } from '../js/dpp.js';

/**
 * Pivoted Cholesky to the end: at each step take the item with the largest leftover diagonal (its squared
 * distance from the span of the pivots so far: greedy volume). Stops when every leftover is below
 * tol x the largest diagonal.
 * Returns { piv, c, height, resid }:
 *  piv[t]   the t-th pivot;
 *  c[t][i]  the Cholesky column of pivot t at item i (c[t][i]^2 is the part of i's squared length that pivot t
 *           explains, given pivots 0..t-1);
 *  height[i] for a pivot, the square root of its leftover when it was taken (a volume factor:
 *           height^2 = det M_{A+i} / det M_A); for any other item, the square root of what is left at the end.
 */
export function pivotedCholesky(M, n, tol = 1e-10) {
  const d = new Float64Array(n);
  for (let i = 0; i < n; i++) d[i] = M[i * n + i];
  const d0 = Math.max(...d), piv = [], c = [], height = new Float64Array(n), taken = new Uint8Array(n);
  for (let t = 0; t < n; t++) {
    let p = -1, best = tol * d0;
    for (let i = 0; i < n; i++) if (!taken[i] && d[i] > best) { best = d[i]; p = i; }
    if (p < 0) break;
    const s = Math.sqrt(d[p]), col = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let v = M[i * n + p];
      for (let u = 0; u < c.length; u++) v -= c[u][i] * c[u][p];
      col[i] = v / s;
    }
    height[p] = s; taken[p] = 1;
    for (let i = 0; i < n; i++) if (!taken[i]) d[i] = Math.max(0, d[i] - col[i] * col[i]);
    piv.push(p); c.push(col);
  }
  for (let i = 0; i < n; i++) if (!taken[i]) height[i] = Math.sqrt(d[i]);
  return { piv, c, height, taken };
}

/** For item i, the pivot (by step) among steps 0..upto-1 that explains the most of it: argmax_s c[s][i]^2. */
function bestPivotStep(c, i, upto) {
  let b = 0, bv = -1;
  for (let s = 0; s < upto; s++) { const v = c[s][i] * c[s][i]; if (v > bv) { bv = v; b = s; } }
  return b;
}

/**
 * Flat blocks: the first K pivots (greedy MAP for M), each followed by the items it explains the most of,
 * sorted by how much it explains. Returns { order, blocks } with blocks[k] = [start row, end row) for pivot k.
 */
export function flatPivotOrder(M, n, K) {
  const pc = pivotedCholesky(M, n), k = Math.min(K, pc.piv.length);
  const step = new Map(pc.piv.slice(0, k).map((p, s) => [p, s]));
  const members = Array.from({ length: k }, () => []);
  for (let i = 0; i < n; i++) {
    if (step.has(i)) continue;
    const s = bestPivotStep(pc.c, i, k);
    members[s].push([i, pc.c[s][i] ** 2]);
  }
  const order = [], blocks = [];
  for (let s = 0; s < k; s++) {
    const start = order.length;
    order.push(pc.piv[s], ...members[s].sort((a, b) => b[1] - a[1]).map((x) => x[0]));
    blocks.push([start, order.length]);
  }
  return { order, blocks };
}

/**
 * Nested: the whole factorisation read as a tree. Parent of a pivot = the earlier pivot that explains the most
 * of it; parent of an item never taken (leftover ~0, e.g. an exact copy) = the pivot that explains the most of it.
 * Height = the leftover length when taken (non-increasing along the pivot order, so a child is never above its
 * parent). Rows: depth-first, children in pivot order (items never taken last, most explained first).
 * Returns { order, parent, height } (parent[i] = -1 for the root).
 */
export function nestedPivotOrder(M, n) {
  const pc = pivotedCholesky(M, n), r = pc.piv.length;
  const parent = new Int32Array(n).fill(-1), children = Array.from({ length: n }, () => []);
  for (let t = 1; t < r; t++) { const p = pc.piv[t], s = bestPivotStep(pc.c, p, t); parent[p] = pc.piv[s]; children[pc.piv[s]].push([p, -t]); }
  for (let i = 0; i < n; i++) {
    if (pc.taken[i]) continue;
    const s = bestPivotStep(pc.c, i, r);
    parent[i] = pc.piv[s]; children[pc.piv[s]].push([i, pc.c[s][i] ** 2 - 1e9]);   // after the pivots, most explained first
  }
  for (const ch of children) ch.sort((a, b) => b[1] - a[1]);
  const order = [], stack = [pc.piv[0]];
  while (stack.length) {
    const v = stack.pop();
    order.push(v);
    const ch = children[v];
    for (let j = ch.length - 1; j >= 0; j--) stack.push(ch[j][0]);
  }
  return { order, parent, height: pc.height };
}

/**
 * Average linkage on the cosine distance 1 - M_ij / sqrt(M_ii M_jj) (UPGMA; Manuel's "blocks" order).
 * Returns the leaf order and the merges { a, b, h } (node ids: items 0..n-1, then n, n+1, ... per merge).
 */
export function averageLinkageOrder(M, n) {
  const D = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) D[i * n + j] = 1 - M[i * n + j] / Math.sqrt(M[i * n + i] * M[j * n + j] || 1);
  let active = Array.from({ length: n }, (_, i) => i);
  const size = new Map(active.map((i) => [i, 1])), id = new Map(active.map((i) => [i, i]));
  const dist = new Map();   // key 'a,b' with a<b (slot indices) -> distance
  const key = (a, b) => (a < b ? `${a},${b}` : `${b},${a}`);
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) dist.set(key(i, j), D[i * n + j]);
  const merges = [], kids = new Map();
  let next = n;
  while (active.length > 1) {
    let ba = -1, bb = -1, bd = Infinity;
    for (let x = 0; x < active.length; x++) for (let y = x + 1; y < active.length; y++) {
      const v = dist.get(key(active[x], active[y]));
      if (v < bd) { bd = v; ba = active[x]; bb = active[y]; }
    }
    const sa = size.get(ba), sb = size.get(bb), node = next++;
    merges.push({ a: id.get(ba), b: id.get(bb), h: bd, node });
    kids.set(node, [id.get(ba), id.get(bb)]);
    for (const o of active) if (o !== ba && o !== bb) dist.set(key(ba, o), (sa * dist.get(key(ba, o)) + sb * dist.get(key(bb, o))) / (sa + sb));
    size.set(ba, sa + sb); id.set(ba, node);
    active = active.filter((o) => o !== bb);
  }
  const order = [], stack = [id.get(active[0])];
  while (stack.length) { const v = stack.pop(); if (v < n) order.push(v); else { const [a, b] = kids.get(v); stack.push(b, a); } }
  return { order, merges };
}

/**
 * Theme blocks, from the K-DPP's own spectral form (Kulesza & Taskar). With L = sum_a lambda_a u_a u_a^T, a K-DPP
 * sample first chooses K eigenvectors ("themes"): theme a is chosen with probability
 *   p_a = lambda_a e_{K-1}(lambda without a) / e_K(lambda),   sum_a p_a = K,
 * and given the chosen set J, item i is in the sample with probability sum_{a in J} u_a(i)^2. So every inclusion
 * probability splits exactly over the themes:
 *   pi_i = sum_a p_a u_a(i)^2.
 * Item i joins the theme with the largest part p_a u_a(i)^2 of its inclusion probability. Blocks in order of p_a;
 * inside a block, sorted by the signed loading u_a(i) (the theme's two ends at the block's two ends).
 * Returns { order, blocks: [start, end, a], p, part(i, a), pi, theme } (theme[i] = the block's eigenvector index).
 */
export function themeOrder(L, n, K) {
  const { values, vectors } = eigSym(L, n);
  const vmax = Math.max(...values, 1e-300), lam = Array.from(values, (v) => Math.max(0, v / vmax));
  const eK = elemSym(lam, K)[K][n];
  const p = lam.map((la, a) => (la > 0 ? (la * elemSym(lam.filter((_, b) => b !== a), K - 1)[K - 1][n - 1]) / eK : 0));
  const part = (i, a) => p[a] * vectors[a][i] ** 2;
  const pi = new Float64Array(n), theme = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    let b = 0, bv = -1;
    for (let a = 0; a < n; a++) { const v = part(i, a); pi[i] += v; if (v > bv) { bv = v; b = a; } }
    theme[i] = b;
  }
  const byP = [...Array(n).keys()].sort((a, b) => p[b] - p[a]);
  const order = [], blocks = [];
  for (const a of byP) {
    const m = [...Array(n).keys()].filter((i) => theme[i] === a);
    if (!m.length) continue;
    const start = order.length;
    order.push(...m.sort((x, y) => vectors[a][y] - vectors[a][x]));
    blocks.push([start, order.length, a]);
  }
  return { order, blocks, p, part, pi, theme };
}
