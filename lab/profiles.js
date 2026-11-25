// Theme profiles: how each completion's inclusion probability splits over the K-DPP's themes (docs/matrix-views.md).
// pi_i = sum_a p_a u_a(i)^2 exactly, and each theme's parts sum to p_a over the completions (sum_i u_a(i)^2 = 1).
import { themeSpectrum } from './orderings.js';

// Muted colours for the themes, in the house style, kept clear of the accent red. Checked with the data-viz validator
// (OKLab ΔE, Machado et al. CVD simulation) on all pairs, together with the accent #c8473c, on the surface #fbfaf8:
// every check passes; the closest pair is ochre against the accent red (CVD ΔE 8.0, normal vision 15.1). No fourth
// muted colour passed with these, so at most three themes get a colour; the rest share one neutral grey.
export const THEME_COLORS = ['#3f74b5', '#b8862a', '#1f9c86'];
export const REST_COLOR = '#c9c1b4';

/**
 * Groups of themes for colouring. Eigenvalues within `merge` (relative) of each other are grouped: inside a (near-)
 * degenerate eigenspace the individual eigenvectors are not determined, but the space is, and so is the group's part
 * of each pi_i when the p's are equal. How many groups get a colour: the effective number of themes, K^2 / sum p_a^2
 * (sum p = K, each p <= 1, so it is at least K), capped by the palette. Colours follow the theme, not its rank: each
 * coloured group takes the colour of the previous coloured group whose eigenspace it overlaps most (overlap above
 * one half), else the first free colour.
 * Returns { groups: [{ eigs, p, color, label }], restP, part(i, g), restPart(i), pi, neff, prev } (pass prev back next time).
 */
export function themeProfiles(L, n, K, prev = [], { merge = 0.05 } = {}) {
  const { lam, vectors, p } = themeSpectrum(L, n, K);
  const idx = [...lam.keys()].filter((a) => p[a] > 0).sort((a, b) => lam[b] - lam[a]);
  const all = [];
  for (const a of idx) {
    const g = all[all.length - 1];
    if (g && (lam[g.eigs[g.eigs.length - 1]] - lam[a]) / lam[g.eigs[0]] < merge) g.eigs.push(a);
    else all.push({ eigs: [a] });
  }
  all.forEach((g) => (g.p = g.eigs.reduce((s, a) => s + p[a], 0)));
  const neff = (K * K) / idx.reduce((s, a) => s + p[a] * p[a], 0);
  const m = Math.min(THEME_COLORS.length, Math.max(1, Math.round(neff)), all.length);
  const groups = all.slice(0, m);
  // Stable colours: match to the previous coloured groups by eigenspace overlap, tr(P_g P_h) / max(dim g, dim h).
  const overlap = (g, h) => { let s = 0; for (const a of g.eigs) for (const v of h.vecs) { let d = 0; for (let i = 0; i < n; i++) d += vectors[a][i] * v[i]; s += d * d; } return s / Math.max(g.eigs.length, h.vecs.length); };
  const pairs = [];
  groups.forEach((g, x) => prev.forEach((h, y) => pairs.push([overlap(g, h), x, y])));
  pairs.sort((a, b) => b[0] - a[0]);
  const used = new Set(), done = new Set();
  for (const [o, x, y] of pairs) {
    if (o <= 0.5 || done.has(x) || used.has(prev[y].color)) continue;
    groups[x].color = prev[y].color; used.add(prev[y].color); done.add(x);
  }
  for (const g of groups) if (!g.color) { g.color = THEME_COLORS.find((c) => !used.has(c)); used.add(g.color); }
  groups.forEach((g) => (g.label = g.eigs.length > 1 ? `themes ${g.eigs.map((a) => idx.indexOf(a) + 1).join('+')}` : `theme ${idx.indexOf(g.eigs[0]) + 1}`));
  const part = (i, g) => g.eigs.reduce((s, a) => s + p[a] * vectors[a][i] ** 2, 0);
  const pi = new Float64Array(n);
  for (let i = 0; i < n; i++) for (const a of idx) pi[i] += p[a] * vectors[a][i] ** 2;
  const restPart = (i) => Math.max(0, pi[i] - groups.reduce((s, g) => s + part(i, g), 0));
  const restP = K - groups.reduce((s, g) => s + g.p, 0);
  return { groups, restP, part, restPart, pi, neff, K, prev: groups.map((g) => ({ color: g.color, vecs: g.eigs.map((a) => Float64Array.from(vectors[a])) })) };
}
