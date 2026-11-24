// The similarity matrices panel: opened from the "Matrices" button in the bottom bar, in the default view and in expert mode.
//   Primal: L = diag(√w) G diag(√w), N×N, over the whole pool, with the weights w the subset is picked with
//           (κ for Most likely, κ/q for Sample), rows and columns in the leaf order of average-linkage clustering on 1 − cos
//           (fixed for a pool), the selected completions marked in the margins.
//   Dual:   the same weighted vectors' D×D matrix in the pool's principal axes (the first 32), from the eigendecomposition of G.
// Expert mode adds a viridis color option, the dendrogram in the margins (colored by group when Groups is on), other row orders
// and the number of axes. The panel redraws only while it is open and only when the pool, the weights or the selection change.
// Pure parts are in matrices.js and expert-lib.js; main.js reaches this file through expert.js.

import { EXPLAIN } from './explain.js';
import { dendrogramLayout, nodeGroups, weightedGram, dualInAxes, normalizeCorr, heatmapPixels, range, viridis, rampBetween, parseHex, fmtSig, weightsFromLog, rowOrder, permuted } from './matrices.js';
import { selectionLogScale, treeOf } from './expert-lib.js';

const DEFAULTS = { color: 'ink', order: 'blocks', r: 32 };
const COLORS = [{ id: 'ink', label: 'Ink' }, { id: 'viridis', label: 'Viridis' }];
const ORDERS = [{ id: 'blocks', label: 'Blocks' }, { id: 'groups', label: 'Groups' }, { id: 'weight', label: 'Weight' }, { id: 'pool', label: 'Pool order' }];
const AXES = [16, 32, 64];

let opt = { ...DEFAULTS }, norm = true, open = false, ctx = null, panel = null, btn = null, lastSig = '';
let wide = null;   // matchMedia, set in initMatrices (Node has none)
const ids = new WeakMap();
let nextId = 1;

const $ = (s) => panel.querySelector(s);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const clip = (t, n = 90) => { const s = String(t).replace(/\s+/g, ' ').trim(); return s.length > n ? s.slice(0, n) + '…' : s; };
const raf = (fn) => { let q = 0; return (...a) => { if (q) return; q = requestAnimationFrame(() => { q = 0; fn(...a); }); }; };

/** Top of the open panel in viewport pixels on wide screens (the plot keeps clear of it), else null. */
export const panelBottom = () => (open && wide?.matches && panel && !panel.hidden ? panel.getBoundingClientRect().top : null);

// ---------------------------------------------------------------- colors
function colormap() {
  if (ctx.isExpert() && opt.color === 'viridis') return { cmap: viridis, stops: [0, 0.25, 0.5, 0.75, 1].map((t) => viridis(t)) };
  const lo = parseHex(css('--surface')) ?? [251, 250, 248], hi = parseHex(css('--ink')) ?? [85, 85, 85];
  return { cmap: rampBetween(lo, hi), stops: [lo, hi] };
}
const colorbar = (sc, { stops }) => `<div class="mp-cbar"><span>${fmtSig(sc.min)}</span><span class="mp-grad" style="background:linear-gradient(90deg,${stops.map((c) => `rgb(${c.join(',')})`).join(',')})"></span><span>${fmtSig(sc.max)}</span></div>`;

// ---------------------------------------------------------------- drawing
function draw(cv, A, n, sc, cmap) {
  cv.width = n; cv.height = n;
  cv.getContext('2d').putImageData(new ImageData(heatmapPixels(A, n, sc, cmap), n, n), 0, 0);
}
const cellAt = (cv, e, n) => {
  const r = cv.getBoundingClientRect();
  return [Math.min(n - 1, Math.max(0, Math.floor(((e.clientY - r.top) / r.height) * n))), Math.min(n - 1, Math.max(0, Math.floor(((e.clientX - r.left) / r.width) * n)))];
};
const cross = (wrap, r, c, n) => {
  const row = wrap.querySelector('.mp-xr'), col = wrap.querySelector('.mp-xc');
  if (r == null) { row.style.display = col.style.display = 'none'; return; }
  row.style.display = col.style.display = 'block';
  row.style.top = `${(100 * (r + 0.5)) / n}%`; col.style.left = `${(100 * (c + 0.5)) / n}%`;
};

/** Wire hover, leave and arrow keys of a heatmap: show(r, c) on a cell, hide() off it. */
function wireGrid(wrap, cv, n, show, hide) {
  let cur = [0, 0];
  cv.onpointermove = raf((e) => { cur = cellAt(cv, e, n); cross(wrap, cur[0], cur[1], n); show(cur[0], cur[1]); });
  cv.onpointerleave = () => { cross(wrap, null, null, n); hide(); };
  wrap.onkeydown = (e) => {
    const d = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }[e.key];
    if (!d) return;
    e.preventDefault();
    cur = [Math.min(n - 1, Math.max(0, cur[0] + d[0])), Math.min(n - 1, Math.max(0, cur[1] + d[1]))];
    cross(wrap, cur[0], cur[1], n); show(cur[0], cur[1]);
  };
  wrap.onblur = () => { cross(wrap, null, null, n); hide(); };
}

function render(force = false) {
  if (!open || !panel) return;
  const H = ctx.H, pool = H.pool(), wts = H.weights(), body = $('.mp-body');
  if (!pool || !wts?.logKappa) { body.innerHTML = '<p class="mp-empty">No completions to show yet.</p>'; lastSig = ''; return; }
  const expert = ctx.isExpert(), grp = expert ? ctx.groups() : null;
  const opts = $('.mp-opts');
  opts.hidden = !expert;
  if (expert) opts.innerHTML = optionsHtml(grp);
  const sel = H.selection(), N = pool.samples.length, G = H.G();
  if (!ids.has(pool)) ids.set(pool, nextId++);
  const sig = [ids.get(pool), wts.mode, wts.T, wts.target, wts.K, sel.join(','), expert, expert ? JSON.stringify(opt) : '', norm, grp ? `${grp.method}${grp.k}${Array.from(grp.label).join('')}` : ''].join('|');
  if (!force && sig === lastSig) return;
  lastSig = sig;

  const w = weightsFromLog(selectionLogScale(wts.mode, wts.logKappa, wts.logq));
  const L = weightedGram(G, N, w), tree = treeOf(G, N), { cmap, stops } = colormap();
  let order = expert ? opt.order : 'blocks';
  if (order === 'groups' && !grp) order = 'blocks';
  const perm = order === 'blocks' ? tree.order : rowOrder(order === 'groups' ? 'cluster' : order, w, grp?.label);
  const pos = new Map(perm.map((i, r) => [i, r]));
  const scL = range(L), showTree = expert && order === 'blocks';

  const ticks = (axis) => sel.map((i) => `<span class="mp-tick" data-i="${i}" title="${escapeHtml(clip(pool.samples[i].text, 60))}" style="${axis === 'x' ? 'left' : 'top'}:${(100 * pos.get(i)) / N}%;${axis === 'x' ? 'width' : 'height'}:max(2px,${100 / N}%)"></span>`).join('');
  const dend = (axis) => {
    const { segs, maxH } = dendrogramLayout(tree.merges, N, perm), ng = grp?.method === 'tree' ? nodeGroups(tree.merges, N, grp.label) : null, byColor = new Map();
    for (const g of segs) {
      const c = ng && ng[g.node] >= 0 ? grp.color(ng[g.node]) : css('--muted'), f = (h) => 1 - h / (maxH || 1);
      const d = axis === 'x' ? `M${g.x1} ${f(g.y1)}L${g.x2} ${f(g.y2)}` : `M${f(g.y1)} ${g.x1}L${f(g.y2)} ${g.x2}`;
      byColor.set(c, (byColor.get(c) ?? '') + d);
    }
    return `<svg class="mp-dend d${axis}" viewBox="${axis === 'x' ? `0 0 ${N} 1` : `0 0 1 ${N}`}" preserveAspectRatio="none" aria-hidden="true">${[...byColor].map(([c, d]) => `<path d="${d}" stroke="${c}" fill="none" stroke-width="1" vector-effect="non-scaling-stroke"/>`).join('')}</svg>`;
  };
  const xh = '<i class="mp-xh mp-xr"></i><i class="mp-xh mp-xc"></i>';
  const orderText = { blocks: 'similar completions next to each other', groups: 'group order, then weight', weight: 'weight order', pool: 'pool order' }[order];

  const dual = dualInAxes(H.eig(), w, expert ? opt.r : 32);
  const A = norm ? normalizeCorr(dual.C, dual.r) : dual.C, m = dual.r;
  let scD = range(A);
  if (norm) { const off = []; for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) if (a !== b) off.push(A[a * m + b]); scD = range(off); }

  body.innerHTML = `<section class="mp-col">
      <div class="mp-sub"><span class="fn">Pairs of completions<button type="button" class="xbtn" data-x="primal" aria-label="What is the matrix of pairs of completions?">?</button></span></div>
      <div class="mp-pgrid${showTree ? ' withd' : ''}">${showTree ? dend('x') + dend('y') : ''}<div class="mp-strip sx">${ticks('x')}</div><div class="mp-strip sy">${ticks('y')}</div>
        <div class="mp-cw" id="mp-pcw" tabindex="0" role="application" aria-label="Matrix of pairs of completions, ${N} by ${N}. Arrow keys move between cells."><canvas id="mp-pcv"></canvas>${xh}</div></div>
      <div class="mp-hover" id="mp-phover">Hover a cell to see the pair.</div>
      ${colorbar(scL, { stops })}
      <p class="mp-cap">L = diag(√w) G diag(√w), ${N} × ${N}, in ${orderText}. Red marks are the ${sel.length} selected completions.</p>
    </section>
    <section class="mp-col">
      <div class="mp-sub"><span class="fn">Pairs of embedding directions<button type="button" class="xbtn" data-x="dual" aria-label="What is the matrix of pairs of embedding directions?">?</button></span></div>
      <div class="mp-dgrid"><div class="mp-cw" id="mp-dcw" tabindex="0" role="application" aria-label="Matrix of pairs of embedding directions, ${m} by ${m}. Arrow keys move between cells."><canvas id="mp-dcv"></canvas>${xh}</div></div>
      <div class="mp-hover" id="mp-dhover">Hover a cell to see the pair.</div>
      ${colorbar(scD, { stops })}
      <label class="toggle mp-norm"><input type="checkbox" id="mp-norm"${norm ? ' checked' : ''}><span>Normalize</span></label>
      <p class="mp-cap">The same weighted vectors, in the pool's first ${m} principal axes (axis 1 varies most).${norm ? ' Entries are divided by √(c<sub>ii</sub> c<sub>jj</sub>), so the diagonal is 1.' : ''}</p>
    </section>`;

  const pcv = $('#mp-pcv'), phov = $('#mp-phover'), phint = phov.textContent;
  draw(pcv, permuted(L, N, perm), N, scL, cmap);
  wireGrid($('#mp-pcw'), pcv, N, (r, c) => {
    const i = perm[r], j = perm[c];
    phov.innerHTML = `<span class="mp-pair">“${escapeHtml(clip(pool.samples[i].text))}”</span><span class="mp-pair">“${escapeHtml(clip(pool.samples[j].text))}”</span>similarity ${fmtSig(G[i * N + j])}, weighted ${fmtSig(L[i * N + j])}`;
    H.mark(i, i === j ? -1 : j);
  }, () => { phov.textContent = phint; H.mark(-1, -1); });
  panel.querySelectorAll('.mp-tick').forEach((t) => {
    t.onmouseenter = () => H.mark(+t.dataset.i, -1);
    t.onmouseleave = () => H.mark(-1, -1);
  });

  const dcv = $('#mp-dcv'), dhov = $('#mp-dhover'), dhint = dhov.textContent;
  draw(dcv, A, m, scD, cmap);
  wireGrid($('#mp-dcw'), dcv, m, (r, c) => { dhov.textContent = `axis ${r + 1} and axis ${c + 1}: ${fmtSig(A[r * m + c])}`; }, () => { dhov.textContent = dhint; });
  $('#mp-norm').onchange = (e) => { norm = e.target.checked; render(); };
}

// ---------------------------------------------------------------- expert options, inside the panel
const seg = (name, list, cur) => `<div class="seg small" role="radiogroup" aria-label="${name}">${list.map((o) => { const id = o.id ?? o, label = o.label ?? o; return `<label><input type="radio" name="m-${name}" value="${id}" data-f="m-${name}-${id}"${String(id) === String(cur) ? ' checked' : ''}><span>${label}</span></label>`; }).join('')}</div>`;
const q = (x, what) => `<button type="button" class="xbtn" data-x="${x}" aria-label="${what}">?</button>`;
const optionsHtml = (grp) => `<div class="mp-opt"><span class="x-lab">Color${q('mcolor', 'What do the color options do?')}</span>${seg('color', COLORS, opt.color)}</div>
  <div class="mp-opt"><span class="x-lab">Order${q('morder', 'What do the orders do?')}</span>${seg('order', ORDERS, opt.order)}${opt.order === 'groups' && !grp ? '<span class="mp-opt-note">Turn on Color by group in the expert panel.</span>' : ''}</div>
  <div class="mp-opt"><span class="x-lab">Axes shown${q('maxes', 'How many axes are shown?')}</span>${seg('axes', AXES, opt.r)}</div>`;
function onOption(e) {
  const f = e.target.dataset?.f;
  if (!f?.startsWith('m-')) return;
  if (f.startsWith('m-color-')) opt.color = f.slice(8);
  else if (f.startsWith('m-order-')) opt.order = f.slice(8);
  else if (f.startsWith('m-axes-')) opt.r = +f.slice(7);
  render(true);
}

// ---------------------------------------------------------------- opening and closing
function setOpen(v) {
  open = v;
  panel.hidden = !v;
  btn.setAttribute('aria-expanded', String(v));
  btn.classList.toggle('on', v);
  document.body.classList.toggle('mat-open', v && wide.matches);
  if (v) render(true);
  else { ctx.H.mark(-1, -1); document.documentElement.style.removeProperty('--mat-h'); }
  requestAnimationFrame(() => ctx.H.redraw());
}

/** Called after the pool, the weights or the selection changed (main.js render); draws only while the panel is open. */
export function matricesRefresh(force = false) { if (open) render(force); }

/** Expert switch turned off: back to the default colors, order and axes. */
export function matricesReset() { opt = { ...DEFAULTS }; matricesRefresh(true); }

export function initMatrices(c) {
  ctx = c;
  wide = matchMedia('(min-width: 901px)');
  panel = document.getElementById('matpanel');
  btn = document.getElementById('matbtn');
  EXPLAIN.matrices = {
    title: 'Similarity matrices',
    body: `<p>Both matrices are built from the weights the subset is picked with: the target quality $\\kappa(y)$ at the current $T$ for <b>Most likely</b>, and $\\kappa(y)/p^{\\circ}_{T'}(y)$ for <b>Sample</b>. Stack the pool's rows $\\sqrt{w_i}\\,\\phi_i$ into a matrix $\\mathbf B$ with $N$ rows. The <b>primal</b> matrix $\\mathbf L = \\mathbf B \\mathbf B^\\top$ is $N \\times N$ (pairs of completions) and the <b>dual</b> matrix $\\mathbf C = \\mathbf B^\\top \\mathbf B$ is $D \\times D$ (pairs of embedding directions). They have the same nonzero eigenvalues.</p>`,
  };
  EXPLAIN.primal = {
    title: 'Pairs of completions',
    body: `<p>Entry $(i, j)$ of $\\mathbf L$ is how similar completions $i$ and $j$ are, $\\phi_i \\cdot \\phi_j$, scaled by $\\sqrt{w_i w_j}$. The diagonal is the weights $w$. The color scale runs from the smallest to the largest entry.</p>
      <p>Rows and columns are ordered so that similar completions sit next to each other (the leaf order of average-linkage clustering on $1 - \\cos$, fixed for the pool), so groups of similar completions are dark blocks along the diagonal. Red marks show the selected set. Its $K \\times K$ submatrix $\\mathbf L_S$ has $\\det \\mathbf L_S$ equal to the subset's score.</p>
      <p>A completion whose weight is far below the largest fades out. When the pool fit is poor, a few rows carry most of the color.</p>`,
  };
  EXPLAIN.dual = {
    title: 'Pairs of embedding directions',
    body: `<p>Entry $(a, b)$ of $\\mathbf C$ is how much the weighted completions vary along embedding directions $a$ and $b$ together. It is shown in the pool's principal axes: the directions along which the completions' embeddings vary most, axis 1 first.</p>
      <p>With equal weights the matrix is diagonal. Unequal weights put mass off the diagonal. It shows no block structure. <b>Normalize</b> divides entry $(a, b)$ by $\\sqrt{c_{aa} c_{bb}}$, so the diagonal is 1 and the colors show the off-diagonal entries.</p>
      <p>$\\mathbf C$ has the same nonzero eigenvalues as the matrix of pairs of completions, so a sampler can work with it instead, which costs less when the embedding size $D$ is below the pool size $N$.</p>`,
  };
  panel.innerHTML = `<div class="mp-head"><span class="mp-title fn">Similarity matrices<button type="button" class="xbtn" data-x="matrices" aria-label="What are the similarity matrices?">?</button></span><button type="button" class="mp-close" aria-label="Close the matrices">×</button></div><div class="mp-opts" hidden></div><div class="mp-body"></div>`;
  panel.querySelector('.mp-opts').addEventListener('change', onOption);
  EXPLAIN.mcolor = { title: 'Matrix colors', body: '<p><b>Ink</b> shades each entry from the page\'s paper color to its ink color. <b>Viridis</b> uses the viridis scale, which separates close values more but is less quiet. Both run from the smallest to the largest entry of the matrix.</p>' };
  EXPLAIN.morder = { title: 'Row order', body: '<p>The order of rows and columns of the matrix of pairs of completions. <b>Blocks</b> puts similar completions next to each other (leaf order of average-linkage clustering) and draws the tree in the margins, in group colors when Groups uses Tree. <b>Groups</b> orders by the groups of the expert panel, then by weight. <b>Weight</b> puts the largest weights first. <b>Pool order</b> is the order the completions were drawn.</p>' };
  EXPLAIN.maxes = { title: 'Axes shown', body: '<p>The matrix of pairs of embedding directions is shown in the pool\'s principal axes. This sets how many of them, from the one with the most variation. Axes with almost no variation are dropped.</p>' };
  btn.onclick = () => setOpen(!open);
  panel.querySelector('.mp-close').onclick = () => setOpen(false);
  // In the capture phase, so an open "?" popover (closed by its own Escape handler) keeps the panel open.
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && open && document.getElementById('xpop').hidden) setOpen(false); }, true);
  // The plot keeps clear of the panel: report its height, and redraw the plot when the panel changes size.
  let last = 0;
  new ResizeObserver(() => {
    if (!open) return;
    const h = Math.ceil(panel.getBoundingClientRect().height);
    document.documentElement.style.setProperty('--mat-h', wide.matches ? `${h}px` : '0px');
    if (h !== last) { last = h; c.H.redraw(); }
  }).observe(panel);
  wide.addEventListener('change', () => { document.body.classList.toggle('mat-open', open && wide.matches); c.H.redraw(); });
}
