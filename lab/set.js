// "Building a set": the kernel matrix and the scatter plot, linked (a separate prototype; see docs/matrix-views.md).
// Click a row of the matrix (or a point) to add that completion to the set; the matrix then shows the kernel conditioned
// on the set, the plot draws the set as red vectors, and hovering a cell draws the pair of vectors it compares.
import * as THREE from 'three';
import { Plot3D } from '../js/plot3d.js';
import { eigSym } from '../js/dpp.js';
import { averageLinkageOrder, flatPivotOrder, nestedPivotOrder, themeOrder } from './orderings.js';
import { condition } from './condition.js';

const $ = (s) => document.querySelector(s);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const DATA = '../data/cache/';
const EMB_LABEL = { 'Qwen/Qwen3-Embedding-4B': 'Qwen3-Embedding-4B', 'Qwen/Qwen3-Embedding-0.6B': 'Qwen3-Embedding-0.6B', 'own:Qwen/Qwen3-14B-AWQ': 'Qwen3 14B own states' };
const sliderToT = (v) => (v >= 1000 ? Infinity : 10 ** (-1 + (4 * v) / 1000));
const fmtT = (T) => (T === Infinity ? '∞' : T >= 100 ? T.toFixed(0) : T >= 10 ? T.toFixed(1) : T.toFixed(2));
const sig = (x) => (x === 0 ? '0' : Math.abs(x) < 1e-3 ? x.toExponential(1) : x.toPrecision(3));

// set: the completions chosen, in the order added. history: the changes, for undo.
const S = { index: null, samples: null, G: null, L: null, n: 0, K: 3, T: 20, target: 'global', mode: 'map', ord: 'theme',
  set: [], history: [], cond: null, pi0: null, res: null, lo: 0, hi: 1, coords: null, c: null, hover: -1 };
const cache = new Map();

// ---------------------------------------------------------------- the plot: the main page's, plus the hovered pair
class LabPlot extends Plot3D {
  setPair(i, j) { this.pair = i >= 0 ? [i, j] : null; this.layout(); }
  layout() {
    super.layout();
    if (!this.pair || !this.dyn || !this.current) return;
    const ink = new THREE.Color(css('--text-2')), O = new THREE.Vector3(), [i, j] = this.pair;
    const mat = new THREE.LineBasicMaterial({ color: ink, transparent: true, opacity: 0.9 });
    for (const k of new Set([i, j])) this.dyn.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([O, this.current[k]]), mat));
    // The angle between the two drawn vectors, as an arc near the origin.
    const a = this.current[i].clone(), b = this.current[j].clone(), la = a.length(), lb = b.length();
    if (i === j || la < 1e-6 || lb < 1e-6) return;
    a.divideScalar(la); b.divideScalar(lb);
    const th = Math.acos(Math.min(1, Math.max(-1, a.dot(b)))), r = 0.35 * Math.min(la, lb);
    if (th < 1e-3) return;
    const pts = Array.from({ length: 25 }, (_, t) => { const u = t / 24; return a.clone().multiplyScalar(Math.sin((1 - u) * th)).add(b.clone().multiplyScalar(Math.sin(u * th))).multiplyScalar(r / Math.sin(th)); });
    this.dyn.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat));
  }
}
let plot = null;

// ---------------------------------------------------------------- data
async function loadIndex() {
  S.index = await (await fetch(DATA + 'index.json')).json();
  $('#pool').innerHTML = S.index.buffers.map((b, i) => `<option value="${i}">${esc(b.pool.replace(/_T(.*)\.json$/, ' · T′ = $1'))}</option>`).join('');
  $('#pool').value = String(Math.max(0, S.index.buffers.findIndex((b) => b.pool === 'story_T1.0.json')));
}

async function loadPool() {
  const b = S.index.buffers[+$('#pool').value], embs = Object.keys(b.G), cur = $('#emb').value;
  $('#emb').innerHTML = embs.map((e) => `<option value="${esc(e)}">${esc(EMB_LABEL[e] ?? e)}</option>`).join('');
  $('#emb').value = embs.includes(cur) ? cur : embs.includes('Qwen/Qwen3-Embedding-4B') ? 'Qwen/Qwen3-Embedding-4B' : embs[0];
  const key = b.pool + '|' + $('#emb').value;
  if (!cache.has(key)) {
    const [pool, buf] = await Promise.all([fetch(DATA + b.pool).then((r) => r.json()), fetch(DATA + b.G[$('#emb').value]).then((r) => r.arrayBuffer())]);
    cache.set(key, { samples: pool.samples ?? pool, raw: new Float32Array(buf) });
  }
  const { samples, raw } = cache.get(key), n = samples.length, M = Math.round(Math.sqrt(raw.length));
  const first = new Map(), c = samples.map((s, i) => (first.has(s.text) ? first.get(s.text) : (first.set(s.text, i), i)));
  const G = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) G[i * n + j] = i === j ? 1 : raw[c[i] * M + c[j]];
  // The plot's axes: G's top three eigenvectors (uncentred PCA), fixed for the pool, so points never move when the set changes.
  const { values, vectors } = eigSym(G, n), top = [...values.keys()].sort((x, y) => values[y] - values[x]).slice(0, 3);
  S.coords = Array.from({ length: n }, (_, i) => top.map((a) => Math.sqrt(Math.max(0, values[a])) * vectors[a][i]));
  Object.assign(S, { samples, G, n, set: [], history: [] });
  plot?.dispose(); plot = null; $('#plot').innerHTML = '';
  rebuild();
}

// Weights, L, the row order (fixed until the settings change) and the plot's lengths; then the set.
function rebuild() {
  const { samples, G, n } = S;
  const lk = samples.map((s) => (S.T === Infinity ? 0 : s.logp / S.T) + (S.target === 'gl' ? s.logq : 0));
  const lw = S.mode === 'sample' ? lk.map((x, i) => x - samples[i].logq) : lk;
  const m = Math.max(...lw), w = lw.map((x) => Math.exp(x - m));
  S.L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) S.L[i * n + j] = Math.sqrt(w[i] * w[j]) * G[i * n + j];
  const mk = Math.max(...lk); S.c = lk.map((x) => Math.exp(0.5 * (x - mk)));   // drawn length √κ, as on the main page
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) { const v = S.L[i * n + j]; if (v < lo) lo = v; if (v > hi) hi = v; }
  Object.assign(S, { lo, hi: hi > lo ? hi : lo + 1 });
  S.res = S.ord === 'pool' ? { order: [...Array(n).keys()] } : S.ord === 'avg' ? averageLinkageOrder(S.L, n) : S.ord === 'flat' ? flatPivotOrder(S.L, n, S.K)
    : S.ord === 'nest' ? nestedPivotOrder(S.L, n) : themeOrder(S.L, n, S.K);
  S.pi0 = condition(S.L, n, [], S.K).pi;
  const keep = [];   // the set under the new weights, as far as it stays valid
  for (const i of S.set.slice(0, S.K)) if (condition(S.L, n, [...keep, i], S.K).ok) keep.push(i);
  if (keep.length !== S.set.length) S.history = [];
  S.set = keep;
  refresh();
}

// ---------------------------------------------------------------- the set
function add(i, at = S.set.length) { S.set.splice(at, 0, i); }
function act(i) {
  const at = S.set.indexOf(i);
  if (at >= 0) { S.set.splice(at, 1); S.history.push({ op: 'remove', i, at }); refresh(); return; }
  if (S.set.length >= S.K || !S.cond.addable[i]) { showCandidate(i, true); return; }
  add(i); S.history.push({ op: 'add', i }); refresh();
}
function undo() {
  const h = S.history.pop();
  if (!h) return;
  if (h.op === 'add') S.set.splice(S.set.indexOf(h.i), 1); else add(h.i, h.at);
  refresh();
}

// Condition on the set and redraw everything.
function refresh() {
  S.cond = condition(S.L, S.n, S.set, S.K);
  drawMatrix(); drawRail(); drawPlot();
  const m = S.set.length;
  $('#matsub').textContent = m ? `L conditioned on the ${m} chosen (L^A): what is left of each comparison` : 'L: every pair of completions compared';
}

// ---------------------------------------------------------------- the matrix
function drawMatrix() {
  const n = S.n, { order } = S.res, A = S.cond.LA, inSet = new Uint8Array(n);
  for (const i of S.set) inSet[i] = 1;
  const c0 = rgb(css('--surface')), c1 = rgb(css('--ink')), ca = rgb(css('--accent'));
  const cv = $('#mat'); cv.width = n; cv.height = n;
  const ctx = cv.getContext('2d'), img = ctx.createImageData(n, n);
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const i = order[r], j = order[c], k = (r * n + c) * 4;
    const col = inSet[i] || inSet[j] ? c0.map((x, ch) => x + (ca[ch] - x) * 0.18) : c0.map((x, ch) => x + (c1[ch] - x) * Math.min(1, Math.max(0, (A[i * n + j] - S.lo) / (S.hi - S.lo))));
    for (let ch = 0; ch < 3; ch++) img.data[k + ch] = Math.round(col[ch]);
    img.data[k + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  // Margin: theme blocks with their strength, or the flat pivots' blocks; the set's rows in red.
  const svg = $('.left svg'), muted = css('--muted'), ink = css('--text-2'), accent = css('--accent');
  svg.setAttribute('viewBox', `0 0 64 ${n}`); svg.setAttribute('preserveAspectRatio', 'none');
  let s = '';
  for (const [a, b] of S.res.blocks ?? []) s += `<path d="M58 ${a + 0.3}H52V${b - 0.3}H58" fill="none" stroke="${muted}" stroke-width="1" vector-effect="non-scaling-stroke"/>`;
  if (S.res.part) for (let r = 0; r < n; r++) { const i = order[r], sh = S.res.part(i, S.res.theme[i]) / S.res.pi[i]; s += `<rect x="${50 - 46 * sh}" y="${r + 0.12}" width="${46 * sh}" height="0.76" fill="${ink}" opacity="0.45"/>`; }
  const pos = new Map(order.map((i, r) => [i, r]));
  for (const i of S.set) s += `<rect x="59" y="${pos.get(i)}" width="5" height="1" fill="${accent}"/>`;
  svg.innerHTML = s;
}
function rgb(c) { const x = document.createElement('canvas').getContext('2d'); x.fillStyle = c; const h = x.fillStyle; return h.startsWith('#') ? [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16)) : h.match(/\d+/g).slice(0, 3).map(Number); }

function crosshair(i, j) {
  const r = $('.xh.r'), c = $('.xh.c');
  if (i < 0) { r.style.display = c.style.display = 'none'; return; }
  const { order } = S.res, n = S.n;
  r.style.display = c.style.display = 'block';
  r.style.top = `${(100 * (order.indexOf(i) + 0.5)) / n}%`; c.style.left = `${(100 * (order.indexOf(j) + 0.5)) / n}%`;
}

function hoverCell(i, j) {
  crosshair(i, j); plot?.setPair(i, j); plot?.highlight(i);
  document.querySelectorAll('.card').forEach((el) => el.classList.toggle('hl', +el.dataset.i === i || +el.dataset.i === j));
  if (i < 0) { $('#pair').textContent = 'Hover a cell: each one compares two vectors of the plot.'; showCandidate(-1); return; }
  const n = S.n, t = (k) => esc(S.samples[k].text.replace(/\s+/g, ' ').slice(0, 70)) + (S.samples[k].text.length > 70 ? '…' : '');
  $('#pair').innerHTML = `<div>${i === j ? `“${t(i)}” with itself` : `“${t(i)}” · “${t(j)}”`}</div>
    <div class="v">cos = ${S.G[i * n + j].toFixed(3)} · L = ${sig(S.L[i * n + j])}${S.set.length ? ` · L^A = ${sig(S.cond.LA[i * n + j])}` : ''}</div>`;
  showCandidate(i);
}

// ---------------------------------------------------------------- the rail
function card(i, k, extra = '') {
  const s = S.samples[i];
  return `<li class="card" data-i="${i}"><div class="txt">${esc(s.text)}</div>
    <div class="meta"><span class="dot"></span><span class="num">${k}</span><span>log p ${s.logp.toFixed(1)}</span>${extra}</div>
    <button type="button" class="x" data-x="${i}" aria-label="Remove from the set" title="Remove (Ctrl/⌘+Z undoes)">×</button></li>`;
}
function drawRail() {
  const m = S.set.length;
  // Chain rule: the set's volume det L_A is the product, in the order added, of each one's leftover when it was added.
  const f = S.set.map((i, k) => condition(S.L, S.n, S.set.slice(0, k), S.K).leftover[i]);
  const vol = f.reduce((a, b) => a * b, 1);
  $('#setsub').textContent = m ? `${m} of K = ${S.K} · volume det L_A = ${sig(vol)}` : `Empty. Click a row of the matrix or a point to add a completion (up to K = ${S.K}).`;
  $('#cards').innerHTML = S.set.map((i, k) => card(i, k + 1, `<span>×${sig(f[k])} volume</span><span>π ${(100 * S.pi0[i]).toFixed(1)}% alone</span>`)).join('') + '<li id="cand"></li>';
  document.querySelectorAll('#cards .card').forEach((el) => {
    el.onmouseenter = () => { plot?.highlight(+el.dataset.i); crosshair(+el.dataset.i, +el.dataset.i); };
    el.onmouseleave = () => { plot?.highlight(-1); crosshair(-1); };
  });
  document.querySelectorAll('#cards .x').forEach((b) => (b.onclick = () => act(+b.dataset.x)));
}
// The hovered completion as a candidate: what adding it would do.
function showCandidate(i, refused = false) {
  const el = $('#cand');
  if (!el) return;
  if (i < 0 || S.set.includes(i)) { el.outerHTML = '<li id="cand"></li>'; return; }
  const s = S.samples[i], full = S.set.length >= S.K, ok = !full && S.cond.addable[i];
  const why = full ? `The set already has K = ${S.K}. Remove one (× or Ctrl/⌘+Z) to add this.` : !S.cond.addable[i] ? 'Already in the span of the set: adding it gives volume 0.'
    : `Click to add: ×${sig(S.cond.leftover[i])} volume · π ${(100 * S.pi0[i]).toFixed(1)}% → ${(100 * S.cond.pi[i]).toFixed(1)}% given the set`;
  el.outerHTML = `<li id="cand" class="card cand${ok ? '' : ' no'}${refused ? ' hl' : ''}"><div class="txt">${esc(s.text)}</div><div class="meta"><span class="dot"></span><span>${why}</span></div></li>`;
}

// ---------------------------------------------------------------- the plot
function drawPlot() {
  if (!plot) {
    plot = new LabPlot($('#plot'), {
      onHover: (i) => { if (i >= 0) hoverCell(i, i); else hoverCell(-1, -1); },
      onTap: (i) => { if (i >= 0) act(i); },
    });
  }
  plot.update({ coords: S.coords, c: S.c, sel: S.set, fitted: false, color: () => css('--ink'), selColor: () => css('--accent'), accent: css('--accent'), text: css('--text-2'), muted: css('--muted'), bg: css('--surface') });
}

// ---------------------------------------------------------------- wiring
async function main() {
  await loadIndex();
  const cv = $('#mat');
  const cell = (e) => { const r = cv.getBoundingClientRect(), n = S.n, f = (v, a, b) => Math.min(n - 1, Math.max(0, Math.floor(((v - a) / b) * n))); return [S.res.order[f(e.clientY, r.top, r.height)], S.res.order[f(e.clientX, r.left, r.width)]]; };
  cv.onpointermove = (e) => hoverCell(...cell(e));
  cv.onpointerleave = () => hoverCell(-1, -1);
  let down = null;
  cv.onpointerdown = (e) => (down = [e.clientX, e.clientY]);
  cv.onpointerup = (e) => { if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) <= 6) act(cell(e)[0]); };
  document.addEventListener('keydown', (e) => { if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key.toLowerCase() === 'z' && !/INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) { e.preventDefault(); undo(); } });
  $('#pool').onchange = loadPool; $('#emb').onchange = loadPool;
  $('#K').onchange = (e) => { S.K = Math.max(1, Math.min(12, +e.target.value || 3)); e.target.value = S.K; rebuild(); };
  $('#T').oninput = (e) => { S.T = sliderToT(+e.target.value); $('#Tout').textContent = fmtT(S.T); };
  $('#T').onchange = () => rebuild();
  $('#target').onchange = (e) => { S.target = e.target.value; rebuild(); };
  $('#mode').onchange = (e) => { S.mode = e.target.value; rebuild(); };
  $('#ord').onchange = (e) => { S.ord = e.target.value; rebuild(); };
  await loadPool();
}
main();
