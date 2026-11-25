// Matrix orderings lab: the pool's DPP kernel L under different row orders, side by side.
// Everything is computed from L = diag(√w) G diag(√w), the matrix the DPP uses.
import { averageLinkageOrder, flatPivotOrder, nestedPivotOrder, themeOrder, pivotedCholesky } from './orderings.js';

const $ = (s, el = document) => el.querySelector(s);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const DATA = '../data/cache/';
const EMB_LABEL = { 'Qwen/Qwen3-Embedding-4B': 'Qwen3-Embedding-4B', 'Qwen/Qwen3-Embedding-0.6B': 'Qwen3-Embedding-0.6B', 'own:Qwen/Qwen3-14B-AWQ': 'Qwen3 14B own states' };
const ORDERS = [
  { id: 'pool', label: 'Pool order' },
  { id: 'avg', label: 'Average linkage (Manuel)' },
  { id: 'flat', label: 'Pivots, flat (K blocks)' },
  { id: 'nest', label: 'Pivots, nested (tree)' },
  { id: 'theme', label: 'Themes (eigenvectors)' },
];
// T slider as on the main page: log10 T from −1 to 3; the right end is ∞.
const sliderToT = (v) => (v >= 1000 ? Infinity : 10 ** (-1 + (4 * v) / 1000));
const fmtT = (T) => (T === Infinity ? '∞' : T >= 100 ? T.toFixed(0) : T >= 10 ? T.toFixed(1) : T.toFixed(2));

const S = { index: null, samples: null, G: null, L: null, n: 0, opts: { K: 3, T: 20, target: 'global', mode: 'map' }, panels: [], picks: [] };
const cache = new Map();

async function loadIndex() {
  S.index = await (await fetch(DATA + 'index.json')).json();
  $('#pool').innerHTML = S.index.buffers.map((b, i) => `<option value="${i}">${esc(b.pool.replace(/_T(.*)\.json$/, ' · T′ = $1'))}</option>`).join('');
  $('#pool').value = String(Math.max(0, S.index.buffers.findIndex((b) => b.pool === 'joke_T1.0.json')));
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
  // Identical strings get identical rows (the embedding is a function of the string), as on the main page.
  const first = new Map(), c = samples.map((s, i) => (first.has(s.text) ? first.get(s.text) : (first.set(s.text, i), i)));
  const G = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) G[i * n + j] = i === j ? 1 : raw[c[i] * M + c[j]];
  Object.assign(S, { samples, G, n });
  rebuild();
}

// The weights and L for the current settings, then every panel.
function rebuild() {
  const { samples, G, n } = S, o = S.opts;
  const lk = samples.map((s) => (o.T === Infinity ? 0 : s.logp / o.T) + (o.target === 'gl' ? s.logq : 0));
  const lw = o.mode === 'sample' ? lk.map((x, i) => x - samples[i].logq) : lk;
  const weigh = (l) => { const m = Math.max(...l), w = l.map((x) => Math.exp(x - m)), A = new Float64Array(n * n); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) A[i * n + j] = Math.sqrt(w[i] * w[j]) * G[i * n + j]; return A; };
  S.L = weigh(lw);
  S.picks = pivotedCholesky(o.mode === 'map' ? S.L : weigh(lk), n).piv.slice(0, o.K);   // greedy (Most likely) picks use κ
  S.panels.forEach(drawPanel);
}

function computeOrder(id) {
  const { L, n } = S;
  if (id === 'pool') return { order: [...Array(n).keys()] };
  if (id === 'avg') return averageLinkageOrder(L, n);
  if (id === 'flat') return flatPivotOrder(L, n, S.opts.K);
  if (id === 'nest') return nestedPivotOrder(L, n);
  return themeOrder(L, n, S.opts.K);
}

// ---------------------------------------------------------------- drawing
const parseRGB = (c) => { const x = document.createElement('canvas').getContext('2d'); x.fillStyle = c; const h = x.fillStyle; return h.startsWith('#') ? [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16)) : h.match(/\d+/g).slice(0, 3).map(Number); };

function makePanel(defaultOrder) {
  const el = document.createElement('section');
  el.className = 'panel';
  el.innerHTML = `<div class="phead"><select class="ord">${ORDERS.map((o) => `<option value="${o.id}">${o.label}</option>`).join('')}</select><span class="note"></span></div>
    <div class="grid"><div class="left"><svg class="margin"></svg></div>
    <div class="cw"><canvas></canvas><i class="xh r"></i><i class="xh c"></i></div></div>
    <div class="scale"><span class="lo"></span><span class="bar"></span><span class="hi"></span></div>`;
  $('#panels').append(el);
  const P = { el, id: defaultOrder, res: null };
  $('.ord', el).value = defaultOrder;
  $('.ord', el).onchange = (e) => { P.id = e.target.value; drawPanel(P); };
  const cv = $('canvas', el);
  cv.onpointermove = (e) => {
    const r = cv.getBoundingClientRect(), n = S.n;
    const row = Math.min(n - 1, Math.max(0, Math.floor(((e.clientY - r.top) / r.height) * n)));
    const col = Math.min(n - 1, Math.max(0, Math.floor(((e.clientX - r.left) / r.width) * n)));
    showPair(P.res.order[row], P.res.order[col]);
  };
  cv.onpointerleave = () => showPair(-1, -1);
  S.panels.push(P);
}

function drawPanel(P) {
  const n = S.n; if (!n) return;
  const t0 = performance.now();
  P.res = computeOrder(P.id);
  const ms = performance.now() - t0, { order } = P.res, A = S.L;
  // Colour: paper to ink over the off-diagonal range.
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) { const v = A[i * n + j]; if (v < lo) lo = v; if (v > hi) hi = v; }
  if (!(hi > lo)) { lo = 0; hi = 1; }
  const c0 = parseRGB(css('--surface')), c1 = parseRGB(css('--ink'));
  const cv = $('canvas', P.el); cv.width = n; cv.height = n;
  const ctx = cv.getContext('2d'), img = ctx.createImageData(n, n);
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const v = A[order[r] * n + order[c]], t = Math.min(1, Math.max(0, (v - lo) / (hi - lo))), k = (r * n + c) * 4;
    for (let ch = 0; ch < 3; ch++) img.data[k + ch] = Math.round(c0[ch] + (c1[ch] - c0[ch]) * t);
    img.data[k + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  $('.lo', P.el).textContent = lo.toPrecision(2); $('.hi', P.el).textContent = hi.toPrecision(2);
  $('.bar', P.el).style.background = `linear-gradient(90deg, ${css('--surface')}, ${css('--ink')})`;
  const pos = new Int32Array(n); order.forEach((i, r) => (pos[i] = r));
  const accent = css('--accent'), muted = css('--muted'), ink = css('--text-2');
  const blocks = P.res.blocks ?? [];
  // Left margin: the tree (nested pivots: leftover length when taken; average linkage: merge distance), else block brackets.
  const left = $('.left svg', P.el);
  left.setAttribute('viewBox', `0 0 70 ${n}`); left.setAttribute('preserveAspectRatio', 'none');
  let ls = '';
  const line = (d, op = 0.7) => `<path d="${d}" fill="none" stroke="${ink}" stroke-width="0.8" vector-effect="non-scaling-stroke" opacity="${op}"/>`;
  if (P.id === 'nest') {
    const { parent, height } = P.res, hmax = height[order[0]] || 1, x = (h) => 64 - (h / hmax) * 60;
    for (let r = 0; r < n; r++) { const i = order[r], p = parent[i]; if (p >= 0) ls += line(`M${x(height[i])} ${r + 0.5}H${x(height[p])}V${pos[p] + 0.5}`); }
  } else if (P.id === 'avg') {
    const { merges } = P.res, hmax = Math.max(...merges.map((m) => m.h)) || 1, x = (h) => 64 - (h / hmax) * 60;
    const yOf = new Map(order.map((i, r) => [i, r + 0.5])), hOf = new Map(order.map((i) => [i, 0]));
    for (const m of merges) {
      const ya = yOf.get(m.a), yb = yOf.get(m.b);
      ls += line(`M${x(hOf.get(m.a))} ${ya}H${x(m.h)}V${yb}H${x(hOf.get(m.b))}`);
      yOf.set(m.node, (ya + yb) / 2); hOf.set(m.node, m.h);
    }
  } else {
    for (const [a, b] of blocks) ls += `<path d="M64 ${a + 0.3}H58V${b - 0.3}H64" fill="none" stroke="${muted}" stroke-width="1" vector-effect="non-scaling-stroke"/>`;
  }
  // Block strength (themes): for each row, the share of the completion's inclusion probability carried by its own
  // theme, r_i = p_a u_a(i)^2 / π_i (a full bar at the dashed line = 100%). Short bars: the block is a weak assignment.
  let strength = '';
  if (P.id === 'theme') {
    const { theme, part, pi } = P.res;
    let sum = 0;
    ls += `<line x1="4" x2="4" y1="0" y2="${n}" stroke="${muted}" stroke-width="1" stroke-dasharray="2 3" vector-effect="non-scaling-stroke"/>`;
    for (let r = 0; r < n; r++) {
      const i = order[r], sh = part(i, theme[i]) / pi[i];
      sum += sh;
      ls += `<rect x="${56 - 52 * sh}" y="${r + 0.12}" width="${52 * sh}" height="0.76" fill="${ink}" opacity="0.55"/>`;
    }
    strength = `mean share ${(100 * sum / n).toFixed(0)}% · `;
  }
  for (const i of S.picks) ls += `<rect x="66" y="${pos[i]}" width="4" height="1" fill="${accent}"/>`;
  left.innerHTML = ls;
  const o = S.opts, nb = blocks.length ? `${blocks.length} blocks · ` : '';
  $('.note', P.el).textContent = `${nb}${strength}T = ${fmtT(o.T)} · ${o.mode === 'map' ? 'κ' : 'κ/q'} · ${ms.toFixed(0)} ms`;
}

function showPair(i, j) {
  S.panels.forEach((Q) => {
    const r = $('.xh.r', Q.el), c = $('.xh.c', Q.el);
    if (i < 0 || !Q.res) { r.style.display = c.style.display = 'none'; return; }
    const n = S.n;
    r.style.display = c.style.display = 'block';
    r.style.top = `${(100 * (Q.res.order.indexOf(i) + 0.5)) / n}%`; c.style.left = `${(100 * (Q.res.order.indexOf(j) + 0.5)) / n}%`;
  });
  if (i < 0) return;
  const n = S.n, s = (a) => S.samples[a];
  const th = S.panels.map((Q) => Q.res).find((r) => r?.part);
  const extra = (a) => {
    const bits = [`log p ${s(a).logp.toFixed(1)}`];
    if (S.picks.includes(a)) bits.push('greedy pick');
    if (th) {
      const t = th.theme[a], rank = th.blocks.findIndex((b) => b[2] === t);
      bits.push(`π ${(100 * th.pi[a]).toFixed(1)}%, ${(100 * th.part(a, t) / th.pi[a]).toFixed(0)}% of it via theme ${rank + 1} (p = ${th.p[t].toFixed(2)})`);
    }
    return bits.join(' · ');
  };
  $('#hover').innerHTML = `<div class="pair">
    <div><div class="k">Row</div><div class="t">${esc(s(i).text)}</div><div class="v">${extra(i)}</div></div>
    <div><div class="k">Column</div><div class="t">${esc(s(j).text)}</div><div class="v">${extra(j)}</div></div></div>
    <div class="v">L = ${S.L[i * n + j].toPrecision(3)} · cos = G = ${S.G[i * n + j].toFixed(4)}</div>`;
}

// ---------------------------------------------------------------- controls
async function main() {
  await loadIndex();
  makePanel('avg'); makePanel('theme');
  $('#pool').onchange = loadPool; $('#emb').onchange = loadPool;
  $('#K').onchange = (e) => { S.opts.K = Math.max(1, Math.min(12, +e.target.value || 3)); e.target.value = S.opts.K; rebuild(); };
  $('#T').oninput = (e) => { S.opts.T = sliderToT(+e.target.value); $('#Tout').textContent = fmtT(S.opts.T); };
  $('#T').onchange = () => rebuild();
  $('#target').onchange = (e) => { S.opts.target = e.target.value; rebuild(); };
  $('#mode').onchange = (e) => { S.opts.mode = e.target.value; rebuild(); };
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => S.panels.forEach(drawPanel));
  await loadPool();
}
main();
