// Matrix orderings lab: the pool's kernel matrix under different row orders, side by side.
import { averageLinkageOrder, flatPivotOrder, nestedPivotOrder, themeOrder, pivotedCholesky } from './orderings.js';

const $ = (s, el = document) => el.querySelector(s);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const DATA = '../data/cache/';
const EMB_LABEL = { 'Qwen/Qwen3-Embedding-4B': 'Qwen3-Embedding-4B', 'Qwen/Qwen3-Embedding-0.6B': 'Qwen3-Embedding-0.6B', 'own:Qwen/Qwen3-14B-AWQ': 'Qwen3 14B own states (256-d projection)' };
const ORDERS = [
  { id: 'pool', label: 'Pool order' },
  { id: 'avg', label: 'Average linkage (Manuel)' },
  { id: 'flat', label: 'Pivots, flat (K blocks)' },
  { id: 'nest', label: 'Pivots, nested (tree)' },
  { id: 'theme', label: 'Themes (eigenvectors)' },
];
// T slider as on the main page: log10 T from −1 to 3, the right end is ∞.
const sliderToT = (v) => (v >= 1000 ? Infinity : 10 ** (-1 + (4 * v) / 1000));
const fmtT = (T) => (T === Infinity ? '∞' : T >= 100 ? T.toFixed(0) : T >= 10 ? T.toFixed(1) : T.toFixed(2));

const S = { index: null, buf: null, samples: null, Gfull: null, items: [], G: null, L: null, n: 0, opts: { rows: 'all', show: 'G', on: 'G', K: 3, T: 20, target: 'global', mode: 'map' }, panels: [], picks: [] };
const cache = new Map();

async function loadIndex() {
  S.index = await (await fetch(DATA + 'index.json')).json();
  const pools = S.index.buffers;
  $('#pool').innerHTML = pools.map((b, i) => `<option value="${i}">${esc(b.pool.replace(/_T(.*)\.json$/, " · T′ = $1"))}</option>`).join('');
  $('#pool').value = String(Math.max(0, pools.findIndex((b) => b.pool === 'joke_T1.0.json')));
}

async function loadPool() {
  const b = S.index.buffers[+$('#pool').value];
  const embs = Object.keys(b.G);
  const cur = $('#emb').value;
  $('#emb').innerHTML = embs.map((e) => `<option value="${esc(e)}">${esc(EMB_LABEL[e] ?? e)}</option>`).join('');
  $('#emb').value = embs.includes(cur) ? cur : embs.includes('Qwen/Qwen3-Embedding-4B') ? 'Qwen/Qwen3-Embedding-4B' : embs[0];
  const key = b.pool + '|' + $('#emb').value;
  if (!cache.has(key)) {
    const [pool, buf] = await Promise.all([fetch(DATA + b.pool).then((r) => r.json()), fetch(DATA + b.G[$('#emb').value]).then((r) => r.arrayBuffer())]);
    cache.set(key, { samples: pool.samples ?? pool, raw: new Float32Array(buf) });
  }
  const { samples, raw } = cache.get(key), N = samples.length, M = Math.round(Math.sqrt(raw.length));
  // Identical strings get identical rows (the embedding is a function of the string), as on the main page.
  const first = new Map(), c = samples.map((s, i) => (first.has(s.text) ? first.get(s.text) : (first.set(s.text, i), i)));
  const G = new Float64Array(N * N);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) G[i * N + j] = i === j ? 1 : raw[c[i] * M + c[j]];
  S.samples = samples; S.Gfull = G; S.firstOf = c;
  rebuild();
}

// Rows (all or distinct), the weights and L; then every panel.
function rebuild() {
  const N = S.samples.length, o = S.opts;
  S.items = o.rows === 'distinct' ? [...Array(N).keys()].filter((i) => S.firstOf[i] === i) : [...Array(N).keys()];
  const n = (S.n = S.items.length), it = S.items;
  const copies = new Map(); S.firstOf.forEach((f) => copies.set(f, (copies.get(f) ?? 0) + 1));
  S.copies = it.map((i) => copies.get(S.firstOf[i]));
  const G = new Float64Array(n * n);
  for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) G[a * n + b] = S.Gfull[it[a] * N + it[b]];
  const lk = it.map((i) => (o.T === Infinity ? 0 : S.samples[i].logp / o.T) + (o.target === 'gl' ? S.samples[i].logq : 0));
  const lw = o.mode === 'sample' ? lk.map((x, a) => x - S.samples[it[a]].logq) : lk;
  const m = Math.max(...lw), w = lw.map((x) => Math.exp(x - m));
  const L = new Float64Array(n * n);
  for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) L[a * n + b] = Math.sqrt(w[a] * w[b]) * G[a * n + b];
  // Greedy (Most likely) picks: the first K pivots of L with the quality weights κ (not κ/q).
  const mk = Math.max(...lk), wk = lk.map((x) => Math.exp(x - mk)), Lk = new Float64Array(n * n);
  for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) Lk[a * n + b] = Math.sqrt(wk[a] * wk[b]) * G[a * n + b];
  S.picks = pivotedCholesky(Lk, n).piv.slice(0, o.K);
  S.G = G; S.L = L; S.w = w;
  S.panels.forEach(drawPanel);
}

function computeOrder(id) {
  const M = S.opts.on === 'L' ? S.L : S.G, n = S.n;
  if (id === 'pool') return { order: [...Array(n).keys()] };
  if (id === 'avg') return averageLinkageOrder(M, n);
  if (id === 'flat') return flatPivotOrder(M, n, S.opts.K);
  if (id === 'nest') return nestedPivotOrder(M, n);
  return themeOrder(M, n);
}

// ---------------------------------------------------------------- drawing
const parseRGB = (c) => { const x = document.createElement('canvas').getContext('2d'); x.fillStyle = c; const h = x.fillStyle; return h.startsWith('#') ? [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16)) : h.match(/\d+/g).slice(0, 3).map(Number); };

function makePanel(defaultOrder) {
  const el = document.createElement('section');
  el.className = 'panel';
  el.innerHTML = `<div class="phead"><select class="ord">${ORDERS.map((o) => `<option value="${o.id}">${o.label}</option>`).join('')}</select><span class="note"></span></div>
    <div class="grid"><div class="corner"></div><div class="top"><svg class="margin"></svg></div><div class="left"><svg class="margin"></svg></div>
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
    showPair(P.res.order[row], P.res.order[col], P);
  };
  cv.onpointerleave = () => showPair(-1, -1);
  S.panels.push(P);
  return P;
}

function drawPanel(P) {
  const n = S.n; if (!n) return;
  const t0 = performance.now();
  P.res = computeOrder(P.id);
  const ms = performance.now() - t0, { order } = P.res;
  const A = S.opts.show === 'L' ? S.L : S.G;
  // Colour: paper to ink over the off-diagonal range of the matrix shown.
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
  $('.lo', P.el).textContent = lo.toFixed(2); $('.hi', P.el).textContent = hi.toFixed(2);
  $('.bar', P.el).style.background = `linear-gradient(90deg, ${css('--surface')}, ${css('--ink')})`;
  const pos = new Int32Array(n); order.forEach((i, r) => (pos[i] = r));
  const accent = css('--accent'), muted = css('--muted'), ink = css('--text-2');
  // Block boundaries (flat pivots, themes).
  const blocks = P.res.blocks ?? [];
  // Top strip: height for the pivot orders; red ticks for the greedy picks.
  const top = $('.top svg', P.el), left = $('.left svg', P.el);
  top.setAttribute('viewBox', `0 0 ${n} 34`); top.setAttribute('preserveAspectRatio', 'none');
  left.setAttribute('viewBox', `0 0 70 ${n}`); left.setAttribute('preserveAspectRatio', 'none');
  let ts = '', ls = '';
  if (P.res.height) {
    const hmax = Math.max(...order.map((i) => P.res.height[i])) || 1;
    for (let r = 0; r < n; r++) { const h = (P.res.height[order[r]] / hmax) * 24; ts += `<rect x="${r + 0.1}" y="${30 - h}" width="0.8" height="${h}" fill="${ink}" opacity="0.55"/>`; }
  }
  for (const i of S.picks) ts += `<rect x="${pos[i]}" y="30" width="1" height="4" fill="${accent}"/>`;
  for (const [a] of blocks) if (a > 0) ts += `<line x1="${a}" x2="${a}" y1="0" y2="34" stroke="${muted}" stroke-width="1" vector-effect="non-scaling-stroke" stroke-dasharray="2 2"/>`;
  // Left margin: the tree (nested pivots: heights; average linkage: merge distances), else block brackets.
  if (P.id === 'nest') {
    const { parent, height } = P.res, hmax = height[order[0]] || 1, x = (h) => 66 - (h / hmax) * 62;
    for (let r = 0; r < n; r++) {
      const i = order[r], p = parent[i];
      if (p < 0) continue;
      const yi = r + 0.5, yp = pos[p] + 0.5;
      ls += `<path d="M${x(height[i])} ${yi}H${x(height[p])}V${yp}" fill="none" stroke="${ink}" stroke-width="0.8" vector-effect="non-scaling-stroke" opacity="0.7"/>`;
    }
  } else if (P.id === 'avg') {
    const { merges } = P.res, hmax = Math.max(...merges.map((m) => m.h)) || 1, x = (h) => 66 - (h / hmax) * 62;
    const yOf = new Map(order.map((i, r) => [i, r + 0.5])), hOf = new Map(order.map((i) => [i, 0]));
    for (const m of merges) {
      const ya = yOf.get(m.a), yb = yOf.get(m.b), xa = x(hOf.get(m.a)), xb = x(hOf.get(m.b)), xm = x(m.h);
      ls += `<path d="M${xa} ${ya}H${xm}V${yb}H${xb}" fill="none" stroke="${ink}" stroke-width="0.8" vector-effect="non-scaling-stroke" opacity="0.7"/>`;
      yOf.set(m.node, (ya + yb) / 2); hOf.set(m.node, m.h);
    }
  } else {
    for (const [a, b] of blocks) ls += `<path d="M64 ${a + 0.3}H58V${b - 0.3}H64" fill="none" stroke="${muted}" stroke-width="1" vector-effect="non-scaling-stroke"/>`;
  }
  for (const i of S.picks) ls += `<rect x="66" y="${pos[i]}" width="4" height="1" fill="${accent}"/>`;
  top.innerHTML = ts; left.innerHTML = ls;
  $('.corner', P.el).textContent = P.res.height ? 'height ↑' : '';
  const nb = blocks.length ? `${blocks.length} blocks · ` : '';
  $('.note', P.el).textContent = `${nb}${n} rows · ${S.opts.show} shown, ordered on ${S.opts.on} · ${ms.toFixed(0)} ms`;
}

function showPair(i, j, P) {
  S.panels.forEach((Q) => {
    const r = $('.xh.r', Q.el), c = $('.xh.c', Q.el);
    if (i < 0 || !Q.res) { r.style.display = c.style.display = 'none'; return; }
    const pos = Q.res.order.indexOf(i), posj = Q.res.order.indexOf(j), n = S.n;
    r.style.display = c.style.display = 'block';
    r.style.top = `${(100 * (pos + 0.5)) / n}%`; c.style.left = `${(100 * (posj + 0.5)) / n}%`;
  });
  if (i < 0) return;
  const n = S.n, it = S.items, s = (a) => S.samples[it[a]];
  const nest = S.panels.map((Q) => Q.res).find((r) => r?.parent);
  const extra = (a) => {
    const bits = [`log p ${s(a).logp.toFixed(1)}`];
    if (S.copies[a] > 1) bits.push(`×${S.copies[a]}`);
    if (S.picks.includes(a)) bits.push('greedy pick');
    if (nest) bits.push(`height ${nest.height[a].toFixed(3)}${nest.parent[a] >= 0 ? '' : ' (root)'}`);
    return bits.join(' · ');
  };
  $('#hover').innerHTML = `<div class="pair">
    <div><div class="k">Row</div><div class="t">${esc(s(i).text)}</div><div class="v">${extra(i)}</div></div>
    <div><div class="k">Column</div><div class="t">${esc(s(j).text)}</div><div class="v">${extra(j)}</div></div></div>
    <div class="v">G = ${S.G[i * n + j].toFixed(4)} · L = ${S.L[i * n + j].toExponential(3)}</div>`;
}

// ---------------------------------------------------------------- controls
function seg(id, key) {
  $(id).querySelectorAll('button').forEach((b) => (b.onclick = () => {
    $(id).querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    S.opts[key] = b.dataset.v;
    $('#Lctl').hidden = S.opts.show !== 'L' && S.opts.on !== 'L';
    rebuild();
  }));
}

async function main() {
  await loadIndex();
  makePanel('avg'); makePanel('nest');
  seg('#rows', 'rows'); seg('#show', 'show'); seg('#on', 'on');
  $('#pool').onchange = loadPool; $('#emb').onchange = loadPool;
  $('#K').onchange = (e) => { S.opts.K = Math.max(1, Math.min(12, +e.target.value || 3)); rebuild(); };
  $('#T').oninput = (e) => { S.opts.T = sliderToT(+e.target.value); $('#Tout').textContent = fmtT(S.opts.T); };
  $('#T').onchange = () => rebuild();
  $('#target').onchange = (e) => { S.opts.target = e.target.value; rebuild(); };
  $('#mode').onchange = (e) => { S.opts.mode = e.target.value; rebuild(); };
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => S.panels.forEach(drawPanel));
  await loadPool();
}
main();
