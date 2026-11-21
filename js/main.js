import * as d3 from 'd3';
import {
  gram, symEig, kernel, spectrum, sampleKDPP, greedyMAP, logZ, logDetSub, inclusionProbs,
  columnLogScale, logWeights, ess, quality, diversity, mulberry32,
} from './dpp.js';
import { Plot3D } from './plot3d.js';

const $ = (s) => document.querySelector(s);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : '—');

// ---------------------------------------------------------------- state
const S = {
  pool: null,        // { prompt, samples, N, D, Tprime, ... }
  phi: null,         // Float32Array[] unit embeddings
  G: null,           // Gram ΦᵀΦ
  logp: null, logq: null,
  proj: null,        // { coords: [N][3], frac: number[] }
  dims: 3,
  mode: 'sample',
  T: { sample: 1, map: 20 },
  K: 10,
  rng: mulberry32(12345),
  sel: [], spec: null, L: null, c: null, pi: null, logP: NaN,
  hover: -1,
};

// T slider: log scale over [0.1, 1000].
const T_MIN = -1, T_MAX = 3;
const sliderToT = (v) => 10 ** (T_MIN + ((T_MAX - T_MIN) * v) / 1000);
const tToSlider = (T) => Math.round(((Math.log10(T) - T_MIN) / (T_MAX - T_MIN)) * 1000);

// ---------------------------------------------------------------- data
function decodeEmb(b64, N, D) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const f = new Float32Array(bytes.buffer);
  return Array.from({ length: N }, (_, i) => f.subarray(i * D, (i + 1) * D));
}

async function loadGallery() {
  const idx = await (await fetch('data/index.json')).json();
  const sel = $('#gallery-select');
  sel.innerHTML = '<option value="" disabled>choose…</option>' + idx.map((g) => `<option value="${g.id}">${escapeHtml(g.prompt)}</option>`).join('');
  sel.onchange = () => sel.value && loadGalleryItem(sel.value);
  sel.value = idx[0].id;
  await loadGalleryItem(idx[0].id);
}

async function loadGalleryItem(id) {
  const rec = await (await fetch(`data/${id}.json`)).json();
  $('#prompt').value = rec.prompt;
  $('#live-status').textContent = `Loaded ${rec.N} precomputed completions (T′ = ${rec.Tprime}). Edit the prompt and press Generate to make your own.`;
  setPool(rec, decodeEmb(rec.emb, rec.N, rec.D));
}

function setPool(pool, phi) {
  S.pool = pool;
  S.phi = phi;
  S.logp = pool.samples.map((s) => s.logp);
  S.logq = pool.samples.map((s) => s.logq);
  const N = pool.samples.length;
  S.G = gram(phi);
  // Uncentered PCA of the unit embeddings via the N×N Gram matrix.
  const { values, vectors } = symEig(S.G, N);
  const tot = values.reduce((a, b) => a + Math.max(0, b), 0);
  S.proj = {
    frac: values.slice(0, 8).map((v) => Math.max(0, v) / tot),
    coords: Array.from({ length: N }, (_, n) => [0, 1, 2].map((k) => Math.sqrt(Math.max(0, values[k])) * vectors[k][n])),
  };
  S.K = Math.min(S.K, N - 1);
  $('#K').max = String(Math.min(20, N - 1));
  $('#K').value = S.K;
  S.rng = mulberry32(12345);
  plot.reset();
  recompute({ newDraw: true });
}

// ---------------------------------------------------------------- inference
function recompute({ newDraw = true } = {}) {
  if (!S.pool) return;
  const N = S.logp.length, K = S.K, T = S.T[S.mode];
  const logScale = columnLogScale(S.mode, S.logp, S.logq, T);
  const { L, c } = kernel(S.G, N, logScale);
  S.L = L;
  S.c = c;
  S.spec = spectrum(L, N);
  S.pi = inclusionProbs(S.spec, K);
  if (S.mode === 'map') {
    S.sel = greedyMAP(L, N, K).set;
  } else if (newDraw || !S.sel.length) {
    S.sel = sampleKDPP(S.spec, K, S.rng) ?? [];
  }
  S.logP = S.sel.length === K ? logDetSub(L, N, S.sel) - logZ(S.spec, K) : NaN;
  render();
}

// ---------------------------------------------------------------- rendering: hero
function render() {
  $('#T-out').textContent = fmtT(S.T[S.mode]);
  plot.update();
  renderEigbars();
  renderReadout();
  renderCards();
  renderVolHint();
  if ($('.pool-table').open) renderTable();
}

const fmtT = (T) => (T < 10 ? T.toFixed(2) : T < 100 ? T.toFixed(1) : T.toFixed(0));

function renderEigbars() {
  const el = $('#eigbars');
  const f = S.proj.frac;
  const max = f[0];
  el.innerHTML = f.map((v, k) => `
    <div class="bar ${k < S.dims ? 'on' : ''}" data-k="${k + 1}" role="button" tabindex="${k < 3 ? 0 : -1}"
      aria-label="Show ${k + 1}D" title="${k < 3 ? `Show ${k + 1}D · ` : ''}${(100 * v).toFixed(1)}%">
      <span>${k + 1}</span><span class="fill" style="width:${(100 * v) / max}%"></span>
    </div>`).join('') +
    `<div class="pct">${(100 * f.slice(0, S.dims).reduce((a, b) => a + b, 0)).toFixed(0)}%<br>shown</div>`;
  el.querySelectorAll('.bar').forEach((b) => {
    const k = +b.dataset.k;
    if (k > 3) { b.style.cursor = 'default'; return; }
    const go = () => { S.dims = k; plot.reset(); render(); };
    b.onclick = go;
    b.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } };
  });
}

function renderReadout() {
  const N = S.logp.length;
  const ok = S.sel.length === S.K;
  const e = ess(logWeights(S.logp, S.logq, S.T.sample));
  $('#readout').innerHTML = !ok
    ? `<div class="stat"><div class="k">No subset</div><div class="s">Fewer than K distinct directions in the pool (kernel rank &lt; K). Lower K or generate more completions.</div></div>`
    : `
    <div class="stat"><div class="k">log P(S)</div><div class="v">${fmt(S.logP)}</div><div class="s">under the pool ${S.K}-DPP</div></div>
    <div class="stat"><div class="k">Quality (mean log p)</div><div class="v">${fmt(quality(S.logp, S.sel))}</div></div>
    <div class="stat"><div class="k">Diversity (log det ΦᵀΦ)</div><div class="v">${fmt(diversity(S.G, N, S.sel))}</div></div>
    ${S.mode === 'sample' ? `<div class="stat"><div class="k">Weight ESS</div><div class="v">${e.toFixed(1)} / ${N}</div><div class="s">w = p^(1/T) / q</div></div>` : ''}`;
}

function renderVolHint() {
  const el = $('#vol-hint');
  if (S.dims === 1) { el.innerHTML = ''; return; }
  el.innerHTML = S.sel.length <= 3 && S.sel.length >= 2
    ? `<svg class="sw-vol" width="18" height="12" viewBox="0 0 18 12" aria-hidden="true"><path d="M1 11L5 1H17L13 11Z"/></svg>volume spanned by the selected set (det L_S = volume²)`
    : `<button id="vol-btn" class="link">Draw the spanned volume (K = 3)</button>`;
  const b = $('#vol-btn');
  if (b) b.onclick = () => { $('#K').value = 3; $('#K').onchange(); };
}

function renderCards() {
  const order = S.sel.slice().sort((a, b) => S.logp[b] - S.logp[a]);
  $('#selected').innerHTML = order.map((i) => {
    const s = S.pool.samples[i];
    return `<div class="card" data-i="${i}" style="border-left-color:${selColor(i)}">${escapeHtml(s.text)}${s.ended ? '' : '…'}
      <div class="meta">#${i} · log p ${fmt(s.logp, 1)} · P(i∈S) ${fmt(S.pi[i], 2)}</div></div>`;
  }).join('');
  $('#selected').querySelectorAll('.card').forEach((el) => {
    el.onmouseenter = () => plot.highlight(+el.dataset.i);
    el.onmouseleave = () => plot.highlight(-1);
  });
}

// ---------------------------------------------------------------- tooltip
const tip = $('#tooltip');
function showTip(i, x, y) {
  if (i < 0) { tip.hidden = true; return; }
  const s = S.pool.samples[i];
  const lw = s.logp / S.T.sample - s.logq;
  tip.innerHTML = `${escapeHtml(s.text)}${s.ended ? '' : '…'}<div class="meta">#${i} · log p ${fmt(s.logp, 2)} · log q ${fmt(s.logq, 2)} · log w ${fmt(lw, 2)}<br>P(i ∈ S) = ${fmt(S.pi[i], 3)}${S.sel.includes(i) ? ' · <b>selected</b>' : ''}</div>`;
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  tip.style.left = Math.min(window.innerWidth - r.width - 8, x + 14) + 'px';
  tip.style.top = Math.min(window.innerHeight - r.height - 8, y + 14) + 'px';
}

// ---------------------------------------------------------------- projection plot
// With K ≤ 3 each selected vector gets its own colour and we draw the
// parallelogram/parallelepiped it spans (paper, Fig. 1); otherwise one accent.
const VEC = ['--series-1', '--series-2', '--series-3'];
function selColor(i) {
  const k = S.sel.indexOf(i);
  return S.sel.length <= 3 && k >= 0 ? css(VEC[k]) : css('--accent');
}
// Faces (as corner lists) and corner points of the parallelotope spanned by vectors V from origin O.
function parallelotope(O, V) {
  const add = (a, b) => a.map((x, d) => x + b[d] - O[d]);
  const corner = (mask) => V.reduce((acc, v, k) => (mask & (1 << k) ? add(acc, v) : acc), O.slice());
  const faces = [];
  for (let a = 0; a < V.length; a++) for (let b = a + 1; b < V.length; b++) {
    const others = [...Array(V.length).keys()].filter((k) => k !== a && k !== b);
    const offs = others.length ? [0, 1 << others[0]] : [0];
    for (const o of offs) faces.push([o, o | (1 << a), o | (1 << a) | (1 << b), o | (1 << b)].map(corner));
  }
  const full = (1 << V.length) - 1;
  const extra = [];
  for (let mask = 1; mask <= full; mask++) if (mask & (mask - 1)) extra.push({ p: corner(mask), far: mask === full });
  return { faces, corners: extra };
}
const colorFor = () => {
  const lp = S.logp.slice().sort((a, b) => a - b);
  const lo = lp[Math.floor(0.05 * (lp.length - 1))], hi = lp[lp.length - 1];
  const sc = d3.scaleLinear().domain([lo, hi]).range([css('--ramp-lo'), css('--ramp-hi')]).interpolate(d3.interpolateLab).clamp(true);
  return (i) => sc(S.logp[i]);
};

class Plot2D {
  constructor(el) { this.el = el; this.svg = null; }
  reset() { this.el.innerHTML = ''; this.svg = null; }
  build() {
    const el = this.el;
    this.W = el.clientWidth || 600;
    this.H = el.clientHeight || 450;
    const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${this.W} ${this.H}`);
    svg.append('defs').append('marker').attr('id', 'arrow').attr('viewBox', '0 0 10 10').attr('refX', 9).attr('refY', 5)
      .attr('markerWidth', 7).attr('markerHeight', 7).attr('orient', 'auto-start-reverse')
      .append('path').attr('d', 'M0,0L10,5L0,10z').attr('fill', css('--accent'));
    this.gOrigin = svg.append('g');
    this.gOrigin.append('path').attr('stroke', css('--muted')).attr('stroke-width', 1);
    this.gOrigin.append('text').attr('class', 'direct-label').attr('font-size', 11).attr('fill', css('--muted')).text('origin');
    this.gPara = svg.append('g');
    this.gTrail = svg.append('g');
    this.gGhost = svg.append('g');
    this.gArrow = svg.append('g');
    this.gDot = svg.append('g');
    this.svg = svg;
    this.jit = S.proj.coords.map((_, i) => ((i * 0.6180339887) % 1) - 0.5);
    svg.on('pointermove', (ev) => {
      const [px, py] = d3.pointer(ev);
      let best = -1, bd = 18 * 18;
      this.pos.forEach(([a, b], i) => { const d = (a - px) ** 2 + (b - py) ** 2; if (d < bd) { bd = d; best = i; } });
      this.highlight(best);
      showTip(best, ev.clientX, ev.clientY);
    }).on('pointerleave', () => { this.highlight(-1); showTip(-1); });
  }
  // Data-space position of item i (scale = 1 for the unit embedding).
  dataXY(i, scale) {
    const c = S.proj.coords[i];
    return S.dims === 1 ? [c[0] * scale, this.jit[i] * 1.6] : [c[0] * scale, c[1] * scale];
  }
  // Fit scales to everything drawn, including the far corner of the spanned volume.
  fit(points) {
    const { W, H } = this, m = 16, dims = S.dims;
    const pad = (a) => { const lo = d3.min(a), hi = d3.max(a), p = 0.06 * (hi - lo || 1); return [lo - p, hi + p]; };
    let [x0, x1] = pad(points.map((q) => q[0]).concat([0]));
    let [y0, y1] = dims === 1 ? [-1, 1] : pad(points.map((q) => q[1]).concat([0]));
    if (dims === 2) {
      const s = Math.min((W - 2 * m) / (x1 - x0), (H - 2 * m) / (y1 - y0));
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      x0 = cx - (W - 2 * m) / s / 2; x1 = cx + (W - 2 * m) / s / 2;
      y0 = cy - (H - 2 * m) / s / 2; y1 = cy + (H - 2 * m) / s / 2;
    }
    this.x = d3.scaleLinear().domain([x0, x1]).range([m, W - m]);
    this.y = d3.scaleLinear().domain([y0, y1]).range([H - m, m]);
  }
  update() {
    if (!this.svg) this.build();
    const N = S.logp.length, dims = S.dims, H = this.H;
    const scaled = $('#scale-toggle').checked;
    const col = colorFor();
    const sel = new Set(S.sel);
    const idx = d3.range(N);
    const ghostD = idx.map((i) => this.dataXY(i, 1));
    const posD = idx.map((i) => (dims === 1 ? [this.dataXY(i, scaled ? S.c[i] : 1)[0], ghostD[i][1]] : this.dataXY(i, scaled ? S.c[i] : 1)));
    const vol = dims === 2 && S.sel.length >= 2 && S.sel.length <= 3;
    const volD = vol ? parallelotope([0, 0], S.sel.map((i) => posD[i])) : { faces: [], corners: [] };
    this.fit([...posD, ...(scaled ? ghostD : []), ...volD.corners.map((c) => c.p)]);
    const px = (q) => [this.x(q[0]), this.y(q[1])];
    const ghostPos = ghostD.map(px);
    this.pos = posD.map(px);
    const ox = this.x(0), oy = this.y(0);
    const t = this.svg.transition().duration(450).ease(d3.easeCubicOut);

    this.gOrigin.select('path').attr('stroke-dasharray', dims === 1 ? '3 3' : null).transition(t)
      .attr('d', dims === 1 ? `M${ox},16V${H - 16}` : `M${ox - 6},${oy}H${ox + 6}M${ox},${oy - 6}V${oy + 6}`);
    this.gOrigin.select('text').transition(t).attr('x', ox + 6).attr('y', dims === 1 ? 26 : oy - 6);
    this.gGhost.selectAll('circle').data(idx).join('circle').attr('r', 3)
      .attr('fill', css('--ghost')).attr('opacity', scaled ? 0.45 : 0)
      .transition(t).attr('cx', (i) => ghostPos[i][0]).attr('cy', (i) => ghostPos[i][1]);
    this.gTrail.selectAll('line').data(idx).join('line')
      .attr('stroke', css('--ghost')).attr('stroke-width', 1).attr('opacity', scaled ? 0.35 : 0)
      .transition(t).attr('x1', (i) => ghostPos[i][0]).attr('y1', (i) => ghostPos[i][1])
      .attr('x2', (i) => this.pos[i][0]).attr('y2', (i) => this.pos[i][1]);
    const few = S.sel.length <= 3;
    const y0 = (i) => (dims === 1 ? this.pos[i][1] : oy);
    this.gArrow.selectAll('line').data(S.sel, (i) => i).join(
      (enter) => enter.append('line').attr('x1', ox).attr('y1', y0).attr('x2', ox).attr('y2', y0),
    ).attr('stroke', (i) => selColor(i)).attr('stroke-width', few ? 2.5 : 2)
      .attr('marker-end', few ? null : 'url(#arrow)')
      .transition(t).attr('x1', ox).attr('y1', y0)
      .attr('x2', (i) => this.pos[i][0]).attr('y2', (i) => this.pos[i][1]);
    // Volume spanned by the selected set (K = 2: parallelogram, K = 3: parallelepiped),
    // from the projected quality-scaled vectors; det L_S is its squared volume.
    const face = (f) => 'M' + f.map((q) => px(q).join(',')).join('L') + 'Z';
    this.gPara.selectAll('path').data(volD.faces).join('path')
      .attr('fill', css('--text')).attr('fill-opacity', 0.04)
      .attr('stroke', css('--text-2')).attr('stroke-opacity', 0.7).attr('stroke-width', 1).attr('stroke-linejoin', 'round')
      .transition(t).attr('d', face);
    this.gPara.selectAll('circle').data(volD.corners).join('circle')
      .attr('fill', css('--text-2')).attr('r', (c) => (c.far ? 4.5 : 2.5))
      .transition(t).attr('cx', (c) => this.x(c.p[0])).attr('cy', (c) => this.y(c.p[1]));
    this.gDot.selectAll('circle').data(idx).join('circle')
      .attr('fill', (i) => col(i))
      .attr('stroke', (i) => (sel.has(i) ? selColor(i) : css('--surface')))
      .attr('stroke-width', (i) => (sel.has(i) ? 2.5 : 1))
      .attr('r', (i) => (sel.has(i) ? 6 : 4.5))
      .sort((a, b) => sel.has(a) - sel.has(b))
      .transition(t).attr('cx', (i) => this.pos[i][0]).attr('cy', (i) => this.pos[i][1]);
  }
  highlight(i) {
    if (!this.svg) return;
    S.hover = i;
    this.gDot.selectAll('circle').attr('stroke', (j) => (j === i ? css('--text') : S.sel.includes(j) ? selColor(j) : css('--surface')));
    document.querySelectorAll('.card').forEach((c) => c.classList.toggle('hl', +c.dataset.i === i));
  }
}

const plot = {
  p2: new Plot2D($('#plot')),
  p3: null,
  reset() {
    this.p2.reset();
    this.p3?.dispose();
    this.p3 = null;
  },
  update() {
    if (S.dims === 3) {
      if (!this.p3) {
        $('#plot').innerHTML = '';
        this.p3 = new Plot3D($('#plot'), { onHover: (i, x, y) => { showTip(i, x, y); this.highlightCards(i); } });
      }
      this.p3.update({
        coords: S.proj.coords, c: S.c, sel: S.sel, scaled: $('#scale-toggle').checked,
        color: colorFor(), selColor, accent: css('--accent'), ghost: css('--ghost'), text: css('--text-2'), bg: css('--surface'),
      });
    } else this.p2.update();
  },
  highlight(i) {
    if (S.dims === 3) { this.p3?.highlight(i); this.highlightCards(i); } else this.p2.highlight(i);
  },
  highlightCards(i) { document.querySelectorAll('.card').forEach((c) => c.classList.toggle('hl', +c.dataset.i === i)); },
};

// ---------------------------------------------------------------- pool table
let sortKey = 'logp', sortDir = -1;
function renderTable() {
  const T = S.T.sample; // log w is PoolDPP-Sampling's importance weight
  const sel = new Set(S.sel);
  const rows = S.pool.samples.map((s, i) => ({ i, text: s.text + (s.ended ? '' : '…'), logp: s.logp, logq: s.logq, logw: s.logp / T - s.logq, pi: S.pi[i] }));
  rows.sort((a, b) => sortDir * (typeof a[sortKey] === 'string' ? a[sortKey].localeCompare(b[sortKey]) : a[sortKey] - b[sortKey]));
  const cols = [['i', '#'], ['text', 'string'], ['logp', 'log p'], ['logq', 'log q'], ['logw', 'log w'], ['pi', 'P(i∈S)']];
  $('#pool-table').innerHTML = `<div class="wrap"><table><thead><tr>${cols.map(([k, n]) => `<th data-k="${k}">${n}${k === sortKey ? (sortDir > 0 ? ' ▲' : ' ▼') : ''}</th>`).join('')}</tr></thead><tbody>${rows.map((r) => `<tr class="${sel.has(r.i) ? 'sel' : ''}"><td class="num">${r.i}</td><td>${escapeHtml(r.text)}</td><td class="num">${fmt(r.logp, 2)}</td><td class="num">${fmt(r.logq, 2)}</td><td class="num">${fmt(r.logw, 2)}</td><td class="num">${fmt(r.pi, 3)}</td></tr>`).join('')}</tbody></table></div>`;
  $('#pool-table').querySelectorAll('th').forEach((th) => (th.onclick = () => {
    const k = th.dataset.k;
    if (k === sortKey) sortDir = -sortDir; else { sortKey = k; sortDir = k === 'text' || k === 'i' ? 1 : -1; }
    renderTable();
  }));
}

// ---------------------------------------------------------------- live mode
let worker = null;
function runLive() {
  worker ??= new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  const N = clampInt($('#live-N').value, 2, 200), Tp = Math.min(2, Math.max(0.2, +$('#live-Tp').value || 1));
  const maxNewTokens = clampInt($('#live-max').value, 8, 256);
  const prompt = $('#prompt').value.trim();
  if (!prompt) return;
  $('#live-go').disabled = true;
  $('#live-stop').hidden = false;
  worker.onmessage = ({ data }) => {
    if (data.type === 'status' || data.type === 'progress') $('#live-status').textContent = data.text;
    if (data.type === 'error') { $('#live-status').textContent = 'Error: ' + data.text; done(); }
    if (data.type === 'done') {
      const D = data.pool.D;
      const phi = Array.from({ length: data.pool.N }, (_, i) => data.emb.subarray(i * D, (i + 1) * D));
      $('#live-status').textContent = `Drew ${data.pool.N} strings on ${data.device}. Drag T, or draw again.`;
      $('#gallery-select').value = '';
      setPool(data.pool, phi);
      done();
    }
  };
  const done = () => { $('#live-go').disabled = false; $('#live-stop').hidden = true; };
  worker.postMessage({ type: 'run', prompt, N, Tprime: Tp, maxNewTokens });
}
const clampInt = (v, lo, hi) => Math.min(hi, Math.max(lo, Math.round(+v || lo)));

// ---------------------------------------------------------------- utils & wiring
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
function setMode(m) {
  S.mode = m;
  $('#T-slider').value = tToSlider(S.T[m]);
  recompute({ newDraw: true });
}

function wire() {
  $('#T-slider').value = tToSlider(S.T[S.mode]);
  let raf = 0;
  $('#T-slider').addEventListener('input', (e) => {
    S.T[S.mode] = sliderToT(+e.target.value);
    $('#T-out').textContent = fmtT(S.T[S.mode]);
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => recompute({ newDraw: true }));
  });
  document.querySelectorAll('input[name=mode]').forEach((r) => (r.onchange = () => setMode(r.value)));
  $('#K').onchange = () => { S.K = clampInt($('#K').value, 1, Math.min(20, S.logp.length - 1)); $('#K').value = S.K; recompute({ newDraw: true }); };
  $('#resample').onclick = () => recompute({ newDraw: true });
  $('#scale-toggle').onchange = () => plot.update();
  $('.pool-table').addEventListener('toggle', () => { if ($('.pool-table').open) renderTable(); });
  $('#live-go').onclick = runLive;
  $('#live-stop').onclick = () => worker?.postMessage({ type: 'stop' });
  let rt = 0;
  window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { plot.reset(); render(); }, 200); });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { plot.reset(); render(); });
}
wire();
loadGallery().catch((e) => { $('#readout').textContent = 'Could not load gallery: ' + e.message; });
