import * as d3 from 'd3';
import {
  gram, symEig, kernel, spectrum, sampleKDPP, greedyMAP, logZ, logDetSub, inclusionProbs,
  columnLogScale, logWeights, ess, quality, diversity, mulberry32,
} from './dpp.js';
import { Plot3D } from './plot3d.js';
import { initExplainers } from './explain.js';
import { MODELS, modelByKey } from './lm.js';

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
  mode: 'map',
  T: { sample: 1, map: 20 },
  K: 3,
  pinned: -1,      // point shown in the inspector
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

const EXAMPLE_LABELS = { sort: 'Sort a list', story: 'Story opening', bike: 'Old bicycle', bank: 'What is a bank?', language: 'Learn a language' };

async function loadGallery() {
  const idx = await (await fetch('data/index.json')).json();
  const el = $('#examples');
  el.innerHTML = '<div class="menu-head">Example prompts</div>' + idx.map((g) => `<button type="button" role="menuitem" data-id="${g.id}"><b>${escapeHtml(EXAMPLE_LABELS[g.id] ?? g.id)}</b><span>${escapeHtml(g.prompt)}</span></button>`).join('');
  el.querySelectorAll('button').forEach((b) => (b.onclick = () => { $('#exmenu').open = false; loadGalleryItem(b.dataset.id); }));
  await loadGalleryItem(idx.find((g) => g.id === 'story')?.id ?? idx[0].id);
}

function markExample(id) {
  document.querySelectorAll('#examples button').forEach((b) => b.toggleAttribute('aria-current', b.dataset.id === id));
}

async function loadGalleryItem(id) {
  const rec = await (await fetch(`data/${id}.json`)).json();
  $('#prompt').value = rec.prompt;
  $('#prompt').dispatchEvent(new Event('input'));
  markExample(id);
  $('#progress').hidden = true;
  $('#live-status').textContent = `Example: ${rec.N} completions from ${modelByKey(MODELS.find((m) => m.repo === rec.model)?.key).name}. Edit the prompt to try your own.`;
  setPool(rec, decodeEmb(rec.emb, rec.N, rec.D));
}

function setPool(pool, phi) {
  $('#labels').innerHTML = '';
  S.pinned = -1;
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
  document.querySelectorAll('input[name=dims]').forEach((r) => (r.checked = +r.value === S.dims));
  const distinct = new Set(S.pool.samples.map((x) => x.text)).size;
  const N = S.pool.samples.length;
  $('#stage-note').textContent = `${N} completions${distinct < N ? ` (${distinct} distinct)` : ''} · ${S.dims === 3 ? 'drag to rotate · tap a point to read it · change dimensions with the bars at right' : 'tap a point to read it · change dimensions with the bars at right'}`;
  $('#subset-meta').textContent = `K = ${S.sel.length} · ${S.mode === 'map' ? 'greedy MAP' : 'K-DPP sample'} at T = ${fmtT(S.T[S.mode])}`;
  $('#T-out').textContent = fmtT(S.T[S.mode]);
  plot.update();
  renderEigbars();
  renderReadout();
  renderCards();
  renderVolHint();
  renderInspect();
  renderLabels();
  if ($('.pool-table').open) renderTable();
}

const fmtT = (T) => (T < 10 ? T.toFixed(2) : T < 100 ? T.toFixed(1) : T.toFixed(0));

function renderEigbars() {
  const el = $('#eigbars');
  const f = S.proj.frac;
  const max = f[0];
  el.innerHTML = `<div class="cap">PCA <button type="button" class="xbtn" data-x="pca" aria-label="About the projection">?</button></div>` + f.map((v, k) => {
    const click = k < 3;
    return `<div class="bar ${k < S.dims ? 'on' : ''}" data-k="${k + 1}" ${click ? `role="button" tabindex="0" aria-label="Show ${k + 1}D"` : ''}
      title="Axis ${k + 1}: ${(100 * v).toFixed(1)}% of variance${click ? ` · click for ${k + 1}D` : ''}">
      <span>${k + 1}</span><span class="fill" style="width:${(100 * v) / max}%"></span></div>`;
  }).join('') + `<div class="pct">${(100 * f.slice(0, S.dims).reduce((a, b) => a + b, 0)).toFixed(0)}% shown</div>`;
  el.querySelectorAll('.bar[role=button]').forEach((b) => {
    const go = () => setDims(+b.dataset.k);
    b.onclick = go;
    b.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } };
  });
}

function setDims(k) { S.dims = k; plot.reset(); render(); }

function renderReadout() {
  const N = S.logp.length;
  const ok = S.sel.length === S.K;
  const e = ess(logWeights(S.logp, S.logq, S.T.sample));
  const stat = (k, v, x) => `<span class="stat">${k} <button type="button" class="xbtn" data-x="${x}" aria-label="What is ${k}?">?</button> <span class="v">${v}</span></span>`;
  $('#readout').innerHTML = !ok
    ? `<span class="stat">No subset of size ${S.K}: there are fewer distinct completions than that. Lower K or generate more.</span>`
    : stat('log P(S)', fmt(S.logP), 'logp') +
      stat('quality', fmt(quality(S.logp, S.sel)), 'quality') +
      stat('diversity', fmt(Math.exp(diversity(S.G, N, S.sel) / S.sel.length), 2), 'diversity') +
      (S.mode === 'sample' ? stat('ESS', `${e.toFixed(0)}/${N}`, 'ess') : '');
}

function renderVolHint() {
  const el = $('#vol-hint');
  if (S.dims === 1) { el.innerHTML = ''; return; }
  el.innerHTML = S.sel.length <= 3 && S.sel.length >= 2
    ? `<svg class="sw-vol" width="18" height="12" viewBox="0 0 18 12" aria-hidden="true"><path d="M1 11L5 1H17L13 11Z"/></svg>spanned volume <button type="button" class="xbtn" data-x="volume" aria-label="What is the spanned volume?">?</button>`
    : `<button id="vol-btn" class="link">Show the spanned volume (K = 3)</button>`;
  const b = $('#vol-btn');
  if (b) b.onclick = () => setK(3);
}

function renderCards() {
  const order = S.sel.slice().sort((a, b) => S.logp[b] - S.logp[a]);
  $('#selected').innerHTML = order.map((i) => {
    const s = S.pool.samples[i];
    return `<li class="card${s.ended ? '' : ' cut'}" data-i="${i}">
      <div class="txt">${escapeHtml(s.text)}</div>
      <div class="meta"><span class="dot" style="background:${selColor(i)}"></span>log p ${fmt(s.logp, 1)}<span>·</span>P(in set) ${fmt(S.pi[i], 2)}</div></li>`;
  }).join('');
  $('#selected').querySelectorAll('.card').forEach((el) => {
    el.onmouseenter = () => plot.highlight(+el.dataset.i);
    el.onmouseleave = () => plot.highlight(-1);
    el.onclick = () => inspect(+el.dataset.i);
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
    const nearest = (ev, r) => {
      const [px, py] = d3.pointer(ev);
      let best = -1, bd = r * r;
      this.pos.forEach(([a, b], i) => { const d = (a - px) ** 2 + (b - py) ** 2; if (d < bd) { bd = d; best = i; } });
      return best;
    };
    svg.on('pointermove', (ev) => {
      if (ev.pointerType !== 'mouse') return;
      const best = nearest(ev, 18);
      this.highlight(best);
      showTip(best, ev.clientX, ev.clientY);
    }).on('pointerleave', () => { this.highlight(-1); showTip(-1); })
      .on('click', (ev) => inspect(nearest(ev, ev.pointerType === 'mouse' ? 18 : 32)));
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
    if (i < 0) i = S.pinned;
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
        this.p3 = new Plot3D($('#plot'), {
          onHover: (i, x, y) => { showTip(i, x, y); this.highlightCards(i < 0 ? S.pinned : i); },
          onTap: (i) => inspect(i),
        });
      }
      this.p3.update({
        coords: S.proj.coords, c: S.c, sel: S.sel, scaled: $('#scale-toggle').checked,
        color: colorFor(), selColor, accent: css('--accent'), ghost: css('--ghost'), text: css('--text-2'), bg: css('--surface'),
      });
    } else this.p2.update();
  },
  highlight(i) {
    if (i < 0) i = S.pinned;
    if (S.dims === 3) { this.p3?.highlight(i); this.highlightCards(i); } else this.p2.highlight(i);
  },
  highlightCards(i) { document.querySelectorAll('.card').forEach((c) => c.classList.toggle('hl', +c.dataset.i === i)); },
};

// ---------------------------------------------------------------- labels
// Small callouts with the opening words of each selected completion. Shown for
// K ≤ 6 so they never swamp the plot; in 3D they follow the points every frame.
const MAX_LABELS = 6;
function labelText(s) { const t = s.text.replace(/\s+/g, ' ').trim(); return t.length > 90 ? t.slice(0, 88) + '…' : t; }
function renderLabels() {
  const layer = $('#labels');
  const show = $('#label-toggle').checked && S.sel.length && S.sel.length <= MAX_LABELS;
  if (!show) { layer.innerHTML = ''; return; }
  const want = new Set(S.sel.map(String));
  layer.querySelectorAll('.lbl').forEach((el) => { if (!want.has(el.dataset.i)) el.remove(); });
  for (const i of S.sel) {
    if (layer.querySelector(`.lbl[data-i="${i}"]`)) continue;
    const el = document.createElement('div');
    el.className = 'lbl';
    el.dataset.i = i;
    el.appendChild(document.createElement('span')).textContent = labelText(S.pool.samples[i]);
    el.onclick = () => inspect(i);
    el.onmouseenter = () => plot.highlight(i);
    el.onmouseleave = () => plot.highlight(-1);
    layer.appendChild(el);
  }
  layer.classList.toggle('anim', S.dims !== 3);
  placeLabels();
}
function placeLabels() {
  const layer = $('#labels');
  const els = [...layer.querySelectorAll('.lbl')];
  if (!els.length) return;
  const W = layer.clientWidth, H = layer.clientHeight;
  const items = els.map((el) => {
    const i = +el.dataset.i;
    const p = S.dims === 3 ? plot.p3?.screenXY(i) : plot.p2.pos?.[i];
    return p && { el, x: p[0], y: p[1], w: el.offsetWidth, h: el.offsetHeight };
  }).filter(Boolean).sort((a, b) => a.y - b.y);
  // Right of the point (left if it would leave the plot), then nudge down to avoid overlaps.
  const placed = [];
  for (const it of items) {
    let left = it.x + 10 + it.w > W - 4 ? it.x - 10 - it.w : it.x + 10;
    left = Math.max(4, left);
    let top = Math.min(Math.max(4, it.y - it.h / 2), H - it.h - 4);
    for (const q of placed) {
      const overlapX = left < q.left + q.w && q.left < left + it.w;
      if (overlapX && top < q.top + q.h + 3 && q.top < top + it.h) top = q.top + q.h + 3;
    }
    placed.push({ left, top, w: it.w, h: it.h });
    it.el.style.transform = `translate(${Math.round(left)}px, ${Math.round(top)}px)`;
  }
}
(function follow3D() {
  if (S.dims === 3 && $('#labels').childElementCount) placeLabels();
  requestAnimationFrame(follow3D);
})();

// ---------------------------------------------------------------- inspector
// Tap or click a point (or a card) to pin its text and numbers under the plot.
function inspect(i) {
  S.pinned = i;
  showTip(-1);
  renderInspect();
  plot.highlight(i);
}
function renderInspect() {
  const el = $('#inspect');
  const i = S.pinned;
  if (i < 0 || !S.pool || i >= S.pool.samples.length) { el.hidden = true; return; }
  const s = S.pool.samples[i];
  const lw = s.logp / S.T.sample - s.logq;
  const inSel = S.sel.includes(i);
  el.hidden = false;
  el.innerHTML = `<div class="ins-txt">${escapeHtml(s.text)}${s.ended ? '' : ' …'}</div>
    <div class="ins-meta">${inSel ? `<span class="pill" style="background:${selColor(i)}">selected</span>` : ''}
      <span>log p ${fmt(s.logp, 2)}</span><span>log q ${fmt(s.logq, 2)}</span><span>log w ${fmt(lw, 2)}</span><span>P(in set) ${fmt(S.pi[i], 3)}</span></div>
    <button class="ins-close" aria-label="Close">×</button>`;
  el.querySelector('.ins-close').onclick = () => inspect(-1);
}

// ---------------------------------------------------------------- pool table
let sortKey = 'logp', sortDir = -1;
function renderTable() {
  const T = S.T.sample; // log w is the sampling mode's importance weight
  const sel = new Set(S.sel);
  const rows = S.pool.samples.map((s, i) => ({ i, text: s.text + (s.ended ? '' : '…'), logp: s.logp, logq: s.logq, logw: s.logp / T - s.logq, pi: S.pi[i] }));
  rows.sort((a, b) => sortDir * (typeof a[sortKey] === 'string' ? a[sortKey].localeCompare(b[sortKey]) : a[sortKey] - b[sortKey]));
  const cols = [['i', '#'], ['text', 'string'], ['logp', 'log p'], ['logq', 'log q'], ['logw', 'log w'], ['pi', 'P(i∈S)']];
  $('#pool-table').innerHTML = `<div class="wrap"><table><thead><tr>${cols.map(([k, n]) => `<th data-k="${k}">${n}${k === sortKey ? (sortDir > 0 ? ' ▲' : ' ▼') : ''}</th>`).join('')}</tr></thead><tbody>${rows.map((r) => `<tr class="${sel.has(r.i) ? 'sel' : ''}"><td class="num" data-k="#">${r.i}</td><td class="str">${escapeHtml(r.text)}</td><td class="num" data-k="log p">${fmt(r.logp, 2)}</td><td class="num" data-k="log q">${fmt(r.logq, 2)}</td><td class="num" data-k="log w">${fmt(r.logw, 2)}</td><td class="num" data-k="P(in set)">${fmt(r.pi, 3)}</td></tr>`).join('')}</tbody></table></div>`;
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
  $('#live-go').hidden = true;
  $('#live-stop').hidden = false;
  $('#progress').hidden = false;
  $('#progress-fill').style.width = '0%';
  $('#progress-last').textContent = '';
  worker.onmessage = ({ data }) => {
    if (data.type === 'status' || data.type === 'progress') $('#live-status').textContent = data.text;
    // Fill the bar during downloads too ("Downloading X… a / b MB").
    const dl = data.type === 'status' && /(\d+) \/ (\d+) MB/.exec(data.text);
    if (dl) { $('#progress').hidden = false; $('#progress-fill').style.width = `${(100 * +dl[1]) / Math.max(1, +dl[2])}%`; $('#progress-last').textContent = ''; }
    if (data.type === 'status' && /^Sampling 0\//.test(data.text)) $('#progress-fill').style.width = '0%';
    if (data.type === 'progress') {
      $('#progress').hidden = false;
      $('#progress-fill').style.width = `${(100 * data.n) / data.N}%`;
      if (data.last?.length) $('#progress-last').textContent = '“' + data.last[data.last.length - 1] + '”';
    }
    if (data.type === 'error') { $('#live-status').textContent = 'Error: ' + data.text; done(); }
    if (data.type === 'done') {
      const D = data.pool.D;
      const phi = Array.from({ length: data.pool.N }, (_, i) => data.emb.subarray(i * D, (i + 1) * D));
      $('#live-status').textContent = `${data.pool.N} completions from ${data.pool.modelName}, on your ${data.device === 'webgpu' ? 'GPU' : 'CPU'}.`;
      $('#progress').hidden = true;
      markExample(null);
      setPool(data.pool, phi);
      done();
    }
  };
  const done = () => { $('#live-go').hidden = false; $('#live-stop').hidden = true; };
  worker.postMessage({ type: 'run', prompt, N, Tprime: Tp, maxNewTokens, model: $('#model').value });
}
const clampInt = (v, lo, hi) => Math.min(hi, Math.max(lo, Math.round(+v || lo)));

// ---------------------------------------------------------------- utils & wiring
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
function setMode(m) {
  S.mode = m;
  $('#resample').disabled = m === 'map';
  $('#T-slider').value = tToSlider(S.T[m]);
  recompute({ newDraw: true });
}

function setK(v) {
  S.K = clampInt(v, 1, Math.min(20, S.logp.length - 1));
  $('#K').value = S.K;
  recompute({ newDraw: true });
}

function wire() {
  initExplainers();
  // Collapsed sampling settings: the summary line shows the current values.
  // Model picker: default to Gemma 3 1B with WebGPU, the small Qwen without; remember the choice.
  // Models marked cpu: null need a GPU with float16 support (WebGPU 'shader-f16').
  let hasGPU = false;
  const fillModels = () => {
    $('#model').innerHTML = MODELS.map((m) => `<option value="${m.key}" ${!hasGPU && !m.cpu ? 'disabled' : ''}>${m.name} · ${(m.gpu / 1000).toFixed(1)} GB${m.cpu ? '' : ' · GPU only'}</option>`).join('');
    let saved = null;
    try { saved = localStorage.getItem('dpp-lm-model'); } catch {}
    const usable = (k) => MODELS.some((m) => m.key === k && (hasGPU || m.cpu));
    $('#model').value = usable(saved) ? saved : hasGPU ? 'qwen3-1.7b' : 'qwen2.5-0.5b';
    advSum(); modelNote();
  };
  const modelNote = () => {
    const m = modelByKey($('#model').value);
    $('#model-note').textContent = `Runs on your ${hasGPU ? 'GPU' : 'CPU'}. First use downloads about ${(m.gpu / 1000).toFixed(1)} GB; it is cached after that.`;
  };
  const advSum = () => ($('#adv-sum').textContent = `${modelByKey($('#model').value).short} · ${$('#live-N').value} samples`);
  $('#model').addEventListener('change', () => { try { localStorage.setItem('dpp-lm-model', $('#model').value); } catch {} advSum(); modelNote(); });
  fillModels();
  navigator.gpu?.requestAdapter().then((a) => { hasGPU = !!a?.features.has('shader-f16'); fillModels(); }).catch(() => {});
  ['#live-N', '#live-Tp', '#live-max'].forEach((id) => $(id).addEventListener('input', advSum));
  advSum();
  $('#resample').disabled = S.mode === 'map';
  $('#T-slider').value = tToSlider(S.T[S.mode]);
  let raf = 0;
  $('#T-slider').addEventListener('input', (e) => {
    S.T[S.mode] = sliderToT(+e.target.value);
    $('#T-out').textContent = fmtT(S.T[S.mode]);
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => recompute({ newDraw: true }));
  });
  document.querySelectorAll('input[name=mode]').forEach((r) => (r.onchange = () => setMode(r.value)));
  $('#K').onchange = () => setK($('#K').value);
  $('#K-minus').onclick = () => setK(S.K - 1);
  $('#K-plus').onclick = () => setK(S.K + 1);
  document.querySelectorAll('input[name=dims]').forEach((r) => (r.onchange = () => setDims(+r.value)));
  $('#resample').onclick = () => recompute({ newDraw: true });
  $('#scale-toggle').onchange = () => { plot.update(); renderLabels(); };
  $('#label-toggle').onchange = () => renderLabels();
  // Close the View menu on an outside click or Escape.
  document.addEventListener('click', (e) => { const m = $('#viewmenu'); if (m.open && !m.contains(e.target)) m.open = false; });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') $('#viewmenu').open = false; });
  $('.pool-table').addEventListener('toggle', () => { if ($('.pool-table').open) renderTable(); });
  $('#live-go').onclick = runLive;
  // Composer: Enter generates, Shift+Enter adds a line; the send button needs a prompt.
  const syncGo = () => ($('#live-go').disabled = !$('#prompt').value.trim());
  $('#prompt').addEventListener('input', syncGo);
  $('#prompt').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); if (!$('#live-go').hidden && !$('#live-go').disabled) runLive(); }
  });
  syncGo();
  // Close the settings menu on an outside click.
  document.addEventListener('click', (e) => { for (const m of [$('#adv'), $('#exmenu')]) if (m.open && !m.contains(e.target)) m.open = false; });
  // Only one composer menu open at a time.
  for (const [a, b] of [['#adv', '#exmenu'], ['#exmenu', '#adv']]) $(a).addEventListener('toggle', () => { if ($(a).open) $(b).open = false; });
  $('#live-stop').onclick = () => worker?.postMessage({ type: 'stop' });
  let rt = 0;
  window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { plot.reset(); render(); }, 200); });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { plot.reset(); render(); });
}
wire();
loadGallery().catch((e) => { $('#readout').textContent = 'Could not load gallery: ' + e.message; });
