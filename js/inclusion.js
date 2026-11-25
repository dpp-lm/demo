// "Color by inclusion probability": color the points of the plot and the dots on the cards by the probability that the
// completion is in a subset drawn from the K-DPP at the current T and K (S.pi, the value the cards print as "inclusion").
// A view option in the View menu for every visitor, off by default and not stored. It uses the same color hooks as Groups
// (expert.js pointColors), so the 2-D plot, the 3-D plot and the card dots follow, and the two colorings exclude each other.
// The scale (square root of p / top, top rounded up from the largest value) lives in expert-lib.js.

import { EXPLAIN } from './explain.js';
import { inclusionTop, inclusionRamp, inclusionShade, inclusionTicks } from './expert-lib.js';
import { rampBetween, parseHex } from './matrices.js';

let on = false, H = null, legend = null, box = null, lastSig = '';
let cache = null;   // { pool, pi, n, undef, top, colors } for the pool and probabilities on screen
const turnedOn = [];

const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const hex = (c) => `#${c.map((x) => x.toString(16).padStart(2, '0')).join('')}`;
const ramp = () => rampBetween(parseHex(css('--ramp-lo')) ?? [195, 189, 179], parseHex(css('--ramp-hi')) ?? [42, 42, 42]);

export const inclusionOn = () => on;
/** fn() runs when the coloring is turned on (Groups turns itself off there). */
export const onInclusionOn = (fn) => turnedOn.push(fn);

/** The colors for the pool on screen, { undef } when no K-DPP exists, or null when the mode is off or there is no pool. */
function state() {
  const info = on ? H?.inclusion() : null;
  if (!info) return null;
  const n = info.pool.samples.length;
  if (cache && cache.pool === info.pool && cache.pi === info.pi && cache.n === n) return cache;
  if (!info.pi) return (cache = { pool: info.pool, pi: info.pi, n, undef: true });
  const values = info.values(), top = inclusionTop(values), rgb = ramp();
  const colors = values.map((p) => { const t = inclusionRamp(p, top); return t === null ? null : hex(rgb(inclusionShade(t))); });
  return (cache = { pool: info.pool, pi: info.pi, n, top, colors });
}

/** For main.js through expert.js: the point and card-dot colors, or null (default colors) when off or when there is no K-DPP. */
export function inclusionColors() {
  const s = state();
  if (!s || s.undef) return null;
  const f = (i) => s.colors[i] ?? css('--ink');
  return { color: f, selColor: f, dot: f };
}

function renderLegend() {
  if (!legend) return;
  const s = state();
  if (!s) { legend.hidden = true; legend.innerHTML = ''; return; }
  legend.hidden = false;
  if (s.undef) { legend.innerHTML = '<span class="x-il-title">Inclusion probability</span><span class="x-chip-note">undefined for this K</span>'; return; }
  const rgb = ramp(), lo = rgb(inclusionShade(0)), hi = rgb(inclusionShade(1));
  legend.innerHTML = `<span class="x-il-title">Inclusion probability</span>
    <span class="x-il-bar"><span class="x-il-grad" style="background:linear-gradient(90deg,${hex(lo)},${hex(hi)})"></span><span class="x-il-ticks">${inclusionTicks(s.top).map((k) => `<span style="left:${100 * k.t}%">${k.label}</span>`).join('')}</span></span>
    <span class="x-chip-note">square-root scale, selected points are outlined</span>`;
}

/** After every render: redraw the legend when what it shows has changed. */
export function inclusionRefresh() {
  if (!legend) return;
  const s = state();
  const sig = !s ? '' : s.undef ? 'undef' : `${s.top}`;
  if (sig === lastSig && legend.hidden === !s) return;
  lastSig = sig;
  renderLegend();
}

/** Turn the coloring on or off, keep the menu's checkbox true to it, and redraw the plot, the cards and the legend. */
export function setInclusion(v) {
  v = !!v;
  const was = on;
  on = v;
  const cb = document.getElementById('incl-toggle');
  if (cb) cb.checked = v;
  if (v && !was) turnedOn.forEach((f) => f());
  lastSig = '';
  H?.redraw();
  inclusionRefresh();
}

export function initInclusion(hooks) {
  H = hooks;
  EXPLAIN.inclcolor = {
    title: 'Color by inclusion probability',
    body: `<p>The inclusion probability of a completion is the chance that a subset drawn from the K-DPP over the completions, at the current $T$ and $K$, contains it. The color of a point is that probability, and the selected points keep their marker.</p>
      <p>In Most likely mode the subset is the greedy pick, while the color still shows the K-DPP's inclusion probability.</p>
      <p>The color follows the square root of the probability, so that small probabilities stay apart. The legend gives the top of the scale.</p>`,
  };
  legend = document.createElement('div');
  legend.id = 'incl-legend';
  legend.className = 'x-legend x-incl';
  legend.hidden = true;
  legend.setAttribute('aria-label', 'Inclusion probability');
  document.querySelector('.stage').append(legend);
  const cb = document.getElementById('incl-toggle');
  cb.checked = false;
  cb.addEventListener('change', () => setInclusion(cb.checked));
}
