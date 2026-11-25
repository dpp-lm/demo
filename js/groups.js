// Expert tool "Groups": color the completions by cluster, name the groups, and show a legend over the plot.
// A coloring aid only: the DPP never sees the groups. Everything is computed from the similarity matrix G alone
// (k-means gets vectors from G's eigendecomposition), so it works for every backend.
// expert.js owns the menu and calls initGroups once; main.js reads the colors through pointColors() (expert.js).

import { EXPLAIN } from './explain.js';
import { GROUP_METHODS, groupLabels, groupNames, groupCounts, groupSlot, gramToVectors, decodeDescribe, newRequestId, treeOf } from './expert-lib.js';
import { groupTexts } from './matrices.js';
import { usageBody } from './usage.js';

const DEFAULTS = { on: false, method: 'tree', k: 6, naming: 'num', llm: false };
const NAMINGS = [{ id: 'num', label: 'Numbers' }, { id: 'kw', label: 'Keywords' }, { id: 'medoid', label: 'Typical member' }];
const K_MIN = 2, K_MAX = 8;
let st = { ...DEFAULTS };
let ctx = null;   // { H, isExpert, rerender }
let legend = null, palette = null, descTimer = 0, lastSig = '';

const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

// ---------------------------------------------------------------- groups of the pool on screen
const CACHE = new WeakMap();   // pool -> { vectors, results: Map(method|k -> group result), names: Map(key -> string[]) }
const cacheOf = (pool) => CACHE.get(pool) ?? (CACHE.set(pool, { vectors: null, results: new Map() }), CACHE.get(pool));

/** The grouping of the pool on screen under the current method and count, or null when grouping is off or there is no pool. */
export function currentGroups() {
  const H = ctx?.H, pool = H?.pool();
  if (!ctx?.isExpert() || !st.on || !pool) return null;
  const c = cacheOf(pool), key = `${st.method}|${st.k}`;
  let g = c.results.get(key);
  if (!g) {
    const G = H.G(), N = pool.samples.length, distinct = new Set(pool.samples.map((x) => x.text)).size;
    const opts = { method: st.method, k: st.k, distinct };
    if (st.method === 'tree') opts.tree = treeOf(G, N);
    if (st.method === 'kmeans') opts.vectors = c.vectors ??= gramToVectors(G, N);
    const r = groupLabels(G, N, opts);
    g = { ...r, method: st.method, counts: groupCounts(r.label, r.k), baseNames: new Map() };
    c.results.set(key, g);
  }
  return g;
}

/** The names shown for the groups: the language model's when they have arrived, else the chosen naming (numbers by default). */
function namesOf(g) {
  const pool = ctx.H.pool(), N = pool.samples.length;
  if (llmOn()) {
    const v = NAMES.get(pool)?.get(nameKey(g));
    if (Array.isArray(v)) return v.map((x, c) => x || `Group ${c + 1}`);
    if (v === 'failed') return Array.from({ length: g.k }, (_, c) => `Group ${c + 1}`);
  }
  if (!g.baseNames.has(st.naming)) {
    g.baseNames.set(st.naming, groupNames(st.naming, { label: g.label, k: g.k, texts: pool.samples.map((x) => x.text), logp: pool.samples.map((x) => x.logp), G: ctx.H.G(), N }));
  }
  return g.baseNames.get(st.naming);
}

const slotColor = (c) => {
  palette ??= Array.from({ length: 8 }, (_, k) => css(`--c${k + 1}`));
  return c < 0 ? css('--text-2') : palette[groupSlot(c) - 1];
};

/** For main.js through expert.js: group colors of the points and card dots, or null (his colors) when grouping is off. */
export function groupColors() {
  const g = currentGroups();
  if (!g) return null;
  const f = (i) => slotColor(g.label[i]);
  return { color: f, selColor: f, dot: f };
}

/** Group id of each completion and the group list of the current grouping, for the similarity matrices (null when off). */
export function groupInfo() {
  const g = currentGroups();
  return g ? { label: g.label, k: g.k, method: g.method, color: slotColor } : null;
}

// ---------------------------------------------------------------- names from the language model
const NAMES = new WeakMap();   // pool -> Map(key -> string[] by group id | 'pending' | 'failed')
const describeInfo = () => {
  const B = ctx?.H.backends[0], b = B?.statusBody;
  return b && b.describe === true && B.up !== false ? { B, groups: Math.min(8, b.maxDescribeGroups ?? 8), texts: Math.min(4, b.maxDescribeTexts ?? 4) } : null;
};
const llmOn = () => ctx.isExpert() && st.on && st.llm && !!describeInfo();
const nameKey = (g) => [ctx.H.embedder?.() ?? '', g.method, g.k, Array.from(g.label).join(',')].join('|');

function scheduleDescribe() {
  clearTimeout(descTimer);
  if (llmOn()) descTimer = setTimeout(describeGroups, 400);
}

async function describeGroups() {
  const g = currentGroups(), info = describeInfo();
  if (!g || !info || !llmOn()) return;
  const H = ctx.H, pool = H.pool(), key = nameKey(g), N = pool.samples.length;
  const cache = NAMES.get(pool) ?? NAMES.set(pool, new Map()).get(pool);
  if (cache.has(key)) { renderLegend(); return; }
  const { ids, groups } = groupTexts(H.G(), N, g.label, g.k, pool.samples.map((x) => x.text), { perGroup: info.texts });
  if (ids.length < 2 || ids.length > info.groups) return;   // the server names 2 to maxDescribeGroups groups
  cache.set(key, 'pending');
  let note = '';
  try {
    const res = await fetch(`${info.B.url}/describe`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: AbortSignal.timeout(30000),
      body: JSON.stringify(usageBody(info.B.url, '/describe', { prompt: pool.prompt ?? '', groups, request_id: newRequestId() })),
    });
    const body = await res.json().catch(() => null);
    if (!res.ok) throw new Error(body?.detail || body?.error || `status ${res.status}`);
    const r = decodeDescribe(body, groups.length), names = Array.from({ length: g.k }, () => '');
    ids.forEach((c, j) => { names[c] = r.names[j] ?? ''; });
    cache.set(key, names);
  } catch (e) {
    cache.set(key, 'failed');
    note = `Could not name the groups (${e?.message || e}).`;
  }
  // An answer for a grouping that is no longer on screen stays in the cache and is not shown.
  const now = currentGroups();
  if (H.pool() !== pool || !llmOn() || !now || nameKey(now) !== key) return;
  if (note) H.setStatus(note);
  renderLegend();
}

// ---------------------------------------------------------------- legend over the plot
function renderLegend() {
  if (!legend) return;
  const g = currentGroups();
  if (!g) { legend.hidden = true; legend.innerHTML = ''; return; }
  const names = namesOf(g);
  legend.hidden = false;
  legend.innerHTML = g.counts.map((n, c) => `<span class="x-chip" title="${escapeHtml(names[c])}: ${n} completions"><span class="x-sw" style="background:${slotColor(c)}"></span><span class="x-chip-name">${escapeHtml(names[c])}</span><span class="x-chip-n">${n}</span></span>`).join('')
    + '<span class="x-chip-note">selected points are outlined</span>';
}

// Re-render only when something the legend shows changed.
function refresh() {
  const g = currentGroups();
  const sig = g ? [ctx.H.pool()?.samples.length, st.method, st.k, st.naming, st.llm, Array.from(g.label).join('')].join('|') : '';
  if (sig === lastSig && (!g || !legend.hidden)) return;
  lastSig = sig;
  renderLegend();
  scheduleDescribe();
}

// ---------------------------------------------------------------- the menu section
function render() {
  const off = !st.on, dis = off ? ' disabled' : '';
  const q = (x, what) => `<button type="button" class="xbtn" data-x="${x}" aria-label="${what}">?</button>`;
  const seg = (name, opts, cur) => `<div class="seg small" role="radiogroup" aria-label="${name}">${opts.map((o) => `<label><input type="radio" name="g-${name}" value="${o.id}" data-f="g-${name}-${o.id}"${o.id === cur ? ' checked' : ''}${dis}><span>${o.label}</span></label>`).join('')}</div>`;
  return `<label class="toggle"><input type="checkbox" data-f="g-on"${st.on ? ' checked' : ''}><span>Color by group${q('groups', 'What are groups?')}</span></label>
    <div class="x-row"><span class="x-lab">Method${q('gmethod', 'What do the grouping methods do?')}</span>${seg('method', GROUP_METHODS, st.method)}</div>
    <div class="x-row"><span class="x-lab" id="g-k-lab">Number${q('gcount', 'How many groups?')}</span>
      <div class="stepper" role="group" aria-labelledby="g-k-lab">
        <button type="button" class="btn icon" data-f="g-minus" aria-label="Fewer groups"${dis}>−</button>
        <input type="number" id="g-k" data-f="g-k" min="${K_MIN}" max="${K_MAX}" value="${st.k}" aria-labelledby="g-k-lab"${dis}>
        <button type="button" class="btn icon" data-f="g-plus" aria-label="More groups"${dis}>+</button>
      </div></div>
    <div class="x-row"><span class="x-lab">Names${q('gnames', 'What do the names show?')}</span>${seg('naming', NAMINGS, st.naming)}</div>
    ${describeInfo() ? `<label class="toggle"><input type="checkbox" data-f="g-llm"${st.llm ? ' checked' : ''}${dis}><span>Described by the language model${q('describe', 'What does describing groups do?')}</span></label>` : ''}`;
}

function setState(patch) {
  st = { ...st, ...patch };
  st.k = Math.min(K_MAX, Math.max(K_MIN, Math.round(+st.k || DEFAULTS.k)));
  ctx.rerender();
  ctx.H.redraw();
  refresh();
}

function onMenu(e) {
  const f = e.target.dataset?.f;
  if (!f?.startsWith('g-')) return;
  if (e.type === 'change') {
    if (f === 'g-on') setState({ on: e.target.checked });
    else if (f === 'g-llm') setState({ llm: e.target.checked });
    else if (f === 'g-k') setState({ k: e.target.value });
    else if (f.startsWith('g-method-')) setState({ method: f.slice(9) });
    else if (f.startsWith('g-naming-')) setState({ naming: f.slice(9) });
  } else if (e.type === 'click') {
    if (f === 'g-minus') setState({ k: st.k - 1 });
    else if (f === 'g-plus') setState({ k: st.k + 1 });
  }
}

export const groupsSection = { id: 'groups', title: 'Groups', render, on: onMenu };

export function initGroups(c) {
  ctx = c;
  EXPLAIN.groups = {
    title: 'Groups',
    body: `<p>Groups only color the completions in the plot and on the cards. They never affect the subset.</p>
      <p>Method, number and names each have their own help.</p>`,
  };
  EXPLAIN.describe = {
    title: 'Described by the language model',
    body: `<p>The GPU server's language model reads a few completions from each group and writes a short name for it. The names are a reading aid and do not affect the selection.</p>
      <p>Until a name arrives, and when the model gives none or the request fails, a group keeps a numbered name.</p>`,
  };
  legend = document.createElement('div');
  legend.id = 'x-legend';
  legend.className = 'x-legend';
  legend.hidden = true;
  legend.setAttribute('aria-label', 'Groups');
  document.querySelector('.stage').append(legend);
  ctx.onPool(() => { lastSig = ''; });
  ctx.onSelection(refresh);
  ctx.onOff(() => { clearTimeout(descTimer); st = { ...DEFAULTS }; lastSig = ''; renderLegend(); ctx.H.redraw(); });
}
