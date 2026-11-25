// Expert tools: a switch in the bottom bar and an "Expert" menu next to View. Everything here sits behind
// the switch except the queue position, which reads GET /queue for any server that reports a queue.
// main.js calls initExpert(hooks) once and a few small functions below; nothing else of it is touched.
//
// hooks = {
//   backends:    the BACKENDS array (the GPU server first),
//   refresh():   rebuild the Model and Embedding lists and re-check the settings (redraws),
//   setStatus(text): the status line under the prompt,
//   openSettings(): open the Settings menu,
//   pool(), G(), weights(), selection(): the pool on screen, its similarity matrix, the weights in use and the selected indices,
//   eig(), embedder(): the eigendecomposition of G and the embedder's repo id,
//   redraw(): redraw the plot and the cards (colors changed), mark(i, j): mark up to two points in the plot (-1 for none),
// }

import { EXPLAIN } from './explain.js';
import { serverExtras, queueText, newRequestId } from './expert-lib.js';
import { groupsSection, initGroups, groupColors, groupInfo } from './groups.js';
import { principledSection, initUnprincipled, npState, npModeT } from './unprincipled.js';
import { initMatrices, matricesRefresh, matricesReset, panelBottom as matPanelBottom } from './matrix-panel.js';

const KEY = 'dpp-expert';
let H = null;
let on = false;
const listeners = { pool: [], selection: [], off: [] };

const read = () => { try { return localStorage.getItem(KEY) === '1'; } catch { return false; } };
const write = (v) => { try { localStorage.setItem(KEY, v ? '1' : '0'); } catch { /* storage may be blocked */ } };

export const isExpert = () => on;
/** Tools register here: onPool(fn) after a new pool is shown, onSelection(fn) after the subset or its readouts change, onOff(fn) when the switch goes off (restore upstream's behavior). */
export const onPool = (fn) => listeners.pool.push(fn);
export const onSelection = (fn) => listeners.selection.push(fn);
export const onOff = (fn) => listeners.off.push(fn);
export const poolChanged = () => { if (on) listeners.pool.forEach((f) => f()); };
export const selectionChanged = () => { matricesRefresh(); if (on) listeners.selection.forEach((f) => f()); };   // the matrices panel is in the default view too
/** Group colors for the points and card dots, or null (his colors) with the switch off or Groups off. */
export const pointColors = () => (on ? groupColors() : null);
/** The "Not super principled" options while the switch is on, else null (the default page). */
export const xopt = () => (on ? npState() : null);
/** T of the picking mode `to` after leaving `from`: one T unless the switch is on and each mode keeps its own. */
export const xModeT = (from, to, T) => (on ? npModeT(from, to, T) : T);
/** Top of the similarity-matrices panel when it is open on a wide screen (viewport pixels), else null. */
export const panelBottom = matPanelBottom;

const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const $ = (s) => document.querySelector(s);

// ---------------------------------------------------------------- queue position (always on)
/** Called when the GPU server's /status has been read: remember its body and whether it has a queue. */
export function noteStatus(B, body) {
  B.statusBody = body ?? null;
  B.queue = !!(body && typeof body.queue === 'object' && body.queue);
  if (on) applyServerExtras();
}

/**
 * Start polling GET {B.url}/queue?id= while a /sample request is pending. Returns { id, stop }: send `id` as
 * request_id, call stop() when the request settles or is cancelled. While the state is "waiting" the status line
 * reads "Waiting in line, N ahead."; when it is not, the line is cleared again. A failed poll is ignored.
 */
export function queueWatch(B, signal, { ms = 2000, fetchFn = (...a) => globalThis.fetch(...a), setStatus = (t) => H?.setStatus(t), getStatus = () => document.getElementById('live-status')?.textContent } = {}) {
  const id = newRequestId();
  let stopped = false, busy = false, shown = null;
  const clear = () => { if (shown !== null && getStatus() === shown) setStatus(''); shown = null; };
  const timer = setInterval(async () => {
    if (stopped || busy) return;
    busy = true;
    try {
      const r = await fetchFn(`${B.url}/queue?id=${encodeURIComponent(id)}`, { cache: 'no-store', signal });
      if (r.ok) {
        const j = await r.json();
        if (stopped || !j || typeof j.state !== 'string') return;
        const text = queueText(j);
        if (text) { if (text !== shown) setStatus(text); shown = text; } else clear();
      }
    } catch { /* a failed poll changes nothing */ } finally { busy = false; }
  }, ms);
  return { id, stop() { stopped = true; clearInterval(timer); clear(); } };
}

// ---------------------------------------------------------------- the GPU server's extra embedders and models
const SERVER = () => H.backends[0];

function applyServerExtras() {
  removeServerExtras();
  const B = SERVER(), x = serverExtras(B.statusBody, { known: B.embedders.map((e) => e.id), defKey: 'qwen3-14b' });
  const added = x.embedders.map((e) => ({ ...e, expert: true }));
  B.embedders = [...B.embedders, ...added];
  const byId = new Map(B.embedders.map((e) => [e.id, e]));
  const derived = x.models.map((m) => ({
    ...B, id: `server:${m.key}`, modelKey: m.key, repo: m.repo, short: m.name, expert: true,
    label: `${m.name} · GPU server${m.state === 'cold' || m.state === 'loading' ? ' (loads on first run)' : ''}`,
    embedders: m.embedders.map((id) => byId.get(id)).filter(Boolean),
    get up() { return B.up; },
  }));
  H.backends.splice(1, 0, ...derived);
}

function removeServerExtras() {
  const B = SERVER();
  B.embedders = B.embedders.filter((e) => !e.expert);
  for (let i = H.backends.length - 1; i > 0; i--) if (H.backends[i].expert) H.backends.splice(i, 1);
}

// ---------------------------------------------------------------- the control panel
// A column under the prompt and subset controls (a section below them on phones), shown while the switch is on.
// Each section is { id, title, render() -> html, on(event) }. Add a section with registerSection({...}); the
// panel re-renders after any change, keeps keyboard focus on the control used, and remembers closed sections.
const SECTIONS = [];
export function registerSection(sec, { before = null } = {}) {
  const at = before ? SECTIONS.findIndex((s) => s.id === before) : -1;
  if (at >= 0) SECTIONS.splice(at, 0, sec); else SECTIONS.push(sec);
}

const CLOSED_KEY = 'dpp-expert-closed';
const readClosed = () => { try { return new Set(JSON.parse(localStorage.getItem(CLOSED_KEY) ?? '[]')); } catch { return new Set(); } };
const writeClosed = (set) => { try { localStorage.setItem(CLOSED_KEY, JSON.stringify([...set])); } catch { /* storage may be blocked */ } };

registerSection(groupsSection);
registerSection(principledSection);   // last

function renderPanel() {
  const body = $('#expert-body');
  if (!body || !on) return;
  const focus = document.activeElement?.dataset?.f;
  const closed = readClosed();
  body.innerHTML = SECTIONS.map((s) => `<details class="xsec" data-sec="${s.id}"${closed.has(s.id) ? '' : ' open'}><summary>${s.title}</summary><div class="xsec-body">${s.render()}</div></details>`).join('');
  if (focus) body.querySelector(`[data-f="${focus}"]`)?.focus();
  fitPanel();
}

/** On wide screens the panel scrolls inside the room left under the controls. */
function fitPanel() {
  const p = $('#xpanel');
  if (!p || p.hidden) return;
  if (!matchMedia('(min-width: 901px)').matches) { p.style.maxHeight = ''; return; }
  const bar = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--bar-h')) || 40;
  p.style.maxHeight = '0px';   // measure the room without the panel's own height
  const top = p.getBoundingClientRect().top;
  p.style.maxHeight = `${Math.max(160, innerHeight - bar - top - 14)}px`;
}

function setOn(v, { save = true } = {}) {
  on = v;
  if (save) write(v);
  $('#expert-toggle').checked = v;
  $('#xpanel').hidden = !v;
  $('#xgo').hidden = !v;
  if (v) { applyServerExtras(); }
  else {
    removeServerExtras();
    listeners.off.forEach((f) => f());
    matricesReset();
  }
  H.refresh();
  if (v) renderPanel();
  matricesRefresh(true);
}

export function initExpert(hooks) {
  H = hooks;
  EXPLAIN.expert = {
    title: 'Expert tools',
    body: `<p>Extra controls for people who know the paper. A panel under the prompt and subset controls groups the completions and has the options the default page leaves out. The similarity matrices get extra options in their own panel. Extra models and embeddings the GPU server offers appear in Settings.</p>
      <p>Turning the switch off puts every setting back to the default page.</p>`,
  };
  EXPLAIN.gmethod = { title: 'Grouping method', body: '<p>How the completions are split into groups. All methods use only the similarity between completions.</p><ul><li><b>Tree</b> cuts the average-linkage tree on $1 - \\cos$, the one that orders the similarity matrices.</li><li><b>Far apart</b> uses complete linkage: no group holds two completions farther apart than the height at which it formed.</li><li><b>Average</b> uses average linkage, which balances that with group size.</li><li><b>k-means</b> builds compact groups around centers, so an isolated completion can join the nearest group.</li></ul>' };
  EXPLAIN.gcount = { title: 'Number of groups', body: '<p>How many groups the completions are cut into, from 2 to 8. Groups are numbered from the largest. Tree, Far apart and Average can return fewer when the completions do not split that far.</p>' };
  EXPLAIN.gnames = { title: 'Group names', body: '<p>What the legend calls each group. <b>Numbers</b> are Group 1, 2, and so on. <b>Keywords</b> picks the word that is most common in the group and rare outside it. <b>Typical member</b> shows the start of the completion most similar to the rest of the group.</p>' };
  const anchor = document.querySelector('.drawer-body > .stage-bar');   // the bar that holds View
  const bar = document.createElement('span');
  bar.className = 'expert-bar';
  bar.innerHTML = `<button type="button" class="matbtn" id="xgo" hidden>Expert</button>
    <label class="toggle expert-switch"><input type="checkbox" id="expert-toggle" role="switch"><span>Expert tools<button type="button" class="xbtn" data-x="expert" aria-label="What are the expert tools?">?</button></span></label>`;
  anchor.after(bar);
  const panel = document.createElement('section');
  panel.id = 'xpanel';
  panel.className = 'step xpanel';
  panel.hidden = true;
  panel.setAttribute('aria-label', 'Expert tools');
  panel.innerHTML = '<h2>Expert tools</h2><div id="expert-body"></div>';
  document.querySelector('.dock').append(panel);
  $('#expert-toggle').addEventListener('change', (e) => { setOn(e.target.checked); if (e.target.checked && !matchMedia('(min-width: 901px)').matches) $('#xpanel').scrollIntoView({ behavior: 'smooth', block: 'start' }); });
  $('#xgo').onclick = () => $('#xpanel').scrollIntoView({ behavior: 'smooth', block: 'start' });
  // The tools: the page's own bits of state they read, and how they redraw.
  const ctx = { H, isExpert, rerender: renderPanel, onPool, onSelection, onOff, groups: groupInfo };
  initGroups(ctx);
  initUnprincipled(ctx);
  initMatrices(ctx);
  for (const type of ['change', 'click']) $('#expert-body').addEventListener(type, (e) => SECTIONS.forEach((s) => s.on?.(e)));
  $('#expert-body').addEventListener('toggle', (e) => {
    const d = e.target;
    if (!d.dataset?.sec) return;
    const c = readClosed();
    if (d.open) c.delete(d.dataset.sec); else c.add(d.dataset.sec);
    writeClosed(c);
  }, true);
  window.addEventListener('resize', fitPanel);
  new ResizeObserver(fitPanel).observe(document.querySelector('.dock'));
  if (read()) setOn(true, { save: false });
}
