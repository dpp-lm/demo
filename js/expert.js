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
// }

import { EXPLAIN } from './explain.js';
import { serverExtras, queueText, newRequestId } from './expert-lib.js';

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
export const selectionChanged = () => { if (on) listeners.selection.forEach((f) => f()); };

const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const $ = (s) => document.querySelector(s);

// ---------------------------------------------------------------- queue position (always on)
/** Called when the GPU server's /status has been read: remember its body and whether it has a queue. */
export function noteStatus(B, body) {
  B.statusBody = body ?? null;
  B.queue = !!(body && typeof body.queue === 'object' && body.queue);
  if (on) { applyServerExtras(); renderMenu(); }
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

// ---------------------------------------------------------------- the menu
const SECTIONS = [
  { id: 'models', title: 'Models and embeddings', render: () => {
    const B = SERVER(), x = serverExtras(B.statusBody, { known: ['qwen3-emb', 'own'], defKey: 'qwen3-14b' });
    const list = (a) => a.map(escapeHtml).join(', ');
    const rows = [];
    if (!B.statusBody) rows.push('<p class="x-note">The GPU server did not answer, so its extra models and embeddings are not listed.</p>');
    else {
      rows.push(x.embedders.length
        ? `<p class="x-note">Embedding choices added to Settings: ${list(x.embedders.map((e) => e.label))}.</p>`
        : '<p class="x-note">The GPU server offers no embeddings beyond the default ones.</p>');
      rows.push(x.models.length
        ? `<p class="x-note">Model choices added to Settings: ${list(x.models.map((m) => m.name))}.</p>`
        : '<p class="x-note">The GPU server reports one model, so the model choice is unchanged.</p>');
    }
    rows.push('<p class="x-act"><button type="button" class="btn" data-xact="settings">Open settings</button></p>');
    return rows.join('');
  } },
];

function renderMenu() {
  const body = $('#expert-body');
  if (!body) return;
  const open = new Set([...body.querySelectorAll('details[open]')].map((d) => d.dataset.sec));
  const first = !body.childElementCount;
  body.innerHTML = SECTIONS.map((s) => `<details class="xsec" data-sec="${s.id}"${first || open.has(s.id) ? ' open' : ''}><summary>${s.title}</summary><div class="xsec-body">${s.render()}</div></details>`).join('');
}

function setOn(v, { save = true } = {}) {
  on = v;
  if (save) write(v);
  $('#expert-toggle').checked = v;
  $('#expertmenu').hidden = !v;
  if (v) { applyServerExtras(); renderMenu(); }
  else {
    $('#expertmenu').open = false;
    removeServerExtras();
    listeners.off.forEach((f) => f());
  }
  H.refresh();
}

export function initExpert(hooks) {
  H = hooks;
  EXPLAIN.expert = {
    title: 'Expert tools',
    body: `<p>Extra controls for people who know the paper: more models and embeddings from the GPU server, and further tools as they are added. They appear in the <b>Expert</b> menu and in Settings.</p>
      <p>Turning the switch off puts every setting back to the default page.</p>`,
  };
  const anchor = document.querySelector('.drawer-body > .stage-bar');   // the bar that holds View
  const bar = document.createElement('span');
  bar.className = 'expert-bar';
  bar.innerHTML = `<details class="viewmenu expertmenu" id="expertmenu" hidden><summary>Expert</summary><div class="menu" id="expert-body"></div></details>
    <label class="toggle expert-switch"><input type="checkbox" id="expert-toggle" role="switch"><span>Expert tools<button type="button" class="xbtn" data-x="expert" aria-label="What are the expert tools?">?</button></span></label>`;
  anchor.after(bar);
  $('#expert-toggle').addEventListener('change', (e) => setOn(e.target.checked));
  $('#expert-body').addEventListener('click', (e) => {
    if (e.target.closest('[data-xact="settings"]')) { $('#expertmenu').open = false; setTimeout(H.openSettings, 0); }   // after the page's own outside-click handler has run
  });
  // Close on an outside click or Escape, like the View and Spectrum menus.
  document.addEventListener('click', (e) => {
    const m = $('#expertmenu');
    if (m.open && !e.composedPath().includes(m) && !e.target.closest('.xpop')) m.open = false;
  });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') $('#expertmenu').open = false; });
  if (read()) setOn(true, { save: false });
}
