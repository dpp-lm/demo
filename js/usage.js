// Usage ids, events and the research notice (rules and formats are in usage-lib.js). All of it is tied to the
// GPU server's GET /status: with `logsRuns` the notice shows, with `usage` the ids and events are on. Every failure is silent.
import { noticeVisible, usageOn, visitorId, newId, withIds, makeThrottle, loadEvent, uiEvent, NOTICE_TEXT } from './usage-lib.js';

const st = { serverUrl: null, status: null, on: false, visitor: null, session: null, loaded: false };
const allow = makeThrottle();
/** Resolves once the GPU server's status has been read (or has failed); the page waits for it, briefly, before it first shows, so the notice does not push the page down later. */
let settle;
export const usageReady = new Promise((r) => { settle = r; });
const get = (id) => document.getElementById(id);

/** Called when the GPU server's status has been read (body null when unreachable). */
export function usageStatus(B, body) {
  st.serverUrl = B.url;
  st.status = body ?? null;
  const on = usageOn(body);
  if (on && !st.visitor) {
    let storage = null;
    try { storage = localStorage; } catch { /* blocked */ }
    st.visitor = storage ? visitorId(storage) : newId();
    st.session = newId();
  }
  st.on = on;
  if (on && !st.loaded) {
    st.loaded = true;
    let expert = false;
    try { expert = !!get('expert-toggle')?.checked; } catch { /* ignore */ }
    send(loadEvent(st, { referrer: document.referrer, expert }));
  }
  usageNotice();
  settle();
}

/** Show or hide the notice for the backend selected now (call after a status or a model change). */
export function usageNotice() {
  const el = get('usage-note');
  if (!el) return;
  el.textContent = NOTICE_TEXT;
  el.hidden = !noticeVisible(get('model')?.value, st.status);
}

/** The body for a request to `url` at `path`, with the ids when it goes to our server and usage is on. */
export function usageBody(url, path, body) {
  return withIds(body, { url, path, serverUrl: st.serverUrl, on: st.on, visitor: st.visitor, session: st.session });
}

/** Report a UI action (names in usage-lib.js). Throttled to one per name per 10 s. */
export function usageUi(name, value) {
  if (!st.on) return;
  const e = uiEvent(st, name, value);
  if (e && allow(name, Date.now())) send(e);
}

function send(body) {
  try {
    fetch(`${st.serverUrl}/event`, { method: 'POST', keepalive: true, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).catch(() => {});
  } catch { /* never block the page */ }
}

/** Listen for the reported actions at the document level, so the tools themselves need no hooks. Call once. */
export function initUsage() {
  document.addEventListener('change', (e) => {
    const t = e.target;
    if (t?.id === 'expert-toggle') usageUi(t.checked ? 'expert_on' : 'expert_off');
    else if (t?.dataset?.f === 'g-on' && t.checked) usageUi('groups_on');
  }, true);
  document.addEventListener('click', (e) => {
    const b = e.target?.closest?.('button');
    if (!b) return;
    if (b.classList.contains('pick-btn') && b.dataset.mode && b.closest('.step-2')) usageUi('mode', b.dataset.mode);
    else if (b.id === 'matbtn') setTimeout(() => { if (b.getAttribute('aria-expanded') === 'true') usageUi('matrices_open'); }, 0);
    else if (b.dataset.f?.startsWith('np-preset-')) usageUi('preset', b.dataset.f.slice(10));
  }, true);
  document.addEventListener('toggle', (e) => { if (e.target?.id === 'specmenu' && e.target.open) usageUi('spectrum_open'); }, true);
}
