// Usage ids, events and the research notice: the pure parts (the DOM and network side is in usage.js).
// Active only for our GPU server, and only while its GET /status says so: `logsRuns` shows the notice,
// `usage` turns on the ids and the events. Other backends never see any of it.

export const VISITOR_KEY = 'dpp-visitor';
export const NOTICE_TEXT = 'Prompts and completions are saved and may be used for research.';
export const THROTTLE_MS = 10000;
/** The UI actions that are reported, by name (anything else is dropped). */
export const UI_NAMES = ['expert_on', 'expert_off', 'mode', 'matrices_open', 'spectrum_open', 'groups_on', 'preset'];
/** The server paths whose JSON body carries the ids. */
export const ID_PATHS = ['/sample', '/embed', '/describe'];

/** A random id of 22 url-safe characters (16 random bytes, base64url without padding). */
export function newId(cryptoObj = globalThis.crypto) {
  const bytes = cryptoObj.getRandomValues(new Uint8Array(16));
  let s = '';
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}
export const isId = (s) => typeof s === 'string' && /^[A-Za-z0-9_-]{22,}$/.test(s);

/** The visitor id from storage, made and stored when missing or malformed. Any storage failure gives a per-load id. */
export function visitorId(storage, cryptoObj = globalThis.crypto) {
  try {
    const old = storage.getItem(VISITOR_KEY);
    if (isId(old)) return old;
  } catch { /* storage may be blocked */ }
  const id = newId(cryptoObj);
  try { storage.setItem(VISITOR_KEY, id); } catch { /* per-load id */ }
  return id;
}

/** The host of a referrer URL, or '' (no path, query or scheme; '' for an empty or unparseable referrer). */
export function referrerHost(ref) {
  try { return ref ? new URL(ref).host : ''; } catch { return ''; }
}

/** True when the status body asks for the research notice and the backend is the GPU server (id server or server:<key>). */
export const noticeVisible = (backendId, status) => /^server(:|$)/.test(String(backendId ?? '')) && status?.logsRuns === true;
export const usageOn = (status) => status?.usage === true;

/**
 * The request body with visitor and session added, when `url` is the GPU server's url and `path` is one of ID_PATHS
 * and usage is on. Otherwise the same body object, unchanged.
 */
export function withIds(body, { url, path, serverUrl, on, visitor, session }) {
  if (!on || !serverUrl || url !== serverUrl || !ID_PATHS.includes(path)) return body;
  return { ...body, visitor, session };
}

/** At most one event per name per `ms`: allow(name, now) is true (and starts the window) or false. */
export function makeThrottle(ms = THROTTLE_MS) {
  const last = new Map();
  return (name, now) => {
    const t = last.get(name);
    if (t !== undefined && now - t < ms) return false;
    last.set(name, now);
    return true;
  };
}

/** The JSON body of POST /event for the page load. */
export const loadEvent = ({ visitor, session }, { referrer, expert }) => ({ visitor, session, type: 'load', referrer: referrerHost(referrer), expert: !!expert });

/** The JSON body of POST /event for a UI action, or null for a name not on the list. `value` is kept only for mode and preset. */
export function uiEvent({ visitor, session }, name, value) {
  if (!UI_NAMES.includes(name)) return null;
  const e = { visitor, session, type: 'ui', name };
  if ((name === 'mode' || name === 'preset') && value != null) e.value = String(value);
  return e;
}
