import test from 'node:test';
import assert from 'node:assert/strict';
import { newId, isId, visitorId, referrerHost, noticeVisible, usageOn, withIds, makeThrottle, loadEvent, uiEvent, VISITOR_KEY, UI_NAMES } from '../js/usage-lib.js';

const mem = () => { const m = new Map(); return { getItem: (k) => m.get(k) ?? null, setItem: (k, v) => m.set(k, v), m }; };

test('newId: 22 url-safe characters, from getRandomValues, different each time', () => {
  const ids = new Set(Array.from({ length: 50 }, () => newId()));
  assert.equal(ids.size, 50);
  for (const id of ids) { assert.equal(id.length, 22); assert.match(id, /^[A-Za-z0-9_-]{22}$/); assert.ok(isId(id)); }
  const fixed = { getRandomValues: (a) => { a.fill(255); return a; } };
  assert.equal(newId(fixed), '_____________________w');
});

test('isId: rejects short, non-string and unsafe values', () => {
  for (const x of ['', 'abc', null, 5, 'a'.repeat(21), 'a'.repeat(21) + '+']) assert.ok(!isId(x));
  assert.ok(isId('a'.repeat(22)) && isId('a-_'.repeat(10)));
});

test('visitorId: stored once, reused, repaired when malformed, per-load id when storage throws', () => {
  const s = mem();
  const a = visitorId(s);
  assert.equal(s.m.get(VISITOR_KEY), a);
  assert.equal(visitorId(s), a);
  s.setItem(VISITOR_KEY, 'short');
  const b = visitorId(s);
  assert.ok(isId(b) && b !== a && s.m.get(VISITOR_KEY) === b);
  const bad = { getItem() { throw new Error('blocked'); }, setItem() { throw new Error('blocked'); } };
  assert.ok(isId(visitorId(bad)));
  const ro = { getItem: () => null, setItem() { throw new Error('quota'); } };
  assert.ok(isId(visitorId(ro)));
});

test('referrerHost: host only, empty when missing or invalid', () => {
  assert.equal(referrerHost('https://news.example.com/a/b?q=secret#x'), 'news.example.com');
  assert.equal(referrerHost('http://localhost:8770/x'), 'localhost:8770');
  assert.equal(referrerHost(''), '');
  assert.equal(referrerHost('not a url'), '');
  assert.equal(referrerHost(undefined), '');
});

test('noticeVisible: only the GPU server (and its models) with logsRuns true', () => {
  assert.ok(noticeVisible('server', { logsRuns: true }));
  assert.ok(noticeVisible('server:qwen3-8b', { logsRuns: true }));
  for (const id of ['modal', 'space', 'servers', 'xserver', undefined]) assert.ok(!noticeVisible(id, { logsRuns: true }));
  for (const s of [null, undefined, {}, { logsRuns: false }, { logsRuns: 'true' }, { logsRuns: 1 }, { usage: true }]) assert.ok(!noticeVisible('server', s));
});

test('usageOn: only the boolean true', () => {
  assert.ok(usageOn({ usage: true }));
  for (const s of [null, {}, { usage: false }, { usage: 'yes' }, { usage: 1 }]) assert.ok(!usageOn(s));
});

test('withIds: ids go only to our server, on its three paths, while usage is on', () => {
  const ctx = { url: 'https://gpu.example', path: '/sample', serverUrl: 'https://gpu.example', on: true, visitor: 'V', session: 'S' };
  const body = { prompt: 'p' };
  assert.deepEqual(withIds(body, ctx), { prompt: 'p', visitor: 'V', session: 'S' });
  assert.deepEqual(body, { prompt: 'p' });
  for (const path of ['/embed', '/describe']) assert.equal(withIds(body, { ...ctx, path }).visitor, 'V');
  assert.equal(withIds(body, { ...ctx, path: '/queue' }), body);
  assert.equal(withIds(body, { ...ctx, path: '/event' }), body);
  assert.equal(withIds(body, { ...ctx, url: 'https://timvieira--dpp-lm-sampler-web.modal.run' }), body);
  assert.equal(withIds(body, { ...ctx, url: 'https://xtimv-dpp-lm-sampler.hf.space' }), body);
  assert.equal(withIds(body, { ...ctx, on: false }), body);
  assert.equal(withIds(body, { ...ctx, serverUrl: null }), body);
});

test('makeThrottle: one per name per 10 s, names independent', () => {
  const allow = makeThrottle();
  assert.ok(allow('mode', 0));
  assert.ok(!allow('mode', 5000));
  assert.ok(!allow('mode', 9999));
  assert.ok(allow('preset', 5000));
  assert.ok(allow('mode', 10000));
  assert.ok(!allow('mode', 19999));
  assert.ok(allow('mode', 20000));
});

test('events: load carries the referrer host and the expert flag, ui only listed names, no extra fields', () => {
  const ids = { visitor: 'V', session: 'S' };
  assert.deepEqual(loadEvent(ids, { referrer: 'https://a.org/p?q=1', expert: true }), { visitor: 'V', session: 'S', type: 'load', referrer: 'a.org', expert: true });
  assert.deepEqual(loadEvent(ids, { referrer: '', expert: 0 }), { visitor: 'V', session: 'S', type: 'load', referrer: '', expert: false });
  assert.deepEqual(uiEvent(ids, 'mode', 'map'), { visitor: 'V', session: 'S', type: 'ui', name: 'mode', value: 'map' });
  assert.deepEqual(uiEvent(ids, 'preset', 'samp'), { visitor: 'V', session: 'S', type: 'ui', name: 'preset', value: 'samp' });
  assert.deepEqual(uiEvent(ids, 'matrices_open', 'ignored'), { visitor: 'V', session: 'S', type: 'ui', name: 'matrices_open' });
  assert.equal(uiEvent(ids, 'typed_prompt', 'secret'), null);
  assert.deepEqual(UI_NAMES, ['expert_on', 'expert_off', 'mode', 'matrices_open', 'spectrum_open', 'groups_on', 'preset']);
});
