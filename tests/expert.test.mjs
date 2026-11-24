import test from 'node:test';
import assert from 'node:assert/strict';
import { queueWatch } from '../js/expert.js';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(cond, ms = 1000) { for (let t = 0; t < ms && !cond(); t += 2) await sleep(2); assert.ok(cond(), 'condition not reached'); }

test('queueWatch: shows the waiting text, clears it when the request runs, stops polling', async () => {
  let state = { state: 'waiting', ahead: 3 };
  const urls = [];
  let text = '', polls = 0;
  const fetchFn = async (url) => { urls.push(url); polls++; return { ok: true, json: async () => state }; };
  const w = queueWatch({ url: 'http://s/dpp_api' }, undefined, { ms: 2, fetchFn, setStatus: (t) => { text = t; }, getStatus: () => text });
  try {
    assert.match(w.id, /^[0-9a-f]{32}$/);
    await until(() => text === 'Waiting in line, 3 ahead.');
    state = { state: 'waiting', ahead: 0 };
    await until(() => text === 'Waiting in line, next up.');
    state = { state: 'running', ahead: 0 };
    await until(() => text === '');
    assert.equal(urls[0], `http://s/dpp_api/queue?id=${w.id}`);
  } finally { w.stop(); }
  await sleep(10);
  const n = polls;
  await sleep(20);
  assert.equal(polls, n);
});

test('queueWatch: stop clears its own text only, ignores failed polls', async () => {
  let text = 'x', fail = true;
  const fetchFn = async () => { if (fail) throw new Error('offline'); return { ok: true, json: async () => ({ state: 'waiting', ahead: 2 }) }; };
  const w = queueWatch({ url: 'u' }, undefined, { ms: 2, fetchFn, setStatus: (t) => { text = t; }, getStatus: () => text });
  try {
    await sleep(15);
    assert.equal(text, 'x');
    fail = false;
    await until(() => text === 'Waiting in line, 2 ahead.');
    text = 'Stopped.';   // another message replaced it
  } finally { w.stop(); }
  assert.equal(text, 'Stopped.');
});
