import test from 'node:test';
import assert from 'node:assert/strict';
import { drawCacheOn, withOffset, offsetMismatch } from '../js/draws-lib.js';

test('drawCacheOn: only our GPU server, only when its status says drawCache: true', () => {
  assert.ok(drawCacheOn('server', { drawCache: true }));
  assert.ok(drawCacheOn('server:qwen3-4b', { drawCache: true }));
  for (const id of ['modal', 'space', 'serverx', '', undefined, null]) assert.ok(!drawCacheOn(id, { drawCache: true }));
  for (const st of [{ drawCache: false }, {}, null, undefined, { drawCache: 'true' }, { drawCache: 1 }]) assert.ok(!drawCacheOn('server', st));
});

test('withOffset: adds the buffer position as offset and drops the seed', () => {
  const body = { prompt: 'p', n: 10, t_prime: 1, max_new_tokens: 64, seed: 5 };
  assert.deepEqual(withOffset(body, { on: true, position: 0 }), { prompt: 'p', n: 10, t_prime: 1, max_new_tokens: 64, offset: 0 });
  assert.deepEqual(withOffset(body, { on: true, position: 30 }).offset, 30);
  assert.equal(body.seed, 5, 'the input is not changed');
});

test('withOffset: the same body object when the cache is off or the position is not a non-negative integer', () => {
  const body = { prompt: 'p', seed: 5 };
  assert.equal(withOffset(body, { on: false, position: 0 }), body);
  for (const position of [-1, 1.5, NaN, undefined, null, '3']) assert.equal(withOffset(body, { on: true, position }), body);
});

test('offsetMismatch: an answer must hold the asked positions; nothing to check when the cache is off', () => {
  assert.ok(!offsetMismatch({ offset: 20 }, { on: true, position: 20 }));
  assert.ok(offsetMismatch({ offset: 0 }, { on: true, position: 20 }));
  assert.ok(offsetMismatch({}, { on: true, position: 20 }));
  assert.ok(!offsetMismatch({}, { on: false, position: 20 }));
  assert.ok(!offsetMismatch({}, { on: true, position: null }));
});
