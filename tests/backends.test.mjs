import test from 'node:test';
import assert from 'node:assert/strict';
import { parseStatus, sampleBody, mapError, BackendError, parseBackendParam, backendConfigs, createBackends, decodePool, decodeEmbed, chooseBackend } from '../js/backends.js';

const b64 = (arr) => Buffer.from(new Float32Array(arr).buffer).toString('base64');
const pool = (N = 2, D = 3) => ({ prompt: 'p', model: 'm', N, D, Tprime: 1, samples: Array.from({ length: N }, (_, i) => ({ text: `s${i}`, logp: -1, logq: -1, ended: true })), emb: b64(Array.from({ length: N * D }, (_, i) => i / 10)) });
const res = (status, body) => ({ ok: status < 400, status, json: async () => body });

test('parseStatus: our shape and the minimal Modal shape', () => {
  const ours = parseStatus({ models: [{ key: 'a', name: 'A', state: 'ready' }, { key: 'b', state: 'cold' }], defaultModel: 'b', embedders: ['minilm', 'qwen3-emb'], maxN: 300 }, { auth: 'none' });
  assert.deepEqual(ours.models.map((m) => m.key), ['a', 'b']);
  assert.equal(ours.def, 'b'); assert.equal(ours.multi, true); assert.deepEqual(ours.embedders, ['minilm', 'qwen3-emb']); assert.equal(ours.maxN, 300);
  const modal = parseStatus({ model: 'qwen3-14b', gpu_minutes_left_this_month: 41 }, { auth: 'hf' });
  assert.deepEqual(modal.models, [{ key: 'qwen3-14b', name: 'qwen3-14b', state: 'ready' }]);
  assert.deepEqual(modal.embedders, ['minilm']); assert.equal(modal.auth, 'hf'); assert.equal(modal.multi, false); assert.equal(modal.budget, 41);
  assert.equal(parseStatus({}, {}).models.length, 0);
  assert.deepEqual(parseStatus({ model: 'x' }, { embedders: ['minilm', 'qwen3-emb'] }).embedders, ['minilm', 'qwen3-emb']);
});

test('sampleBody: model only when multi, embedder only when offered or different', () => {
  const req = { prompt: 'hi', n: 30, t_prime: 1.2, max_new_tokens: 64, seed: 7, model: 'qwen3-14b', embedder: 'qwen3-emb' };
  assert.deepEqual(sampleBody(req, { multi: true, embedders: ['minilm', 'qwen3-emb'] }), { prompt: 'hi', n: 30, t_prime: 1.2, max_new_tokens: 64, seed: 7, model: 'qwen3-14b', embedder: 'qwen3-emb' });
  const m = sampleBody({ ...req, embedder: 'minilm' }, { multi: false, embedders: ['minilm'] });
  assert.equal('model' in m, false); assert.equal('embedder' in m, false);
  assert.ok(Number.isInteger(sampleBody({ ...req, seed: undefined }, { embedders: ['minilm'] }).seed));
});

test('mapError: codes and messages', () => {
  assert.equal(mapError(400, { detail: 'The prompt is empty.' }).code, 'bad_request');
  assert.equal(mapError(401, { detail: 'Sign in' }).code, 'unauthorized');
  assert.equal(mapError(429, { detail: 'used up' }).message, 'used up');
  assert.equal(mapError(503, {}).code, 'unavailable');
  assert.equal(mapError(502, {}).code, 'unavailable');
  assert.match(mapError(429, {}).message, /limit/);
  assert.ok(mapError(400, { error: { message: 'x' } }) instanceof BackendError);
});

test('parseBackendParam: only http(s) URLs', () => {
  assert.deepEqual(parseBackendParam('?backend=https://a.example.com/api/'), { url: 'https://a.example.com/api', origin: 'https://a.example.com' });
  assert.equal(parseBackendParam('?backend=http://localhost:8000').origin, 'http://localhost:8000');
  for (const bad of ['javascript:alert(1)', 'ftp://x.com', '/relative', 'file:///etc/passwd', '']) assert.equal(parseBackendParam(`?backend=${encodeURIComponent(bad)}`), null);
  assert.equal(parseBackendParam('?x=1'), null);
  const cfgs = backendConfigs([{ id: 'server', kind: 'http', url: '/dpp_api' }, { bad: true }], '?backend=http://localhost:9');
  assert.deepEqual(cfgs.map((c) => c.id), ['custom', 'server']);
});

test('http backend: status, sample request and decoding, error mapping, embed', async () => {
  const calls = [];
  const fetch = async (url, opt = {}) => {
    calls.push([url, opt.method ?? 'GET', opt.body ? JSON.parse(opt.body) : null, opt.headers]);
    if (url.endsWith('/status')) return res(200, { models: [{ key: 'q', name: 'Q', state: 'ready' }], embedders: ['minilm', 'qwen3-emb'], embed: true });
    if (url.endsWith('/sample')) return JSON.parse(opt.body).prompt === 'bad' ? res(400, { detail: 'The prompt is empty.' }) : res(200, pool());
    if (url.endsWith('/embed')) return res(200, { embedder: 'qwen3-emb', D: 3, emb: b64([1, 2, 3, 4, 5, 6]) });
    return res(404, {});
  };
  const [b] = createBackends([{ id: 's', kind: 'http', url: 'http://x/api', auth: 'hf' }], { fetch, getToken: () => 'tok' });
  const caps = await b.status();
  assert.equal(caps.up, true); assert.equal(caps.embed, true);
  const out = await b.sample({ prompt: 'hi', n: 2, t_prime: 1, max_new_tokens: 8, seed: 1, model: 'q', embedder: 'qwen3-emb' });
  assert.equal(out.emb.length, 6); assert.equal(out.pool.N, 2); assert.equal('emb' in out.pool, false);
  const sample = calls.find((c) => c[0].endsWith('/sample'));
  assert.equal(sample[1], 'POST'); assert.equal(sample[2].model, 'q'); assert.equal(sample[2].embedder, 'qwen3-emb'); assert.equal(sample[3].Authorization, 'Bearer tok');
  await assert.rejects(b.sample({ prompt: 'bad', n: 2, t_prime: 1, max_new_tokens: 8 }), (e) => e.code === 'bad_request' && /empty/.test(e.message));
  const em = await b.embed({ texts: ['a', 'b'], embedder: 'qwen3-emb', model: 'q', prompt: 'hi' });
  assert.equal(em.D, 3); assert.equal(em.emb.length, 6);
  assert.deepEqual(calls.find((c) => c[0].endsWith('/embed'))[2], { texts: ['a', 'b'], embedder: 'qwen3-emb', model: 'q', prompt: 'hi' });
});

test('http backend: unreachable status and sample', async () => {
  const [b] = createBackends([{ id: 's', kind: 'http', url: 'http://x' }], { fetch: async () => { throw new TypeError('fail'); } });
  assert.equal((await b.status()).up, false);
  await assert.rejects(b.sample({ prompt: 'x', n: 2, t_prime: 1, max_new_tokens: 8 }), (e) => e.code === 'unavailable');
});

test('decodePool and decodeEmbed reject mismatched sizes', () => {
  const bad = pool(2, 3); bad.D = 4;
  assert.throws(() => decodePool(bad), BackendError);
  assert.throws(() => decodePool({ samples: [] }), BackendError);
  assert.throws(() => decodeEmbed({}), BackendError);
  const k = pool(); delete k.emb; k.kernels = [{ layer: 12, gram: [] }];
  assert.throws(() => decodePool(k), /kernels but no embedding/);
});

test('chooseBackend: first backend with the model, embedder fallback is reported', () => {
  const mk = (id, models, embedders, up = true) => ({ id, caps: { up, models: models.map((key) => ({ key })), embedders } });
  const bs = [mk('server', ['a'], ['minilm', 'qwen3-emb'], false), mk('modal', ['a'], ['minilm']), mk('browser', ['z'], ['minilm'])];
  const c = chooseBackend(bs, { model: 'a', embedder: 'qwen3-emb' });
  assert.equal(c.backend.id, 'modal'); assert.equal(c.embedder, 'minilm'); assert.equal(c.fellBack, true);
  assert.equal(chooseBackend(bs, { model: 'a', embedder: 'minilm' }).fellBack, false);
  assert.equal(chooseBackend(bs, { model: 'nope', embedder: 'minilm' }), null);
  assert.equal(chooseBackend(bs, { model: 'a', embedder: 'minilm' }, (b) => b.id !== 'modal'), null);
});
