import test from 'node:test';
import assert from 'node:assert/strict';
import { gram, columnLogScale, mulberry32 } from '../js/dpp.js';
import { displayScale, matchPreset, PRESETS, presetDiff, weightLogScale, resolveWeight, paperWeight, weightEss, readoutNotes, effectiveK, normText, distinctCount, gramToVectors, newRequestId, queueText, decodeDescribe, serverExtras, repoLabel } from '../js/expert-lib.js';

test('displayScale: quality is the same in both modes, importance divides by q, paper follows the mode, unit is 1', () => {
  const logp = [-3, -5, -1, -9, -4], logq = [-3.2, -4.1, -1.5, -8, -4.4];
  for (const T of [0.5, 1, 20]) {
    const qm = displayScale('quality', 'map', logp, logq, T), qs = displayScale('quality', 'sample', logp, logq, T);
    assert.deepEqual(qm, qs);
    assert.equal(Math.max(...qm), 1);
    const sm = displayScale('importance', 'map', logp, logq, T), sw = displayScale('importance', 'sample', logp, logq, T);
    assert.deepEqual(sm, sw); // importance is the same in both modes
    assert.notDeepEqual(sw, qs);
    // an unknown choice behaves as the mode's paper weight
    assert.deepEqual(displayScale('paper', 'map', logp, logq, T), qm);
    assert.deepEqual(displayScale('paper', 'sample', logp, logq, T), sw);
    assert.ok(Math.abs(sw[2] / sw[0] - Math.exp(0.5 * ((logp[2] - logp[0]) / T - (logq[2] - logq[0])))) < 1e-12);
  }
  assert.deepEqual(displayScale('unit', 'sample', logp, logq, 3), [1, 1, 1, 1, 1]);
});

test('matchPreset', () => {
  assert.equal(matchPreset({ mode: 'map', Tprime: '1.5', K: 3, T: 3 }), 'map');
  assert.equal(matchPreset({ mode: 'map', Tprime: 1.5, K: 3, T: 7.5 }), 'map');
  assert.equal(matchPreset({ mode: 'sample', Tprime: '1.0', K: 3, T: 0.82 }), 'samp');
  assert.equal(matchPreset({ mode: 'sample', Tprime: 1, K: 3, T: 0.43 }), 'samp');
  assert.equal(matchPreset({ mode: 'map', Tprime: 1.5, K: 4, T: 3 }), null);
  assert.equal(matchPreset({ mode: 'map', Tprime: 1.0, K: 3, T: 3 }), null);
  assert.equal(matchPreset({ mode: 'map', Tprime: 1.5, K: 3, T: 4 }), null);
  assert.equal(matchPreset({ mode: 'sample', Tprime: 1.5, K: 3, T: 0.82 }), null);
  assert.equal(matchPreset({ mode: 'sample', Tprime: 1, K: 3, T: 3 }), null);
});

test('presets use the paper DPP weight with the lock on; the match needs the resolved weight', () => {
  assert.equal(PRESETS.map.weight, 'quality'); assert.equal(PRESETS.map.lock, true);
  assert.equal(PRESETS.samp.weight, 'importance'); assert.equal(PRESETS.samp.lock, true);
  assert.equal(matchPreset({ mode: 'map', Tprime: 1.5, K: 3, T: 3, weight: 'quality' }), 'map');
  assert.equal(matchPreset({ mode: 'map', Tprime: 1.5, K: 3, T: 3, weight: 'importance' }), null);
  assert.equal(matchPreset({ mode: 'sample', Tprime: 1, K: 3, T: 0.82, weight: 'importance' }), 'samp');
  assert.equal(matchPreset({ mode: 'sample', Tprime: 1, K: 3, T: 0.82, weight: 'quality' }), null);
  assert.equal(matchPreset({ mode: 'sample', Tprime: 1, K: 3, T: 0.82, weight: 'unit' }), null);
  // without a weight the match is as before
  assert.equal(matchPreset({ mode: 'map', Tprime: 1.5, K: 3, T: 3 }), 'map');
});

test('weightLogScale: the paper weight of each mode equals columnLogScale; the families do not depend on the mode', () => {
  const logp = [-3, -5, -1, -9, -4], logq = [-3.2, -4.1, -1.5, -8, -4.4];
  for (const T of [0.3, 1, 20]) {
    assert.deepEqual(weightLogScale('quality', 'map', logp, logq, T), columnLogScale('map', logp, logq, T));
    assert.deepEqual(weightLogScale('importance', 'sample', logp, logq, T), columnLogScale('sample', logp, logq, T));
    // tolerant fallback: an unknown value is the mode's paper weight
    assert.deepEqual(weightLogScale('paper', 'map', logp, logq, T), columnLogScale('map', logp, logq, T));
    assert.deepEqual(weightLogScale(undefined, 'sample', logp, logq, T), columnLogScale('sample', logp, logq, T));
    for (const mode of ['map', 'sample']) {
      assert.deepEqual(weightLogScale('quality', mode, logp, logq, T), columnLogScale('map', logp, logq, T));
      assert.deepEqual(weightLogScale('importance', mode, logp, logq, T), columnLogScale('sample', logp, logq, T));
      assert.deepEqual(weightLogScale('unit', mode, logp, logq, T), [0, 0, 0, 0, 0]);
    }
  }
  assert.equal(resolveWeight('paper', 'map'), 'quality'); assert.equal(resolveWeight(null, 'sample'), 'importance'); assert.equal(resolveWeight('unit', 'map'), 'unit'); assert.equal(paperWeight('sample'), 'importance');
});

test('presetDiff lists the changed settings in reading order', () => {
  const a = { mode: 'map', T: 3, K: 3, tp: '1.5', len: 'quality' };
  assert.deepEqual(presetDiff(a, { ...a }), []);
  assert.deepEqual(presetDiff(a, { mode: 'sample', T: 0.82, K: 3, tp: 1.0, len: 'importance' }), ['mode', 'T', 'tp', 'len']);
  const w = { ...a, weight: 'quality', lock: false };
  assert.deepEqual(presetDiff(w, { ...w, lock: true, weight: 'importance', len: 'importance', K: 4 }), ['K', 'weight', 'len', 'lock']);
  assert.deepEqual(presetDiff(a, { ...a, K: 4, T: 3 * (1 + 1e-12), tp: 1.5 }), ['K']);
  assert.deepEqual(presetDiff(a, { ...a, len: 'unit', K: 5, T: 7.5 }), ['T', 'K', 'len']);
});

test('readoutNotes: low ESS and few distinct completions', () => {
  assert.deepEqual(readoutNotes({ ess: 169, K: 3, N: 200, distinct: 187 }), []);
  const n = readoutNotes({ ess: 1.2, K: 3, N: 200, distinct: 187 });
  assert.equal(n.length, 1); assert.match(n[0].text, /about 1 completions/);
  const d = readoutNotes({ ess: 50, K: 3, N: 200, distinct: 12 });
  assert.equal(d.length, 1); assert.match(d[0].text, /Only 12 distinct/);
  assert.equal(readoutNotes({ ess: 1, K: 5, N: 20, distinct: 3 }).length, 2);
  // pools of 30 completions with d = 1, 2, 3 distinct texts and K = 3
  for (const d of [1, 2, 3]) {
    const kinds = readoutNotes({ ess: 30, K: 3, N: 30, distinct: d }).map((x) => x.kind);
    assert.equal(kinds.includes('distinct'), d < 3, `d=${d}`);
    assert.equal(kinds.includes('one'), d === 3);
    assert.equal(effectiveK(3, d, 30), d);
  }
  assert.equal(effectiveK(10, 200, 30), 10);
  assert.equal(effectiveK(50, 200, 30), 29);
});

test('normText and distinctCount ignore whitespace differences', () => {
  assert.equal(normText('  Why did\n the  chicken\t'), 'Why did the chicken');
  const t = (x) => ({ text: x });
  assert.equal(distinctCount([t('A joke.'), t('A  joke. '), t('\nA joke.'), t('Another joke.'), t('   ')]), 3);
  assert.equal(distinctCount([]), 0);
});

test('readoutNotes: near-duplicates note when half the pool is one answer', () => {
  const n = readoutNotes({ ess: 100, K: 3, N: 200, distinct: 80, near: { size: 150 } });
  assert.deepEqual(n.map((x) => x.kind), ['near']); assert.match(n[0].text, /150 of 200/);
  assert.deepEqual(readoutNotes({ ess: 100, K: 3, N: 200, distinct: 80, near: { size: 60 } }), []);
  // not repeated when the few-distinct note is already shown
  assert.deepEqual(readoutNotes({ ess: 100, K: 3, N: 200, distinct: 12, near: { size: 190 } }).map((x) => x.kind), ['distinct']);
});

test('weightEss: unit is N, quality tends to N at large T, importance is smaller when q is uneven', () => {
  const logp = [-3, -5, -1, -9, -4, -7], logq = [-3.2, -4.1, -1.5, -8, -4.4, -6];
  const N = logp.length;
  for (const mode of ['map', 'sample']) {
    assert.ok(Math.abs(weightEss('unit', mode, logp, logq, 3) - N) < 1e-9);
    assert.ok(N - weightEss('quality', mode, logp, logq, 1e6) < 1e-3);
    assert.ok(weightEss('quality', mode, logp, logq, 0.2) < weightEss('quality', mode, logp, logq, 5));
  }
  // same weight, same ESS in both picking modes; Σw²: manual check for importance
  assert.equal(weightEss('importance', 'map', logp, logq, 2), weightEss('importance', 'sample', logp, logq, 2));
  const w = logp.map((lp, i) => Math.exp(lp / 2 - logq[i]));
  const s1 = w.reduce((a, b) => a + b, 0), s2 = w.reduce((a, b) => a + b * b, 0);
  assert.ok(Math.abs(weightEss('importance', 'sample', logp, logq, 2) - (s1 * s1) / s2) < 1e-9);
});

test('gramToVectors reproduces a Gram matrix', () => {
  const rnd = mulberry32(3), N = 12, D = 5;
  const phi = Array.from({ length: N }, () => { const v = Float32Array.from({ length: D }, () => rnd() - 0.5); const n = Math.hypot(...v); return v.map((x) => x / n); });
  const G = gram(phi), V = gramToVectors(G, N);
  assert.equal(V.length, N);
  assert.equal(V[0].length, D); // rank D: only the positive eigenvalues are kept
  const G2 = gram(V);
  for (let i = 0; i < N * N; i++) assert.ok(Math.abs(G[i] - G2[i]) < 1e-4, `entry ${i}`);
  for (const r of V) assert.ok(Math.abs(Math.hypot(...r) - 1) < 1e-5);
});

test('gramToVectors on an indefinite or non-unit Gram', () => {
  // Off-diagonals of 0.9 and -0.9 together are not positive semidefinite for 3 points.
  const G = Float64Array.from([1, 0.9, -0.9, 0.9, 1, 0.9, -0.9, 0.9, 1]);
  const V = gramToVectors(G, 3);
  for (const r of V) { assert.ok(r.every(Number.isFinite)); assert.ok(Math.abs(Math.hypot(...r) - 1) < 1e-5); }
  // identical rows stay identical
  const G1 = Float64Array.from([1, 1, 1, 1]);
  const W = gramToVectors(G1, 2);
  assert.ok(Math.abs(W[0][0] * W[1][0] - 1) < 1e-6);
});

test('queueText and newRequestId', () => {
  assert.equal(queueText({ state: 'waiting', ahead: 3 }), 'Waiting in line, 3 ahead.');
  assert.equal(queueText({ state: 'waiting', ahead: 1 }), 'Waiting in line, 1 ahead.');
  assert.equal(queueText({ state: 'waiting', ahead: 0 }), 'Waiting in line, next up.');
  assert.equal(queueText({ state: 'running', ahead: 0 }), null);
  assert.equal(queueText({ state: 'unknown', ahead: 0 }), null);
  assert.equal(queueText(null), null);
  const id = newRequestId();
  assert.match(id, /^[A-Za-z0-9_-]{1,64}$/);
  assert.notEqual(id, newRequestId());
});

test('decodeDescribe checks the group count and trims names', () => {
  assert.deepEqual(decodeDescribe({ names: [' a ', 3], model: 'm', ms: 5 }, 2), { names: ['a', ''], model: 'm', ms: 5 });
  assert.throws(() => decodeDescribe({ names: ['a'] }, 2));
  assert.throws(() => decodeDescribe(null, 1));
});

const STATUS = {
  models: [{ key: 'qwen3-14b', name: 'Qwen3 14B', repo: 'Qwen/Qwen3-14B-AWQ', state: 'ready' }],
  embedders: { minilm: 'sentence-transformers/all-MiniLM-L6-v2', 'qwen3-emb': 'Qwen/Qwen3-Embedding-0.6B', 'qwen3-emb-4b': 'Qwen/Qwen3-Embedding-4B', mpnet: 'sentence-transformers/all-mpnet-base-v2', own: null },
  ownModels: ['qwen3-14b'],
};

test('serverExtras: the live /status shape adds three embedders and no model', () => {
  const x = serverExtras(STATUS);
  assert.deepEqual(x.embedders.map((e) => [e.id, e.label]), [['minilm', 'all-MiniLM-L6-v2'], ['qwen3-emb-4b', 'Qwen3-Embedding-4B'], ['mpnet', 'all-mpnet-base-v2']]);
  assert.deepEqual(x.models, []);
});

test('serverExtras: models appear by themselves when the server reports more than one', () => {
  const st = { ...STATUS, models: [...STATUS.models,
    { key: 'qwen3-8b', name: 'Qwen3 8B', repo: 'Qwen/Qwen3-8B', state: 'cold' },
    { key: 'big', name: 'Big', repo: 'org/Big', state: 'unavailable' },
    { key: 'gemma', name: 'Gemma', repo: 'google/gemma', state: 'ready' }] };
  const x = serverExtras(st);
  assert.deepEqual(x.models.map((m) => [m.key, m.name, m.repo, m.state]), [['qwen3-8b', 'Qwen3 8B', 'Qwen/Qwen3-8B', 'cold'], ['gemma', 'Gemma', 'google/gemma', 'ready']]);
  // `own` only for models the server lists in ownModels; the default's `qwen3-emb` is included, the id itself is not duplicated
  assert.deepEqual(x.models[0].embedders, ['qwen3-emb', 'minilm', 'qwen3-emb-4b', 'mpnet']);
  assert.deepEqual(serverExtras({ ...st, ownModels: ['qwen3-14b', 'gemma'] }).models[1].embedders.at(-1), 'own');
});

test('serverExtras: missing, malformed or minimal status gives nothing', () => {
  for (const st of [null, undefined, {}, { embedders: ['minilm'] }, { models: 'x', embedders: null }, { model: 'm' }]) assert.deepEqual(serverExtras(st), { embedders: [], models: [] });
  assert.deepEqual(serverExtras({ models: [{ key: 'a' }, { key: 'qwen3-14b' }] }).models.map((m) => m.key), ['a']);
  assert.equal(repoLabel('a/b/c'), 'c');
});
