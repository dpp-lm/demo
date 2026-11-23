// Precompute gallery pools with exactly the code path the browser uses (js/lm.js).
// Usage: NODE_USE_ENV_PROXY=1 node scripts/build-gallery.mjs [N] [maxNewTokens] [id,id,...|all] [model key]
import * as T from '@huggingface/transformers';
import { writeFileSync, mkdirSync, existsSync, readFileSync } from 'node:fs';
import { loadLM, loadEmbedder, samplePool, embed, modelByKey, DEFAULT_EMBEDDER } from '../js/lm.js';
import { mulberry32 } from '../js/dpp.js';

const N = Number(process.argv[2] ?? 100);
const maxNewTokens = Number(process.argv[3] ?? 48);
const only = process.argv[4] && process.argv[4] !== 'all' ? new Set(process.argv[4].split(',')) : null;
const M = modelByKey(process.argv[5] ?? 'qwen3-1.7b');
const Tprime = 1.0;
const DTYPE = M.cpu; // 4-bit weights, float32 activations: log p is exact (tests/check-lm.mjs)

export const PROMPTS = [
  { id: 'sort', prompt: 'Implement a sorting algorithm from scratch in Python.' },
  { id: 'story', prompt: 'Write the opening sentence of a short story. Reply with one sentence only.' },
  { id: 'bike', prompt: 'Give one creative use for an old bicycle, in one sentence.' },
  { id: 'bank', prompt: 'What does "bank" mean? Answer in one sentence.' },
  { id: 'language', prompt: 'What is the best way to learn a new language? Answer in one sentence.' },
  { id: 'joke', prompt: 'Tell me a joke.' },
];

const out = new URL('../data/', import.meta.url);
mkdirSync(out, { recursive: true });
const lm = await loadLM(T, { model: M.repo, dtype: DTYPE });
const embedder = await loadEmbedder(T, { model: DEFAULT_EMBEDDER, dtype: 'fp32' });
const index = [];
for (const [k, p] of PROMPTS.entries()) {
  if (only && !only.has(p.id)) {
    const f = new URL(`${p.id}.json`, out);
    if (existsSync(f)) { const r = JSON.parse(readFileSync(f)); index.push({ id: p.id, prompt: r.prompt, N: r.N, Tprime: r.Tprime, model: r.model }); }
    continue;
  }
  const t0 = Date.now();
  const pool = await samplePool(T, lm, {
    prompt: p.prompt, chat: M.chat ?? {}, N, Tprime, maxNewTokens, batchSize: 10, rng: mulberry32(1000 + k),
    onProgress: (n) => process.stdout.write(`\r${p.id}: ${n}/${N}`),
  });
  const phi = await embed(embedder, pool.map((s) => s.text));
  const D = phi[0].length;
  const flat = new Float32Array(N * D);
  phi.forEach((v, i) => flat.set(v, i * D));
  const rec = {
    id: p.id, prompt: p.prompt, model: M.repo, modelName: M.name, dtype: DTYPE, embedder: DEFAULT_EMBEDDER,
    Tprime, maxNewTokens, N, D,
    samples: pool.map(({ text, logp, logq, nTokens, ended }) => ({ text, logp: +logp.toFixed(5), logq: +logq.toFixed(5), nTokens, ended })),
    emb: Buffer.from(flat.buffer).toString('base64'),
  };
  writeFileSync(new URL(`${p.id}.json`, out), JSON.stringify(rec));
  const ended = pool.filter((s) => s.ended).length;
  index.push({ id: p.id, prompt: p.prompt, N, Tprime, model: M.repo });
  console.log(`\r${p.id}: ${N} samples, ${ended} ended, ${((Date.now() - t0) / 1000).toFixed(0)}s`);
}
writeFileSync(new URL('index.json', out), JSON.stringify(index, null, 1));
