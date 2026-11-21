// Slow integration check (downloads ~0.5GB): recorded log p / log q must match
// an independent teacher-forced forward pass. Run: node tests/check-lm.mjs
import * as T from '@huggingface/transformers';
import { loadLM, samplePool, chatPrompt, eosIds, modelByKey } from '../js/lm.js';

const M = modelByKey(process.argv[4] ?? 'qwen2.5-0.5b');
const dt = process.argv[3] ?? 'q4';
const lm = await loadLM(T, { model: M.repo, dtype: dt, extra: dt === M.cpu ? M.cpuExtra ?? {} : {} });
console.log('model', M.name);
const Tp = Number(process.argv[2] ?? 1.0);
const t0 = Date.now();
const pool = await samplePool(T, lm, { prompt: (process.argv[5] ?? 'Name a fruit and say one thing about it.'), chat: M.chat ?? {}, N: 4, Tprime: Tp, maxNewTokens: 24, batchSize: 4 });
console.log('gen secs', (Date.now() - t0) / 1000);
const eos = [...eosIds(lm.model, lm.tokenizer)];
const text = chatPrompt(lm.tokenizer, (process.argv[5] ?? 'Name a fruit and say one thing about it.'), null, M.chat ?? {});
let worst = 0;
for (const s of pool) {
  const pIds = lm.tokenizer(text, { add_special_tokens: false }).input_ids.tolist()[0].map(Number);
  const tgt = s.ids;
  const ids = [...pIds, ...tgt];
  const input_ids = new T.Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
  const attention_mask = new T.Tensor('int64', BigInt64Array.from(ids.map(() => 1n)), [1, ids.length]);
  const { logits } = await lm.model({ input_ids, attention_mask });
  const V = logits.dims[2];
  let lp = 0, lq = 0;
  for (let t = 0; t < tgt.length; t++) {
    const row = logits.data.subarray((pIds.length - 1 + t) * V, (pIds.length + t) * V);
    const lse = (sc) => { let m = -Infinity; for (const v of row) m = Math.max(m, v * sc); let z = 0; for (const v of row) z += Math.exp(v * sc - m); return m + Math.log(z); };
    lp += row[tgt[t]] - lse(1); lq += row[tgt[t]] / Tp - lse(1 / Tp);
  }
  const d = Math.max(Math.abs(lp - s.logp), Math.abs(lq - s.logq));
  if (tgt.length === s.nTokens) worst = Math.max(worst, d);
  console.log(JSON.stringify(s.text.slice(0, 60)), 'ended', s.ended, 'ntok', s.nTokens, tgt.length, 'logp', s.logp.toFixed(3), lp.toFixed(3), 'logq', s.logq.toFixed(3), lq.toFixed(3));
}
console.log('max abs diff (round-tripped strings):', worst);
