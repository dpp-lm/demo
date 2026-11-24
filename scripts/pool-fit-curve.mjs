// Pool fit θ̂ vs T for every precomputed pool. Usage: node scripts/pool-fit-curve.mjs K out.json; then python3 scripts/pool-fit-plot.py out.json docs/figures/pool-fit-vs-T.png
// θ̂(T) exactly as the page computes it (js/main.js recompute), for every precomputed pool.
import { readFileSync, writeFileSync } from 'node:fs';
import { kdpp } from '../js/dpp.js';
const dir = new URL('../data/cache/', import.meta.url).pathname;
const idx = JSON.parse(readFileSync(dir + 'index.json'));
const K = +(process.argv[2] ?? 3);
const Ts = [...Array(41).keys()].map((i) => 10 ** (-1.3 + i * (4.3 / 40))).concat([Infinity]);   // 0.05 … 1000, ∞
const out = [];
for (const b of idx.buffers) {
  const pool = JSON.parse(readFileSync(dir + b.pool)); const samples = pool.samples ?? pool;
  const N = samples.length;
  const raw = new Float32Array(readFileSync(dir + b.G['Qwen/Qwen3-Embedding-0.6B']).buffer.slice(0));
  const M = Math.round(Math.sqrt(raw.length));
  // same rows for identical strings (as the page does)
  const first = new Map(), c = samples.map((s, i) => (first.has(s.text) ? first.get(s.text) : (first.set(s.text, i), i)));
  const G = new Float64Array(N * N); for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) G[i * N + j] = raw[c[i] * M + c[j]];
  const logp = samples.map((s) => s.logp), logq = samples.map((s) => s.logq);
  for (const target of ['global', 'gl']) {
    const fit = Ts.map((T) => {
      const lk = logp.map((lp, i) => (T === Infinity ? 0 : lp / T) + (target === 'gl' ? logq[i] : 0));
      const lw = lk.map((x, i) => x - logq[i]);
      const pi = kdpp(G, N, lw, K).pi;
      let s2 = 0; for (const p of pi) s2 += (p / K) ** 2;
      return (K * K * N * s2) / (N - K + 1);
    });
    out.push({ id: b.pool.split('_')[0], Tprime: b.Tprime, target, N, fit });
  }
}
writeFileSync(process.argv[3], JSON.stringify({ K, Ts: Ts.map((t) => (t === Infinity ? null : t)), rows: out }));
console.log(out.length, 'curves');
