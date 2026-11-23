// Download the precomputed pools and embeddings that are not kept in git.
// Usage: node scripts/fetch-data.mjs [base URL of a deployed copy of the demo]
import { mkdir, writeFile, access } from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import { dirname } from 'node:path';

const base = (process.argv[2] ?? 'https://zueri-monstro.rufous-hamlet.ts.net/dpp-lm-demo').replace(/\/$/, '');
const cat = JSON.parse(readFileSync('data/models.json', 'utf8'));
const extra = cat.embedders.filter((e) => !e.repo || !e.default).map((e) => e.key).filter((k) => k !== 'minilm');
const embDirs = [...new Set([...cat.embedders.map((e) => e.key).filter((k) => k !== 'minilm'), ...extra])];

const files = [];
for (const [id, byModel] of Object.entries(cat.examples)) {
  for (const [model, tps] of Object.entries(byModel)) {
    for (const tp of tps) {
      files.push(`data/pools/${model}/${id}_T${tp}.json`);
      for (const e of embDirs) files.push(`data/emb/${e}/${model}/${id}_T${tp}.bin`);
    }
  }
}

let got = 0, kept = 0, failed = 0;
for (const f of files) {
  try { await access(f); kept++; continue; } catch {}
  const r = await fetch(`${base}/${f}`);
  if (!r.ok) { failed++; console.error(`${r.status} ${f}`); continue; }
  await mkdir(dirname(f), { recursive: true });
  await writeFile(f, Buffer.from(await r.arrayBuffer()));
  got++;
}
console.log(`${got} downloaded, ${kept} already present, ${failed} failed, of ${files.length}`);
