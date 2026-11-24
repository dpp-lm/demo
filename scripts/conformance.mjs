// Check a backend against docs/BACKEND.md.
//   node scripts/conformance.mjs <base-url> [--token <hf token>] [--gradio] [--no-sample]
// Prints one line per check and exits non-zero if any check fails. A sample run costs GPU time
// (n = 4, 16 tokens); --no-sample checks only /status, the limits and the error format.
const args = process.argv.slice(2);
const url = args.find((a) => !a.startsWith('--'))?.replace(/\/$/, '');
const opt = (k) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : null; };
const token = opt('--token') || process.env.HF_TOKEN || null;
const gradio = args.includes('--gradio'), noSample = args.includes('--no-sample');
if (!url) { console.error('usage: node scripts/conformance.mjs <base-url> [--token T] [--gradio] [--no-sample]'); process.exit(2); }

let failed = 0;
const check = (name, ok, extra = '') => {
  extra = String(extra).slice(0, 160); console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${extra ? ` — ${extra}` : ''}`); if (!ok) failed++; };
const headers = { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) };

async function http(method, path, body) {
  const r = await fetch(url + path, { method, headers, body: body ? JSON.stringify(body) : undefined });
  let json = null; try { json = await r.json(); } catch {}
  return { status: r.status, json, cors: r.headers.get('access-control-allow-origin') };
}
// Gradio's queue protocol (its /call endpoint drops error messages, so it cannot check error codes).
async function gradioCall(api, data) {
  const cfg = await (await fetch(`${url}/config`, { headers })).json();
  const fn = cfg.dependencies.findIndex((d) => d.api_name === api);
  if (fn < 0) return { error: `no "${api}" function` };
  const session = Math.random().toString(36).slice(2);
  const j = await fetch(`${url}/gradio_api/queue/join`, { method: 'POST', headers, body: JSON.stringify({ data, fn_index: fn, trigger_id: cfg.dependencies[fn].id ?? null, session_hash: session, event_data: null }) });
  if (!j.ok) return { error: `HTTP ${j.status}` };
  const text = await (await fetch(`${url}/gradio_api/queue/data?session_hash=${session}`, { headers })).text();
  for (const line of text.split('\n')) {
    if (!line.startsWith('data:')) continue;
    const m = JSON.parse(line.slice(5));
    if (m.msg !== 'process_completed') continue;
    return m.success ? { json: m.output.data[0] } : { error: String(m.output?.error ?? '') };
  }
  return { error: 'no answer' };
}

function checkG(name, out, n) {
  if (!out.G) { check(`${name}: has G`, !!out.emb, out.emb ? 'sends emb + D instead (allowed for now)' : 'no G'); return; }
  const f = new Float32Array(Uint8Array.from(Buffer.from(out.G, 'base64')).buffer);
  check(`${name}: G is ${n}×${n}`, f.length === n * n, `${f.length} values`);
  let sym = 0, diag = 0, range = true;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const g = f[i * n + j];
    if (i === j) diag = Math.max(diag, Math.abs(g - 1)); else sym = Math.max(sym, Math.abs(g - f[j * n + i]));
    if (!(g >= -1.0001 && g <= 1.0001)) range = false;
  }
  check(`${name}: G symmetric, diagonal 1, entries in [−1, 1]`, sym === 0 && diag === 0 && range, `asym ${sym}, diag ${diag}`);
}
function checkSample(out, n, maxTok, tp) {
  check('sample: N, maxNewTokens, Tprime as requested', out.N === n && out.maxNewTokens === maxTok && Math.abs(out.Tprime - tp) < 1e-9, `${out.N}, ${out.maxNewTokens}, ${out.Tprime}`);
  check('sample: model is Qwen/Qwen3-14B-AWQ', out.model === 'Qwen/Qwen3-14B-AWQ', out.model);
  const s = out.samples ?? [];
  check('sample: n samples with text, logp, logq, nTokens, ended', s.length === n && s.every((x) => typeof x.text === 'string' && Number.isFinite(x.logp) && Number.isFinite(x.logq) && Number.isInteger(x.nTokens) && typeof x.ended === 'boolean'));
  check('sample: logp, logq ≤ 0; nTokens ≤ max', s.every((x) => x.logp <= 1e-9 && x.logq <= 1e-9 && x.nTokens <= maxTok && x.nTokens >= 1));
  check('sample: unfinished completions used all tokens', s.every((x) => x.ended || x.nTokens === maxTok));
  checkG('sample', out, n);
}

if (gradio) {
  const bad = await gradioCall('sample', ['hi', 'qwen3-14b', 201, 1.0, 8, 0]);
  check('limits: n = 201 refused with bad_request', /^bad_request: /.test(bad.error ?? ''), bad.error);
  if (!noSample) {
    const r = await gradioCall('sample', ['Name a colour.', 'qwen3-14b', 4, 1.0, 16, 1]);
    check('sample: answered', !!r.json, r.error);
    if (r.json) {
      checkSample(r.json, 4, 16, 1.0);
      const e = await gradioCall('embed', [r.json.samples.map((x) => x.text), 'qwen3-emb']);
      check('embed: answered', !!e.json, e.error);
      if (e.json) {
        checkG('embed', e.json, 4);
        if (e.json.G && r.json.G) check('embed: same G as sample', e.json.G === r.json.G || maxDiff(e.json.G, r.json.G) < 1e-5, `max diff ${maxDiff(e.json.G, r.json.G)}`);
      }
    }
  }
} else {
  const st = await http('GET', '/status');
  check('status: 200', st.status === 200, `HTTP ${st.status}`);
  const j = st.json ?? {};
  check('status: version 1', j.version === 1, String(j.version));
  check('status: limits 200 / 256', j.limits?.maxN === 200 && j.limits?.maxNewTokens === 256, JSON.stringify(j.limits));
  check('status: offers qwen3-emb', !!j.embedders?.['qwen3-emb'], JSON.stringify(j.embedders));
  const bad = await http('POST', '/sample', { prompt: 'hi', n: 201, t_prime: 1, max_new_tokens: 8 });
  check('limits: n = 201 refused with 400 bad_request', bad.status === 400 && bad.json?.code === 'bad_request', `HTTP ${bad.status} ${JSON.stringify(bad.json)}`);
  const tok = await http('POST', '/sample', { prompt: 'hi', n: 2, t_prime: 1, max_new_tokens: 300 });
  check('limits: max_new_tokens = 300 refused with 400 bad_request', tok.status === 400 && tok.json?.code === 'bad_request', `HTTP ${tok.status}`);
  if (!noSample) {
    const r = await http('POST', '/sample', { prompt: 'Name a colour.', n: 4, t_prime: 1.0, max_new_tokens: 16, seed: 1, model: 'qwen3-14b', embedder: 'qwen3-emb' });
    check('sample: 200', r.status === 200, `HTTP ${r.status} ${r.status !== 200 ? JSON.stringify(r.json) : ''}`);
    if (r.status === 200) {
      checkSample(r.json, 4, 16, 1.0);
      const e = await http('POST', '/embed', { texts: r.json.samples.map((x) => x.text), embedder: 'qwen3-emb' });
      check('embed: 200', e.status === 200, `HTTP ${e.status}`);
      if (e.status === 200) {
        checkG('embed', e.json, 4);
        if (e.json.G && r.json.G) check('embed: same G as sample', maxDiff(e.json.G, r.json.G) < 1e-5, `max diff ${maxDiff(e.json.G, r.json.G)}`);
      }
    }
  }
}
function maxDiff(a, b) {
  const f = (s) => new Float32Array(Uint8Array.from(Buffer.from(s, 'base64')).buffer);
  const x = f(a), y = f(b); let m = 0;
  for (let i = 0; i < x.length; i++) m = Math.max(m, Math.abs(x[i] - y[i]));
  return m;
}
console.log(failed ? `\n${failed} check(s) failed` : '\nall checks passed');
process.exit(failed ? 1 : 0);
