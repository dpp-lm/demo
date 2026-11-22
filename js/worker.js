// Live pool construction in a Web Worker: same code path as the gallery builder.
import * as T from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
import { loadLM, loadEmbedder, samplePool, embed, modelByKey, DEFAULT_EMBEDDER } from './lm.js';

let lm = null, lmKey = null, embedder = null, stop = false, device = null, f16 = false;

// WebGPU if available; float16 weights only when the GPU supports shader-f16.
async function pickDevice() {
  try {
    const a = await self.navigator?.gpu?.requestAdapter();
    if (a) { f16 = a.features.has('shader-f16'); return 'webgpu'; }
  } catch {}
  return 'wasm';
}

function progressCb(what) {
  const seen = new Map();
  return (e) => {
    if (e.status === 'progress' && e.total) {
      seen.set(e.file, [e.loaded, e.total]);
      let l = 0, t = 0;
      for (const [a, b] of seen.values()) { l += a; t += b; }
      postMessage({ type: 'status', text: `${what}… ${(l / 1e6).toFixed(0)} / ${(t / 1e6).toFixed(0)} MB` });
    }
  };
}

self.onmessage = async ({ data }) => {
  if (data.type === 'stop') { stop = true; return; }
  if (data.type !== 'run') return;
  stop = false;
  try {
    device ??= await pickDevice();
    const M = modelByKey(data.model);
    const dtype = device === 'webgpu' && f16 ? 'q4f16' : M.cpu;
    if (!dtype) throw new Error(`${M.name} needs a GPU with float16 support, which this browser does not offer. Pick a smaller model.`);
    if (!lm || lmKey !== M.key) {
      if (lm) { try { await lm.model.dispose(); } catch {} lm = null; }
      // transformers.js reports progress for cache reads too, so say which one this is.
      let fromCache = false;
      try { fromCache = !!(await (await caches.open('transformers-cache')).match(`https://huggingface.co/${M.repo}/resolve/main/onnx/model_${dtype}.onnx`)); } catch {}
      const verb = fromCache ? `Loading ${M.name} from this browser` : `Downloading ${M.name}`;
      postMessage({ type: 'status', text: `${verb}…` });
      lm = await loadLM(T, { model: M.repo, device, dtype, extra: dtype === M.cpu ? M.cpuExtra ?? {} : {}, progress_callback: progressCb(verb) });
      lmKey = M.key;
    }
    if (!embedder) {
      postMessage({ type: 'status', text: 'Loading the embedder…' });
      embedder = await loadEmbedder(T, { model: DEFAULT_EMBEDDER, device: 'wasm', dtype: 'fp32', progress_callback: progressCb('Loading the embedder') });
    }
    const t0 = performance.now();
    postMessage({ type: 'status', text: `Sampling 0/${data.N}…` });
    const pool = await samplePool(T, lm, {
      prompt: data.prompt, chat: M.chat ?? {}, N: data.N, Tprime: data.Tprime, maxNewTokens: data.maxNewTokens,
      batchSize: device === 'webgpu' ? 10 : 5,
      shouldStop: () => stop,
      onProgress: (n, N, last) => {
        const s = ((performance.now() - t0) / 1000).toFixed(0);
        postMessage({ type: 'progress', n, N, last: last.map((x) => x.text), text: `Sampling ${n}/${N}… (${s}s)` });
      },
    });
    if (pool.length < 2) throw new Error('Stopped before enough samples were drawn.');
    postMessage({ type: 'status', text: `Embedding ${pool.length} strings…` });
    const phi = await embed(embedder, pool.map((s) => s.text));
    const D = phi[0].length;
    const flat = new Float32Array(pool.length * D);
    phi.forEach((v, i) => flat.set(v, i * D));
    postMessage({
      type: 'done', device,
      pool: {
        prompt: data.prompt, model: M.repo, modelName: M.name, embedder: DEFAULT_EMBEDDER, Tprime: data.Tprime,
        maxNewTokens: data.maxNewTokens, N: pool.length, D,
        samples: pool.map(({ text, logp, logq, nTokens, ended }) => ({ text, logp, logq, nTokens, ended })),
      },
      emb: flat,
    }, [flat.buffer]);
  } catch (err) {
    postMessage({ type: 'error', text: String(err?.message ?? err) });
  }
};
