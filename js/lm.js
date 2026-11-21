// Pool construction (paper §2, §4).
//
// Draws N strings i.i.d. from the locally annealed proposal q = p°_{T'} (Eq. 1–2)
// and records, for every string, the exact log p(y) and log q(y), including the
// end-of-string factor. These two numbers are all the importance weights
// w = κ/q = p^{1/T} / p°_{T'} (Eq. 6, 8) need.
//
// The sampling happens inside our own logits processor: it sees the model's raw
// logits, samples a token from softmax(logits / T'), records both log-probs, and
// then masks every other token, so the library's greedy step simply picks it.
// This keeps library defaults (top-k, top-p, repetition penalty, ...) from silently
// changing the proposal. The module takes the transformers.js namespace as an
// argument so the same code runs in the browser worker and in Node.

export const DEFAULT_LM = 'onnx-community/Qwen2.5-0.5B-Instruct';
export const DEFAULT_EMBEDDER = 'Xenova/all-MiniLM-L6-v2';

function logSumExp(x, scale) {
  let m = -Infinity;
  for (let i = 0; i < x.length; i++) { const v = x[i] * scale; if (v > m) m = v; }
  let s = 0;
  for (let i = 0; i < x.length; i++) s += Math.exp(x[i] * scale - m);
  return m + Math.log(s);
}

// Sample index from softmax(x * scale) by inverse CDF.
function sampleSoftmax(x, scale, lse, u) {
  let acc = 0;
  for (let i = 0; i < x.length; i++) {
    acc += Math.exp(x[i] * scale - lse);
    if (u < acc) return i;
  }
  // Numerical slack: return the argmax.
  let best = 0;
  for (let i = 1; i < x.length; i++) if (x[i] > x[best]) best = i;
  return best;
}

export function eosIds(model, tokenizer) {
  let ids = model.generation_config?.eos_token_id ?? model.config?.eos_token_id ?? tokenizer.eos_token_id;
  if (!Array.isArray(ids)) ids = [ids];
  return new Set(ids.filter((v) => v !== null && v !== undefined).map(Number));
}

export async function loadLM(T, { model = DEFAULT_LM, device, dtype, progress_callback } = {}) {
  const tokenizer = await T.AutoTokenizer.from_pretrained(model, { progress_callback });
  const lm = await T.AutoModelForCausalLM.from_pretrained(model, { device, dtype, progress_callback });
  return { tokenizer, model: lm, name: model };
}

export async function loadEmbedder(T, { model = DEFAULT_EMBEDDER, device, dtype, progress_callback } = {}) {
  const pipe = await T.pipeline('feature-extraction', model, { device, dtype, progress_callback });
  return { pipe, name: model };
}

// Unit-norm embeddings φ(y) (paper footnote 5), as an array of Float32Array.
export async function embed(embedder, texts, batchSize = 16) {
  const out = [];
  for (let i = 0; i < texts.length; i += batchSize) {
    const chunk = texts.slice(i, i + batchSize).map((t) => (t.trim().length ? t : '(empty)'));
    const t = await embedder.pipe(chunk, { pooling: 'mean', normalize: true });
    const [b, d] = t.dims;
    for (let j = 0; j < b; j++) out.push(Float32Array.from(t.data.subarray(j * d, (j + 1) * d)));
  }
  return out;
}

export function chatPrompt(tokenizer, prompt, system) {
  const messages = [];
  if (system) messages.push({ role: 'system', content: system });
  messages.push({ role: 'user', content: prompt });
  return tokenizer.apply_chat_template(messages, { tokenize: false, add_generation_prompt: true });
}

/**
 * Draw a pool of N completions from q = p°_{T'}.
 * Returns [{ text, logp, logq, nTokens, ended }], where ended=false means the
 * string hit maxNewTokens (then logp/logq are prefix log-probs; see note in UI).
 */
export async function samplePool(T, { tokenizer, model }, {
  prompt, system = null, N = 50, Tprime = 1.0, maxNewTokens = 64, batchSize = 10,
  rng = Math.random, onProgress = () => {}, shouldStop = () => false,
}) {
  const { LogitsProcessor, LogitsProcessorList } = T;
  const eos = eosIds(model, tokenizer);
  const eosList = [...eos];
  const text = chatPrompt(tokenizer, prompt, system);
  const promptIds = tokenizer(text, { add_special_tokens: false });
  const promptLen = promptIds.input_ids.dims.at(-1);
  const pool = [];
  const invT = 1 / Tprime;

  class ProposalSampler extends LogitsProcessor {
    constructor(B) {
      super();
      this.logp = new Float64Array(B);
      this.logq = new Float64Array(B);
      this.nTok = new Int32Array(B);
      this.done = new Uint8Array(B);
      this.step = 0;
    }
    _call(input_ids, logits) {
      const [B, V] = logits.dims;
      const data = logits.data;
      for (let b = 0; b < B; b++) {
        const row = data.subarray(b * V, (b + 1) * V);
        let chosen;
        if (this.done[b]) {
          chosen = eosList[0];
        } else {
          const lse1 = logSumExp(row, 1);
          const lseT = logSumExp(row, invT);
          chosen = sampleSoftmax(row, invT, lseT, rng());
          this.logp[b] += row[chosen] - lse1;
          this.logq[b] += row[chosen] * invT - lseT;
          this.nTok[b] += 1;
          if (eos.has(chosen)) this.done[b] = 1;
        }
        row.fill(-Infinity);
        row[chosen] = 0;
      }
      this.step++;
      return logits;
    }
  }

  while (pool.length < N && !shouldStop()) {
    const B = Math.min(batchSize, N - pool.length);
    const enc = tokenizer(Array(B).fill(text), { add_special_tokens: false });
    const proc = new ProposalSampler(B);
    const list = new LogitsProcessorList();
    list.push(proc);
    const out = await model.generate({
      ...enc,
      max_new_tokens: maxNewTokens,
      do_sample: false,
      num_beams: 1,
      temperature: 1.0,
      top_k: 0,
      top_p: 1.0,
      repetition_penalty: 1.0,
      no_repeat_ngram_size: 0,
      min_length: 0,
      min_new_tokens: null,
      bad_words_ids: null,
      forced_bos_token_id: null,
      forced_eos_token_id: null,
      begin_suppress_tokens: null,
      logits_processor: list,
    });
    const seqs = out.tolist();
    for (let b = 0; b < B; b++) {
      const gen = seqs[b].slice(promptLen).map(Number);
      const cut = gen.findIndex((t) => eos.has(t));
      const ids = cut >= 0 ? gen.slice(0, cut) : gen;
      pool.push({
        text: tokenizer.decode(ids, { skip_special_tokens: true }).trim(),
        logp: proc.logp[b],
        logq: proc.logq[b],
        nTokens: proc.nTok[b],
        ids: cut >= 0 ? gen.slice(0, cut + 1) : gen,
        ended: !!proc.done[b],
      });
    }
    onProgress(pool.length, N, pool.slice(-B));
  }
  return pool;
}
