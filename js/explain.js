// Small "?" buttons that open a plain-language explainer with the math typeset.
// Any element with class "xbtn" and data-x="<key>" opens EXPLAIN[key].

export const EXPLAIN = {
  T: {
    title: 'Temperature T',
    body: `<p>Each completion gets a quality score $\\kappa(y)$ from the model's probability $p(y)$ of the whole completion, sharpened or flattened by $T$ (see <b>Target quality</b> in the settings: by default $\\kappa(y) = p(y)^{1/T}$).</p>
      <ul>
        <li><b>Small $T$</b> exaggerates differences in likelihood, so the most likely completions win (quality).</li>
        <li><b>Large $T$</b> flattens them; at the right end, $T = \\infty$, they count equally and only how different the completions are matters (diversity).</li>
      </ul>
      <p>One $T$ serves both <b>Most likely</b> and <b>Sample</b>. Sample is reliable only when the completions represent the target well; the <b>pool fit</b> under the plot says how well. $T$ is unrelated to the sampling temperature $T'$ ($T'$ tempers each token as it is drawn, $T$ the probability of the whole completion), so matching them means nothing. With the default target the fit is good only near $T = 1$, and only for completions drawn at $T' = 1$.</p>`,
  },
  subset: {
    title: 'How the subset is chosen',
    body: `<p>Think of each completion as an arrow. Its <b>direction</b> is its meaning (the embedding $\\phi(y)$) and its <b>length</b> is $\\sqrt{\\kappa(y)}$. A set $S$ of $K$ completions scores the squared volume its arrows span:
      $$\\det \\mathbf L_S = \\Big(\\prod_{y\\in S}\\kappa(y)\\Big)\\,\\det\\!\\big(\\Phi_S^\\top \\Phi_S\\big).$$
      The volume is large when the arrows are long (likely completions) and point different ways (different meanings), and zero when two are the same. This is a determinantal point process (DPP).</p>
      <p><b>Most likely</b> picks the set with the largest volume among the completions, adding one at a time. Exact search is NP-hard; the greedy answer is within a factor $(K!)^2$ of the best.</p>
      <p><b>Sample</b> draws a random set from the DPP over all strings, approximated with the completions: each is weighted by $\\kappa(y)/p^{\\circ}_{T'}(y)$ to undo how it was drawn. Press Sample again for another draw. The plot draws the same arrow lengths in both modes, so switching changes which points are selected, not how long the arrows are.</p>`,
  },
  volume: {
    title: 'Spanned volume',
    body: `<p>The grey box is spanned by the selected arrows. Its squared volume is $\\det \\mathbf L_S$, the subset's score: long arrows (likely completions) pointing in different directions (different meanings) make a big box; near-duplicates flatten it.</p>
      <p>The box is drawn only when the plot's axes contain the selected arrows ($K \\le 2$ in 2-D, $K \\le 3$ in 3-D), so it is exactly the volume being scored, not a projection of it. The axes stay put as the selection changes; <i>Fit axes to selection</i> turns them to the current one.</p>`,
  },
  logp: {
    title: 'log P(S)',
    body: `<p>The probability of exactly this set under the DPP over the completions:
      $$P(S) = \\frac{\\det \\mathbf L_S}{Z}, \\qquad Z = \\sum_{|S'| = K} \\det \\mathbf L_{S'},$$
      where $\\mathbf L_S$ is the kernel restricted to $S$ and the sum is over all $K$-element subsets $S'$ of the completions. With <b>Most likely</b>, it is the value of the objective that was maximized; with <b>Sample</b>, the probability of the set that was drawn (using the importance-weighted kernel).</p>`,
  },
  quality: {
    title: 'Quality',
    body: `<p>The average $\\log p(y)$ over the subset: how likely the model finds these completions. Closer to 0 is more likely.</p>`,
  },
  diversity: {
    title: 'Diversity',
    body: `<p>How different the selected completions are: the log of the squared volume their unit-length embeddings span,
      $$\\log\\det\\big(\\Phi_S^\\top\\Phi_S\\big).$$</p>
      <p>It is at most 0, reached when the completions point in completely different directions, and goes to $-\\infty$ as two of them become duplicates.</p>`,
  },
  fit: {
    title: 'Pool fit',
    body: `<p>How well these $N$ completions can stand in for all strings when aiming at the target at this $T$. The paper's Theorem 5.2 bounds how far the sets <b>Sample</b> draws are from the target:
      $$\\mathrm{KL} \\le \\theta = \\frac{K^2 \\rho}{N - K + 1},$$
      $$\\rho = \\mathbb E_{y \\sim p^{\\circ}_{T'}}\\Big[\\Big(\\frac{\\pi_1(y)}{p^{\\circ}_{T'}(y)}\\Big)^2\\Big],$$
      where $\\pi_1(y)$ is the probability that a set drawn from the target contains $y$, divided by $K$. The page estimates $\\rho$ from the completions' inclusion probabilities and shows $\\theta$ in nats.</p>
      <p>It is an <b>optimistic</b> estimate: it can only see completions the pool contains. Large values mean the pool is a poor match: Sample may be far from its target, and Most likely may miss sets the target favours. Moving $T$ toward $T'$, or sampling at a $T'$ closer to $T$, helps.</p>`,
  },
  inclusion: {
    title: 'Inclusion probability',
    body: `<p>The probability that a set drawn from the DPP over the completions contains this completion:
      $$\\Pr(i \\in S) = \\sum_{S \\ni i,\\ |S| = K} P(S),$$
      summed over the $K$-element subsets that contain completion $i$. Copies of the same text can never be in a set together, so for a repeated completion the probabilities of its copies are added. They are shown for the kernel of the current mode, like log P(S).</p>`,
  },
  target: {
    title: 'Target quality',
    body: `<p>What the subset should favour, besides diversity: the quality $\\kappa(y)$ of each completion.</p>
      <ul>
        <li><b>Global (paper):</b> $\\kappa(y) = p(y)^{1/T}$. Strings that are likely as a whole, sharpened or flattened by $T$.</li>
        <li><b>Global and local:</b> $\\kappa(y) = p(y)^{1/T}\\, p^{\\circ}_{T'}(y)$. Strings that are likely both as a whole and token by token at $T'$. Because the completions are generated token by token at $T'$, they represent this target especially well.</li>
      </ul>
      <p>At $T = \\infty$ the global factor is flat, so Global and local becomes the local probability alone.</p>`,
  },
  model: {
    title: 'Model',
    body: `<p>The language model that writes the completions, and the machine it runs on. All three choices are the same model, <b>Qwen3 14B</b> with its weights stored in 4 bits (AWQ), so they give the same probabilities; they differ only in where they run.</p>
      <p>The <b>GPU server</b> is shared by everyone using this demo. <b>Modal</b> and the <b>Hugging Face Space</b> start a GPU on demand and need a Hugging Face sign-in.</p>`,
  },
  embedder: {
    title: 'Embedding',
    body: `<p>Turns each completion $y$ into a unit-length vector $\\phi(y)$, its direction in the plot. Completions whose vectors point the same way count as similar, and the subset is chosen to span a large volume.</p>
      <p><b>Qwen3-Embedding</b> is a model trained for this: texts with similar meaning get nearby vectors. <b>Qwen3 14B’s own states</b> uses the language model itself: its last hidden state, averaged over the completion’s tokens, shifted by a fixed mean and scaled to unit length.</p>`,
  },
  N: {
    title: 'Samples N',
    body: `<p>How many completions are drawn from the model. The subset of $K$ is then chosen from these $N$.</p>
      <p>More samples give the choice more to work with and make <b>Sample</b> follow the intended distribution more closely, but take longer to generate.</p>`,
  },
  maxtok: {
    title: 'Max tokens',
    body: `<p>The longest a completion may be, in tokens (pieces of words). A completion that reaches the limit is cut off; it is marked with “…”.</p>
      <p>For a cut-off completion only the probability of its beginning is known, so its quality score is less accurate. Raise the limit if many completions are cut off.</p>`,
  },
  Tprime: {
    title: 'Sampling temperature T′',
    body: `<p>Controls how the completions are generated: every next-token distribution is raised to the power $1/T'$ and renormalized. $T' = 1$ samples the model as it is. Higher $T'$ gives more varied but less coherent text, which is why diversity comes from choosing the subset afterwards instead.</p>`,
  },
  pca: {
    title: 'Spectrum',
    body: `<p>The completions' embeddings have hundreds of dimensions. The bars show how much of their spread lies along each of the first principal axes (uncentred, so arrows start at the origin, where volume is measured from). A pool whose spread sits in very few axes has little room for diverse subsets.</p>
      <p>If the plot was first drawn with $K$ no more than its dimensions, or after <i>Fit axes to selection</i>, its first axes are fitted to the selected completions instead; otherwise it shows the first principal axes.</p>`,
  },
};

let openBtn = null;

function typeset(el) {
  window.renderMathInElement?.(el, {
    delimiters: [{ left: '$$', right: '$$', display: true }, { left: '$', right: '$', display: false }],
    throwOnError: false,
  });
}

export function closeExplainer() {
  const pop = document.getElementById('xpop');
  pop.hidden = true;
  openBtn?.setAttribute('aria-expanded', 'false');
  openBtn = null;
}

function openExplainer(btn) {
  const item = EXPLAIN[btn.dataset.x];
  if (!item) return;
  const pop = document.getElementById('xpop');
  pop.innerHTML = `<div class="xp-title">${item.title}</div>${item.body}<button type="button" class="xp-close" aria-label="Close">×</button>`;
  typeset(pop);
  pop.querySelector('.xp-close').onclick = closeExplainer;
  pop.hidden = false;
  openBtn?.setAttribute('aria-expanded', 'false');
  openBtn = btn;
  btn.setAttribute('aria-expanded', 'true');
  // Place below the button (above if there is no room), clamped to the viewport.
  const r = btn.getBoundingClientRect();
  const w = Math.min(360, window.innerWidth - 24);
  pop.style.width = w + 'px';
  const left = Math.max(12, Math.min(window.innerWidth - w - 12, r.left + r.width / 2 - w / 2));
  const h = pop.offsetHeight;
  let top = r.bottom + 8;
  if (top + h > window.innerHeight - 8 && r.top - h - 8 > 8) top = r.top - h - 8;
  pop.style.left = left + window.scrollX + 'px';
  pop.style.top = Math.max(8, top) + window.scrollY + 'px';
}

export function initExplainers() {
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.xbtn');
    if (btn) {
      e.preventDefault(); // don't toggle the checkbox or focus the input of a surrounding <label>
      e.stopPropagation();
      if (openBtn === btn) closeExplainer(); else openExplainer(btn);
      return;
    }
    if (!e.target.closest('#xpop')) closeExplainer();
  }, true);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeExplainer(); });
  window.addEventListener('resize', closeExplainer);
}
