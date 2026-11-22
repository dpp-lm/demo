// Small "?" buttons that open a plain-language explainer with the math typeset.
// Any element with class "xbtn" and data-x="<key>" opens EXPLAIN[key].

export const EXPLAIN = {
  T: {
    title: 'Temperature T',
    body: `<p>Each completion gets a quality score
      $$\\kappa(y) = p(y)^{1/T},$$
      where $p(y)$ is the model's probability of the whole completion.</p>
      <ul>
        <li><b>Small $T$</b> exaggerates differences in likelihood, so the most likely completions win (quality).</li>
        <li><b>Large $T$</b> flattens them: every completion counts about equally, and only how different they are matters (diversity).</li>
      </ul>
      <p>Drag it and watch the arrows in the plot grow and shrink.</p>`,
  },
  subset: {
    title: 'How the subset is chosen',
    body: `<p>Think of each completion as an arrow. Its <b>direction</b> is its meaning (the embedding $\\phi(y)$) and its <b>length</b> is its quality $\\sqrt{\\kappa(y)}$. A set $S$ scores the squared volume its arrows span:
      $$\\det \\mathbf L_S = \\Big(\\prod_{y\\in S}\\kappa(y)\\Big)\\,\\det\\!\\big(\\Phi_S^\\top \\Phi_S\\big).$$
      The volume is large when the arrows are long (likely completions) and point different ways (different meanings), and zero when two are the same. This is a determinantal point process (DPP).</p>
      <p><b>Most likely</b> picks the set with the largest volume, adding one completion at a time. Exact search is NP-hard; the greedy answer is within a factor $(K!)^2$ of the best.</p>
      <p><b>Sample</b> draws a random set with probability proportional to its volume, so you see different good sets. Press Resample to draw again.</p>`,
  },
  volume: {
    title: 'Spanned volume',
    body: `<p>The grey box is spanned by the selected arrows. Its squared volume, $\\det \\mathbf L_S$, is the subset's score: long arrows (likely completions) pointing in different directions (different meanings) make a big box; near-duplicates flatten it.</p>
      <p>The plot is a projection of 384-dimensional embeddings, so the box you see is a shadow of the real one.</p>`,
  },
  logp: {
    title: 'log P(S)',
    body: `<p>How probable this subset is among all size-$K$ subsets of the completions:
      $$P(S) = \\frac{\\det \\mathbf L_S}{\\sum_{|S'| = K} \\det \\mathbf L_{S'}}.$$
      The most likely subset is the one that maximizes it.</p>`,
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
  ess: {
    title: 'Effective sample size',
    body: `<p>The completions are generated at sampling temperature $T'$, but the subset should follow $\\kappa = p^{1/T}$. So <b>Sample</b> reweights each completion by
      $$w(y) = \\frac{p(y)^{1/T}}{p^{\\circ}_{T'}(y)}.$$
      The effective sample size $(\\sum w)^2 / \\sum w^2$ says how many completions effectively count. When $T = T'$ every weight is equal; far from $T'$ a few completions dominate.</p>`,
  },
  Tprime: {
    title: 'Sampling temperature T′',
    body: `<p>Controls how the completions are generated: every next-token distribution is raised to the power $1/T'$ and renormalized. $T' = 1$ samples the model as it is. Higher $T'$ gives more varied but less coherent text, which is why diversity comes from choosing the subset afterwards instead.</p>`,
  },
  scale: {
    title: 'Scale points by quality',
    body: `<p>When on, each point sits at length $\\sqrt{\\kappa(y)}$ along its embedding direction: the same arrows the score measures, so changing $T$ grows or shrinks them. When off, every completion is drawn at unit length, showing meaning only.</p>`,
  },
  pca: {
    title: 'Projection',
    body: `<p>The embeddings have 384 dimensions. The plot shows the top principal axes (uncentered, so arrows start at the origin, where volume is measured from). The bars show how much of the spread each axis captures; click 1, 2 or 3 to change the view.</p>`,
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
