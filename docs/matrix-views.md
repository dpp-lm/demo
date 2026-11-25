# Matrix views: design notes

A record of our discussion (25 Nov 2026) about bringing matrix views into the demo, starting from Manuel's
matrix panel. Nothing here is built yet. Decisions taken so far are marked **Decided**, and open questions
**Open**.

## Contents

1. [What Manuel built](#1-what-manuel-built)
2. [The views we are considering](#2-the-views-we-are-considering)
3. [The kernel, its eigendecomposition and PCA](#3-the-kernel-its-eigendecomposition-and-pca)
4. [Open questions](#4-open-questions)
5. [References](#5-references)

Notation: N completions with unit embeddings φ_i, the rows of Φ (N × D). The similarity matrix is
G = ΦΦᵀ, with G_ii = 1. The quality of completion i is κ_i, and the DPP's kernel is
L = diag(√w) G diag(√w), where the weights w are κ for Most likely and κ/q for Sample. The page uses a
K-DPP: a fixed set size K.

## 1. What Manuel built

His `js/matrix-panel.js` and `js/matrices.js` (his main, e769c13) have two views:

- **"Pairs of completions" (the primal).** L for the whole pool, N × N, with the weights the subset was
  picked with.
  - Rows and columns are in the leaf order of average-linkage clustering on 1 − cos, so similar completions
    form blocks on the diagonal.
  - The selected completions are marked in the margins, and hovering a cell shows the pair.
  - Expert mode adds a dendrogram in the margins, other row orders (groups, weight, pool order), viridis
    colour, and the selected set's own matrix L_S.
- **"Dual".** The weighted vectors' D × D matrix C = BᵀB (B has rows √w_i φ_i), shown in the pool's first
  32 principal axes (computed from the eigendecomposition of G) and normalised to correlations.

The primal is readable. The dual is an unlabelled 32 × 32 grid; it is hard to say what a visitor learns
from it.

## 2. The views we are considering

### A. The similarity matrix, reordered into blocks (VAT). High priority

This is the standard picture of a kernel, or of a covariance matrix in GP work. In clustering it is called
VAT, "visual assessment of cluster tendency" (Bezdek & Hathaway 2002). Manuel's primal view is an instance.

**Decided:**
- We want this, and it goes high on the priority queue.
- A dendrogram is welcome only if its ordering is coherent with the DPP and its geometry. We will not block
  on a method like average linkage, whose relationship to volume is arbitrary.

**Proposal: order the rows by pivoted Cholesky on G.** This is exactly the greedy volume procedure, so the
ordering comes from the DPP's own geometry.

- **Pivot order.** Pivot t is the completion farthest from the span of pivots 1 … t−1, the one whose
  leftover length after projecting out those pivots is largest. So the first rows are the most different
  completions in the pool.
- **Blocks.** Every other completion i joins the block of the pivot that explains the most of it. That is
  the step t where its leftover length shrinks the most, i.e. the largest R_it² in the Cholesky factor.
  Within a block, rows are sorted by that amount.
- **What it shows:**
  - Blocks are "near-copies of pivot t", with "near" meaning the same thing as in the volume.
  - The pivot sequence takes the place of a dendrogram, with no linkage rule.
  - A margin strip can show each completion's leftover length when its pivot was taken: tall where it adds
    a new direction, short where it is a near-copy.
- **One consistent story.** This uses the same operation as the plot's fitted axes and greedy MAP
  (section 3): greedy volume picks the pivots, the plot's axes are the pivots, and the matrix's blocks are
  the pivots' near-copies.

**Nested version: one Cholesky factorisation read as a tree.** No thresholds.

Run pivoted Cholesky to the end (until the leftover is negligible), then read the factor as a tree:
- **Order:** the pivots, coarse to fine.
- **Parent:** each completion's parent is the earlier pivot that explains the most of it (the largest
  R_ts²).
- **Height:** each node's height is its leftover length when it was taken, a continuous number, so no level
  cut-offs are needed.
- **Matrix order:** depth-first along the tree, so the blocks nest. The flat proposal above is this tree cut
  below the first K pivots.

**Is it meaningful for the DPP?** Its numbers are exact DPP quantities, but the tree describes greedy MAP's
view, not the sampling distribution.

- **Meaningful:**
  - **Heights are volume factors.** Leftover² of i given earlier pivots A is det G_{A∪i} / det G_A: the
    factor by which adding i multiplies the volume. The chain rule makes any set's volume the product of
    these factors.
  - **Parents are the strongest repulsion.** The share of i's leftover removed by pivot s is the squared
    partial correlation ρ²(i, s | A). For the L-ensemble's set weights,

    det L_{A∪{i,s}} · det L_A / (det L_{A∪i} · det L_{A∪s}) = 1 − ρ²(i, s | A)

    exactly. So the parent is the earlier pivot that most suppresses also having i.
  - **The first K pivots are greedy MAP** for the matrix the tree is built on. On L they are the page's
    Most likely selection. On G they are greedy MAP with quality switched off, which for the Global target
    at T = ∞ is the DPP itself (κ = 1, so L = G). So a G-tree is the pool's DPP picture at T = ∞, and an
    L-tree is the picture at the current settings.
- **Not meaningful:**
  - **The sampling distribution.** The tree is one greedy path and says nothing about inclusion
    probabilities or other sets. Use C (repulsion) or D (themes) for that.
  - **Dasgupta and Long's guarantee.** It is about cluster radius (k-center), not volume. We borrow their
    construction, not their theorem.
  - **The screening literature.** It explains sparse Cholesky factors for spatial kernels: a reason blocks
    might appear, not a DPP property, and unproven for high-dimensional embeddings.
- **Stability.** Near-ties between leftovers can flip a pivot and reorder the tree. That matters for a
  picture that should stay put.

**The sampling-side alternative: theme blocks.** A K-DPP samples by first choosing K eigenvectors (exact
probabilities λ_a·e_{K−1}/e_K), then items within them. So grouping each completion under the eigenvector it
loads on most gives blocks grounded in how Sample works.
- **The catch:** eigenvectors are rarely localised (completions load on several, with arbitrary signs), so
  theme blocks are likely muddier.
- **Working view:** the greedy tree is the structural, Most likely side, and what we use to order the
  matrix. The themes are the distributional, Sample side, and what we use to explain Sample. Whether the
  first pivots line up with the top eigenvectors is itself worth showing.
- **Tim:** the greedy interpretation is probably as good as we'll get for an ordering. Agreed, with the
  themes as the sampling-side counterpart.

**Literature for the ordering and the nesting:**
- **Pivoted Cholesky is farthest-point sampling in feature space.** The pivot is the point farthest from
  the span of those chosen, and the factor is an implicit Gram–Schmidt (G. Shabat, 2026, arXiv 2601.03706,
  a short note).
- **Hierarchy from a farthest-first order.** Dasgupta and Long (2005), "Performance guarantees for
  hierarchical clustering", JCSS 70(4): the parent is the nearest earlier point, and for every k the
  k-clustering is within 8× the optimal radius. Our version measures distance to the span of the earlier
  pivots, not to the nearest one; the span distance is what matches volume.
- **Coarse-to-fine orderings and screening.**
  - Schäfer, Sullivan and Owhadi (2021, SIAM J. Sci. Comput., arXiv 1706.02205) and Schäfer, Katzfuss and
    Owhadi (2021, arXiv 2004.14455) use the maximin ordering before Cholesky.
  - In that ordering, points group into levels of scale, and fine points are nearly uncorrelated given the
    coarse ones, so the factor is nearly sparse (Chen and Sanz-Alonso, arXiv 2412.08820).
  - Proved for spatial kernels in low dimension; to be checked for our embeddings.
- **Randomly pivoted Cholesky** (Chen, Epperly, Tropp and Webber, arXiv 2207.06503; CPAM 2025): pivots are
  drawn in proportion to the residual diagonal. It reports good cluster recovery in kernel spectral
  clustering.

  **Correction to the discussion:** this is not DPP sampling. Drawing in proportion to the residual
  diagonal is exact only for a projection DPP (the second phase of the spectral sampler). For an
  L-ensemble K-DPP the sequence probability is not ∝ det L_S, and exact sequential sampling needs the
  conditioned K-DPP's inclusion probabilities (the e_k terms). It is a cousin of DPP sampling, not an
  instance.

**Next check:** prototype the flat and nested orders on a few presets and compare the picture with Manuel's
average-linkage order.

**Open: G or L in the matrix?**
- G is pure similarity, fixed for a pool. L is similarity × quality, and it changes with T, the target and
  the mode.
- Lean (Claude's, not decided): show G, so the order and picture stay put while T is dragged, and show
  quality in a margin strip as √κ bars that follow T. That keeps the quality × diversity split
  (L = diag(q)·S·diag(q), Kulesza & Taskar) visible.

**Not porting as is:**
- Viridis. The house style is ink on paper, which is also Manuel's default.
- Average-linkage order and its dendrogram (see above).

### B. Conditioning (the Schur complement)

**The operation.** Conditioning on a set A replaces the kernel by L − L_{·A} L_A⁻¹ L_{A·}. Its diagonal is
each completion's leftover quality once A is "explained away".

**Shared by greedy and sampling.** Conditioning a K-DPP on including A gives a (K − |A|)-DPP whose kernel is
that Schur complement. The two methods differ only in the next step:
- **Greedy MAP** takes the completion with the largest leftover.
- **Sampling one at a time** draws the next completion in proportion to its inclusion probability under the
  conditioned DPP.

In the spectral sampler (Kulesza & Taskar), the item phase draws each next item in proportion to the
leftover diagonal of the projection kernel of the chosen eigenvectors, then conditions.

**The GP view.** With L as a GP covariance, conditioning is the posterior covariance after observing A, and
the diagonal is the posterior variance. So greedy DPP MAP is "observe the point you are most uncertain
about", the greedy max-entropy rule of GP sensor placement (Krause, Singh & Guestrin 2008). The chain rule
gives log det L_S = Σ_t log (leftover at step t).

**Tim:** it is clearly a fundamental operation, but the value for exploring the distribution is not clear
yet. It does sound fun to play with.

**Claimed value: a "what if" control.** Pin a completion, and every other completion's inclusion is
recomputed given that one is in. For example, pin a skeleton joke: the other skeleton jokes' inclusion
drops, unrelated jokes' rises. That shows the distribution's structure directly. It would be a feature of
the matrix view (hover or click a row), not a separate panel.

**Already on the page.** The 3-D view's fitted axes condition on the selected set: the axes beyond K are
the PCA of G − AAᵀ (section 3).

### C. Pairwise repulsion

The signature of a DPP is negative correlation: two similar completions are less likely to appear together
than if they were included independently.
- **The measure:** the ratio π_ij / (π_i π_j), which is below 1 for repelling pairs. Shown as a matrix, it
  makes "diverse" measurable rather than visual.
- **Tim: this is the marginal kernel.** For a standard DPP (an L-ensemble), K = L(L + I)⁻¹ and
  P(i, j ∈ S) = K_ii K_jj − K_ij², so the repulsion is the squared correlation K_ij² / (K_ii K_jj).
- **The catch for the page.** A K-DPP (fixed size) has no marginal kernel, so π_ij must be computed
  directly:
  - one row costs one conditioning, the same computation as B;
  - all N² pairs take about 200 conditionings, a few seconds for N = 200.

So B and C are the same computation. Hovering a row could show that row's repulsion cheaply.

### D. Themes and effective dimension

**Decided:** we want to discuss themes, and the effective dimension is agreed.

**Themes.**
- **What they are.** The eigenvectors of the kernel, each a pattern over the completions: a direction in
  meaning space (from the dual), with each completion's loading on it (from the primal).
- **Naming.** Name a theme by the completions at its two extremes, e.g. "skeleton jokes ↔ atom jokes".
  Eigenvectors have arbitrary sign, so the two ends are a pair.
- **Exact probabilities for Sample.** A K-DPP sample first chooses K eigenvectors, then items within them
  (Kulesza & Taskar 2011). So the probability that theme a is among those chosen is exact and computable
  from the spectrum:

  P(a) = λ_a · e_{K−1}(λ without a) / e_K(λ)

  where e_k is the k-th elementary symmetric polynomial. This answers "which themes will a Sample cover?"
  with a number. It is meaningful for L's eigenvalues: the DPP that is actually sampled, with the Sample
  weights.

**Effective dimension: Σ_a λ_a / (λ_a + 1).**
- This is the expected size of the DPP with kernel L, and the effective degrees of freedom of the
  corresponding kernel ridge regression or GP.
- It explains "K too large for this pool" from the spectrum, without a threshold (backlog item).

### The dual view: push on it, don't drop it

**Tim:** the dual view does not need to struggle, so we push on it.

- **Why it belongs.** C = Σ_i w_i φ_i φ_iᵀ is the weighted second-moment matrix in embedding space. It has
  the same nonzero eigenvalues as L, and its eigenvectors are the themes as directions in meaning space. So
  the dual and D are one topic.
- **Why Manuel's version is hard to read, and the fixes:**
  1. **Unnamed axes.** The axes are raw principal axes. Fix: name each axis by its extreme completions.
  2. **The diagonal is just the spectrum.** In its own eigenbasis the dual is diagonal. Fix: hold the axes
     fixed (from G, the pool's structure) and show how T, the target and the mode re-weight them. At low T,
     the likely completions' theme should dominate; at high T, weight spreads out. Off-diagonal entries
     then show how the quality weights mix the pool's themes.
  3. **No link to Sample.** Fix: add each theme's exact probability of being chosen by Sample (above).

## 3. The kernel, its eigendecomposition and PCA

### One factorisation

Take the SVD Φ = U Σ Vᵀ. Then:

| Object | Formula | Size and meaning |
|---|---|---|
| The kernel (primal) | G = ΦΦᵀ = U Σ² Uᵀ | N × N: similarities between completions |
| The dual | ΦᵀΦ = V Σ² Vᵀ | D × D: second moments in embedding space |
| PCA coordinates | ΦV = UΣ | each completion's position on the principal axes |

- **Shared eigenvalues.** All three have the same eigenvalues, λ = σ².
- **Two descriptions of the same directions.** U (the kernel's eigenvectors) and V (PCA's axes) describe
  the same directions: V as directions in meaning space, U as each completion's coordinate on them.
- **PCA from the kernel alone.** The PCA coordinates are U√Λ, which needs only G, never Φ. This is kernel
  PCA (Schölkopf, Smola & Müller 1998), and it is why the page can draw everything from G.
- **Shares of variance.** Total variance is trace G = N for unit vectors, so axis a explains λ_a / N
  (`frac` in `project()`).
- **The DPP's normaliser.** Summed over all K-subsets, det L_S = e_K(λ), using L's eigenvalues, so the
  spectrum alone fixes the normalisation. A K-subset with positive volume exists only if at least K
  eigenvalues are nonzero.
- **The GP connection.** G as a GP covariance has the same eigenvectors, there called the Karhunen–Loève or
  Mercer expansion. A random function from the GP is Σ_a √λ_a z_a u_a, with z_a independent standard
  normals. So the themes are also the GP's typical patterns of variation over the completions.

### Three choices hidden in "do PCA", and what the page does

1. **Centring: the page does not centre.**
   - Standard PCA subtracts the mean, i.e. uses HGH with H = I − 11ᵀ/N. The page's PCA is uncentred, which
     matches the DPP: the paper's cosine kernel measures volume from the origin.
   - The cost is the bouquet: the first uncentred axis is mostly the shared mean direction (52% of the
     spectrum for the story prompt).
   - Centring would remove the bouquet but make the kernel depend on the pool, which leaves the paper's
     fixed kernel and Theorem 5.2. That is option B in the bouquet backlog item, ruled out.
2. **G's axes, not L's.**
   - The plot uses G's eigenvectors, which are fixed for a pool, and scales each point by √κ.
   - The DPP's own matrix is L = DGD with D = diag(√w), and its eigenvectors differ: the weights rotate the
     axes toward the high-weight completions.
   - So the drawn axes are the pool's, while the DPP's themes are L's, and those move with T, the target
     and the mode.
3. **Fitted axes when K ≤ 3, which are not PCA at all** (`project()`, `js/main.js:290`).
   - The first K axes come from a Cholesky of the selected completions (Gram–Schmidt), so the drawn volume
     is exactly det G_S, with no loss from projecting.
   - The remaining axes are the top PCA axes of the Schur complement G − AAᵀ: the pool conditioned on the
     selected set.
   - So the 3-D view already does "condition on S, then PCA what's left", the same operation as B and as
     pivoted Cholesky in A.
   - PCA can only lose volume (the volume in its top-3 plane is at most the true one); the fitted axes
     remove that loss for the selected set.

## 4. Open questions

1. **G or L as the basis for each view.** G answers "what's in the pool" and is stable for a pool. L answers
   "what this DPP, at this T, with this target and mode, cares about", and it moves.
   - The matrix (A): lean G, with quality in the margin (above).
   - Themes (D) and the dual: the axes from G, re-weighted by L. Sample's theme probabilities must use L.
2. **The VAT details:** the pivoted-Cholesky ordering, the margin strips (leftover length, √κ), where the
   matrix lives on the page, and how it links to the plot and the cards (hover a row to highlight the point
   and card, as Manuel's does).
3. **Whether conditioning (B) earns a place.** Is a "what if" pin control worth it, and is its cost of one
   conditioning per row acceptable?
4. **Naming themes:** extreme completions, or something else? How many themes to show?
5. **The tree on G or on L:** the T = ∞ picture, which is stable, or the current settings', which moves.
   This is question 1 again, in DPP terms.

## 5. References

- J. C. Bezdek and R. J. Hathaway (2002). VAT: a tool for visual assessment of (cluster) tendency.
- A. Kulesza and B. Taskar (2011). k-DPPs: fixed-size determinantal point processes.
- A. Kulesza and B. Taskar (2012). Determinantal point processes for machine learning (the quality ×
  diversity decomposition, the marginal kernel, conditioning, the spectral sampler).
- B. Schölkopf, A. Smola and K.-R. Müller (1998). Nonlinear component analysis as a kernel eigenvalue
  problem (kernel PCA).
- C. E. Rasmussen and C. K. I. Williams (2006). Gaussian processes for machine learning.
- A. Krause, A. Singh and C. Guestrin (2008). Near-optimal sensor placements in Gaussian processes.
- S. Dasgupta and P. M. Long (2005). Performance guarantees for hierarchical clustering. JCSS 70(4).
- G. Shabat (2026). The geometry of the pivot: a note on lazy pivoted Cholesky and farthest point sampling.
  arXiv 2601.03706.
- F. Schäfer, T. J. Sullivan and H. Owhadi (2021). Compression, inversion, and approximate PCA of dense
  kernel matrices at near-linear computational complexity. arXiv 1706.02205.
- F. Schäfer, M. Katzfuss and H. Owhadi (2021). Sparse Cholesky factorization by Kullback–Leibler
  minimization. arXiv 2004.14455.
- J. Chen and D. Sanz-Alonso. Precision and Cholesky factor estimation for Gaussian processes. arXiv
  2412.08820.
- Y. Chen, E. N. Epperly, J. A. Tropp and R. J. Webber (2025). Randomly pivoted Cholesky: practical
  approximation of a kernel matrix with few entry evaluations. CPAM; arXiv 2207.06503.
- L. Chen, G. Zhang and H. Zhou (2018). Fast greedy MAP inference for determinantal point process to
  improve recommendation diversity.
