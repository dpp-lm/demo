import json, sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
d = json.load(open(sys.argv[1])); K = d['K']
Ts = np.array([t if t is not None else np.nan for t in d['Ts']])
xinf = 3000.0
x = np.where(np.isnan(Ts), xinf, Ts)
col = {1.0: '#2a78d6', 1.5: '#eb6834'}
ink, muted, grid = '#2b2b2a', '#6f6e69', '#e4e2dc'
N = d['rows'][0]['N']; floor = K * K / (N - K + 1); ceil = K * N / (N - K + 1)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, dpi=200)
titles = [('global', r'Global (paper):  $\kappa = p^{1/T}$'), ('gl', r"Global and local:  $\kappa = p^{1/T}\, p^{\circ}_{T'}$")]
for ax, (target, title) in zip(axes, titles):
    for r in d['rows']:
        if r['target'] != target: continue
        f = np.array(r['fit']); c = col[r['Tprime']]
        ax.plot(x[:-1], f[:-1], color=c, lw=1.2, alpha=0.75)
        ax.plot([x[-2], x[-1]], [f[-2], f[-1]], color=c, lw=1.2, alpha=0.75, ls=(0, (2, 2)))
        ax.plot([x[-1]], [f[-1]], 'o', ms=3, color=c)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(0.035, 5)
    for y in (floor, ceil): ax.axhline(y, color=muted, lw=1, ls=(0, (1, 2)))
    ax.axvline(20, color=muted, lw=1, ls=(0, (4, 3)))
    for tp in (1.0, 1.5): ax.axvline(tp, color=col[tp], lw=1, alpha=0.35)
    ax.set_xticks([0.1, 1, 10, 100, 1000, xinf]); ax.set_xticklabels(['0.1', '1', '10', '100', '1000', '∞'])
    ax.set_title(title, fontsize=11, color=ink, loc='left')
    ax.set_xlabel('T  (global temperature)', color=ink, fontsize=10)
    ax.grid(True, which='major', color=grid, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color(muted)
    ax.tick_params(colors=muted, labelsize=9)
    ax.text(0.055, ceil * 1.12, f'ceiling KN/(N−K+1) = {ceil:.2f}: all weight on K completions', color=muted, fontsize=8)
axes[0].text(2500, floor * 1.15, f'floor K²/(N−K+1) = {floor:.3f}: weights uniform', color=muted, fontsize=8, ha='right')
axes[0].text(22, 0.2, 'page default\nT = 20', color=muted, fontsize=8, va='bottom')
axes[1].text(22, 1.2, 'page default\nT = 20', color=muted, fontsize=8, va='bottom')
axes[0].set_ylabel(r'pool fit $\hat\theta$ (nats)', color=ink)
axes[1].legend([Line2D([], [], color=col[1.0], lw=1.5), Line2D([], [], color=col[1.5], lw=1.5)],
               ["pools drawn at T′ = 1.0", "pools drawn at T′ = 1.5"], frameon=False, fontsize=9, loc='center right', bbox_to_anchor=(1.0, 0.42))
fig.suptitle(f'Pool fit vs T: 9 example prompts × 2 values of T′, N = {N}, K = {K}, Qwen3-Embedding-0.6B. Faint vertical lines mark T = T′.', fontsize=9.5, color=muted, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(sys.argv[2], facecolor='white')
