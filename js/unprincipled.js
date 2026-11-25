// Expert section "Not super principled": options the upstream page left out because he cannot defend them. They are kept here, marked
// as such. The section holds only state and controls. main.js reads the state through xopt() (expert.js) at a few `// expert:` hooks:
//   recompute()   the weight override (kernelFamily, weightScale), the drawn lengths and point radii, the open cone and the fixed axes,
//   setMode()     one T or one T per picking mode (xModeT),
//   plots         per-point radius, unit positions, no drawn volume while the cone is open.
// hooks used from main (ctx.H): weights(), ctrl.apply({ mode, T, K, Tprime, target }), ctrl.tp(), recompute(newDraw).

import { EXPLAIN } from './explain.js';
import { NP_NOTE, NP_DEFAULTS, npIsDefault, kernelFamily, drawnFamily, modeT, presetNote, PRESETS, matchPreset } from './expert-lib.js';

let st = { ...NP_DEFAULTS };
let ctx = null, Tby = null, lastPreset = null, lastSig = '';

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const q = (x, what) => `<button type="button" class="xbtn" data-x="${x}" aria-label="${what}">?</button>`;
const fmtT = (T) => (T === Infinity ? '∞' : T < 10 ? T.toFixed(2) : T < 100 ? T.toFixed(1) : T.toFixed(0));

/** The state of the section (read by main.js through expert.js: xopt()). */
export const npState = () => st;

/** T of the picking mode `to` after leaving `from` (main.js setMode): one T unless each mode keeps its own. */
export function npModeT(from, to, T) {
  return modeT({ shared: st.sharedT, Tby, from, to, T });
}

// ---------------------------------------------------------------- the section
const seg = (name, list, cur) => `<div class="pick" role="group" aria-label="${name}">${list.map((o) => `<button type="button" class="pick-btn" data-f="np-${o.f}" aria-pressed="${o.id === cur}">${o.label}</button>`).join('')}</div>`;
const sel = (f, label, x, what, opts, cur, dis = false) => `<label class="x-row"><span class="x-lab">${label}${q(x, what)}</span><select class="x-select" data-f="${f}"${dis ? ' disabled' : ''}>${opts.map(([v, t]) => `<option value="${v}"${v === cur ? ' selected' : ''}>${t}</option>`).join('')}</select></label>`;
const toggle = (f, label, x, what, on, dis = false) => `<label class="toggle"><input type="checkbox" data-f="${f}"${on ? ' checked' : ''}${dis ? ' disabled' : ''}><span>${label}${q(x, what)}</span></label>`;

function nowState() {
  const w = ctx.H.weights(), tp = ctx.H.ctrl.tp();
  return { w, tp, fam: kernelFamily(st, w.mode) };
}

function render() {
  const { w, tp, fam } = nowState();
  const ticks = PRESETS[w.mode === 'map' ? 'map' : 'samp'].Ts;
  const match = matchPreset({ mode: w.mode, Tprime: tp, K: w.K, T: w.T, weight: w.target === 'global' ? fam : 'target' });
  const note = presetNote({ match, last: lastPreset, now: { mode: w.mode, T: w.T, K: w.K, tp, weight: w.target === 'global' ? fam : 'target', len: st.len, lock: true } });
  const other = w.mode === 'map' ? 'sample' : 'map', Tother = Tby?.[other] ?? w.T;
  const unitDrawn = drawnFamily(st, w.mode) === 'unit';
  return `<p class="x-note">${NP_NOTE}</p>
    <div class="x-row"><span class="x-lab">Paper settings${q('np-preset', 'What do the paper settings do?')}</span>${seg('Paper settings', [{ id: 'map', f: 'preset-map', label: PRESETS.map.label }, { id: 'samp', f: 'preset-samp', label: PRESETS.samp.label }], match)}${note ? `<span class="x-note">${esc(note)}</span>` : ''}</div>
    <div class="x-row"><span class="x-lab">Paper T for ${w.mode === 'map' ? 'Most likely' : 'Sample'}${q('np-paperT', 'What are the paper T values?')}</span><div class="x-ticks">${ticks.map((t) => `<button type="button" class="x-tick" data-f="np-T-${t}" aria-pressed="${Math.abs(t - w.T) <= 1e-6 * t}">${t}</button>`).join('')}</div></div>
    ${toggle('np-shared', 'Same T for both picking modes', 'np-shared', 'What does one T for both modes do?', st.sharedT)}
    ${st.sharedT ? '' : `<p class="x-note">T is <span class="x-num">${fmtT(w.T)}</span> for ${w.mode === 'map' ? 'Most likely' : 'Sample'} and <span class="x-num">${fmtT(Tother)}</span> for ${other === 'map' ? 'Most likely' : 'Sample'}.</p>`}
    ${sel('np-weight', 'DPP weight', 'np-weight', 'What does the DPP weight override do?', [['follow', 'Follows mode and target'], ['quality', 'Quality, κ'], ['importance', 'Importance, κ / q'], ['unit', 'Unit, 1']], st.weight)}
    ${sel('np-len', 'Vector length', 'np-len', 'What does the vector length do?', [['follow', 'Follows the weight (√κ unless overridden)'], ['quality', 'Quality, √κ'], ['importance', 'Importance, √(κ / q)'], ['unit', 'Unit length']], st.len)}
    ${toggle('np-size', 'Point size shows quality', 'np-size', 'What does point size do?', st.sizeQ)}
    ${toggle('np-ghost', 'Show unit positions', 'np-ghost', 'What are the faint dots?', st.ghost, unitDrawn)}
    ${unitDrawn ? '<p class="x-note">Points are already at unit length, so there is nothing to compare.</p>' : ''}
    ${toggle('np-cone', 'Open the cone', 'np-cone', 'What does opening the cone do?', st.cone, w.dims === 1)}
    ${w.dims === 1 ? '<p class="x-note">The cone opens from 2-D on. Spectrum sets the dimensions.</p>' : ''}
    <div class="x-row"><span class="x-lab">Axes${q('np-axes', 'What do the axes options do?')}</span>${seg('Axes', [{ id: 'fit', f: 'axes-fit', label: 'Fitted at first draw' }, { id: 'fixed', f: 'axes-fixed', label: 'Principal axes' }], st.axes)}</div>
    ${w.K > 3 ? '<p class="x-note">For K above 3 both choices give the same axes.</p>' : ''}`;
}

function setState(patch, { redraw = true, newDraw = false } = {}) {
  const was = st;
  st = { ...st, ...patch };
  if ('sharedT' in patch) {
    Tby = st.sharedT ? null : {};   // a mode not visited yet starts at the T in use
  }
  syncNotes();
  if (redraw) ctx.H.recompute(newDraw || was.weight !== st.weight);
  ctx.rerender();
}

function onMenu(e) {
  const f = e.target.dataset?.f;
  if (!f?.startsWith('np-')) return;
  const click = e.type === 'click', change = e.type === 'change';
  if (click && f.startsWith('np-preset-')) {
    const key = f.slice(10), p = PRESETS[key];
    lastPreset = key;
    st = { ...st, weight: 'follow', len: 'follow' };
    syncNotes();
    ctx.H.ctrl.apply({ mode: p.mode, T: p.T, K: p.K, Tprime: p.Tprime, target: 'global' });
    ctx.rerender();
  } else if (click && f.startsWith('np-T-')) { ctx.H.ctrl.apply({ T: +f.slice(5) }); ctx.rerender(); }
  else if (click && f.startsWith('np-axes-')) setState({ axes: f.slice(8) });
  else if (change && f === 'np-shared') setState({ sharedT: e.target.checked }, { redraw: false });
  else if (change && f === 'np-weight') setState({ weight: e.target.value }, { newDraw: true });
  else if (change && f === 'np-len') setState({ len: e.target.value });
  else if (change && f === 'np-size') setState({ sizeQ: e.target.checked });
  else if (change && f === 'np-ghost') setState({ ghost: e.target.checked });
  else if (change && f === 'np-cone') setState({ cone: e.target.checked });
}

export const principledSection = { id: 'unprincipled', title: 'Not super principled', render, on: onMenu };

// ---------------------------------------------------------------- notes next to his controls
let modeNote = null, targetNote = null;
const WORD = { quality: 'quality', importance: 'importance', unit: 'unit' };
function syncNotes() {
  const on = ctx.isExpert() && st.weight !== 'follow';
  if (modeNote) { modeNote.hidden = !on; if (on) modeNote.textContent = `DPP weight overridden in the expert panel: ${WORD[st.weight]}.`; }
  if (targetNote) { targetNote.hidden = !on; if (on) targetNote.textContent = `The DPP weight is overridden in the expert panel (${WORD[st.weight]}), so this choice no longer sets it.`; }
}

/** Re-render the panel after the selection changed, only when something this section shows changed. */
function refresh() {
  const { w, tp } = nowState();
  const sig = [w.mode, w.T, w.K, w.target, w.dims, tp, st.weight, Tby ? `${Tby.map}/${Tby.sample}` : ''].join('|');
  if (sig === lastSig) return;
  lastSig = sig;
  ctx.rerender();
}

// ---------------------------------------------------------------- init
const np = (title, body) => ({ title, body });

export function initUnprincipled(c) {
  ctx = c;
  EXPLAIN['np-preset'] = np('Paper settings', `<p>These buttons set the picking mode, $T$, $K$, $T'$ and the target quality to operating points from the paper (Ambrosia, $K = 3$, 200 completions).</p>
    <ul><li><b>MAP, paper</b>: Most likely on pools drawn at $T' = 1.5$, with $T = 3$ (the paper sweeps $T \\in \\{1, 3, 7.5, 10\\}$).</li>
    <li><b>Samp, paper</b>: Sample on pools drawn at $T' = 1.0$, with $T = 0.82$ (the paper sweeps $T \\in \\{0.43, 0.54, 0.67, 0.82, 1\\}$).</li></ul>
    <p>A button also sets the target to Global and the DPP weight and vector length to Follows. It stays pressed while the picking mode, $T'$ and $K$ match and $T$ is on the paper's grid. After a press, the note under the buttons names what has changed since. $T'$ is set through the Temp. $T'$ field of Settings, and the pool at that $T'$ must be stored for the prompt (the examples have 1.0 and 1.5).</p>`);
  EXPLAIN['np-paperT'] = np('Paper T', `<p>The values of $T$ the paper sweeps for the current picking mode. A press sets $T$ only, for this mode if each mode keeps its own $T$.</p>`);
  EXPLAIN['np-shared'] = np('Same T for both picking modes', `<p>On, one $T$ serves Most likely and Sample, as on the default page, so the picture stays the same when you switch. Off, each mode remembers its own $T$ and switching the mode restores that mode's $T$. The paper uses different values for the two (for example 3 for MAP and 0.82 for Samp).</p>`);
  EXPLAIN['np-weight'] = np('DPP weight', `<p>The weight $w(y)$ sets how much each completion counts in the kernel $L_{ij} = \\sqrt{w_i}\\,\\phi_i\\cdot\\phi_j\\,\\sqrt{w_j}$. The default page uses $\\kappa(y)$ for Most likely and $\\kappa(y)/p^{\\circ}_{T'}(y)$ for Sample, with $\\kappa$ the target quality of the Settings.</p>
    <ul><li><b>Quality</b>: $w = \\kappa$, in either mode.</li><li><b>Importance</b>: $w = \\kappa / q$, in either mode.</li><li><b>Unit</b>: $w = 1$, so $T$ has no effect on the kernel.</li></ul>
    <p>A choice other than Follows replaces the weight that picks the subset in the current mode. The similarity matrices show the weights in use. The pool fit under the plot is still computed for the default weights and still describes the target of the Settings.</p>`);
  EXPLAIN['np-len'] = np('Vector length', `<p>How long each arrow is drawn. It changes the picture, never the selection. <b>Follows the weight</b> is locked to the DPP weight: $\\sqrt{\\kappa}$ unless the DPP weight is overridden, so the drawn box is the one the kernel scores in Most likely. <b>Quality</b> draws $\\sqrt{\\kappa(y)}$ and <b>Importance</b> draws $\\sqrt{\\kappa(y)/q(y)}$, each relative to the largest. <b>Unit length</b> draws every arrow at length 1, showing directions only.</p>
    <p>When the drawn length is not the kernel's weight, the line under the plot says which arrows the volume belongs to.</p>`);
  EXPLAIN['np-size'] = np('Point size shows quality', `<p>Shows quality as the radius of each point: proportional to $\\sqrt{\\kappa(y)}$ relative to the likeliest completion, with a floor of 35% so every point stays visible. It does not change the arrow lengths. With Vector length set to Unit length the radius is then the only sign of quality.</p>`);
  EXPLAIN['np-ghost'] = np('Unit positions', `<p>The faint dots mark each completion's direction at unit length, where it sits with no length scaling. The solid points are the same completions at the chosen vector length. A faint dot is drawn only where it differs from its solid point.</p>`);
  EXPLAIN['np-cone'] = np('Open the cone', `<p>The completions of one prompt have embeddings pointing in nearly the same direction, so the points sit in a narrow cone around their mean direction. Opening the cone stretches every direction perpendicular to that mean by a factor $s$ and leaves the mean direction alone. $s$ is the ratio of the points' spread along the mean direction to their spread across it, between 1 and 4.</p>
    <p>The stretch is one linear map, so every volume is multiplied by $s^2$ in 3-D (by $s$ in 2-D) and ratios between volumes are unchanged. The kernel and the selection are unchanged. The box is not drawn while the cone is open.</p>`);
  EXPLAIN['np-axes'] = np('Axes', `<p>The default page sets the axes once per pool, when the points are first drawn, and never turns them afterwards. <b>Fitted at first draw</b> is its choice for $K \\le 3$: the first $K$ axes span the arrows selected at that moment, so the box is exactly their volume (for a later selection it is the volume's shadow on these axes).</p>
    <p><b>Principal axes</b> uses plain uncentered principal axes of the pool for every $K$. Neither choice moves the cloud when the selection, picking mode, $T$ or $K$ change. Switching between them frames the plot again.</p>`);
  // Notes beside his own controls, shown while the weight is overridden.
  const row = document.querySelector('.step-2 .mode-row');
  modeNote = document.createElement('p');
  modeNote.className = 'x-wt-note';
  modeNote.hidden = true;
  row?.after(modeNote);
  targetNote = document.createElement('p');
  targetNote.className = 'x-wt-note';
  targetNote.id = 'x-target-note';
  targetNote.hidden = true;
  document.getElementById('target')?.closest('.mode-row')?.after(targetNote);   // upstream moved Target to its own row under the mode
  ctx.onSelection(refresh);
  ctx.onPool(() => { lastSig = ''; });
  ctx.onOff(() => {
    const touched = !npIsDefault(st);
    st = { ...NP_DEFAULTS };
    Tby = null; lastPreset = null; lastSig = '';
    syncNotes();
    if (touched) ctx.H.recompute(true);
  });
}
