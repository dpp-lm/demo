// 3D view: quality-scaled embeddings as points, the selected set as arrows from the
// origin, and (for K ≤ 3) the parallelepiped whose squared volume the DPP scores.
// Everything drawn is scaled into a unit bounding sphere (radius 1); sizes below are in that frame.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { easeInOut, pairSelected } from './shared.js';
import { tipSheet } from './layout.js';

function rod(a, b, r, color) {
  const d = new THREE.Vector3().subVectors(b, a);
  const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, d.length(), 8), new THREE.MeshBasicMaterial({ color }));
  m.position.copy(a).addScaledVector(d, 0.5);
  m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.clone().normalize());
  return m;
}

// Depth cues: faces write depth (back sides faint, front sides a bit stronger and tinted), edges are drawn solid where
// visible and dashed and dim where something nearer hides them (GreaterDepth pass). renderOrder: faces 1-2, visible 3, hidden 4.
function addFaces(group, tri, o) {
  const geo = new THREE.BufferGeometry().setFromPoints(tri);
  const grey = new THREE.Color(o.text), tint = grey; // neutral in both colour modes
  const mk = (color, opacity, side, order) => {
    const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity, side, depthWrite: true, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }));
    m.renderOrder = order;
    group.add(m);
  };
  mk(grey, 0.06, THREE.BackSide, 1);
  mk(tint, 0.13, THREE.FrontSide, 2);
}
function addEdges(group, pts, color) {
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const vis = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.8 }));
  vis.renderOrder = 3;
  const hid = new THREE.LineSegments(geo.clone(), new THREE.LineDashedMaterial({ color, transparent: true, opacity: 0.35, dashSize: 0.03, gapSize: 0.02, depthFunc: THREE.GreaterDepth, depthWrite: false }));
  hid.computeLineDistances();
  hid.renderOrder = 4;
  group.add(vis, hid);
}
function addRod(group, a, b, r, color) {
  const vis = rod(a, b, r, color), hid = rod(a, b, r, color);
  vis.material.transparent = true; vis.renderOrder = 3;
  Object.assign(hid.material, { transparent: true, opacity: 0.6, depthFunc: THREE.GreaterDepth, depthWrite: false });
  hid.renderOrder = 4;
  group.add(vis, hid);
}
// Rod from a to b with a small cone arrowhead at b.
function addArrow(group, a, b, r, color) {
  const d = new THREE.Vector3().subVectors(b, a), len = d.length();
  if (len < 1e-6) return;
  const dir = d.clone().normalize();
  const head = Math.min(0.07, len * 0.35);
  addRod(group, a, b.clone().addScaledVector(dir, -head * 0.9), r, color);
  const cone = new THREE.Mesh(new THREE.ConeGeometry(Math.max(0.018, r * 2.6), head, 14), new THREE.MeshBasicMaterial({ color, transparent: true }));
  cone.position.copy(b).addScaledVector(dir, -head / 2);
  cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
  cone.renderOrder = 3;
  const hid = cone.clone();
  hid.material = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.6, depthFunc: THREE.GreaterDepth, depthWrite: false });
  hid.renderOrder = 4;
  group.add(cone, hid);
}

let dotTex = null;
function roundDot() {
  if (dotTex) return dotTex;
  const c = document.createElement('canvas');
  c.width = c.height = 32;
  const g = c.getContext('2d');
  g.fillStyle = '#fff';
  g.beginPath(); g.arc(16, 16, 14, 0, 2 * Math.PI); g.fill();
  dotTex = new THREE.CanvasTexture(c);
  return dotTex;
}

const R_SEL = 0.018, R_FREE = 0.010, R_FLOOR = 0.006;

export class Plot3D {
  constructor(el, { onHover, onTap }) {
    this.el = el;
    this.onHover = onHover;
    this.onTap = onTap;
    const W = el.clientWidth || 600, H = el.clientHeight || 450;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.setSize(W, H, false);
    el.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, W / H, 0.01, 100);
    this.camera.position.set(1.6, 1.2, 2.2);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.userRotate = !matchMedia('(prefers-reduced-motion: reduce)').matches; // cleared by the first drag
    this.controls.autoRotate = this.userRotate;
    this.controls.autoRotateSpeed = 0.6;
    this.renderer.domElement.addEventListener('pointerdown', () => { this.userRotate = false; this.controls.autoRotate = false; });
    this.group = new THREE.Group();
    this.scene.add(this.group);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const dl = new THREE.DirectionalLight(0xffffff, 0.8);
    dl.position.set(2, 3, 4);
    this.scene.add(dl);
    this.mouse = new THREE.Vector2();
    this.hovered = -1; this.hovered2 = -1;
    this.targets = null;
    this.current = null;
    this.goal = null; // where the orbit target is heading (the centre of the drawn bounding sphere)
    // Hover tooltips for mice; a tap (pointer down and up without dragging) inspects
    // the nearest point on screen within a finger-sized radius.
    const cv = this.renderer.domElement;
    cv.addEventListener('pointermove', (e) => { if (e.pointerType === 'mouse') this.pick(e); });
    cv.addEventListener('pointerleave', () => { this.hovered = -1; this.onHover(-1); });
    let down = null;
    cv.addEventListener('pointerdown', (e) => (down = [e.clientX, e.clientY]));
    cv.addEventListener('pointerup', (e) => {
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 6) return;
      this.onTap?.(this.nearest(e.clientX, e.clientY, e.pointerType === 'mouse' ? 16 : 32));
    });
    this.alive = true;
    const loop = () => {
      if (!this.alive) return;
      requestAnimationFrame(loop);
      this.tick();
      this.controls.update();
      if (this.scene.fog) { const d = this.camera.position.distanceTo(this.controls.target); this.scene.fog.near = d - 0.4; this.scene.fog.far = d + 1.8; }
      this.renderer.render(this.scene, this.camera);
    };
    loop();
  }

  // Bounding sphere of everything drawn, in the data frame: the drawn points, the ghost unit points when shown,
  // the origin, and all corners of the parallelepiped. Centre = bounding-box centre, radius = farthest distance from it.
  frame(o) {
    if (o.fixedFrame) {
      // Fixed axes: frame the unit positions and the origin only (the scaled points lie between them), so the camera does not
      // follow the selection or T. The radius has a constant margin for the parallelepiped.
      const pts = [[0, 0, 0], ...o.coords.map((c) => [c[0], c[1], c[2]])];
      // Room for the parallelepiped: the far corner of a typical triple lies near 2.5 times the mean unit position.
      const n = o.coords.length || 1;
      pts.push([0, 1, 2].map((d) => 2.5 * o.coords.reduce((a, c) => a + c[d], 0) / n));
      const lo = [0, 1, 2].map((d) => Math.min(...pts.map((p) => p[d]))), hi = [0, 1, 2].map((d) => Math.max(...pts.map((p) => p[d])));
      const center = lo.map((l, d) => (l + hi[d]) / 2);
      let r = 0;
      for (const p of pts) r = Math.max(r, Math.hypot(p[0] - center[0], p[1] - center[1], p[2] - center[2]));
      return { center, radius: (r || 1) * 1.05 };
    }
    const pts = [[0, 0, 0]];
    const f = (i) => (o.scaled ? o.c[i] : 1);
    o.coords.forEach((c, i) => {
      pts.push([c[0] * f(i), c[1] * f(i), c[2] * f(i)]);
      if (o.scaled) pts.push([c[0], c[1], c[2]]);
    });
    const K = o.sel.length;
    if (K >= 2 && K <= 3) {
      const v = o.sel.map((i) => [0, 1, 2].map((d) => o.coords[i][d] * f(i)));
      for (let mask = 1; mask < 1 << K; mask++) pts.push([0, 1, 2].map((d) => v.reduce((a, x, k) => (mask & (1 << k) ? a + x[d] : a), 0)));
    }
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const p of pts) for (let d = 0; d < 3; d++) { lo[d] = Math.min(lo[d], p[d]); hi[d] = Math.max(hi[d], p[d]); }
    const center = lo.map((l, d) => (l + hi[d]) / 2);
    let r = 0;
    for (const p of pts) r = Math.max(r, Math.hypot(p[0] - center[0], p[1] - center[1], p[2] - center[2]));
    return { center, radius: r || 1 };
  }

  // o.morph: a snapshot() of the previous view; the selected set glides to its new place and the other points cross-fade.
  update(o, morph = null) {
    this.o = o;
    const N = o.coords.length;
    this.scene.fog = new THREE.Fog(o.bg, 1, 10); // near/far follow the camera distance every frame
    const { center, radius } = this.frame(o);
    const s = 1 / radius;
    this.scale = s;
    const cw = new THREE.Vector3(center[0] * s, center[1] * s, center[2] * s);
    this.centerW = cw;
    if (!this.camSet) {
      // Place the camera so the unit sphere fits the narrower of the vertical and horizontal fields of view, with 10% margin.
      this.camSet = true;
      const vh = (this.camera.fov * Math.PI) / 360, hh = Math.atan(Math.tan(vh) * this.camera.aspect);
      const dist = 1.1 / Math.sin(Math.min(vh, hh));
      this.controls.target.copy(cw);
      this.camera.position.copy(cw).addScaledVector(new THREE.Vector3(1.1, 0.8, 1.55).normalize(), dist);
      this.goal = null;
    } else this.goal = cw; // the orbit target and the camera glide by the same step in tick(), so the view direction and distance stay
    this.targets = o.coords.map((c, i) => {
      const f = (o.scaled ? o.c[i] : 1) * s;
      return new THREE.Vector3(c[0] * f, c[1] * f, c[2] * f);
    });
    this.rFree = this.freeRadius();
    this.initCurrent(N, morph);
    this.rebuildStatic();
  }

  // Radius of the non-selected balls: 0.010, shrunk to 0.35 × the median nearest-neighbour distance when crowded (floor 0.006).
  freeRadius() {
    const sel = new Set(this.o.sel);
    const P = this.targets.filter((_, i) => !sel.has(i));
    if (P.length < 2) return R_FREE;
    const nn = P.map((p, i) => { let b = Infinity; for (let j = 0; j < P.length; j++) if (j !== i) b = Math.min(b, p.distanceToSquared(P[j])); return Math.sqrt(b); });
    nn.sort((x, y) => x - y);
    const med = nn[Math.floor(nn.length / 2)];
    return Math.max(R_FLOOR, Math.min(R_FREE, 0.35 * med));
  }

  initCurrent(N, morph) {
    this.endMorph();
    if (morph?.instant) this.current = this.targets.map((v) => v.clone());
    else if (morph && morph.kind === '3d') {
      this.current = this.targets.map((v) => v.clone());
      this.beginMorph(morph);
    } else if (!this.current || this.current.length !== N) this.current = this.targets.map((v) => v.clone());
  }

  // Everything needed to start a morph from the current view: positions, colours, ball radii, the selected set.
  snapshot() {
    const o = this.o;
    if (!o || !this.current) return null;
    const sel = new Set(o.sel);
    return {
      kind: '3d', pos: this.current.map((v) => v.clone()), sel: o.sel.slice(),
      colors: o.coords.map((_, i) => new THREE.Color(sel.has(i) ? o.selColor(i) : o.color(i))),
      scales: this.current.map((_, i) => this.radius(i, sel, false)),
    };
  }

  // Ball radius in the unit frame (the sphere geometry has radius 1).
  radius(i, sel, hover) {
    return (sel.has(i) ? R_SEL : this.rFree) * (hover ? 1.5 : 1) * (this.o?.rs ? this.o.rs[i] : 1);
  }

  beginMorph(snap) {
    const o = this.o;
    const oldSel = new Set(snap.sel);
    const idx = snap.pos.map((_, i) => i).filter((i) => !oldSel.has(i));
    const mat = new THREE.MeshLambertMaterial({ transparent: true, opacity: 1, depthWrite: false });
    const old = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 12, 8), mat, idx.length);
    const m = new THREE.Matrix4();
    idx.forEach((i, k) => {
      const r = snap.scales[i];
      old.setMatrixAt(k, m.makeScale(r, r, r).setPosition(snap.pos[i]));
      old.setColorAt(k, snap.colors[i]);
    });
    old.instanceMatrix.needsUpdate = true;
    if (old.instanceColor) old.instanceColor.needsUpdate = true;
    this.scene.add(old);
    const px = snap.pos.map((v) => v.x), nx = this.targets.map((v) => v.x);
    const dbg = window.__dppDebug;
    if (dbg) { dbg.morphs++; dbg.lastKind = '3d'; }
    this.morph = { t0: performance.now(), dur: 500, snap, old, pairs: pairSelected(snap.sel, px, o.sel, nx) };
  }

  endMorph() {
    const mo = this.morph;
    if (!mo) return;
    if (window.__dppDebug) window.__dppDebug.lastMorphMs = Math.round(performance.now() - mo.t0);
    this.morph = null;
    this.scene.remove(mo.old);
    mo.old.geometry.dispose(); mo.old.material.dispose(); mo.old.dispose?.();
    if (this.mesh) { Object.assign(this.mesh.material, { transparent: false, opacity: 1, depthWrite: true }); this.mesh.material.needsUpdate = true; }
  }

  tickMorph() {
    const mo = this.morph;
    const t = Math.min(1, (performance.now() - mo.t0) / mo.dur), e = easeInOut(t);
    for (const [ni, oi] of mo.pairs) this.current[ni].lerpVectors(mo.snap.pos[oi], this.targets[ni], e);
    if (this.mesh) this.mesh.material.opacity = e;
    mo.old.material.opacity = 1 - e;
    this.layout();
    if (t >= 1) { this.endMorph(); this.layout(); }
  }

  rebuildStatic() {
    const o = this.o;
    const N = o.coords.length;
    for (const ch of [...this.group.children]) { this.group.remove(ch); ch.geometry?.dispose(); ch.material?.dispose(); }
    // Axes from the origin.
    const axMat = new THREE.LineBasicMaterial({ color: o.text, transparent: true, opacity: 0.35 });
    for (const d of [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) {
      const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(...d).multiplyScalar(0.9)]);
      this.group.add(new THREE.Line(g, axMat));
    }
    // Faint floor grid through the origin (as in the paper's Fig. 1), sized to cover the frame.
    const grid = new THREE.GridHelper(2.4, 24, o.text, o.text);
    grid.material.transparent = true;
    grid.material.opacity = 0.12;
    grid.material.depthWrite = false;
    grid.position.set(this.centerW.x, 0, this.centerW.z);
    this.group.add(grid);
    // Ghosts: the unit position of each point, same colour as the point, only where the scaled point is not already (almost) there.
    const s = this.scale;
    this.ghostIdx = []; this.ghostPos = new Map(); this.ghostGeo = null;
    if (o.scaled && o.ghosts !== false) {
      o.coords.forEach((c, i) => {
        const g = new THREE.Vector3(c[0] * s, c[1] * s, c[2] * s);
        if (g.distanceTo(this.targets[i]) >= 0.03) { this.ghostIdx.push(i); this.ghostPos.set(i, g); }
      });
      const gpos = new Float32Array(this.ghostIdx.length * 3);
      this.ghostIdx.forEach((i, k) => gpos.set(this.ghostPos.get(i).toArray(), k * 3));
      const gg = new THREE.BufferGeometry();
      gg.setAttribute('position', new THREE.BufferAttribute(gpos, 3));
      gg.setAttribute('color', new THREE.BufferAttribute(new Float32Array(this.ghostIdx.length * 3), 3));
      this.ghostGeo = gg;
      this.paintGhosts();
      this.group.add(new THREE.Points(gg, new THREE.PointsMaterial({ vertexColors: true, size: 0.024, map: roundDot(), alphaTest: 0.2, transparent: true, opacity: 0.6, depthWrite: false })));
    }
    // Points.
    this.mesh = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 12, 8), new THREE.MeshLambertMaterial(), N);
    if (this.morph) Object.assign(this.mesh.material, { transparent: true, opacity: 0, depthWrite: false });
    const sel = new Set(o.sel);
    for (let i = 0; i < N; i++) this.mesh.setColorAt(i, new THREE.Color(sel.has(i) ? o.selColor(i) : o.color(i)));
    this.group.add(this.mesh);
    // Arrows + parallelepiped edges are rebuilt every tick from current positions.
    this.dyn = new THREE.Group();
    this.group.add(this.dyn);
    this.layout();
  }

  paintGhosts() {
    if (!this.ghostGeo) return;
    const o = this.o, sel = new Set(o.sel), col = this.ghostGeo.getAttribute('color'), c = new THREE.Color();
    this.ghostIdx.forEach((i, k) => { c.set(sel.has(i) ? o.selColor(i) : o.color(i)); col.setXYZ(k, c.r, c.g, c.b); });
    col.needsUpdate = true;
  }

  // New colour functions (a legend chip dims other clusters): repaint balls, ghosts and arrows without moving anything.
  recolor(color, selColor) {
    if (!this.o || !this.mesh) return;
    this.o.color = color; this.o.selColor = selColor;
    const sel = new Set(this.o.sel);
    for (let i = 0; i < this.o.coords.length; i++) this.mesh.setColorAt(i, new THREE.Color(sel.has(i) ? selColor(i) : color(i)));
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
    this.paintGhosts();
    this.layout();
  }

  layout() {
    if (!this.mesh) return;
    const o = this.o;
    const m = new THREE.Matrix4();
    const sel = new Set(o.sel);
    // During a morph the selected points leave the cross-fading mesh and are drawn as solid spheres below.
    const morphing = !!this.morph;
    this.current.forEach((p, i) => {
      const r = morphing && sel.has(i) ? 0 : this.radius(i, sel, i === this.hovered || i === this.hovered2);
      m.makeScale(r, r, r).setPosition(p);
      this.mesh.setMatrixAt(i, m);
    });
    this.mesh.instanceMatrix.needsUpdate = true;
    for (const ch of [...this.dyn.children]) { this.dyn.remove(ch); ch.geometry?.dispose(); ch.material?.dispose?.(); }
    if (morphing) {
      for (const i of o.sel) {
        const s = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 8), new THREE.MeshLambertMaterial({ color: new THREE.Color(o.selColor(i)) }));
        s.position.copy(this.current[i]);
        s.scale.setScalar(this.radius(i, sel, i === this.hovered || i === this.hovered2));
        this.dyn.add(s);
      }
    }
    const few = o.sel.length <= 3;
    const O = new THREE.Vector3();
    // Edges from the origin to each selected point: one colour per item, with an arrowhead.
    for (const i of o.sel) {
      const p = this.current[i];
      const len = p.length();
      if (len < 1e-4) continue;
      const color = new THREE.Color(o.selColor(i));
      if (few) addArrow(this.dyn, O, p, 0.006, color);
      else this.dyn.add(new THREE.ArrowHelper(p.clone().normalize(), O, len, color, Math.min(0.06, len * 0.3), Math.min(0.03, len * 0.15)));
    }
    if (o.sel.length > 3) {
      // K > 3: an open triangulated sheet through the tips (a cloth over the arrow tips), see-through from both sides.
      const P = o.sel.map((i) => this.current[i]);
      const { tris } = tipSheet(P.map((p) => [p.x, p.y, p.z]));
      if (tris.length) {
        addFaces(this.dyn, tris.flatMap((t) => t.map((i) => P[i])), o);
        const seen = new Set(), seg = [];
        for (const t of tris) for (let k = 0; k < 3; k++) {
          const a = t[k], b = t[(k + 1) % 3], id = a < b ? `${a},${b}` : `${b},${a}`;
          if (!seen.has(id)) { seen.add(id); seg.push(P[a], P[b]); }
        }
        addEdges(this.dyn, seg, new THREE.Color(o.text));
      }
    }
    if (o.sel.length >= 2 && few) {
      const v = o.sel.map((i) => this.current[i]);
      const n = v.length;
      const corner = (mask) => v.reduce((c, x, k) => (mask & (1 << k) ? c.add(x) : c), new THREE.Vector3());
      const grey = new THREE.Color(o.text);
      // Translucent faces (depth-writing, back and front sides tinted differently).
      const tri = [];
      for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) {
        const others = [...Array(n).keys()].filter((k) => k !== a && k !== b);
        for (const off of others.length ? [0, 1 << others[0]] : [0]) {
          const q = [off, off | (1 << a), off | (1 << a) | (1 << b), off | (1 << b)].map(corner);
          tri.push(q[0], q[1], q[2], q[0], q[2], q[3]);
        }
      }
      addFaces(this.dyn, tri, o);
      // Edges (solid where visible, dashed where hidden), except the ones from the origin (drawn above in colour).
      const seg = [];
      for (let a = 0; a < 1 << n; a++) for (let k = 0; k < n; k++) {
        const b = a | (1 << k);
        if (b !== a && a !== 0) seg.push(corner(a), corner(b));
      }
      addEdges(this.dyn, seg, grey);
    }
  }

  tick() {
    // Auto-rotation pauses while a morph runs, then resumes unless the user has dragged.
    this.controls.autoRotate = this.userRotate && !this.morph;
    // The orbit target glides to the new frame centre; the camera moves by the same step, so the view does not jump.
    if (this.goal) {
      const d = this.goal.clone().sub(this.controls.target);
      if (d.lengthSq() < 1e-8) { this.controls.target.copy(this.goal); this.goal = null; }
      else { d.multiplyScalar(0.18); this.camera.position.add(d); this.controls.target.add(d); }
    }
    if (!this.targets || !this.current) return;
    if (this.morph) { this.tickMorph(); return; }
    let moving = false;
    this.current.forEach((p, i) => {
      const t = this.targets[i];
      if (p.distanceToSquared(t) > 1e-10) { p.lerp(t, 0.18); moving = true; }
    });
    if (moving) this.layout();
  }

  // Balls are small, so hovering uses the nearest point on screen (10 px) instead of a ray hit.
  pick(e) {
    if (!this.mesh) return;
    const i = this.nearest(e.clientX, e.clientY, 10);
    if (i !== this.hovered || this.hovered2 >= 0) { this.hovered = i; this.hovered2 = -1; this.layout(); }
    this.onHover(i, e.clientX, e.clientY);
  }

  // Index of the point drawn nearest to a screen position, or -1 if none within r px.
  nearest(x, y, r) {
    if (!this.current) return -1;
    const rect = this.renderer.domElement.getBoundingClientRect();
    let best = -1, bd = r * r, bz = Infinity;
    const v = new THREE.Vector3();
    const test = (p, i) => {
      v.copy(p).project(this.camera);
      const sx = rect.left + ((v.x + 1) / 2) * rect.width, sy = rect.top + ((1 - v.y) / 2) * rect.height;
      const d = (sx - x) ** 2 + (sy - y) ** 2;
      if (d < bd || (d === bd && v.z < bz)) { bd = d; bz = v.z; best = i; }
    };
    this.current.forEach((p, i) => {
      test(p, i);
      // A point collapsed onto the origin is reached through its unit-position ghost.
      if (this.ghostPos?.has(i) && this.targets[i].length() < 0.02) test(this.ghostPos.get(i), i);
    });
    return best;
  }

  // Position of point i in CSS pixels relative to the canvas.
  screenXY(i) {
    if (!this.current?.[i]) return null;
    const v = this.current[i].clone().project(this.camera);
    const w = this.renderer.domElement.clientWidth, h = this.renderer.domElement.clientHeight;
    return [((v.x + 1) / 2) * w, ((1 - v.y) / 2) * h];
  }

  highlight(i, j = -1) { this.hovered = i; this.hovered2 = j; this.layout(); }

  dispose() {
    this.endMorph();
    this.alive = false;
    this.controls.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
