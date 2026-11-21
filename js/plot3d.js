// 3D view: quality-scaled embeddings as points, the selected set as arrows from the
// origin, and (for K ≤ 3) the parallelepiped whose squared volume the DPP scores.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

function rod(a, b, r, color) {
  const d = new THREE.Vector3().subVectors(b, a);
  const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, d.length(), 8), new THREE.MeshBasicMaterial({ color }));
  m.position.copy(a).addScaledVector(d, 0.5);
  m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.clone().normalize());
  return m;
}

export class Plot3D {
  constructor(el, { onHover }) {
    this.el = el;
    this.onHover = onHover;
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
    this.controls.autoRotate = !matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.controls.autoRotateSpeed = 0.6;
    this.renderer.domElement.addEventListener('pointerdown', () => (this.controls.autoRotate = false));
    this.group = new THREE.Group();
    this.scene.add(this.group);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const dl = new THREE.DirectionalLight(0xffffff, 0.8);
    dl.position.set(2, 3, 4);
    this.scene.add(dl);
    this.ray = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this.hovered = -1;
    this.targets = null;
    this.current = null;
    this.renderer.domElement.addEventListener('pointermove', (e) => this.pick(e));
    this.renderer.domElement.addEventListener('pointerleave', () => { this.hovered = -1; this.onHover(-1); });
    this.alive = true;
    const loop = () => {
      if (!this.alive) return;
      requestAnimationFrame(loop);
      this.tick();
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };
    loop();
  }

  update(o) {
    this.o = o;
    const N = o.coords.length;
    // Fit points and (for K ≤ 3) the far corner of the spanned volume inside the unit ball.
    let m = 0;
    for (const c of o.coords) m = Math.max(m, Math.hypot(...c));
    if (o.sel.length >= 2 && o.sel.length <= 3) {
      const far = [0, 1, 2].map((d) => o.sel.reduce((acc, i) => acc + o.coords[i][d] * (o.scaled ? o.c[i] : 1), 0));
      m = Math.max(m, Math.hypot(...far));
    }
    this.scale = 1 / (m || 1);
    if (!this.camSet) {
      this.camSet = true;
      const cx = o.coords.reduce((a, c) => a + c[0], 0) / N * this.scale;
      this.controls.target.set(cx / 2, 0, 0);
      this.camera.position.set(cx / 2 + 1.1, 0.8, 1.55);
    }
    const s = this.scale;
    this.targets = o.coords.map((c, i) => {
      const f = (o.scaled ? o.c[i] : 1) * s;
      return new THREE.Vector3(c[0] * f, c[1] * f, c[2] * f);
    });
    if (!this.current || this.current.length !== N) this.current = this.targets.map((v) => v.clone());
    this.rebuildStatic();
  }

  rebuildStatic() {
    const o = this.o;
    const N = o.coords.length;
    for (const ch of [...this.group.children]) { this.group.remove(ch); ch.geometry?.dispose(); ch.material?.dispose(); }
    // Axes through the origin.
    const axMat = new THREE.LineBasicMaterial({ color: o.text, transparent: true, opacity: 0.35 });
    for (const d of [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) {
      const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(...d).multiplyScalar(0.9)]);
      this.group.add(new THREE.Line(g, axMat));
    }
    // Faint floor grid through the origin (as in the paper's Fig. 1).
    const grid = new THREE.GridHelper(2, 16, o.text, o.text);
    grid.material.transparent = true;
    grid.material.opacity = 0.12;
    grid.material.depthWrite = false;
    this.group.add(grid);
    // Ghosts: unit embeddings.
    const s = this.scale;
    const gpos = new Float32Array(N * 3);
    o.coords.forEach((c, i) => gpos.set([c[0] * s, c[1] * s, c[2] * s], i * 3));
    const gg = new THREE.BufferGeometry();
    gg.setAttribute('position', new THREE.BufferAttribute(gpos, 3));
    this.group.add(new THREE.Points(gg, new THREE.PointsMaterial({ color: o.ghost, size: 0.02, transparent: true, opacity: o.scaled ? 0.5 : 0 })));
    // Points.
    const sphere = new THREE.SphereGeometry(0.018, 12, 8);
    this.mesh = new THREE.InstancedMesh(sphere, new THREE.MeshLambertMaterial(), N);
    const sel = new Set(o.sel);
    for (let i = 0; i < N; i++) this.mesh.setColorAt(i, new THREE.Color(sel.has(i) ? o.selColor(i) : o.color(i)));
    this.group.add(this.mesh);
    // Arrows + parallelepiped edges are rebuilt every tick from current positions.
    this.dyn = new THREE.Group();
    this.group.add(this.dyn);
    this.layout();
  }

  layout() {
    if (!this.mesh) return;
    const o = this.o;
    const m = new THREE.Matrix4();
    const sel = new Set(o.sel);
    this.current.forEach((p, i) => {
      const r = sel.has(i) ? 1.6 : i === this.hovered ? 1.8 : 1;
      m.makeScale(r, r, r).setPosition(p);
      this.mesh.setMatrixAt(i, m);
    });
    this.mesh.instanceMatrix.needsUpdate = true;
    for (const ch of [...this.dyn.children]) { this.dyn.remove(ch); ch.geometry?.dispose(); ch.material?.dispose?.(); }
    const few = o.sel.length <= 3;
    const O = new THREE.Vector3();
    // Edges from the origin to each selected point: arrows for large K, plain coloured
    // rods (one colour per item) for K ≤ 3, as in the paper's Fig. 1.
    for (const i of o.sel) {
      const p = this.current[i];
      const len = p.length();
      if (len < 1e-4) continue;
      const color = new THREE.Color(o.selColor(i));
      if (few) this.dyn.add(rod(O, p, 0.006, color));
      else this.dyn.add(new THREE.ArrowHelper(p.clone().normalize(), O, len, color, Math.min(0.06, len * 0.3), Math.min(0.03, len * 0.15)));
    }
    if (o.sel.length >= 2 && few) {
      const v = o.sel.map((i) => this.current[i]);
      const n = v.length;
      const corner = (mask) => v.reduce((c, x, k) => (mask & (1 << k) ? c.add(x) : c), new THREE.Vector3());
      const grey = new THREE.Color(o.text);
      // Translucent faces.
      const tri = [];
      for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) {
        const others = [...Array(n).keys()].filter((k) => k !== a && k !== b);
        for (const off of others.length ? [0, 1 << others[0]] : [0]) {
          const q = [off, off | (1 << a), off | (1 << a) | (1 << b), off | (1 << b)].map(corner);
          tri.push(q[0], q[1], q[2], q[0], q[2], q[3]);
        }
      }
      const fg = new THREE.BufferGeometry().setFromPoints(tri);
      this.dyn.add(new THREE.Mesh(fg, new THREE.MeshBasicMaterial({ color: grey, transparent: true, opacity: 0.08, side: THREE.DoubleSide, depthWrite: false })));
      // Dark edges, except the ones from the origin (drawn above in colour).
      const seg = [];
      for (let a = 0; a < 1 << n; a++) for (let k = 0; k < n; k++) {
        const b = a | (1 << k);
        if (b !== a && a !== 0) seg.push(corner(a), corner(b));
      }
      this.dyn.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(seg), new THREE.LineBasicMaterial({ color: grey, transparent: true, opacity: 0.75 })));
      // Corner dots; the far corner is larger.
      const full = (1 << n) - 1;
      for (let mask = 1; mask <= full; mask++) {
        if (!(mask & (mask - 1))) continue;
        const dot = new THREE.Mesh(new THREE.SphereGeometry(mask === full ? 0.016 : 0.009, 10, 8), new THREE.MeshBasicMaterial({ color: grey }));
        dot.position.copy(corner(mask));
        this.dyn.add(dot);
      }
    }
  }

  tick() {
    if (!this.targets || !this.current) return;
    let moving = false;
    this.current.forEach((p, i) => {
      const t = this.targets[i];
      if (p.distanceToSquared(t) > 1e-10) { p.lerp(t, 0.18); moving = true; }
    });
    if (moving) this.layout();
  }

  pick(e) {
    if (!this.mesh) return;
    const r = this.renderer.domElement.getBoundingClientRect();
    this.mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this.ray.setFromCamera(this.mouse, this.camera);
    const hit = this.ray.intersectObject(this.mesh)[0];
    const i = hit ? hit.instanceId : -1;
    if (i !== this.hovered) { this.hovered = i; this.layout(); }
    this.onHover(i, e.clientX, e.clientY);
  }

  highlight(i) { this.hovered = i; this.layout(); }

  dispose() {
    this.alive = false;
    this.controls.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
