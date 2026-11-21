// 3D view: quality-scaled embeddings as points, the selected set as arrows from the
// origin, and (for K ≤ 3) the parallelepiped whose squared volume the DPP scores.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

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
    this.controls.autoRotate = true;
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
    // Centre the camera target on the unit-embedding cloud, keep the origin visible.
    if (!this.scale) {
      let m = 0;
      for (const c of o.coords) m = Math.max(m, Math.hypot(...c));
      this.scale = 1 / (m || 1);
      const cx = o.coords.reduce((a, c) => a + c[0], 0) / N * this.scale;
      this.controls.target.set(cx / 2, 0, 0);
      this.camera.position.set(cx / 2 + 1.4, 1.0, 2.0);
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
    for (let i = 0; i < N; i++) this.mesh.setColorAt(i, new THREE.Color(sel.has(i) ? o.accent : o.color(i)));
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
    const accent = new THREE.Color(o.accent);
    for (const i of o.sel) {
      const p = this.current[i];
      const len = p.length();
      if (len < 1e-4) continue;
      const a = new THREE.ArrowHelper(p.clone().normalize(), new THREE.Vector3(), len, accent, Math.min(0.06, len * 0.3), Math.min(0.03, len * 0.15));
      this.dyn.add(a);
    }
    if (o.sel.length >= 2 && o.sel.length <= 3) {
      const v = o.sel.map((i) => this.current[i]);
      const corners = [];
      const n = v.length;
      for (let mask = 0; mask < 1 << n; mask++) {
        const c = new THREE.Vector3();
        for (let k = 0; k < n; k++) if (mask & (1 << k)) c.add(v[k]);
        corners.push(c);
      }
      const pts = [];
      for (let a = 0; a < corners.length; a++) for (let k = 0; k < n; k++) {
        const b = a | (1 << k);
        if (b !== a) pts.push(corners[a], corners[b]);
      }
      const g = new THREE.BufferGeometry().setFromPoints(pts);
      this.dyn.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: accent, transparent: true, opacity: 0.5 })));
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
