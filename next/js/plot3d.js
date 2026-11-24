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

// The x, y, z axes through the origin, from −1 to 1 (the length of every embedding). Coloured per
// vertex so the near end is darker than the far end (see shadeDepth).
function axesLines(scale) {
  // Drawn first and without depth writes, so points, vectors and the volume always sit on top.
  const g = new THREE.Group(), mat = new THREE.LineBasicMaterial({ vertexColors: true, depthWrite: false });
  const line = (pts, ink) => {
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(pts.length * 3), 3));
    const l = new THREE.Line(geo, mat);
    l.renderOrder = -1; l.userData.ink = ink;   // how dark the line is, near side (shadeDepth)
    return l;
  };
  for (const d of [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) {
    const pts = []; for (let t = -1; t <= 1.0001; t += 0.05) pts.push(new THREE.Vector3(d[0] * t, d[1] * t, d[2] * t));
    g.add(line(pts, 0.6));
  }
  g.scale.setScalar(scale);
  return g;
}

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
    // Orthographic: no perspective, so every point is drawn the same size at any depth and
    // lengths (quality) are drawn to one scale. The view holds the unit ball on any aspect.
    const a = W / H, h = Math.max(0.78, 0.9 / a);
    this.camera = new THREE.OrthographicCamera(-h * a, h * a, h, -h, -50, 50);
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
    // When the plot area changes size (window resize, the docked controls measuring themselves),
    // resize the canvas and re-frame, so the drawing is never stretched or cut off.
    this.ro = new ResizeObserver(() => {
      const W = el.clientWidth, H = el.clientHeight;
      if (!W || !H) return;
      this.renderer.setSize(W, H, false);
      if (this.o && this.camSet) { const f = this.frameFor(this.o, this.scale); this.frameH = f.h; this.goal = null; this.setHalfHeight(f.h); }
    });
    this.ro.observe(el);
    const loop = () => {
      if (!this.alive) return;
      requestAnimationFrame(loop);
      this.tick();
      this.controls.update();
      this.shadeDepth();
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
    const s = this.scale;
    // Placed once, looking at the origin; framed when it first appears, and again, smoothly, when it
    // switches between fitted axes (K ≤ 3) and principal axes.
    if (!this.camSet) {
      // Look down from 35°: a vector lying near the floor is foreshortened at most to sin 35° ≈ 0.57
      // of its length as the view turns, so no selected vector can collapse onto the origin.
      this.camSet = true;
      const el = (35 * Math.PI) / 180, az = (40 * Math.PI) / 180;
      this.controls.target.set(0, 0, 0);
      this.camera.position.set(3 * Math.cos(el) * Math.sin(az), 3 * Math.sin(el), 3 * Math.cos(el) * Math.cos(az));
      this.camera.lookAt(0, 0, 0);
      this.camera.updateMatrixWorld();
      const frame = this.frameFor(o, s);
      this.frameH = frame.h;
      this.setHalfHeight(frame.h);
    } else if (this.lastFitted !== o.fitted) {
      const frame = this.frameFor(o, s);
      this.frameH = frame.h;
      this.goal = frame;
    }
    this.lastFitted = o.fitted;
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
    // No axes or floor grid: the only meaningful frame is the origin, from which the scored volume is measured.
    // The x, y, z axes through the origin, which the view turns about.
    this.globe = axesLines(this.scale);
    this.globeInk = new THREE.Color(o.text ?? '#555555');
    this.group.add(this.globe);
    // Points.
    // Radius tied to the zoom: about 1.5 px on a 600-px-tall plot, the same as the 2-D dots.
    const sphere = new THREE.SphereGeometry(0.0055 * (this.frameH ?? this.camera.top), 10, 8);
    this.mesh = new THREE.InstancedMesh(sphere, new THREE.MeshLambertMaterial(), N);
    const sel = new Set(o.sel);
    // Base colours; shadeDepth() fades each toward the background by its distance from the viewer.
    this.base = Array.from({ length: N }, (_, i) => new THREE.Color(sel.has(i) ? o.selColor(i) : o.color(i)));
    this.isSel = Array.from({ length: N }, (_, i) => sel.has(i));
    this.bgCol = new THREE.Color(o.bg ?? '#fbfaf8');
    for (let i = 0; i < N; i++) this.mesh.setColorAt(i, this.base[i]);
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
      m.makeScale(1, 1, 1).setPosition(p);   // all points the same size; colour marks the selected ones
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
      const z = this.frameH ?? 1;   // widths follow the zoom, so they look the same at any frame
      if (few) this.dyn.add(rod(O, p, 0.0055 * z, color));
      else this.dyn.add(new THREE.ArrowHelper(p.clone().normalize(), O, len, color, Math.min(0.06 * z, len * 0.3), Math.min(0.03 * z, len * 0.15)));
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
    }
  }

  // Depth shading: a point's colour moves toward the background the farther it is from the viewer
  // (up to 60% for grey points, 35% for the selected ones), so the cloud's depth reads as it turns.
  shadeDepth() {
    if (!this.mesh || !this.current || !this.base) return;
    const fwd = new THREE.Vector3(); this.camera.getWorldDirection(fwd);
    const ds = this.current.map((p) => p.dot(fwd));
    let lo = Infinity, hi = -Infinity; for (const d of ds) { if (d < lo) lo = d; if (d > hi) hi = d; }
    const span = Math.max(hi - lo, 1e-6), c = new THREE.Color();
    ds.forEach((d, i) => {
      const t = (d - lo) / span;   // 0 nearest, 1 farthest
      c.copy(this.base[i]).lerp(this.bgCol, (this.isSel[i] ? 0.35 : 0.6) * t);
      this.mesh.setColorAt(i, c);
    });
    this.mesh.instanceColor.needsUpdate = true;
    // The axes: near end `ink` of the way from the background to the ink colour, far side 5%.
    if (this.globe) {
      for (const line of this.globe.children) {
        const pos = line.geometry.attributes.position, col = line.geometry.attributes.color;
        for (let k = 0; k < pos.count; k++) {
          const t = (1 - (pos.getX(k) * fwd.x + pos.getY(k) * fwd.y + pos.getZ(k) * fwd.z)) / 2;   // 1 nearest
          c.copy(this.bgCol).lerp(this.globeInk, 0.05 + (line.userData.ink - 0.05) * t);
          col.setXYZ(k, c.r, c.g, c.b);
        }
        col.needsUpdate = true;
      }
    }
  }

  // The view turns about the origin, which is the centre of the frame. The frame holds every point (and
  // so every selected vector) at every angle of a turn about the vertical axis, seen from the current
  // elevation; the spanned volume's far corner may pass the edge. A point at height y and distance ρ from the vertical axis stays within
  // |x| ≤ ρ and |y'| ≤ |y| cos(el) + ρ sin(el) on screen.
  frameFor(o, s) {
    const pts = o.coords.map((c, i) => c.map((x) => x * (o.scaled ? o.c[i] : 1) * s));
    const dir = new THREE.Vector3().subVectors(this.camera.position, this.controls.target).normalize();
    const el = Math.asin(Math.max(-1, Math.min(1, dir.y))), ce = Math.cos(el), se = Math.abs(Math.sin(el));
    let mx = 0, my = 0;
    for (const [x, y, z] of pts) { const rho = Math.hypot(x, z); mx = Math.max(mx, rho); my = Math.max(my, Math.abs(y) * ce + rho * se); }
    const W = this.el.clientWidth || 600, H = this.el.clientHeight || 450;
    return { ctr: new THREE.Vector3(0, 0, 0), h: Math.max(0.1, 1.06 * Math.max(my, mx * H / W)) };
  }
  setHalfHeight(h) {
    const W = this.el.clientWidth || 600, H = this.el.clientHeight || 450, a = W / H;
    Object.assign(this.camera, { left: -h * a, right: h * a, top: h, bottom: -h });
    this.camera.updateProjectionMatrix();
  }

  tick() {
    if (this.goal) {
      // Glide the view to the new frame: move the target (and the camera with it) and the zoom.
      const g = this.goal, d = new THREE.Vector3().subVectors(g.ctr, this.controls.target).multiplyScalar(0.15);
      this.controls.target.add(d); this.camera.position.add(d);
      const h = this.camera.top + (g.h - this.camera.top) * 0.15;
      this.setHalfHeight(h);
      if (d.length() < 1e-4 && Math.abs(g.h - h) < 1e-4) this.goal = null;
    }
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
    // The nearest point on screen within 10 px (the dots are too small to hit exactly).
    const i = this.nearest(e.clientX, e.clientY, 10);
    if (i !== this.hovered) { this.hovered = i; this.layout(); }
    this.onHover(i, e.clientX, e.clientY);
  }

  // Index of the point drawn nearest to a screen position, or -1 if none within r px.
  nearest(x, y, r) {
    if (!this.current) return -1;
    const rect = this.renderer.domElement.getBoundingClientRect();
    let best = -1, bd = r * r, bz = Infinity;
    const v = new THREE.Vector3();
    this.current.forEach((p, i) => {
      v.copy(p).project(this.camera);
      const sx = rect.left + ((v.x + 1) / 2) * rect.width, sy = rect.top + ((1 - v.y) / 2) * rect.height;
      const d = (sx - x) ** 2 + (sy - y) ** 2;
      if (d < bd || (d === bd && v.z < bz)) { bd = d; bz = v.z; best = i; }
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

  highlight(i) { this.hovered = i; this.layout(); }

  dispose() {
    this.alive = false;
    this.ro?.disconnect();
    this.controls.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
