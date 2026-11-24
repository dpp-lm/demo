// Finds content spilling out of its box: every descendant of a bordered container must lie inside it.
const { chromium } = require('playwright');
const WIDTHS = [320, 360, 390, 430, 600, 700, 768, 900, 1024, 1280, 1440];
const CONTAINERS = '.step, .stage, .composer, .chipmenu .menu, .viewmenu .menu, .card, .stage-foot, .stage-bar, .wip, .pick, .stepper, .rail, .drawer, .dock';  // boxes with a visible edge or background
(async () => {
  const b = await chromium.launch(); let problems = 0;
  for (const w of WIDTHS) {
    const mobile = w <= 600;
    const p = await b.newPage({ viewport: { width: w, height: 900 }, isMobile: mobile, hasTouch: mobile });
    const errs = []; p.on('pageerror', e => errs.push(e.message));
    await p.goto(process.argv[2] || 'http://localhost:8765/', { waitUntil: 'networkidle' }); await p.waitForTimeout(1500);
    for (const state of ['plain', 'settings', 'examples', 'sample', 'details', 'view']) {
      await p.evaluate((st) => {
        for (const id of ['#adv', '#viewmenu', '#exmenu']) document.querySelector(id).open = false;
        if (st === 'settings') document.querySelector('#adv').open = true;
        // Details: the drawer (wide screens) or the collapsible section (phones); the View menu lives in it.
        const det = document.querySelector('#details');
        if (st === 'details' || st === 'view') { if (!det.classList.contains('open')) (innerWidth > 900 ? document.querySelector('#details-btn') : document.querySelector('#details-toggle')).click(); }
        else if (det.classList.contains('open')) (innerWidth > 900 ? document.querySelector('#details-close') : document.querySelector('#details-toggle')).click();
        if (st === 'view') document.querySelector('#viewmenu').open = true;
        if (st === 'examples') document.querySelector('#exmenu').open = true;
        if (st === 'sample') document.querySelector('.pick-btn[data-mode=sample]').click();
      }, state);
      await p.waitForTimeout(400);
      const found = await p.evaluate((sel) => {
        const out = [];
        const vis = (e) => { const cs = getComputedStyle(e); const r = e.getBoundingClientRect(); return cs.visibility !== 'hidden' && cs.display !== 'none' && r.width > 0 && r.height > 0; };
        const name = (e) => e.tagName.toLowerCase() + (e.id ? '#' + e.id : '') + (e.className && typeof e.className === 'string' ? '.' + e.className.trim().split(/\s+/).join('.') : '');
        for (const c of document.querySelectorAll(sel)) {
          if (!vis(c)) continue;
          const cr = c.getBoundingClientRect();
          // A container that scrolls (the rail) may hold more than it shows: check only the other axis.
          const csy = getComputedStyle(c).overflowY, scrollsY = csy === 'auto' || csy === 'scroll';
          for (const e of c.querySelectorAll('*')) {
            if (!vis(e) || e.closest('.labels, canvas, svg, .xpop') && e !== c) continue;
            if (e.closest('.chipmenu .menu, .viewmenu .menu') && !c.matches('.chipmenu .menu, .viewmenu .menu') && c.contains(e.closest('.chipmenu .menu, .viewmenu .menu'))) continue; // pop-up menus may extend beyond their anchor
            // What is visible: the box, cut by any ancestor (inside c) that clips its overflow.
            const r = e.getBoundingClientRect().toJSON();
            for (let a = e.parentElement; a && a !== c; a = a.parentElement) {
              if (getComputedStyle(a).overflowX === 'visible' && getComputedStyle(a).overflowY === 'visible') continue;
              const ar = a.getBoundingClientRect();
              r.left = Math.max(r.left, ar.left); r.right = Math.min(r.right, ar.right); r.top = Math.max(r.top, ar.top); r.bottom = Math.min(r.bottom, ar.bottom);
            }
            const d = Math.max(cr.left - r.left, r.right - cr.right, scrollsY ? 0 : cr.top - r.top, scrollsY ? 0 : r.bottom - cr.bottom);
            if (d > 0.5) out.push(`${name(e)} spills ${d.toFixed(1)}px out of ${name(c)}`);
          }
        }
        // Open menus must be on top: sample points across each one and check nothing covers them.
        for (const m of document.querySelectorAll('details[open] > .menu')) {
          const r = m.getBoundingClientRect();
          for (const fx of [0.1, 0.5, 0.9]) for (const fy of [0.1, 0.5, 0.9]) {
            const x = r.left + fx * r.width, y = r.top + fy * r.height;
            if (x < 0 || y < 0 || x >= innerWidth || y >= innerHeight) continue;
            const hit = document.elementFromPoint(x, y);
            if (hit && !m.contains(hit)) out.push(`${name(m)} is covered by ${name(hit)} at (${Math.round(x)}, ${Math.round(y)})`);
          }
        }
        // Open menus stay inside the window (sideways always; vertically where the page does not scroll).
        for (const m of document.querySelectorAll('details[open] > .menu')) {
          const r = m.getBoundingClientRect(), pageScrolls = document.documentElement.scrollHeight > innerHeight + 1;
          if (r.left < -0.5 || r.right > innerWidth + 0.5 || (!pageScrolls && (r.top < -0.5 || r.bottom > innerHeight + 0.5))) out.push(`${name(m)} is off screen (${Math.round(r.left)}, ${Math.round(r.top)}, ${Math.round(r.right)}, ${Math.round(r.bottom)})`);
        }
        if (document.documentElement.scrollWidth > innerWidth + 0.5) out.push(`page scrolls sideways (${document.documentElement.scrollWidth} > ${innerWidth})`);
        return [...new Set(out)];
      }, CONTAINERS);
      for (const f of found.slice(0, 6)) console.log(`${w}px ${state}: ${f}`);
      problems += found.length;
    }
    if (errs.length) console.log(w, 'errors', errs);
    await p.close();
  }
  console.log(problems ? `${problems} problems` : 'no spills');
  await b.close();
  process.exitCode = problems ? 1 : 0;
})();
