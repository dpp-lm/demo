// Finds content spilling out of its box: every descendant of a bordered container must lie inside it.
const { chromium } = require('playwright');
const WIDTHS = [320, 360, 390, 430, 600, 700, 768, 900, 1024, 1280, 1440];
const CONTAINERS = '.step, .stage, .composer, .chipmenu .menu, .viewmenu .menu, .card, .stage-foot, .stage-bar, .wip, .pool-table, .eigbars, .pick, .stepper, .tfield, .mode-row';
(async () => {
  const b = await chromium.launch(); let problems = 0;
  for (const w of WIDTHS) {
    const mobile = w <= 600;
    const p = await b.newPage({ viewport: { width: w, height: 900 }, isMobile: mobile, hasTouch: mobile });
    const errs = []; p.on('pageerror', e => errs.push(e.message));
    await p.goto(process.argv[2] || 'http://localhost:8765/', { waitUntil: 'networkidle' }); await p.waitForTimeout(1500);
    for (const state of ['plain', 'settings', 'view', 'examples', 'sample']) {
      await p.evaluate((st) => {
        for (const id of ['#adv', '#viewmenu', '#exmenu']) document.querySelector(id).open = false;
        if (st === 'settings') document.querySelector('#adv').open = true;
        if (st === 'view') document.querySelector('#viewmenu').open = true;
        if (st === 'examples') document.querySelector('#exmenu').open = true;
        if (st === 'sample') document.querySelector('.pick-btn[data-mode=sample]').click();
      }, state);
      await p.waitForTimeout(300);
      const found = await p.evaluate((sel) => {
        const out = [];
        const vis = (e) => { const cs = getComputedStyle(e); const r = e.getBoundingClientRect(); return cs.visibility !== 'hidden' && cs.display !== 'none' && r.width > 0 && r.height > 0; };
        const name = (e) => e.tagName.toLowerCase() + (e.id ? '#' + e.id : '') + (e.className && typeof e.className === 'string' ? '.' + e.className.trim().split(/\s+/).join('.') : '');
        for (const c of document.querySelectorAll(sel)) {
          if (!vis(c)) continue;
          const cr = c.getBoundingClientRect();
          for (const e of c.querySelectorAll('*')) {
            if (!vis(e) || e.closest('.labels, canvas, svg, .xpop') && e !== c) continue;
            if (e.closest('.chipmenu .menu, .viewmenu .menu') && !c.matches('.chipmenu .menu, .viewmenu .menu') && c.contains(e.closest('.chipmenu .menu, .viewmenu .menu'))) continue; // pop-up menus may extend beyond their anchor
            const r = e.getBoundingClientRect();
            const d = Math.max(cr.left - r.left, r.right - cr.right, cr.top - r.top, r.bottom - cr.bottom);
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
})();
