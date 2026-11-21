// Live-mode end-to-end check in headless Chromium: node tests/live.cjs <url> <out.png>
const { chromium } = require('playwright');
(async () => {
  const [url, out] = process.argv.slice(2);
  const proxy = process.env.HTTPS_PROXY ? { server: process.env.HTTPS_PROXY } : undefined;
  const browser = await chromium.launch({ proxy: proxy && { ...proxy, bypass: 'localhost,127.0.0.1' } });
  const page = await browser.newPage({ viewport: { width: 1280, height: 1000 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.fill('#prompt', 'Name a color and say one thing about it, in one sentence.');
  await page.click('#adv summary');
  await page.fill('#live-N', '12');
  await page.fill('#live-max', '24');
  await page.click('#live-go');
  const t0 = Date.now();
  let last = '';
  while (Date.now() - t0 < 900000) {
    const st = await page.textContent('#live-status');
    if (st !== last) { console.log(((Date.now() - t0) / 1000).toFixed(0) + 's', st); last = st; }
    if (/^Drew|^Error/.test(st)) break;
    await page.waitForTimeout(2000);
  }
  await page.waitForTimeout(1500);
  await page.screenshot({ path: out });
  console.log('cards:', await page.$$eval('.card', (c) => c.length));
  console.log(errors.length ? 'errors:\n' + errors.join('\n') : 'no console errors');
  await browser.close();
})();
