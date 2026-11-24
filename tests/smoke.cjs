// Headless smoke test: node tests/smoke.cjs <url> <outdir>
const { chromium } = require('playwright');
(async () => {
  const [url, out] = process.argv.slice(2);
  const browser = await chromium.launch();
  const errors = [];
  for (const [name, vp] of [['desktop', { width: 1280, height: 900 }], ['phone', { width: 390, height: 844 }]]) {
    const page = await browser.newPage({ viewport: vp, deviceScaleFactor: 1 });
    page.on('console', (m) => { if (m.type() === 'error') errors.push(`[${name}] ${m.text()}`); });
    page.on('pageerror', (e) => errors.push(`[${name}] ${e.message}`));
    await page.goto(url, { waitUntil: 'networkidle' });
    await page.waitForSelector('.card', { timeout: 20000 });
    await page.waitForTimeout(2500);
    await page.screenshot({ path: `${out}/${name}-hero.png` });
    if (name === 'desktop') {
      const sl = await page.$('#T-slider');
      await sl.evaluate((el) => { el.value = 250; el.dispatchEvent(new Event('input')); });
      await page.waitForTimeout(800);
      await page.screenshot({ path: `${out}/desktop-lowT.png` });
      await page.click('.pick-btn[data-mode=sample]');
      await page.waitForTimeout(500);
      await page.click('.pick-btn[data-mode=map]');
      await page.waitForTimeout(800);
      await page.screenshot({ path: `${out}/desktop-map.png` });
      await page.click('#eigbars .bar[data-k="2"]');
      await page.waitForTimeout(1500);
      await page.screenshot({ path: `${out}/desktop-2d.png` });
      await page.click('#eigbars .bar[data-k="1"]');
      await page.waitForTimeout(800);
      await page.screenshot({ path: `${out}/desktop-1d.png` });
      await page.evaluate(() => document.querySelector('.pool-table').open = true);
      await page.waitForTimeout(1500);
      await page.screenshot({ path: `${out}/desktop-full.png`, fullPage: true });
    }
    await page.close();
  }
  console.log(errors.length ? errors.join('\n') : 'no console errors');
  await browser.close();
})();
