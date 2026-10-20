/* Run with the bundled Node runtime (or a runtime with Playwright installed). */
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
let playwright;
try { playwright = require('playwright'); }
catch { playwright = require(path.resolve(path.dirname(process.execPath), '../node_modules/playwright')); }

(async () => {
  const root = path.resolve(__dirname, '..');
  const out = path.resolve(root, '../原图主色验证_2026-10-19');
  await fs.mkdir(out, { recursive: true });
  const input = path.resolve(root, '../专色匹配验证_2026-10-19/input.png');
  const browser = await playwright.chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('http://127.0.0.1:8768/');
    await page.locator('#file-single').setInputFiles(input);
    await page.locator('#check-go').click();
    const card = page.locator('.source-colors');
    await card.waitFor({ state: 'visible', timeout: 90000 });
    const before = await card.innerText();
    assert(before.includes('#DC1E32') && before.includes('#1941CD'), before);
    assert(before.includes('RGB 220, 30, 50') && before.includes('RGB 25, 65, 205'), before);
    assert(!before.includes('#FFFFFF'), before);
    await card.locator('.palette-swatch').first().click();
    await page.locator('#metric-strip button.m').first().click();
    assert.equal(await page.locator('#metric-strip button.m .exp').first().isVisible(), true);
    await page.locator('#metric-strip').screenshot({ path: path.join(out, 'desktop-colors.png') });
    // Use the real form and conversion path again, with the optional recoloring enabled.
    await page.evaluate(() => {
      document.querySelector('#spot_colors').value = '#00ff00';
      document.querySelector('#spot_matching').classList.remove('off');
      startConvert();
    });
    await page.waitForFunction(() => state.phase === 'result' && state.result?.stats?.spot_matching?.enabled,
                               null, { timeout: 90000 });
    assert.equal(await card.innerText(), before);
    const result = await page.evaluate(() => ({ palette: state.result.metrics.source_colors,
      matching: state.result.stats.spot_matching, svg: state.result.svg }));
    assert(result.matching.matched_layers > 0);
    assert(result.svg.toLowerCase().includes('#00ff00'));
    await page.waitForFunction(() => document.querySelector('#toast-host').children.length === 0,
                               null, { timeout: 15000 });
    await page.setViewportSize({ width: 390, height: 844 });
    await card.scrollIntoViewIfNeeded();
    const layout = await card.evaluate(el => {
      const r = el.getBoundingClientRect();
      return { left: r.left, right: r.right, viewport: innerWidth,
        overflow: el.scrollWidth > el.clientWidth + 1 };
    });
    assert(layout.left >= 0 && layout.right <= layout.viewport + 1 && !layout.overflow, JSON.stringify(layout));
    await card.screenshot({ path: path.join(out, 'mobile-colors.png') });
    assert.deepEqual(errors, []);
    const report = { palette: result.palette, matching: result.matching,
      checks: { actual_upload_conversion: true, source_colors_survive_recoloring: true,
        white_background_excluded: true, metric_clicks: true, mobile_card_fits: true, page_errors: errors } };
    await fs.writeFile(path.join(out, 'browser-verification.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
