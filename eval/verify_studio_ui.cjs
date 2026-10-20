/* Browser regression: real conversion, color editing/export, history and responsive layout. */
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
let playwright;
try { playwright = require('playwright'); }
catch { playwright = require(path.join(process.env.USERPROFILE, '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright')); }
const { Resvg } = require('./tools/node_modules/@resvg/resvg-js');

const root = path.resolve(__dirname, '..');
const out = path.resolve(root, '../UI改版验证_2026-10-20');
const fixture = '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="512"><rect width="640" height="512" fill="white"/><path fill="#DC1E32" d="M80 65H285V447H80ZM120 110V400H245V110Z" fill-rule="evenodd"/><path fill="#1941CD" d="M355 65H560V447H355ZM395 110V400H520V110Z" fill-rule="evenodd"/></svg>';

function pixels(svg) {
  const rendered = new Resvg(svg).render();
  const data = rendered.pixels;
  let visible = 0;
  let green = 0;
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] > 128) visible += 1;
    if (data[i] === 34 && data[i + 1] === 170 && data[i + 2] === 102 && data[i + 3] > 128) green += 1;
  }
  return { visible, green };
}

(async () => {
  await fs.mkdir(out, { recursive: true });
  const input = path.join(out, 'two-colors.png');
  await fs.writeFile(input, new Resvg(fixture).render().asPng());
  const browser = await playwright.chromium.launch({ channel: 'msedge', headless: true });
  const errors = [];
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.env.ZIJI_TEST_URL || 'http://127.0.0.1:8768/');
    await page.waitForFunction(() => document.querySelectorAll('#preset-group button').length > 0);
    await page.locator('.quick-sample').first().waitFor();
    assert.equal(await page.locator('#sidebar').isVisible(), false);
    assert.equal(await page.locator('#canvas-toolbar').isVisible(), false);
    assert.equal(await page.locator('#action-bar').isVisible(), false);
    assert.equal(await page.locator('a[href="/docs"]').isVisible(), false);
    await page.screenshot({ path: path.join(out, 'desktop-empty.png'), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({ path: path.join(out, 'mobile-empty.png'), fullPage: true });
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.locator('#file-single').setInputFiles(input);
    await page.waitForFunction(() => state.inspect !== null);
    assert.equal(await page.locator('#sidebar').isVisible(), false);
    await page.locator('#check-go').click();
    await page.waitForFunction(() => state.phase === 'result', null, { timeout: 90000 });
    assert.equal(await page.locator('#sidebar').isVisible(), true);
    assert.equal(await page.locator('#param-card').getAttribute('open'), null);
    assert.equal(await page.locator('#result-details').getAttribute('open'), null);
    const geometry = await page.evaluate(() => {
      const canvas = document.querySelector('#canvas').getBoundingClientRect();
      const panel = document.querySelector('#sidebar').getBoundingClientRect();
      return { canvas: { x: canvas.x, right: canvas.right, width: canvas.width, height: canvas.height }, panel: { x: panel.x, width: panel.width } };
    });
    assert(geometry.canvas.right < geometry.panel.x && geometry.canvas.width > 900 && geometry.canvas.height > 550, JSON.stringify(geometry));
    assert.equal(await page.locator('[data-layer-color]').count(), 2);
    const original = await page.evaluate(() => currentSvg());
    await page.waitForFunction(() => [...document.querySelectorAll('#ls-out,#side-out')].every(img => img.complete));
    await page.screenshot({ path: path.join(out, 'desktop-result.png'), fullPage: true });

    // Real picker input events update both the SVG preview and downloaded file.
    await page.locator('[data-layer-color]').first().evaluate(input => {
      input.value = '#22aa66';
      input.dispatchEvent(new Event('input', { bubbles: true }));
      input.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await page.waitForFunction(() => !!state.editedSvg);
    assert.equal(await page.locator('#layer-vector').isVisible(), true);
    assert.equal(await page.locator('[data-view="diff"]').isDisabled(), true);
    const downloadEvent = page.waitForEvent('download');
    await page.locator('#act-primary').click();
    const download = await downloadEvent;
    const downloadedPath = path.join(out, 'edited-download.svg');
    await download.saveAs(downloadedPath);
    const edited = await fs.readFile(downloadedPath, 'utf8');
    assert.equal(edited, await page.evaluate(() => currentSvg()));
    assert(pixels(edited).green > 5000);
    await page.locator('[data-layer-toggle]').first().click();
    const hidden = await page.evaluate(() => currentSvg());
    assert(pixels(hidden).visible < pixels(edited).visible * .8);
    await page.locator('[data-layer-toggle]').first().click();
    assert.equal(pixels(await page.evaluate(() => currentSvg())).visible, pixels(edited).visible);
    await page.waitForFunction(() => document.querySelector('#vector-out').complete);
    await page.screenshot({ path: path.join(out, 'desktop-edited.png'), fullPage: true });

    // Edited history must reopen without borrowing another image as its original.
    await page.locator('#act-close').click();
    await page.locator('#open-history').click();
    await page.locator('.hist').first().click();
    assert.equal(await page.evaluate(() => currentSvg()), edited);
    assert.equal(await page.locator('#rerun-convert').isDisabled(), true);
    assert.equal(await page.locator('[data-view="slider"]').isDisabled(), true);
    await page.locator('#reset-layers').click();
    assert.equal(await page.evaluate(() => currentSvg()), original);

    // All inspection modes and the keyboard comparison handle remain usable.
    await page.locator('#file-single').setInputFiles(input);
    await page.waitForFunction(() => state.inspect !== null);
    await page.locator('#check-go').click();
    await page.waitForFunction(() => state.phase === 'result', null, { timeout: 90000 });
    await page.locator('[data-view="side"]').click();
    assert.equal(await page.locator('#layer-side').isVisible(), true);
    await page.locator('[data-view="diff"]').click();
    assert.equal(await page.locator('#layer-diff').isVisible(), true);
    await page.locator('[data-view="code"]').click();
    assert.equal(await page.locator('#layer-code').textContent(), await page.evaluate(() => currentSvg()));
    await page.locator('[data-view="slider"]').click();
    await page.locator('#ls-handle').focus();
    await page.keyboard.press('ArrowRight');
    assert.equal(await page.locator('#ls-handle').getAttribute('aria-valuenow'), '52');
    await page.locator('#result-details > summary').click();
    await page.locator('#metric-strip button.m').first().click();
    assert.equal(await page.locator('#metric-strip button.m .exp').first().isVisible(), true);
    await page.locator('#result-details > summary').click();

    // Advanced parameters still mark results stale and offer an exact restoration.
    await page.locator('#param-card > summary').click();
    await page.locator('[data-preset="quality"]').click();
    assert.equal(await page.evaluate(() => state.stale), true);
    await page.locator('#restore-params').click();
    assert.equal(await page.evaluate(() => state.stale), false);
    await page.locator('#param-card > summary').click();

    // Desktop/tablet boundary keeps either a readable side panel or a clean stack.
    await page.setViewportSize({ width: 1024, height: 900 });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    const tabletWide = await page.evaluate(() => {
      const panel = document.querySelector('#sidebar').getBoundingClientRect();
      const work = document.querySelector('#work').getBoundingClientRect();
      return { panelLeft: panel.left, workRight: work.right };
    });
    assert(tabletWide.panelLeft >= tabletWide.workRight);
    await page.setViewportSize({ width: 800, height: 900 });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    const tabletStack = await page.evaluate(() => {
      const panel = document.querySelector('#sidebar').getBoundingClientRect();
      const work = document.querySelector('#work').getBoundingClientRect();
      return { panelTop: panel.top, workBottom: work.bottom };
    });
    assert(tabletStack.panelTop >= tabletStack.workBottom);

    // Narrow screens stack the panel below the canvas and keep controls accessible.
    await page.setViewportSize({ width: 390, height: 844 });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    const mobile = await page.evaluate(() => {
      const panel = document.querySelector('#sidebar').getBoundingClientRect();
      const work = document.querySelector('#work').getBoundingClientRect();
      return { panelTop: panel.top, workBottom: work.bottom };
    });
    assert(mobile.panelTop >= mobile.workBottom);
    await page.screenshot({ path: path.join(out, 'mobile-result.png'), fullPage: true });
    await page.setViewportSize({ width: 1440, height: 1000 });

    // Gradients and linked effects retain their definitions and separate controls.
    const specialSvg = '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="120"><defs><linearGradient id="g"><stop stop-color="#FF0000"/><stop offset="1" stop-color="#0000FF"/></linearGradient><filter id="f"><feGaussianBlur in="SourceAlpha" stdDeviation="2"/><feFlood flood-color="#00FF00" flood-opacity=".5"/><feComposite in2="SourceAlpha" operator="in"/></filter></defs><g fill="url(#g)"><path d="M10 10H90V100H10Z"/></g><g fill="#123456" opacity=".6"><path d="M110 10H190V100H110Z" filter="url(#f)"/></g></svg>';
    const special = await page.evaluate(svg => {
      const model = SVGColorLayers.parse(svg);
      const check = (condition) => { if (!condition) throw new Error('Special SVG assertion failed'); };
      check(SVGColorLayers.serialize(model) === svg);
      const gradient = model.layers.find(layer => layer.gradient);
      const solid = model.layers.find(layer => layer.color === '#123456');
      const effect = model.layers.find(layer => layer.kind === 'effect');
      check(gradient && solid && effect);
      solid.color = '#ABCDEF';
      effect.visible = false;
      const changed = SVGColorLayers.serialize(model);
      const doc = new DOMParser().parseFromString(changed, 'image/svg+xml');
      check(doc.querySelector('g').getAttribute('fill') === 'url(#g)');
      check(doc.querySelector('stop').getAttribute('stop-color') === '#FF0000');
      check(doc.querySelector('g[opacity]').getAttribute('opacity') === '.6');
      check(doc.querySelector('feFlood').getAttribute('flood-opacity') === '0');
      return { layers: model.layers.length, retainsGradient: true, retainsOpacity: true, independentEffect: true };
    }, specialSvg);

    // Real queue conversion and ZIP download keep a separate output format control.
    await page.locator('[data-mode="batch"]').click();
    await page.locator('#batch-output-format').selectOption('pdf');
    assert.equal(await page.evaluate(() => state.fmt), 'pdf');
    await page.locator('#batch-output-format').selectOption('svg');
    assert.equal(await page.locator('#color-layer-card').isVisible(), false);
    await page.locator('#file-batch').setInputFiles([input]);
    await page.locator('#batch-run').click();
    await page.waitForFunction(() => !state.batchRunning && state.queue[0]?.status === 'ok', null, { timeout: 90000 });
    const zipEvent = page.waitForEvent('download');
    await page.locator('#batch-zip').click();
    await (await zipEvent).saveAs(path.join(out, 'batch.zip'));
    await page.screenshot({ path: path.join(out, 'desktop-batch.png'), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.setViewportSize({ width: 1440, height: 1000 });

    // Render a real bundled lettering sample, rather than only the geometric fixture.
    await page.goto('http://127.0.0.1:8768/?sample=13_cartoon.png');
    await page.waitForFunction(() => state.phase === 'result', null, { timeout: 90000 });
    await page.locator('[data-view="vector"]').click();
    await page.waitForFunction(() => document.querySelector('#vector-out').complete);
    await page.screenshot({ path: path.join(out, 'desktop-artwork.png'), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({ path: path.join(out, 'mobile-artwork.png'), fullPage: true });
    assert.deepEqual(errors, []);
    const report = { realUploadConversion: true, hiddenInitialSettings: true, rightPanel: geometry, colorEditExportMatches: true,
      hiddenLayerChangesPixels: true, exactReset: true, editedHistory: true, responsiveBreakpoints: true,
      mobileNoOverflow: true, staleParameterRestore: true,
      specialSvg: special, inspectionViews: true, batchFormatSelector: true, batchConversionAndZip: true, realArtworkPreview: true, pageErrors: errors };
    await fs.writeFile(path.join(out, 'verification.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
