/* Browser regression: real conversion, color editing/export, history and responsive layout. */
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
let playwright;
try { playwright = require('playwright'); }
catch { playwright = require(path.join(process.env.USERPROFILE, '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright')); }
const { Resvg } = require('./tools/node_modules/@resvg/resvg-js');

const root = path.resolve(__dirname, '..');
const out = path.resolve(root, '../UI功能目录验证_2026-10-21');
const fixture = '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="512"><rect width="640" height="512" fill="white"/><path fill="#DC1E32" d="M80 40H285V472H80ZM120 85V427H245V85Z" fill-rule="evenodd"/><path fill="#1941CD" d="M355 40H560V472H355ZM395 85V427H520V85Z" fill-rule="evenodd"/></svg>';

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
    await page.locator('#stage').dispatchEvent('dragenter');
    assert.equal(await page.locator('#empty-state').evaluate(el => el.classList.contains('drag')), true);
    await page.screenshot({ path: path.join(out, 'portal-drag.png'), fullPage: true });
    await page.locator('#stage').dispatchEvent('dragleave');
    assert.equal(await page.locator('#empty-state').evaluate(el => el.classList.contains('drag')), false);
    await page.screenshot({ path: path.join(out, 'desktop-empty.png'), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({ path: path.join(out, 'mobile-empty.png'), fullPage: true });
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.locator('#file-single').setInputFiles(input);
    await page.waitForFunction(() => state.inspect !== null);
    assert.equal(await page.locator('#sidebar').isVisible(), false);
    // Clean uploads progress from inspection to conversion without an extra confirmation.
    await page.waitForFunction(() => state.phase === 'result', null, { timeout: 90000 });
    assert.equal(await page.locator('#sidebar').isVisible(), true);
    assert.equal(await page.locator('#param-card').getAttribute('open'), null);
    assert.equal(await page.locator('#result-details').getAttribute('open'), null);
    assert.equal(await page.locator('#editor-tools').isVisible(), true);
    for (const id of ['color-layer-card', 'preset-card', 'param-card', 'source-card', 'result-details', 'export-tool']) {
      assert.equal(await page.locator(`#${id}`).isVisible(), false, `Initial tool content should be hidden: ${id}`);
    }
    const geometry = await page.evaluate(() => {
      const canvas = document.querySelector('#canvas').getBoundingClientRect();
      const panel = document.querySelector('#sidebar').getBoundingClientRect();
      return { canvas: { x: canvas.x, right: canvas.right, width: canvas.width, height: canvas.height }, panel: { x: panel.x, width: panel.width } };
    });
    assert(geometry.panel.x + geometry.panel.width <= geometry.canvas.x
      && geometry.canvas.width > 900 && geometry.canvas.height > 650, JSON.stringify(geometry));
    await page.locator('#sidebar-toggle').click();
    await page.waitForTimeout(260);
    const collapsedGeometry = await page.evaluate(() => {
      const canvas = document.querySelector('#canvas').getBoundingClientRect();
      const panel = document.querySelector('#sidebar').getBoundingClientRect();
      return { canvasWidth: canvas.width, panelWidth: panel.width };
    });
    assert(collapsedGeometry.panelWidth < 70 && collapsedGeometry.canvasWidth > geometry.canvas.width + 150,
      JSON.stringify(collapsedGeometry));
    assert.equal(await page.locator('#sidebar-toggle').getAttribute('aria-expanded'), 'false');
    // Top download must also open its inspector when the whole sidebar was collapsed.
    await page.locator('#toolbar-download').click();
    assert.equal(await page.locator('#sidebar-toggle').getAttribute('aria-expanded'), 'true');
    assert.equal(await page.locator('#export-tool').isVisible(), true);
    await page.locator('#tool-back').click();
    await page.waitForTimeout(260);
    assert.equal(await page.locator('[data-layer-color]').count(), 2);
    const original = await page.evaluate(() => currentSvg());
    await page.waitForFunction(() => [...document.querySelectorAll('#ls-out,#side-out')].every(img => img.complete));
    await page.screenshot({ path: path.join(out, 'desktop-result.png'), fullPage: true });

    // Real picker input events update both the SVG preview and downloaded file.
    await page.locator('[data-tool="colors"]').click();
    assert.equal(await page.locator('#editor-tools').isVisible(), false);
    assert.equal(await page.locator('#preset-card').isVisible(), false);
    await page.waitForTimeout(200);
    await page.screenshot({ path: path.join(out, 'desktop-color-panel.png'), fullPage: true });
    await page.locator('[data-layer-color]').first().evaluate(input => {
      input.value = '#22aa66';
      input.dispatchEvent(new Event('input', { bubbles: true }));
      input.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await page.waitForFunction(() => !!state.editedSvg);
    assert.equal(await page.locator('#layer-vector').isVisible(), true);
    assert.equal(await page.locator('[data-view="diff"]').isDisabled(), true);
    await page.locator('#toolbar-download').click();
    assert.equal(await page.locator('#color-layer-card').isVisible(), false);
    const downloadEvent = page.waitForEvent('download');
    await page.locator('#act-primary').click();
    const download = await downloadEvent;
    const downloadedPath = path.join(out, 'edited-download.svg');
    await download.saveAs(downloadedPath);
    const edited = await fs.readFile(downloadedPath, 'utf8');
    assert.equal(edited, await page.evaluate(() => currentSvg()));
    assert(pixels(edited).green > 5000);
    await page.locator('#tool-back').click();
    await page.locator('[data-tool="colors"]').click();
    await page.locator('[data-layer-toggle]').first().click();
    const hidden = await page.evaluate(() => currentSvg());
    assert(pixels(hidden).visible < pixels(edited).visible * .8);
    await page.locator('[data-layer-toggle]').first().click();
    assert.equal(pixels(await page.evaluate(() => currentSvg())).visible, pixels(edited).visible);
    await page.waitForFunction(() => document.querySelector('#vector-out').complete);
    await page.screenshot({ path: path.join(out, 'desktop-edited.png'), fullPage: true });

    // Edited history must reopen without borrowing another image as its original.
    await page.locator('#editor-close').click();
    await page.locator('#open-history').click();
    await page.locator('.hist').first().click();
    assert.equal(await page.evaluate(() => currentSvg()), edited);
    assert.equal(await page.locator('#rerun-convert').isDisabled(), true);
    assert.equal(await page.locator('[data-view="slider"]').isDisabled(), true);
    await page.locator('[data-tool="colors"]').click();
    await page.locator('#reset-layers').click();
    assert.equal(await page.evaluate(() => currentSvg()), original);

    // All inspection modes and the keyboard comparison handle remain usable.
    await page.locator('#file-single').setInputFiles(input);
    await page.waitForFunction(() => state.inspect !== null);
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
    await page.locator('[data-tool="metrics"]').click();
    assert.equal(await page.locator('#verdict').isVisible(), true);
    await page.locator('#metric-strip button.m').first().click();
    assert.equal(await page.locator('#metric-strip button.m .exp').first().isVisible(), true);
    await page.locator('#tool-back').click();

    // Advanced parameters still mark results stale and offer an exact restoration.
    await page.locator('[data-tool="presets"]').click();
    await page.locator('[data-preset="quality"]').click();
    assert.equal(await page.evaluate(() => state.stale), true);
    await page.locator('#tool-back').click();
    await page.locator('[data-tool="advanced"]').click();
    await page.locator('#restore-params').click();
    assert.equal(await page.evaluate(() => state.stale), false);
    await page.locator('#tool-back').click();
    await page.locator('[data-tool="source"]').click();
    assert.equal(await page.locator('#source-card').isVisible(), true);
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('#editor-tools').isVisible(), true);

    // Desktop/tablet boundary keeps either a readable side panel or a clean stack.
    await page.setViewportSize({ width: 1024, height: 900 });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    const tabletWide = await page.evaluate(() => {
      const panel = document.querySelector('#sidebar').getBoundingClientRect();
      const work = document.querySelector('#work').getBoundingClientRect();
      return { panelRight: panel.right, workLeft: work.left };
    });
    assert(tabletWide.panelRight <= tabletWide.workLeft);
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
    await page.locator('[data-tool="colors"]').click();
    assert.equal(await page.locator('#color-layer-card').isVisible(), true);
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.locator('#tool-back').click();
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
    const preservedSingleSvg = await page.evaluate(() => currentSvg());
    await page.locator('#editor-batch').click();
    assert.equal(await page.locator('#batch-drop-zone').isVisible(), true);
    assert.equal(await page.locator('#batch-queue-card').isVisible(), false);
    await page.locator('#batch-drop-zone').dispatchEvent('dragenter');
    assert.equal(await page.locator('#batch-drop-zone').evaluate(el => el.classList.contains('is-dragging')), true);
    await page.locator('#batch-drop-zone').dispatchEvent('dragleave');
    await page.locator('.page').evaluate(el => Promise.all(el.getAnimations().map(animation => animation.finished)));
    await page.screenshot({ path: path.join(out, 'batch-empty.png'), fullPage: true });
    await page.locator('#batch-output-format').selectOption('pdf');
    assert.equal(await page.evaluate(() => state.fmt), 'pdf');
    await page.locator('#batch-output-format').selectOption('svg');
    assert.equal(await page.locator('#color-layer-card').isVisible(), false);
    await page.locator('#file-batch').setInputFiles([input]);
    const uploadBytes = (await fs.readFile(input)).toString('base64');
    await page.evaluate(base64 => {
      const bytes = Uint8Array.from(atob(base64), char => char.charCodeAt(0));
      const transfer = new DataTransfer();
      transfer.items.add(new File([bytes], 'dropped-second.png', { type: 'image/png' }));
      document.querySelector('#batch-drop-zone').dispatchEvent(new DragEvent('drop', { dataTransfer: transfer, bubbles: true, cancelable: true }));
    }, uploadBytes);
    assert.equal(await page.evaluate(() => state.queue.length), 2);
    assert.equal(await page.locator('#mode-group .batch-queue-count').textContent(), '2');
    await page.locator('#batch-run').click();
    await page.waitForFunction(() => !state.batchRunning && state.queue[0]?.status === 'ok', null, { timeout: 90000 });
    const zipEvent = page.waitForEvent('download');
    await page.locator('#batch-zip').click();
    await (await zipEvent).saveAs(path.join(out, 'batch.zip'));
    await page.screenshot({ path: path.join(out, 'desktop-batch.png'), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({ path: path.join(out, 'batch-mobile.png'), fullPage: true });
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.locator('#mode-group [data-mode="single"]').click();
    assert.equal(await page.evaluate(() => currentSvg()), preservedSingleSvg);
    assert.equal(await page.locator('#editor-batch').isVisible(), true);
    await page.screenshot({ path: path.join(out, 'editor-batch-entry.png'), fullPage: true });

    // Render a real bundled lettering sample, rather than only the geometric fixture.
    await page.goto(`${process.env.ZIJI_TEST_URL || 'http://127.0.0.1:8768/'}?sample=13_cartoon.png`);
    await page.waitForFunction(() => state.phase === 'result', null, { timeout: 90000 });
    await page.locator('[data-view="vector"]').click();
    await page.waitForFunction(() => document.querySelector('#vector-out').complete);
    await page.screenshot({ path: path.join(out, 'desktop-artwork.png'), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({ path: path.join(out, 'mobile-artwork.png'), fullPage: true });

    // A real warned upload must skip the confirmation screen too, retaining its warnings.
    await page.goto(process.env.ZIJI_TEST_URL || 'http://127.0.0.1:8768/');
    await page.waitForFunction(() => document.querySelectorAll('#preset-group button').length > 0);
    await page.evaluate(() => {
      window.uploadPhases = [];
      window.uploadObserver = new MutationObserver(records => {
        for (const record of records) window.uploadPhases.push(record.oldValue);
        window.uploadPhases.push(document.body.dataset.phase);
      });
      window.uploadObserver.observe(document.body, { attributes: true, attributeFilter: ['data-phase'], attributeOldValue: true });
    });
    await page.locator('#file-single').setInputFiles(path.join(root, 'samples/images/01_hei_solid.png'));
    await page.waitForFunction(() => state.phase === 'result', null, { timeout: 90000 });
    const warnedUpload = await page.evaluate(() => {
      window.uploadObserver.disconnect();
      return { phases: window.uploadPhases, warnings: state.inspect.warnings };
    });
    assert(warnedUpload.warnings.length > 0, 'Regression image should trigger quality warnings');
    assert(warnedUpload.phases.includes('checking') && warnedUpload.phases.includes('converting'));
    assert(!warnedUpload.phases.includes('inspect'), JSON.stringify(warnedUpload.phases));
    assert.equal(await page.locator('#check-card').isVisible(), false);
    await page.evaluate(() => openEditorTool('metrics'));
    for (const warning of warnedUpload.warnings) {
      assert((await page.locator('#metrics-note').textContent()).includes(warning));
    }
    await page.screenshot({ path: path.join(out, 'automatic-warned-upload.png'), fullPage: true });
    assert.deepEqual(errors, []);
    const report = { realUploadConversion: true, automaticCleanUpload: true, automaticWarnedUpload: warnedUpload, dragFeedback: true,
      hiddenInitialSettings: true, toolDirectoryAndPanels: true, topDownloadOpensExport: true,
      leftEditorPanel: geometry, collapsibleEditorPanel: collapsedGeometry,
      colorEditExportMatches: true,
      hiddenLayerChangesPixels: true, exactReset: true, editedHistory: true, responsiveBreakpoints: true,
      mobileNoOverflow: true, staleParameterRestore: true,
      specialSvg: special, inspectionViews: true, batchFormatSelector: true, batchDropAndQueueCount: true,
      batchPreservesSingleResult: true, batchConversionAndZip: true, realArtworkPreview: true, pageErrors: errors };
    await fs.writeFile(path.join(out, 'verification.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
