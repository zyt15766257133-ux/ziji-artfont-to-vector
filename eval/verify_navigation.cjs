/* Real browser navigation, retained edits, batch queue and in-flight cancellation. */
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
let playwright;
try { playwright = require('playwright'); }
catch { playwright = require(path.join(process.env.USERPROFILE, '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright')); }
const root = path.resolve(__dirname, '..');
const url = process.env.ZIJI_TEST_URL || 'http://127.0.0.1:8768/';
const out = path.resolve(root, '../UI功能目录验证_2026-10-21');
const first = path.join(root, 'samples/images/01_hei_solid.png');
const second = path.join(root, 'samples/images/13_cartoon.png');

(async () => {
  await fs.mkdir(out, { recursive: true });
  const browser = await playwright.chromium.launch({ channel: 'msedge', headless: true });
  const errors = [];
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    page.on('pageerror', error => errors.push(error.message));
    let jobs = 0;
    page.on('request', request => { if (request.method() === 'POST' && request.url().endsWith('/api/jobs')) jobs += 1; });
    const ready = async () => {
      await page.goto(url);
      await page.waitForFunction(() => document.querySelectorAll('#preset-group button').length > 0);
    };
    const result = () => page.waitForFunction(() => state.phase === 'result' && state.mode === 'single', null, { timeout: 90000 });
    const home = () => page.waitForFunction(() => state.phase === 'empty' && state.mode === 'single');
    const menu = () => page.waitForFunction(() => state.phase === 'result' && !state.editorTool);
    await ready();
    await page.locator('#file-single').setInputFiles(first);
    await result();
    await page.locator('[data-tool="colors"]').click();
    await page.locator('[data-layer-color]').first().evaluate(input => {
      input.value = '#22aa66';
      input.dispatchEvent(new Event('input', { bubbles: true }));
      input.dispatchEvent(new Event('change', { bubbles: true }));
    });
    await page.waitForFunction(() => !!state.editedSvg);
    const edited = await page.evaluate(() => currentSvg());
    await page.goBack();
    await menu();
    await page.goBack();
    await home();
    assert.equal(page.url(), url);
    assert.equal(await page.locator('#empty-state').isVisible(), true);
    await page.screenshot({ path: path.join(out, 'navigation-home.png'), fullPage: true });
    const completedJobs = jobs;
    await page.goForward();
    await result();
    assert.equal(await page.evaluate(() => currentSvg()), edited);
    assert.equal(await page.evaluate(() => !!state.file), true);
    assert.equal(await page.evaluate(() => state.historyOnly), false);
    await page.waitForFunction(() => document.querySelector('#ls-in').complete && document.querySelector('#ls-in').naturalWidth > 0);
    await page.goForward();
    await page.waitForFunction(() => state.editorTool === 'colors');
    assert.equal(await page.locator('[data-layer-color]').first().inputValue(), '#22aa66');
    assert.equal(jobs, completedJobs, 'Forward must reuse the result without another conversion');
    await page.locator('#toolbar-download').click();
    await page.goBack();
    await menu();
    await page.goForward();
    await page.waitForFunction(() => state.editorTool === 'export');
    await page.locator('#tool-back').click();
    await menu();
    await page.locator('#editor-back').click();
    await home();
    await page.goForward();
    await result();
    await page.screenshot({ path: path.join(out, 'navigation-result.png'), fullPage: true });

    // A second upload must not overwrite the first browser-history entry.
    await page.locator('#file-single').setInputFiles(second);
    await result();
    const nextSvg = await page.evaluate(() => currentSvg());
    assert.notEqual(nextSvg, edited);
    await page.goBack();
    await page.waitForFunction(() => state.fileName === '01_hei_solid.png' && state.phase === 'result');
    assert.equal(await page.evaluate(() => currentSvg()), edited);
    await page.goForward();
    await page.waitForFunction(() => state.fileName === '13_cartoon.png' && state.phase === 'result');
    assert.equal(await page.evaluate(() => currentSvg()), nextSvg);
    await page.locator('#editor-batch').click();
    await page.locator('#file-batch').setInputFiles([first, second]);
    await page.goBack();
    await result();
    await page.goForward();
    await page.waitForFunction(() => state.mode === 'batch');
    assert.equal(await page.evaluate(() => state.queue.length), 2);
    await page.setViewportSize({ width: 390, height: 844 });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.evaluate(() => Promise.all(document.getAnimations().map(animation => animation.finished.catch(() => {}))));
    await page.screenshot({ path: path.join(out, 'navigation-batch-mobile.png'), fullPage: true });
    await page.locator('#page-back').click();
    await result();
    await page.locator('#editor-close').click();
    await home();
    await page.goBack();
    await result();
    assert.equal(await page.evaluate(() => currentSvg()), nextSvg);

    // Returning during image reading must ignore the late inspection response.
    await ready();
    let releaseInspect;
    const inspectGate = new Promise(resolve => { releaseInspect = resolve; });
    await page.route('**/api/inspect', async route => {
      const response = await route.fetch();
      await inspectGate;
      await route.fulfill({ response });
    });
    const beforeReadJobs = jobs;
    await page.locator('#file-single').setInputFiles(first);
    await page.waitForFunction(() => state.phase === 'checking');
    await page.locator('#page-back').click();
    await home();
    releaseInspect();
    await page.waitForLoadState('networkidle');
    assert.equal(jobs, beforeReadJobs);
    assert.equal(await page.evaluate(() => state.phase), 'empty');
    await page.unroute('**/api/inspect');

    // A job submitted just before Back must be cancelled when its response arrives.
    await ready();
    let releaseJob, submittedId;
    const jobGate = new Promise(resolve => { releaseJob = resolve; });
    await page.route('**/api/jobs', async route => {
      if (route.request().method() !== 'POST') { await route.continue(); return; }
      const response = await route.fetch();
      submittedId = (await response.json()).job_id;
      await jobGate;
      await route.fulfill({ response });
    });
    await page.locator('#file-single').setInputFiles(first);
    await page.waitForFunction(() => state.phase === 'converting');
    await page.goBack();
    await home();
    const cancelled = page.waitForRequest(request => request.method() === 'DELETE' && request.url().endsWith(`/api/jobs/${submittedId}`));
    releaseJob();
    await cancelled;
    await page.waitForLoadState('networkidle');
    assert.equal(await page.evaluate(() => state.phase), 'empty');
    assert.deepEqual(errors, []);
    const report = { browserBackAndForward: true, pageBackButtons: true, editsAndSourceRetained: true,
      forwardWithoutConversion: true, toolSwitchUsesOneLevel: true, multipleUploadsRestored: true,
      batchQueueRetained: true, mobileNoOverflow: true, closeResultRestorable: true,
      lateInspectionIgnored: true, lateJobCancelled: true, pageErrors: errors };
    await fs.writeFile(path.join(out, 'navigation-verification.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
