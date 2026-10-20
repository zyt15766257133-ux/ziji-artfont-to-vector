/* 字迹 · 艺术字转矢量 —— 前端交互（无框架） */
const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];

const PARAM_IDS = ['tolerance', 'max_nodes', 'max_colors', 'gradient_levels', 'filter_speckle', 'corner_angle'];
const PARAM_LABELS = {
  tolerance: '容差', max_nodes: '节点上限', max_colors: '颜色层',
  gradient_levels: '渐变色阶', filter_speckle: '杂点过滤', corner_angle: '角点角',
};
const PRESET_DEFAULTS = {
  fidelity: {},
  pixel_exact: {},
  quality: { tolerance: 0.25, max_nodes: 500, max_colors: 16, gradient_levels: 14, filter_speckle: 2, corner_angle: 55 },
  balanced: { tolerance: 0.5, max_nodes: 500, max_colors: 12, gradient_levels: 10, filter_speckle: 4, corner_angle: 62 },
  compact: { tolerance: 0.8, max_nodes: 500, max_colors: 8, gradient_levels: 8, filter_speckle: 8, corner_angle: 70 },
};
const PRESET_NAMES = { fidelity: '保真曲线', pixel_exact: '原像素路径', quality: '高质量', balanced: '均衡', compact: '精简' };
const PARAM_TIPS = {
  tolerance: {
    title: '轮廓贴合容差（像素）',
    rows: [['调小', '路径更贴原图边缘，细节保留多，节点变多、转换变慢'],
           ['调大', '轮廓更平滑简洁，细小笔触与尖角会被抹平']],
    dft: '默认 0.5px（均衡档）· 验收看轮廓还原度 ≥90%',
  },
  max_nodes: {
    title: '单字节点上限（个）',
    rows: [['调小', '每个字的节点更少、文件更小，细节损失'],
           ['调大', '细节保留多，但路径难编辑、文件变大']],
    dft: '默认 500 · 需求要求单字 50~500 个节点',
  },
  max_colors: {
    title: '颜色层上限（层）',
    rows: [['调小', '相近颜色合并成大色块，多色字会丢色'],
           ['调大', '颜色分层更细，碎色层与节点随之增多']],
    dft: '默认 12（均衡档）· 单色字用不上多层',
  },
  gradient_levels: {
    title: '渐变色阶（级）',
    rows: [['调小', '渐变过渡变快，可能出现色带断层'],
           ['调大', '过渡更平滑，文件更大、转换更慢']],
    dft: '默认 10（均衡档）· 需求要求渐变字 ≥8 级',
  },
  filter_speckle: {
    title: '杂点过滤（像素）',
    rows: [['调小', '小笔触与飞白细节保留多，噪点也一起留下'],
           ['调大', '背景杂点清得干净，细小笔画可能被误删']],
    dft: '默认 4（均衡档）· 书法飞白类建议调小',
  },
  corner_angle: {
    title: '角点判定角（度）',
    rows: [['调小', '更多转折被判为尖角，方正字更挺括，圆润字会毛糙'],
           ['调大', '转角更圆润连贯，尖角特征会被磨钝']],
    dft: '默认 62°（均衡档）· 书法手写类建议调大',
  },
  auto_verify: {
    title: '第三方引擎兜底',
    rows: [['开', '自检不达标时自动调 vtracer（本地库）对比取优，结果里标注实际引擎'],
           ['关', '只用自研轮廓引擎，转换更快、结果完全可预期']],
    dft: '默认开 · 敏感素材可关闭，图片不上传任何云端',
  },
  strict_input: {
    title: '严格校验素材',
    rows: [['开', '主体占比、模糊、背景等不合格时直接拒绝转换（需求 5.6）'],
           ['关', '异常素材给提示但继续转换，方便先看效果']],
    dft: '默认关',
  },
};
const FAIL_TEXT = {
  'contour>=0.90': '轮廓还原度低于 90%',
  'feature>=0.85': '字体特征保留低于 85%',
  node_50_500: '单字节点数不在 50~500 区间',
  holes_preserved: '有内孔（镂空）丢失',
  no_specks: '输出中出现孤立碎点',
  no_breaks: '出现断笔（笔画被切碎）',
  'spur_ratio<=0.02': '毛刺像素偏多',
  'waviness': '直线不平直（长直边波浪度超出该档容差）',
  'color_dev<=0.03': '单色色值偏差超过 3%',
  'color_dev<=0.08': '多色主色偏差超过 8%',
  'gradient_levels>=8': '渐变色阶不足 8 级',
};
const STAGE_ORDER = ['preprocess', 'vectorize', 'verify'];
const STAGE_TEXT = {
  effects: '拟合光晕与软阴影，并检查局部明暗',
  preprocess: '阶段 1 / 3：预处理 · 去背景、亚像素边缘、倾斜校正',
  vectorize: '阶段 2 / 3：矢量化 · 主色分层、轮廓拟合、节点预算',
  verify: '阶段 3 / 3：复核 · 还原度、内孔、颜色偏差',
};

const state = {
  mode: 'single',
  phase: 'empty',          // empty | inspect | converting | result
  stale: false,            // 结果来自旧参数
  preset: 'fidelity',
  result: null,            // 当前结果（含 svg / metrics）
  paramsUsed: null,        // 转换时的参数快照
  fmt: 'svg',
  view: 'slider',
  zoom: 1, fitWidth: 0,
  file: null, fileName: '', sourceUrl: null, alignedUrl: null, previewUrl: null,
  inspect: null, stage: 'preprocess',
  jobId: null, pollTimer: null,
  history: [], queue: [], batchFilter: 'all', batchRunning: false,
};

/* ------------------------------------------------------------ 基础工具 */
function toast(message, kind = '', ms = 3600) {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = message;
  $('#toast-host').appendChild(el);
  setTimeout(() => el.remove(), ms);
}

function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}

function svgDataUrl(svg) {
  return `data:image/svg+xml;base64,${btoa(unescape(encodeURIComponent(svg)))}`;
}

function base64ToBlob(b64, type) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type });
}

const pct = (v) => `${((v ?? 0) * 100).toFixed(1)}%`;
const num = (v, d = 0) => (v ?? 0).toFixed(d);
const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, (ch) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[ch]));

/* ------------------------------------------------------------ 参数 */
function readParams() {
  const p = { preset: state.preset, engine: 'auto' };
  if (['fidelity', 'pixel_exact'].includes(state.preset)) {
    p.strict_input = !$('#strict_input').classList.contains('off');
    if (state.preset === 'fidelity') p.soft_effects = !$('#soft_effects').classList.contains('off');
    if ($('#glyph_count').value) p.glyph_count = Number($('#glyph_count').value);
  } else {
    PARAM_IDS.forEach((id) => { p[id] = Number($(`#${id}`).value); });
    p.auto_verify = !$('#auto_verify').classList.contains('off');
    p.strict_input = !$('#strict_input').classList.contains('off');
    const glyphs = $('#glyph_count').value;
    if (glyphs) p.glyph_count = Number(glyphs);
  }
  p.spot_matching = state.preset !== 'pixel_exact' && !$('#spot_matching').classList.contains('off');
  const spot = $('#spot_colors').value.trim();
  if (spot && p.spot_matching) p.spot_colors = spot;
  return p;
}

function buildForm(extra = {}, params = readParams()) {
  const p = params;
  const fd = new FormData();
  Object.entries(p).forEach(([k, v]) => fd.append(k, v));
  Object.entries(extra).forEach(([k, v]) => fd.append(k, v));
  return fd;
}

function syncOutputs() {
  const units = { tolerance: ' px', max_nodes: '', max_colors: '', gradient_levels: '', filter_speckle: '', corner_angle: '°' };
  const defaults = PRESET_DEFAULTS[state.preset] || {};
  PARAM_IDS.forEach((id) => {
    const el = $(`#${id}-out`);
    const v = Number($(`#${id}`).value);
    const d = defaults[id];
    el.innerHTML = `${$(`#${id}`).value}${units[id]}`
      + (d !== undefined && v !== d ? ` <span class="def">默认 ${d}</span>` : '');
  });
}

function applyPreset(key) {
  const table = PRESET_DEFAULTS[key];
  if (!table) return;
  state.preset = key;
  const sourceMode = ['fidelity', 'pixel_exact'].includes(key);
  const pixelMode = key === 'pixel_exact';
  [...PARAM_IDS, 'auto_verify'].forEach((id) => { $(`#${id}`).disabled = sourceMode; });
  $('#spot_matching').disabled = pixelMode;
  $('#spot_colors').disabled = pixelMode;
  $('#soft-effects-row').hidden = key !== 'fidelity';
  Object.entries(table).forEach(([k, v]) => { $(`#${k}`).value = v; });
  syncOutputs();
  renderPresetSummary();
}

function renderPresetSummary() {
  const p = readParams();
  if (['fidelity', 'pixel_exact'].includes(p.preset)) {
    $('#preset-summary').innerHTML = `<span class="pill">${p.preset === 'fidelity' ? '纯色 / 浅色纸底自动去底' : '保留原背景'}</span><span class="pill">不强制节点上限</span>`
      + (p.preset === 'fidelity' ? `<span class="pill">柔光补偿${p.soft_effects ? '开' : '关'}</span>` : '');
    return;
  }
  $('#preset-summary').innerHTML = [
    `容差 <b>${p.tolerance}px</b>`,
    `颜色层 <b>${p.max_colors}</b>`,
    `节点上限 <b>${p.max_nodes}</b>`,
  ].map((t) => `<span class="pill">${t}</span>`).join('');
}

function paramsEqual(a, b) {
  if (!a || !b) return true;
  return [...PARAM_IDS, 'preset', 'engine', 'auto_verify', 'strict_input', 'soft_effects', 'glyph_count', 'spot_matching', 'spot_colors']
    .every((id) => String(a[id] ?? '') === String(b[id] ?? ''));
}

/* 参数改动 → 结果降级为旧参数产物 */
function markParamsChanged() {
  renderPresetSummary();
  if (state.phase !== 'result') { render(); return; }
  state.stale = !paramsEqual(readParams(), state.paramsUsed);
  render();
}

function restoreParams() {
  if (!state.paramsUsed) return;
  state.preset = state.paramsUsed.preset || state.preset;
  $('#glyph_count').value = state.paramsUsed.glyph_count ?? '';
  $('#spot_colors').value = state.paramsUsed.spot_colors ?? '';
  $('#spot_matching').classList.toggle('off', !state.paramsUsed.spot_matching);
  $('#spot_matching').setAttribute('aria-pressed', String(!!state.paramsUsed.spot_matching));
  $$('#preset-group button').forEach((button) => button.classList.toggle('on', button.dataset.preset === state.preset));
  applyPreset(state.preset);
  PARAM_IDS.forEach((id) => { if (state.paramsUsed[id] !== undefined) $(`#${id}`).value = state.paramsUsed[id]; });
  $('#auto_verify').classList.toggle('off', !state.paramsUsed.auto_verify);
  $('#strict_input').classList.toggle('off', !state.paramsUsed.strict_input);
  $('#soft_effects').classList.toggle('off', state.paramsUsed.soft_effects === false);
  $('#soft_effects').setAttribute('aria-pressed', String(state.paramsUsed.soft_effects !== false));
  syncOutputs();
  state.stale = false;
  renderPresetSummary();
  render();
}

/* ------------------------------------------------------------ 渲染 */
function setPhase(phase) {
  state.phase = phase;
  if (phase !== 'result') state.stale = false;
  render();
}

function render() {
  const { mode, phase, stale } = state;
  const single = mode === 'single';

  $('#pane-batch').hidden = single;
  $('#stage').hidden = !single;
  $('#action-bar').hidden = !single;
  $('#check-card').hidden = !(single && phase === 'inspect');

  // 画布四态
  $('#empty-state').hidden = !(phase === 'empty');
  $('#inspect-preview').hidden = !(phase === 'inspect');
  $('#loader').hidden = !(phase === 'converting');
  const showResultView = phase === 'result';
  $('#layer-slider').hidden = !(showResultView && state.view === 'slider');
  $('#layer-side').hidden = !(showResultView && state.view === 'side');
  $('#layer-diff').hidden = !(showResultView && state.view === 'diff');
  $('#layer-code').hidden = !(showResultView && state.view === 'code');
  $('#metric-strip').hidden = !showResultView;

  renderVerdict();
  renderActions();

  // 参数区：转换中锁定；结果过期时露出“恢复参数”
  $('#param-box').classList.toggle('locked', phase === 'converting');
  $$('.field[data-param]').forEach((el) => {
    const id = el.dataset.param;
    el.classList.toggle('moved', !!(!$(`#${id}`).disabled && phase === 'result' && state.paramsUsed
      && Number($(`#${id}`).value) !== Number(state.paramsUsed[id])));
  });
  $('#restore-params').hidden = !(single && phase === 'result' && stale);
  $('#health').classList.toggle('dirty', single && phase === 'result' && stale);
  if (single && phase === 'result' && stale) {
    $('#health span').textContent = '参数已改动';
  } else if ($('#health').classList.contains('ok')) {
    $('#health span').textContent = $('#health').dataset.ver || '服务正常';
  }

  // 结果区整体降级
  $('#stage').style.opacity = (phase === 'result' && stale) ? '.62' : '';
}

function renderVerdict() {
  const box = $('#verdict');
  const { phase, stale, inspect, result } = state;
  if (state.mode === 'batch') { box.hidden = true; return; }

  let tone = 'good', ico = '✓', title = '', body = '', ms = '';
  if (phase === 'empty') { box.hidden = true; return; }

  if (phase === 'inspect') {
    const q = inspect?.quality || {};
    const warn = (inspect?.warnings || []).length || (q.invalid_reasons || []).length;
    tone = warn ? 'warn' : 'good';
    ico = warn ? '!' : '✓';
    title = warn ? '素材预检：有提醒项' : '素材预检通过';
    body = warn
      ? `${[...(inspect?.warnings || []), ...(q.invalid_reasons || []).map(reasonText)].slice(0, 2).join('；')}。可以继续转换，效果可能受影响。`
      : '素材干净清晰，可以直接转换。';
    ms = '预检完成';
  } else if (phase === 'converting') {
    tone = 'info'; ico = '⟳';
    title = `正在矢量化　${state.fileName}`;
    body = STAGE_TEXT[state.stage || 'preprocess'];
    ms = '转换中';
  } else if (phase === 'result' && stale) {
    tone = 'warn'; ico = '!';
    title = '参数已改动，重新转换后生效';
    const previous = state.paramsUsed?.tolerance == null ? (PRESET_NAMES[state.paramsUsed?.preset] || '自定义') : `容差 ${state.paramsUsed.tolerance}px`;
    body = `下面的结果来自上一次参数（${previous}）。重新转换后会自动替换，这次结果保留在「最近转换」里。`;
    ms = `上次 ${Math.round(result?.elapsed_ms || 0)} ms`;
  } else if (phase === 'result') {
    const m = result?.metrics || {};
    const fails = resultFailures(m);
    if (m.source_fidelity) {
      tone = 'warn'; ico = '!';
      title = result.stats?.algorithm === 'native_ink_coverage' ? '纹理保留：透明像素覆盖路径'
        : result.stats?.representation === 'pixel_grid' ? '原像素路径：放大保留像素网格' : '保真曲线：仍为近似结果';
      body = `${m.background_removal?.applied ? '合成回原背景后，' : '原图回渲'}${m.source_fidelity.excluded_background_artifact_pixels ? '排除已清理的背景杂物，' : ''}主体平均误差 ${num(m.source_fidelity.active_mae, 3)}/255，最大误差 ${num(m.source_fidelity.max_error, 2)}/255。${(result.warnings || [])[0] || ''}`;
      if (['nonuniform_border', 'no_foreground'].includes(m.background_removal?.reason)) {
        tone = 'bad';
        title = '自动去背景未成功：结果仍含原背景';
        body = '当前背景无法可靠分离，请更换干净纯色或透明底素材。';
      }
    } else if (m.input_invalid) {
      tone = 'bad'; ico = '!';
      title = '建议更换素材';
      body = `素材判定异常（${(m.input_invalid_reasons || []).length} 项），继续调整参数收益有限。`;
    } else if (fails.length) {
      const shape = fails.some((f) => f.includes('轮廓') || f.includes('特征'));
      tone = shape ? 'bad' : 'warn'; ico = '!';
      title = shape ? '建议更换素材或调整参数' : '建议调整参数';
      body = `未达标项：${fails.join('、')}。可切「高质量」档或减少颜色层数后重新转换。`;
    } else {
      tone = 'warn'; ico = '!';
      title = '自动检查完成，请确认效果';
      body = `轮廓诊断分 ${pct(m.shape_iou)}，特征综合分 ${pct(m.feature_retention)}。请放大检查笔画、转角和颜色，实际编辑兼容性待验证。`;
    }
    ms = `${Math.round(result?.elapsed_ms || 0)} ms`;
  }

  box.hidden = false;
  box.className = `verdict ${tone}`;
  $('#verdict-ico').textContent = ico;
  $('#verdict-title').textContent = title;
  $('#verdict-body').textContent = body;
  $('#verdict-ms').textContent = ms;
}

function renderActions() {
  const { phase, stale, mode } = state;
  if (mode === 'batch') return;
  const primary = $('#act-primary');
  const show = (sel, on) => { $(sel).hidden = !on; };

  show('#act-repick', phase === 'inspect' || phase === 'result');
  show('#act-close', phase === 'inspect' || phase === 'result');
  show('#act-copy', phase === 'result');
  show('#act-compare', phase === 'result' && !!state.result?.preview_png);
  show('#act-download-old', phase === 'result' && stale);
  show('#format-group', phase === 'result');

  primary.classList.remove('ghost');
  primary.classList.add('primary');
  if (phase === 'empty') {
    primary.textContent = `下载 ${state.fmt.toUpperCase()}`;
    primary.disabled = true;
  } else if (phase === 'inspect') {
    primary.textContent = '继续转换';
    primary.disabled = false;
  } else if (phase === 'converting') {
    primary.textContent = '取消转换';
    primary.disabled = false;
    primary.classList.remove('primary');
    primary.classList.add('ghost');
  } else if (stale) {
    primary.textContent = '重新转换';
    primary.disabled = false;
  } else {
    primary.textContent = `下载 ${state.fmt.toUpperCase()}`;
    primary.disabled = false;
  }

  $('#file-name').textContent = state.fileName || '还没有素材';
  if (phase === 'empty') $('#file-meta').textContent = '选好文件后自动进入预检与转换';
  else if (phase === 'inspect') $('#file-meta').textContent = '预检完成，等待确认';
  else if (phase === 'converting') $('#file-meta').textContent = '按 Esc 取消';
  else {
    const d = state.result || {};
    $('#file-meta').textContent =
      `${d.size?.width ?? '?'} × ${d.size?.height ?? '?'} · 引擎 ${d.metrics?.engine || 'contour'} · ${Math.round(d.elapsed_ms || 0)} ms`;
  }
}

function reasonText(r) {
  return {
    background_complex: '背景复杂杂乱', blurry: '图像严重模糊', glare: '大面积过曝',
    no_foreground: '未识别到前景', subject_small: '主体占画面低于 60%', photo_like: '疑似照片素材',
    resolution_small: '分辨率低于 500×500',
  }[r] || r;
}

function failList(checks) {
  return Object.entries(checks || {})
    .filter(([k, v]) => !v && k !== 'passed')
    .map(([k]) => FAIL_TEXT[k] || k);
}

function resultFailures(metrics) {
  const failures = failList(metrics.checks);
  const items = metrics.assessment?.items || {};
  for (const item of Object.values(items)) {
    if (item.status === 'fail') failures.push(item.label);
  }
  return [...new Set(failures)];
}

/* ------------------------------------------------------------ 指标 */
function renderSourceColors(palette) {
  const colors = (Array.isArray(palette?.colors) ? palette.colors : [])
    .filter(color => /^#[0-9a-f]{6}$/i.test(color?.hex || '')).slice(0, 8);
  const note = !palette ? '重新转换后显示原图颜色' : !colors.length ? '未识别到可见颜色'
    : palette.scope === 'foreground' ? '已排除检测到的背景；渐变与柔边以代表色展示'
      : '按可见原图取色，可能包含背景颜色；渐变以代表色展示';
  return `<section class="m source-colors" aria-label="原图主色">
    <div class="palette-heading"><b>原图主色</b><span>${colors.length ? `${colors.length} 种代表色 · HEX / RGB` : '自动识别'}</span></div>
    ${colors.length ? `<ul class="palette-list">${colors.map(color => {
      const hex = color.hex.toUpperCase();
      const rgb = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
      return `<li class="palette-color"><span class="palette-swatch" style="background-color:${hex}" aria-hidden="true"></span>
        <span><strong>${hex}</strong><small>RGB ${rgb.join(', ')}</small></span></li>`;
    }).join('')}</ul>` : ''}
    <p class="palette-note">${note}</p></section>`;
}

function renderMetrics(data) {
  const m = data.metrics || {};
  const q = data.quality || {};
  const c = m.checks || {};
  const colorCheck = Object.entries(c).find(([k]) => k.startsWith('color_dev'));
  const chips = [
    { k: '轮廓诊断分', v: pct(m.shape_iou), ok: m.assessment?.items?.contour_raw?.status === 'pass', n: `容差分 ${pct(m.contour_fidelity)}`, e: '相对预处理掩膜的原始 IoU，容差分另列；原始输入及关键笔画需要目视检查' },
    { k: '特征保留', v: pct(m.feature_retention), ok: c['feature>=0.85'], n: `边界 ${num(m.boundary_f1_2px, 2)}`, e: '衬线、笔锋、尖角等字体特征的保留程度，≥85% 合格' },
    { k: '平均节点/字', v: num(m.node_per_glyph), ok: m.assessment?.items?.node_average?.status === 'pass', n: `共 ${m.node_total} 锚点`, e: '总锚点除以字数；真实字数及逐字 50～500 节点需要分别核实' },
    { k: '颜色诊断值', v: pct(m.color_deviation), ok: m.assessment?.items?.color?.status === 'pass', n: '阈值 ≤3%', e: '单色主色值或多色主色匹配偏差；颜色位置、面积和透明度需另验' },
    { k: '内孔保持', v: `${m.holes_out}/${m.holes_orig}`, ok: m.holes_preserved, n: '镂空保持', e: '日、回、品这类字的镂空数量，输出不得少于原图' },
    { k: '直线平直度', v: m.straight_waviness == null ? '—' : `${m.straight_waviness}px`, ok: m.straight_waviness == null ? true : (c['waviness'] !== false), n: `容差 ${m.waviness_limit ?? 0.18}px`, e: '长直边的波浪幅度，横平竖直的字越小越挺括；原图本身弯曲的字与书法手写不考核' },
    { k: '渐变色阶', v: m.color_levels_out, ok: undefined, n: m.is_gradient_art ? '自动识别渐变' : '适用性待确认', e: '色阶数量作辅助；独立素材标签与实际过渡效果需确认' },
    { k: '轮廓杂讯', v: `${m.stroke_breaks ?? 0}/${m.speck_count ?? 0}`, ok: c['no_breaks'] !== false && c['no_specks'] !== false, n: '断笔/碎点', e: '断笔数 / 孤立碎点数，均应为 0' },
    { k: '转换耗时', v: `${Math.round(data.elapsed_ms)} ms`, ok: data.elapsed_ms <= 5000, n: '阈值 ≤5s', e: '服务内完整转换耗时，含指标和预览；1000×1000 以内阈值 5 秒，网页等待另计' },
    { k: '输出引擎', v: m.engine === 'vtracer' ? 'vtracer' : '轮廓引擎', ok: true, n: m.engine === 'vtracer' ? '第三方兜底命中' : '自研引擎', e: '本次实际使用的矢量化引擎，由「第三方引擎兜底」开关控制 vtracer 是否参与对比取优' },
    { k: '素材判定', v: m.input_invalid ? '异常' : q.severity === 'warning' ? '提醒' : '合格', ok: !m.input_invalid, n: `主体 ${pct(q.orig_bbox_ratio ?? q.subject_bbox_ratio)}`, e: '背景、模糊、过曝、主体占比等输入质量判定' },
  ];
  if (m.source_fidelity) {
    const f = m.source_fidelity;
    chips.splice(0, chips.length,
      { k: '主体平均误差', v: `${num(f.active_mae, 3)}/255`, n: f.excluded_background_artifact_pixels ? '排除已清理背景杂物' : m.background_removal?.applied ? '合成回原背景对照' : '原图逐像素对照', e: m.background_removal?.applied ? '将透明结果合成回原底色后对照；主动清除的边框及背景杂物不计作字形损失，此误差不代表透明度真值已验收' : '直接对原始 RGBA 在黑底、白底合成，取较大的误差；不使用处理后的掩膜替代原图' },
      { k: '最大误差', v: `${num(f.max_error, 2)}/255`, n: '单通道最差像素', e: '全图最差像素误差，不能用全图均值掩盖局部细节损失' },
      { k: '明显差异像素', v: f.pixels_over_8, n: '误差 >8/255', e: '任一通道在黑白背景上合成误差大于 8 的像素数量' },
      { k: 'SVG 锚点', v: m.node_total, n: data.stats?.effect_editing?.follows_body ? '字身与光晕共用路径' : data.stats?.effect_compensation?.selected ? '字身与柔光分层' : '不强制节点上限', e: '本档不强制单字 500 节点；柔光候选仅在局部复核通过后减少色块路径' },
      { k: '表示方式', v: data.stats?.representation === 'pixel_grid' ? '像素网格' : '近似曲线', n: m.background_removal?.applied ? '已去底 · 透明输出' : data.stats?.preserves_background === false ? '透明背景' : '保留原画布和背景', e: '像素路径不是平滑字形重建；曲线结果仍需放大检查' },
      { k: '转换耗时', v: `${Math.round(data.elapsed_ms)} ms`, n: '含回渲复核', e: '完整服务耗时，不承诺 5 秒内完成' });
    const backgroundCheck = m.background_removal?.export_verification;
    if (backgroundCheck) chips.push({ k: '透明底复核', v: backgroundCheck.passed ? '已通过' : '未通过',
      ok: backgroundCheck.passed, n: '检查实际导出结果',
      e: '将导出的 SVG 重新渲染，检查检测到的空白区域是否透明、主要笔画是否丢失；柔光按效果范围单独检查。' });
    const debris = m.background_removal?.removed_artifact_pixels;
    if (debris) chips.push({ k: '背景杂物', v: '已清理', n: '边框及外围独立杂物',
      e: '清理与主体分离的截图长边框及外围小杂物，保留附近点画、细笔画和相连光晕；与字身重叠的内容无法据此可靠分离。' });
    const effect = data.stats?.effect_compensation;
    if (effect) {
      chips.push({ k: '柔光补偿', v: effect.selected ? '已采用' : '沿用原流程',
        n: effect.selected ? `${data.stats.effects.length} 个柔光效果层` : '局部校验后选择',
        e: '分别检查字身边缘、弱光晕和明暗偏差；不适用或误差未改善时保留原结果' });
      if (effect.selected) chips.push({ k: '光晕明暗偏差', v: `${num(effect.local.halo_luma_bias, 3)}/255`,
        n: '正数偏亮，负数偏暗', e: '原图确定的光晕区域内，输出减原图的 sRGB 加权通道均值；不是还原率' });
      if (effect.selected) chips.push({ k: '柔光编辑',
        v: data.stats?.effect_editing?.follows_body ? '字身联动' : '独立发光源',
        n: '兼容 SVG 滤镜',
        e: data.stats?.effect_editing?.follows_body
          ? '光晕附着在字身复合路径上；释放复合路径后的效果及其他编辑器仍需核验'
          : '多发光源独立保留；修改字身节点后需同步调整发光源' });
    }
  }
  const spot = data.stats?.spot_matching;
  if (spot?.enabled) chips.push({ k: '专色匹配', v: `${spot.matched_layers} 个纯色层已改色`,
    n: 'SVG 渐变层保持原样', e: '按填写的 HEX 色板匹配 RGB 最近色；主动改色会增加与原图的颜色差异。渐变若已被转换为纯色色块，也会按纯色层处理。' });
  $('#metric-strip').innerHTML = chips.map((ch) => `
    <button class="m ${ch.ok === false ? 'warn' : ch.ok ? '' : ''}" type="button">
      <div class="k">${ch.k}</div>
      <div class="v ${ch.ok && ch.ok !== false ? 'ok' : ''}">${ch.v}</div>
      <div class="n">${ch.n}</div>
      <div class="exp" hidden>${ch.e}</div>
    </button>`).join('') + renderSourceColors(m.source_colors);
  $$('#metric-strip button.m').forEach((btn) => btn.addEventListener('click', () => {
    const exp = btn.querySelector('.exp');
    exp.hidden = !exp.hidden;
  }));
}

/* ------------------------------------------------------------ 对比与缩放 */
function setWipe(percent) {
  const value = Math.max(2, Math.min(98, percent));
  $('#layer-slider').style.setProperty('--wipe', `${value}%`);
  $('#ls-handle').setAttribute('aria-valuenow', String(Math.round(value)));
  const pctEl = $('#ls-pct');
  if (pctEl) pctEl.textContent = `${Math.round(value)}%`;
}

function fitView() {
  const body = $('#canvas');
  const maxW = Math.max(280, body.clientWidth - 60);
  const maxH = 420;
  const ar = (state.result?.size?.width || 1) / (state.result?.size?.height || 1);
  let w = maxW;
  let h = w / ar;
  if (h > maxH) { h = maxH; w = h * ar; }
  state.fitWidth = w;
  state.zoom = 1;
  applyZoom();
}

function applyZoom() {
  const w = Math.round(state.fitWidth * state.zoom);
  ['#layer-slider', '#layer-side', '#layer-diff'].forEach((sel) => {
    $(sel)?.style.setProperty('--view-w', `${w}px`);
  });
  const ar = (state.result?.size?.width || 1) / (state.result?.size?.height || 1);
  $('#layer-slider')?.style.setProperty('--view-ar', String(ar));
}

function switchView(view) {
  state.view = view;
  $$('#view-group .seg, #view-group button').forEach((b) => b.classList.toggle('on', b.dataset.view === view));
  render();
  if (view !== 'code') requestAnimationFrame(applyZoom);
}

/* ------------------------------------------------------------ 转换流程 */
async function inspectFile(file) {
  state.file = file;
  state.fileName = file.name;
  state.sourceUrl = URL.createObjectURL(file);
  state.inspect = null;
  setPhase('inspect');
  $('#check-thumb').src = state.sourceUrl;
  $('#check-title').textContent = file.name;
  $('#check-sub').textContent = '正在预检素材…';
  $('#check-flags').innerHTML = '';
  try {
    const fd = new FormData();
    fd.append('file', file);
    const res = await fetch('/api/inspect', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.problems?.length) {
      toast(data.problems.join('；'), 'bad', 5000);
      setPhase('empty');
      return;
    }
    state.inspect = data;
    $('#inspect-preview').src = state.sourceUrl;
    renderInspect(data);
  } catch (err) {
    toast(`预检失败：${err.message}`, 'bad', 5000);
    setPhase('empty');
  }
}

function renderInspect(data) {
  const q = data.quality || {};
  const size = data.size || {};
  $('#check-title').textContent = `${data.filename}　·　${size.width} × ${size.height}`;
  $('#check-sub').textContent = data.warnings?.length ? data.warnings.join('；') : '素材检查通过';
  const flags = [
    { t: '背景干净', ok: !(q.invalid_reasons || []).includes('background_complex') },
    { t: '清晰度', ok: !(q.invalid_reasons || []).includes('blurry') },
    { t: `主体占比 ${pct(q.orig_bbox_ratio ?? q.subject_bbox_ratio)}`, ok: (q.orig_bbox_ratio ?? q.subject_bbox_ratio) >= 0.6 },
    { t: q.photo_like ? '疑似照片' : '艺术字素材', ok: !q.photo_like },
  ];
  if (data.background_removal && state.preset === 'fidelity') {
    const bg = data.background_removal;
    const labels = {
      uniform_background: '可自动去除纯色底',
      light_paper_background: '可自动去除浅色纸底',
      shaded_light_paper_background: '可自动去除浅色渐变纸底',
      existing_transparency: '保留已有透明背景',
    };
    flags.unshift({ t: labels[bg.reason] || '背景无法可靠自动去除',
      ok: bg.applied || bg.reason === 'existing_transparency' });
    if (!bg.applied && bg.reason !== 'existing_transparency') {
      $('#check-sub').textContent = '背景无法可靠自动分离，继续转换将保留原背景；建议更换干净素材。';
    }
  }
  $('#check-flags').innerHTML = flags
    .map((f) => `<span class="flag ${f.ok ? 'ok' : 'warn'}">${f.ok ? '✓' : '⚠'} ${f.t}</span>`).join('');
  $('#check-card').classList.toggle('ok', !(data.warnings || []).length && q.severity === 'ok');
  renderVerdict();
}

async function startConvert() {
  if (!state.file) return;
  state.paramsUsed = { ...readParams() };
  state.startedAt = performance.now();
  setPhase('converting');
  updateLoader('preprocess', 0);
  try {
    const fd = buildForm({}, state.paramsUsed);
    fd.append('file', state.file);
    fd.append('with_preview', 'true');
    const res = await fetch('/api/jobs', { method: 'POST', body: fd });
    const response = await res.json();
    if (!res.ok || !response.job_id) throw new Error(response.detail || response.error || '服务返回异常');
    const { job_id } = response;
    state.jobId = job_id;
    pollJob(job_id);
  } catch (err) {
    toast(`无法开始转换：${err.message}`, 'bad', 5000);
    setPhase('inspect');
  }
}

function pollJob(jobId) {
  fetch(`/api/jobs/${jobId}`)
    .then((r) => r.json())
    .then((data) => {
      if (state.jobId !== jobId) return;
      if (data.status === 'done') {
        state.jobId = null;
        data.result.frontend_wait_ms = Math.round(performance.now() - state.startedAt);
        showResult(data.result);
      } else if (data.status === 'error') {
        state.jobId = null;
        toast(data.error || '转换失败', 'bad', 6000);
        setPhase(state.result ? 'result' : 'inspect');
      } else if (data.status === 'cancelled') {
        state.jobId = null;
        toast('已取消，素材与参数已保留', 'warn');
        setPhase(state.result ? 'result' : 'inspect');
      } else {
        updateLoader(data.stage, data.elapsed_ms);
        state.pollTimer = setTimeout(() => pollJob(jobId), 250);
      }
    })
    .catch(() => {
      if (state.jobId === jobId) state.pollTimer = setTimeout(() => pollJob(jobId), 600);
    });
}

function updateLoader(stage, elapsedMs) {
  const idx = Math.max(0, STAGE_ORDER.indexOf(stage === 'effects' ? 'vectorize' : stage));
  state.stage = stage;
  $$('#loader-steps .step').forEach((el, i) => {
    el.classList.toggle('done', i < idx);
    el.classList.toggle('now', i === idx);
  });
  $('#loader-bar').style.width = `${[24, 62, 88][idx]}%`;
  $('#loader-elapsed').textContent = `${((elapsedMs || 0) / 1000).toFixed(1)}s`;
  if (state.phase === 'converting') {
    $('#verdict-body').textContent = STAGE_TEXT[stage] || '';
    if (['fidelity', 'pixel_exact'].includes(state.preset)) {
      $('#verdict-body').textContent = ({preprocess:'读取原图', vectorize:'保留原图细节并生成路径', effects:'分离并拟合光晕与软阴影', verify:'与原图逐像素比较'})[stage] || '';
      $$('#loader-steps .step').forEach((el, i) => {
        el.textContent = ['1 读取原图','2 生成保真路径','3 对照原图检查'][i];
      });
    }
    $('#verdict-ms').textContent = `已用 ${((elapsedMs || 0) / 1000).toFixed(1)}s`;
  }
}

function cancelJob() {
  if (state.jobId) {
    fetch(`/api/jobs/${state.jobId}`, { method: 'DELETE' }).catch(() => {});
    state.jobId = null;
  }
  clearTimeout(state.pollTimer);
  setPhase(state.result ? 'result' : 'inspect');
  toast('已取消', 'warn');
}

/* 关闭回空态：结果已入「最近转换」，无需确认 */
function closeToEmpty() {
  if (state.jobId) {
    fetch(`/api/jobs/${state.jobId}`, { method: 'DELETE' }).catch(() => {});
    state.jobId = null;
  }
  clearTimeout(state.pollTimer);
  const srcUrl = state.sourceUrl;
  const aligned = state.alignedUrl;
  if (srcUrl) URL.revokeObjectURL(srcUrl);
  if (aligned && aligned !== srcUrl) URL.revokeObjectURL(aligned);
  state.file = null;
  state.fileName = '';
  state.sourceUrl = null;
  state.alignedUrl = null;
  state.previewUrl = null;
  state.inspect = null;
  state.result = null;
  state.paramsUsed = null;
  state.stale = false;
  ['#ls-in', '#ls-out', '#side-in', '#side-out', '#diff-img', '#inspect-preview'].forEach((s) => { $(s).src = ''; });
  $('#layer-code').textContent = '';
  setPhase('empty');
  toast('已关闭，结果保存在「最近转换」');
}

function showResult(data) {
  state.result = data;
  state.stale = false;
  setPhase('result');

  const svgUrl = svgDataUrl(data.svg);
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = svgUrl;

  if (state.alignedUrl && state.alignedUrl !== state.sourceUrl) URL.revokeObjectURL(state.alignedUrl);
  state.alignedUrl = data.source_png
    ? URL.createObjectURL(base64ToBlob(data.source_png, 'image/png'))
    : state.sourceUrl;

  $('#ls-in').src = state.alignedUrl || '';
  $('#ls-out').src = svgUrl;
  $('#side-in').src = state.alignedUrl || '';
  $('#side-out').src = svgUrl;
  $('#layer-code').textContent = data.svg;
  if (data.preview_png) {
    $('#diff-img').src = `data:image/png;base64,${data.preview_png}`;
  }

  renderMetrics(data);
  requestAnimationFrame(() => { fitView(); setWipe(50); });
  render();
  pushHistory(data);
  if (data.warnings?.length) toast(data.warnings.slice(0, 2).join('；'), 'warn', 5000);
}

/* ------------------------------------------------------------ 历史 */
const HISTORY_KEY = 'ziji.history.v1';

function loadHistory() {
  try {
    state.history = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
  } catch { state.history = []; }
  renderHistory();
}

function saveHistory() {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(state.history));
  } catch {
    // 超额时丢掉最早的完整数据，只留记录
    while (state.history.length > 4) {
      state.history.splice(Math.floor(state.history.length / 2), 1);
      try {
        localStorage.setItem(HISTORY_KEY, JSON.stringify(state.history));
        break;
      } catch { /* 继续瘦身 */ }
    }
  }
}

function pushHistory(data) {
  const item = {
    filename: data.filename, stem: data.stem, time: Date.now(),
    params: { ...(state.paramsUsed || readParams()) }, fmt: state.fmt, elapsed_ms: data.elapsed_ms,
    metrics: data.metrics, quality: data.quality, size: data.size, stats: data.stats,
    svg: data.svg, warnings: data.warnings || [],
  };
  state.history.unshift(item);
  if (state.history.length > 20) state.history.length = 20;
  saveHistory();
  renderHistory();
}

function verdictShort(item) {
  if (!item.metrics) return { text: '—', bad: true };
  if (item.metrics.source_fidelity) return { text: `误差 ${num(item.metrics.source_fidelity.active_mae, 3)}/255`, bad: false };
  const fails = resultFailures(item.metrics);
  if (item.metrics.input_invalid) return { text: '建议换素材', bad: true };
  if (!fails.length) return { text: pct(item.metrics.contour_fidelity), bad: false };
  return { text: `建议调整 ${fails.length} 项`, bad: true };
}

function renderHistory() {
  $('#history-count').hidden = !state.history.length;
  $('#history-count').textContent = state.history.length;
  $('#history-list').innerHTML = state.history.map((it, i) => {
    const v = verdictShort(it);
    return `<button class="hist ${state.result && it.stem === state.result.stem && !state.stale && i === 0 ? 'on' : ''}" data-i="${i}">
      <img src="${svgDataUrl(it.svg)}" alt="">
      <div><b>${escapeHtml(it.filename)}</b><span>${new Date(it.time).toLocaleString('zh-CN', { hour12: false })} · ${PRESET_NAMES[it.params?.preset] || '自定义'}档${it.params?.tolerance == null ? '' : ` · ${it.params.tolerance}px`}</span></div>
      <div class="v"><b class="${v.bad ? 'bad' : ''}">${v.text}</b><span>${it.fmt.toUpperCase()}</span></div>
    </button>`;
  }).join('') || '<p class="empty-row">还没有转换记录</p>';
  $$('#history-list .hist').forEach((btn) => btn.addEventListener('click', () => openHistory(Number(btn.dataset.i))));
}

function openHistory(i) {
  const it = state.history[i];
  if (!it) return;
  state.result = {
    filename: it.filename, stem: it.stem, svg: it.svg, metrics: it.metrics,
    quality: it.quality, size: it.size, elapsed_ms: it.elapsed_ms, warnings: it.warnings, stats: it.stats,
  };
  state.paramsUsed = it.params;
  PARAM_IDS.forEach((id) => { if (it.params?.[id] !== undefined) $(`#${id}`).value = it.params[id]; });
  $('#spot_colors').value = it.params?.spot_colors ?? '';
  $('#spot_matching').classList.toggle('off', !it.params?.spot_matching);
  $('#spot_matching').setAttribute('aria-pressed', String(!!it.params?.spot_matching));
  syncOutputs();
  state.fileName = it.filename;
  state.stale = false;
  setPhase('result');
  const svgUrl = svgDataUrl(it.svg);
  $('#ls-out').src = svgUrl;
  $('#ls-in').src = state.alignedUrl || '';
  $('#side-out').src = svgUrl;
  $('#side-in').src = state.alignedUrl || '';
  $('#layer-code').textContent = it.svg;
  renderMetrics(state.result);
  requestAnimationFrame(() => { fitView(); setWipe(50); });
  render();
  renderHistory();
  closeDrawer();
}

function openDrawer() {
  $('#drawer').hidden = false;
  $('#drawer-mask').hidden = false;
}
function closeDrawer() {
  $('#drawer').hidden = true;
  $('#drawer-mask').hidden = true;
}

/* ------------------------------------------------------------ 批量 */
function addToQueue(files) {
  const valid = files.filter((f) => /\.(png|jpe?g|webp)$/i.test(f.name));
  const skipped = files.length - valid.length;
  if (!valid.length) { toast('没有可用的图片文件（仅支持 PNG / JPG / WEBP）', 'warn'); return; }
  valid.forEach((file) => state.queue.push({ file, status: 'idle' }));
  renderQueue();
  toast(`已加入 ${valid.length} 张，共 ${state.queue.length} 张` + (skipped ? `（跳过 ${skipped} 个非图片文件）` : ''));
}

function queueVerdict(item) {
  if (item.status === 'bad') return { text: item.error || '失败', cls: 'bad' };
  if (!item.metrics) return { text: '待转换', cls: 'idle' };
  if (item.metrics.source_fidelity) return { text: '原图误差已测，效果待确认', cls: 'warn' };
  const fails = resultFailures(item.metrics);
  if (item.metrics.input_invalid) return { text: '建议换素材', cls: 'bad' };
  if (!fails.length) return { text: '效果待确认', cls: 'warn' };
  return { text: '建议调整', cls: 'warn' };
}

function renderQueue() {
  const q = state.queue;
  const filtered = q.map((item, i) => ({ item, i })).filter(({ item }) => {
    if (state.batchFilter === 'fail') return item.status === 'bad';
    if (state.batchFilter === 'done') return item.status === 'ok';
    return true;
  });

  $('#queue-body').innerHTML = filtered.map(({ item, i }) => {
    const m = item.metrics;
    const v = queueVerdict(item);
    const badge = {
      idle: '<span class="badge idle">待转换</span>',
      run: '<span class="badge run">转换中</span>',
      ok: '<span class="badge ok">完成</span>',
      bad: `<span class="badge bad">${escapeHtml(item.error || '失败')}</span>`,
    }[item.status];
    const act = item.status === 'ok'
      ? '<button class="rowlink" data-act="save" data-i="' + i + '">下载</button>'
      : item.status === 'bad'
        ? '<button class="rowlink" data-act="retry" data-i="' + i + '">重试</button>' : '';
    return `<tr class="${item.status === 'bad' ? 'bad' : ''}">
      <td title="${escapeHtml(item.file.name)}">${i + 1}. ${escapeHtml(item.file.name)}</td>
      <td><span class="badge ${v.cls}">${escapeHtml(v.text)}</span></td>
      <td class="num">${m?.source_fidelity ? `误差 ${num(m.source_fidelity.active_mae, 3)}/255` : m ? pct(m.contour_fidelity) : '—'}</td>
      <td class="num">${m?.node_per_glyph == null ? '—' : num(m.node_per_glyph)}</td>
      <td class="num">${item.elapsed_ms ? `${Math.round(item.elapsed_ms)} ms` : '—'}</td>
      <td>${badge}</td>
      <td>${act}</td>
    </tr>`;
  }).join('') || '<tr><td colspan="7" class="empty-row">队列为空，先添加图片</td></tr>';

  $$('#queue-body .rowlink').forEach((btn) => btn.addEventListener('click', () => {
    const item = state.queue[Number(btn.dataset.i)];
    if (btn.dataset.act === 'retry' && state.batchRunning) return;
    if (btn.dataset.act === 'retry') { item.status = 'idle'; item.error = null; renderQueue(); runBatch([item]); }
    if (btn.dataset.act === 'save') saveQueueItem(item);
  }));

  const done = q.filter((x) => x.status === 'ok').length;
  const failed = q.filter((x) => x.status === 'bad').length;
  $('#queue-title').textContent = `队列 ${q.length} 张`;
  $('#queue-sub').textContent = `完成 ${done} · 失败 ${failed} · 输出 ${state.fmt.toUpperCase()}`;
  $('#batch-ms').textContent = q.length ? `完成 ${done} / ${q.length}` : '队列为空';
  $('#batch-run').disabled = state.batchRunning || !q.some((x) => x.status === 'idle');
  $('#batch-zip').hidden = !done;
  $('#batch-zip').disabled = state.batchRunning;
  $('#batch-run').hidden = !!done && !q.some((x) => x.status === 'idle');
}

async function runBatch(only = null) {
  if (state.batchRunning) return;
  const targets = only || state.queue.filter((x) => x.status === 'idle');
  if (!targets.length) return;
  const batchParams = Object.freeze({ ...readParams() });
  const batchStarted = performance.now();
  state.batchRunning = true;
  const pending = [...targets];
  const work = async () => {
    for (;;) {
      const item = pending.shift();
      if (!item) return;
      item.status = 'run';
      renderQueue();
      try {
        const fd = buildForm({}, batchParams);
        fd.append('file', item.file);
        fd.append('with_preview', 'false');
        const res = await fetch('/api/convert', { method: 'POST', body: fd });
        const json = await res.json().catch(() => ({ error: '服务返回异常' }));
        if (!res.ok || json.error) throw new Error(json.error || '转换失败');
        item.status = 'ok';
        item.metrics = json.metrics;
        item.elapsed_ms = json.elapsed_ms;
        item.svg = json.svg;
        item.stem = json.stem;
        item.params = { ...batchParams };
        item.sourcePath = item.file.webkitRelativePath || item.file.name;
      } catch (err) {
        item.status = 'bad';
        item.error = err.message || '转换失败';
      }
      renderQueue();
    }
  };
  await Promise.all(Array.from({ length: Math.min(3, pending.length) }, work));
  state.batchRunning = false;
  state.batchWallMs = Math.round(performance.now() - batchStarted);
  renderQueue();
  const ok = targets.filter((t) => t.status === 'ok').length;
  toast(ok === targets.length ? `批量完成：${ok} 张，可下载 ZIP` : `完成 ${ok}/${targets.length}，失败的可单张重试`, ok === targets.length ? '' : 'warn', 5000);
}

function saveQueueItem(item) {
  if (state.fmt === 'svg') {
    download(new Blob([item.svg], { type: 'image/svg+xml' }), `${item.stem}.svg`);
    return;
  }
  const fd = new FormData();
  fd.append('svg', item.svg);
  fd.append('fmt', state.fmt);
  fd.append('stem', item.stem);
  fetch('/api/export', { method: 'POST', body: fd })
    .then((res) => {
      if (!res.ok) throw new Error('导出失败，请重试');
      return res.blob();
    })
    .then((blob) => download(blob, `${item.stem}.${state.fmt}`))
    .catch((err) => toast(err.message, 'bad'));
}

async function exportBlob(item, fmt = state.fmt) {
  if (fmt === 'svg') return new TextEncoder().encode(item.svg);
  const fd = new FormData();
  fd.append('svg', item.svg);
  fd.append('fmt', fmt);
  fd.append('stem', item.stem);
  const res = await fetch('/api/export', { method: 'POST', body: fd });
  if (!res.ok) throw new Error('导出失败，请重试');
  return new Uint8Array(await res.arrayBuffer());
}

async function buildZip() {
  if (state.batchRunning) return;
  const items = state.queue.filter((x) => x.status === 'ok' && x.svg);
  if (!items.length) { toast('没有可打包的结果', 'warn'); return; }
  const files = [];
  const used = new Set();
  const mapping = [];
  const fmt = state.fmt;
  for (const item of items) {
    let stem = (item.stem || 'image').replace(/[<>:"/\\|?*]/g, '_');
    if (stem.toLowerCase().startsWith('_batch')) stem = `image${stem}`;
    let unique = stem;
    let suffix = 2;
    while (used.has(unique.toLowerCase())) unique = `${stem}__${suffix++}`;
    used.add(unique.toLowerCase());
    const outputName = `${unique}.${fmt}`;
    files.push({ name: outputName, data: await exportBlob(item, fmt) });
    mapping.push({ source: item.sourcePath || item.file.name, output: outputName, options: item.params });
    files.push({
      name: `${unique}.json`,
      data: new TextEncoder().encode(JSON.stringify({
        filename: item.sourcePath || item.file.name, options: item.params,
        elapsed_ms: Math.round(item.elapsed_ms), metrics: item.metrics,
      }, null, 2)),
    });
  }
  files.push({ name: '_batch_summary.json', data: new TextEncoder().encode(JSON.stringify({
    count: items.length, conversion_wall_ms: state.batchWallMs, files: mapping,
  }, null, 2)) });
  download(zipStore(files), `vectorized-${Date.now()}.zip`);
  toast(`ZIP 已打包（${items.length} 个矢量文件）`);
}

function downloadQueueJson() {
  const items = state.queue.filter((x) => x.metrics);
  if (!items.length) { toast('还没有可导出的指标', 'warn'); return; }
  const blob = new Blob([JSON.stringify(items.map((x) => ({
    filename: x.sourcePath || x.file.webkitRelativePath || x.file.name,
    options: x.params, elapsed_ms: x.elapsed_ms, metrics: x.metrics,
  })), null, 2)], { type: 'application/json' });
  download(blob, `metrics-${Date.now()}.json`);
}

/* 纯前端 ZIP（存储方式），逐张进度与单独重试不依赖服务端打包 */
const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes) {
  let c = 0xFFFFFFFF;
  for (let i = 0; i < bytes.length; i += 1) c = CRC_TABLE[(c ^ bytes[i]) & 0xFF] ^ (c >>> 8);
  return (c ^ 0xFFFFFFFF) >>> 0;
}

function zipStore(files) {
  const enc = new TextEncoder();
  const chunks = [];
  const central = [];
  let offset = 0;
  const now = new Date();
  const dosTime = ((now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() >> 1)) & 0xFFFF;
  const dosDate = (((now.getFullYear() - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate()) & 0xFFFF;
  files.forEach(({ name, data }) => {
    const nameBytes = enc.encode(name);
    const header = new DataView(new ArrayBuffer(30));
    header.setUint32(0, 0x04034b50, true);
    header.setUint16(4, 20, true); header.setUint16(6, 0x0800, true);
    header.setUint16(8, 0, true); header.setUint16(10, dosTime, true); header.setUint16(12, dosDate, true);
    header.setUint32(14, crc32(data), true); header.setUint32(18, data.length, true); header.setUint32(22, data.length, true);
    header.setUint16(26, nameBytes.length, true); header.setUint16(28, 0, true);
    chunks.push(new Uint8Array(header.buffer), nameBytes, data);
    const cd = new DataView(new ArrayBuffer(46));
    cd.setUint32(0, 0x02014b50, true);
    cd.setUint16(4, 20, true); cd.setUint16(6, 20, true); cd.setUint16(8, 0x0800, true);
    cd.setUint16(10, 0, true); cd.setUint16(12, dosTime, true); cd.setUint16(14, dosDate, true);
    cd.setUint32(16, crc32(data), true); cd.setUint32(20, data.length, true); cd.setUint32(24, data.length, true);
    cd.setUint16(28, nameBytes.length, true);
    cd.setUint32(42, offset, true);
    central.push(new Uint8Array(cd.buffer), nameBytes);
    offset += 30 + nameBytes.length + data.length;
  });
  const cdSize = central.reduce((acc, x) => acc + x.length, 0);
  const eocd = new DataView(new ArrayBuffer(22));
  eocd.setUint32(0, 0x06054b50, true);
  eocd.setUint16(8, files.length, true); eocd.setUint16(10, files.length, true);
  eocd.setUint32(12, cdSize, true); eocd.setUint32(16, offset, true);
  return new Blob([...chunks, ...central, new Uint8Array(eocd.buffer)], { type: 'application/zip' });
}

/* ------------------------------------------------------------ 参数注解气泡 */
function tipPopEl() {
  let el = $('#tip-pop');
  if (!el) {
    el = document.createElement('div');
    el.id = 'tip-pop';
    el.hidden = true;
    document.body.appendChild(el);
  }
  return el;
}

function showTip(btn, key) {
  const t = PARAM_TIPS[key];
  if (!t) return;
  const el = tipPopEl();
  el.innerHTML = `<b>${t.title}</b>`
    + t.rows.map(([k, v]) => `<div class="r"><em>${k}</em><span>${v}</span></div>`).join('')
    + `<div class="dft">${t.dft}</div>`;
  el.hidden = false;
  const rect = btn.getBoundingClientRect();
  const w = el.offsetWidth;
  const h = el.offsetHeight;
  let left = rect.left - w - 10;
  if (left < 8) left = rect.right + 10;
  left = Math.max(8, Math.min(left, window.innerWidth - w - 8));
  const top = Math.max(8, Math.min(rect.top - h / 2, window.innerHeight - h - 8));
  el.style.left = `${left}px`;
  el.style.top = `${top}px`;
}

function hideTip() {
  const el = $('#tip-pop');
  if (el) el.hidden = true;
  $$('.tip.on').forEach((b) => b.classList.remove('on'));
}

function bindTips() {
  let pinned = null;
  $$('.tip').forEach((btn) => {
    const key = btn.dataset.tip;
    btn.addEventListener('mouseenter', () => { if (!pinned) showTip(btn, key); });
    btn.addEventListener('mouseleave', () => { if (!pinned) hideTip(); });
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (pinned === btn) { pinned = null; hideTip(); return; }
      pinned = btn;
      $$('.tip.on').forEach((b) => b.classList.remove('on'));
      btn.classList.add('on');
      showTip(btn, key);
    });
  });
  document.addEventListener('click', (e) => {
    if (pinned && !e.target.closest('.tip') && !e.target.closest('#tip-pop')) { pinned = null; hideTip(); }
  });
}

/* ------------------------------------------------------------ 素材入口 */
function pickFile() { $('#file-single').click(); }

function useFile(file) {
  if (!file) return;
  if (!/image\/(png|jpeg|webp)/.test(file.type) && !/\.(png|jpe?g|webp)$/i.test(file.name)) {
    toast('只支持 PNG / JPG / WEBP', 'bad');
    return;
  }
  if (file.size > 5 * 1024 * 1024) {
    toast(`文件 ${(file.size / 1048576).toFixed(1)}MB 超过 5MB 上限`, 'bad', 5000);
    return;
  }
  inspectFile(file);
}

async function loadSample(name, opts = {}) {
  const auto = opts.auto !== false;
  try {
    const list = await fetch('/api/samples').then((r) => r.json()).then((r) => r.samples);
    if (!list.length) { toast('没有可用示例', 'bad'); return; }
    const pick = name ? (list.find((x) => x.file === name) || list[0])
                      : list[Math.floor(Math.random() * list.length)];
    const blob = await fetch(pick.url).then((r) => r.blob());
    const file = new File([blob], pick.file, { type: blob.type || 'image/png' });
    toast(`示例：${pick.style || pick.file}`);
    useFile(file);
    // 示例素材直接进入转换，方便演示；auto 为 false 时停在预检
    if (auto) setTimeout(() => startConvert(), 60);
  } catch (err) {
    toast(`示例载入失败：${err.message}`, 'bad', 5000);
  }
}

function openSampleModal() {
  $('#sample-mask').hidden = false;
  $('#sample-modal').hidden = false;
  renderSampleGrid();
}

function closeSampleModal() {
  $('#sample-mask').hidden = true;
  $('#sample-modal').hidden = true;
}

async function renderSampleGrid() {
  try {
    const list = await fetch('/api/samples').then((r) => r.json()).then((r) => r.samples);
    $('#sample-grid').innerHTML = list.map((s) => `
      <button class="sample-cell" type="button" data-file="${s.file}">
        <img src="${s.url}" alt=""><b>${s.text || s.file}</b><span>${s.style || ''}</span>
      </button>`).join('') || '<p class="empty-row">样本目录为空</p>';
    $$('#sample-grid .sample-cell').forEach((btn) => btn.addEventListener('click', () => {
      closeSampleModal();
      loadSample(btn.dataset.file);
    }));
  } catch (err) {
    toast(`样本列表载入失败：${err.message}`, 'bad', 5000);
  }
}

/* ------------------------------------------------------------ 事件绑定 */
function switchMode(mode) {
  state.mode = mode;
  $$('#mode-group button').forEach((b) => b.classList.toggle('on', b.dataset.mode === mode));
  render();
  renderQueue();
}

function bind() {
  // 顶栏
  $('#mode-group').addEventListener('click', (e) => {
    const btn = e.target.closest('button');
    if (btn) switchMode(btn.dataset.mode);
  });
  $('#open-history').addEventListener('click', openDrawer);
  $('#drawer-close').addEventListener('click', closeDrawer);
  $('#drawer-mask').addEventListener('click', closeDrawer);
  $('#history-clear').addEventListener('click', () => {
    state.history = [];
    saveHistory();
    renderHistory();
    toast('历史已清空');
  });
  $('#history-export').addEventListener('click', () => {
    const withSvg = state.history.filter((x) => x.svg);
    if (!withSvg.length) { toast('历史里没有可导出的文件', 'warn'); return; }
    const files = withSvg.map((x) => ({ name: `${x.stem}.svg`, data: new TextEncoder().encode(x.svg) }));
    download(zipStore(files), `history-${Date.now()}.zip`);
    toast(`已导出 ${files.length} 个 SVG`);
  });

  // 参数
  $('#preset-group').addEventListener('click', (e) => {
    const btn = e.target.closest('button');
    if (!btn) return;
    $$('#preset-group button').forEach((b) => b.classList.toggle('on', b === btn));
    $('#preset-hint').textContent = btn.dataset.desc || '';
    applyPreset(btn.dataset.preset);
    markParamsChanged();
  });
  PARAM_IDS.forEach((id) => $(`#${id}`).addEventListener('input', () => { syncOutputs(); markParamsChanged(); }));
  ['auto_verify', 'strict_input', 'soft_effects', 'spot_matching'].forEach((id) => $(`#${id}`).addEventListener('click', (e) => {
    const btn = e.currentTarget;
    btn.classList.toggle('off');
    btn.setAttribute('aria-pressed', String(!btn.classList.contains('off')));
    markParamsChanged();
  }));
  $('#glyph_count').addEventListener('input', markParamsChanged);
  $('#spot_colors').addEventListener('input', markParamsChanged);
  $('#restore-params').addEventListener('click', restoreParams);

  // 视图 / 底色 / 缩放
  $('#view-group').addEventListener('click', (e) => {
    const btn = e.target.closest('button');
    if (btn) switchView(btn.dataset.view);
  });
  $('#bg-group').addEventListener('click', (e) => {
    const btn = e.target.closest('button');
    if (!btn) return;
    $$('#bg-group button').forEach((b) => b.classList.toggle('on', b === btn));
    $('#canvas').classList.remove('bg-light', 'bg-dark');
    if (btn.dataset.bg !== 'checker') $('#canvas').classList.add(`bg-${btn.dataset.bg}`);
  });
  $('#zoom-in').addEventListener('click', () => { state.zoom = Math.min(4, state.zoom * 1.25); applyZoom(); });
  $('#zoom-out').addEventListener('click', () => { state.zoom = Math.max(0.3, state.zoom / 1.25); applyZoom(); });
  $('#zoom-reset').addEventListener('click', fitView);

  // 对比滑块：指针 + 键盘
  const slider = $('#layer-slider');
  let dragging = false;
  const move = (clientX) => {
    const rect = slider.getBoundingClientRect();
    setWipe(((clientX - rect.left) / rect.width) * 100);
  };
  slider.addEventListener('pointerdown', (e) => {
    dragging = true;
    slider.setPointerCapture(e.pointerId);
    move(e.clientX);
    $('#ls-handle').focus();
  });
  slider.addEventListener('pointermove', (e) => { if (dragging) move(e.clientX); });
  slider.addEventListener('pointerup', () => { dragging = false; });
  slider.addEventListener('pointercancel', () => { dragging = false; });
  $('#ls-handle').addEventListener('keydown', (e) => {
    const now = Number($('#ls-handle').getAttribute('aria-valuenow')) || 50;
    const step = e.shiftKey ? 10 : 2;
    if (e.key === 'ArrowLeft') setWipe(now - step);
    else if (e.key === 'ArrowRight') setWipe(now + step);
    else if (e.key === 'Home') setWipe(2);
    else if (e.key === 'End') setWipe(98);
    else return;
    e.preventDefault();
  });

  // 上传
  $('#pick-file').addEventListener('click', (e) => { e.stopPropagation(); pickFile(); });
  $('#load-sample').addEventListener('click', (e) => { e.stopPropagation(); openSampleModal(); });
  $('#sample-close').addEventListener('click', closeSampleModal);
  $('#sample-mask').addEventListener('click', closeSampleModal);
  $('#sample-random').addEventListener('click', () => { closeSampleModal(); loadSample(); });
  $('#empty-state').addEventListener('click', pickFile);
  $('#act-repick').addEventListener('click', pickFile);
  $('#act-close').addEventListener('click', closeToEmpty);
  $('#check-repick').addEventListener('click', pickFile);
  $('#check-go').addEventListener('click', startConvert);
  $('#file-single').addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    if (file) useFile(file);
    e.target.value = '';
  });

  const stage = $('#stage');
  ['dragenter', 'dragover'].forEach((ev) => stage.addEventListener(ev, (e) => {
    e.preventDefault();
    $('#empty-state').classList.add('drag');
  }));
  ['dragleave', 'drop'].forEach((ev) => stage.addEventListener(ev, (e) => {
    e.preventDefault();
    $('#empty-state').classList.remove('drag');
  }));
  stage.addEventListener('drop', (e) => {
    const files = [...(e.dataTransfer?.files || [])];
    if (!files.length) return;
    useFile(files[0]);
    if (files.length > 1) {
      addToQueue(files.slice(1));
      toast('其余文件已加入批量队列', 'warn', 4000);
    }
  });
  document.addEventListener('paste', (e) => {
    const items = [...(e.clipboardData?.items || [])].filter((x) => x.type.startsWith('image/'));
    if (!items.length) return;
    const file = items[0].getAsFile();
    if (file) { useFile(file); toast('已粘贴剪贴板图片'); }
  });

  // 动作条
  $('#act-primary').addEventListener('click', () => {
    const { phase, stale } = state;
    if (phase === 'inspect') startConvert();
    else if (phase === 'converting') cancelJob();
    else if (phase === 'result' && stale) startConvert();
    else downloadResult();
  });
  $('#act-download-old').addEventListener('click', downloadResult);
  $('#act-copy').addEventListener('click', async () => {
    if (!state.result) return;
    try {
      await navigator.clipboard.writeText(state.result.svg);
      toast('SVG 源码已复制到剪贴板');
    } catch {
      toast('复制失败，请用「源码」视图手动复制', 'warn');
    }
  });
  $('#act-compare').addEventListener('click', () => {
    const data = state.result;
    if (data?.preview_png) download(base64ToBlob(data.preview_png, 'image/png'), `${data.stem}_比对.png`);
  });
  $('#format-group').addEventListener('click', (e) => {
    const btn = e.target.closest('button');
    if (!btn) return;
    state.fmt = btn.dataset.fmt;
    $$('#format-group button').forEach((b) => b.classList.toggle('on', b === btn));
    renderActions();
    renderQueue();
  });

  // 批量
  $('#batch-add').addEventListener('click', () => $('#file-batch').click());
  $('#batch-folder').addEventListener('click', () => $('#file-folder').click());
  $('#file-batch').addEventListener('change', (e) => { addToQueue([...e.target.files]); e.target.value = ''; });
  $('#file-folder').addEventListener('change', (e) => { addToQueue([...e.target.files]); e.target.value = ''; });
  $('#batch-run').addEventListener('click', () => runBatch());
  $('#batch-zip').addEventListener('click', () => buildZip().catch((err) => toast(err.message, 'bad')));
  $('#batch-json').addEventListener('click', downloadQueueJson);
  $('#batch-filter').addEventListener('click', (e) => {
    const btn = e.target.closest('button');
    if (!btn) return;
    state.batchFilter = btn.dataset.filter;
    $$('#batch-filter button').forEach((b) => b.classList.toggle('on', b === btn));
    renderQueue();
  });
  $('#batch-clear').addEventListener('click', () => { $('#clear-pop').hidden = false; });
  $('#clear-cancel').addEventListener('click', () => { $('#clear-pop').hidden = true; });
  $('#clear-confirm').addEventListener('click', () => {
    if (state.batchRunning) return;
    state.queue = [];
    $('#clear-pop').hidden = true;
    renderQueue();
    toast('队列已清空');
  });

  // 键盘
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (!$('#tip-pop')?.hidden) { hideTip(); return; }
    if (!$('#sample-modal').hidden) { closeSampleModal(); return; }
    if (!$('#clear-pop').hidden) { $('#clear-pop').hidden = true; return; }
    if (!$('#drawer').hidden) { closeDrawer(); return; }
    if (state.phase === 'converting') cancelJob();
    else if (state.phase === 'inspect' || state.phase === 'result') closeToEmpty();
  });

  bindTips();

  window.addEventListener('resize', () => {
    if (state.phase === 'result' && state.view !== 'code') fitView();
  });
}

function downloadResult() {
  const data = state.result;
  if (!data) return;
  if (state.fmt === 'svg') {
    download(new Blob([data.svg], { type: 'image/svg+xml' }), `${data.stem}.svg`);
    return;
  }
  const fd = new FormData();
  fd.append('svg', data.svg);
  fd.append('fmt', state.fmt);
  fd.append('stem', data.stem);
  fetch('/api/export', { method: 'POST', body: fd })
    .then((res) => {
      if (!res.ok) throw new Error('导出失败');
      return res.blob();
    })
    .then((blob) => { download(blob, `${data.stem}.${state.fmt}`); toast(`已导出 ${state.fmt.toUpperCase()}`); })
    .catch((err) => toast(err.message, 'bad'));
}

async function addSamplesToQueue(names) {
  const list = await fetch('/api/samples').then((r) => r.json()).then((r) => r.samples);
  for (const name of names) {
    const pick = list.find((x) => x.file === name.trim());
    if (!pick) continue;
    const blob = await fetch(pick.url).then((r) => r.blob());
    state.queue.push({ file: new File([blob], pick.file, { type: blob.type || 'image/png' }), status: 'idle' });
  }
  renderQueue();
  runBatch();
}

/* ------------------------------------------------------------ 初始化 */
async function init() {
  bind();

  fetch('/api/health')
    .then((r) => r.json())
    .then((h) => {
      $('#health').classList.add('ok');
      $('#health').dataset.ver = `服务正常 v${h.version}`;
      $('#health span').textContent = $('#health').dataset.ver;
    })
    .catch(() => {
      $('#health').classList.add('bad');
      $('#health span').textContent = '服务不可用';
    });

  const cfg = await fetch('/api/presets').then((r) => r.json());
  $('#preset-group').innerHTML = cfg.presets.map((p, i) => `
    <button class="${p.key === state.preset ? 'on' : ''}" data-preset="${p.key}" data-desc="${p.desc}">${p.name}</button>`).join('');
  $('#preset-hint').textContent = cfg.presets.find((p) => p.key === state.preset).desc;

  const d = cfg.defaults;
  PARAM_IDS.forEach((id) => { if (d[id] !== undefined) $(`#${id}`).value = d[id]; });
  applyPreset(state.preset);
  syncOutputs();
  renderPresetSummary();
  loadHistory();
  render();
  renderQueue();

  // 演示直达：?sample=13_cartoon.png&stale=1&history=1&inspect=1&mode=batch&samples=a,b,c
  const qs = new URLSearchParams(location.search);
  if (qs.get('mode') === 'batch') switchMode('batch');
  if (qs.has('samples') && state.mode === 'batch') {
    await addSamplesToQueue((qs.get('samples') || '').split(',').filter(Boolean));
  } else if (qs.get('sample')) {
    await loadSample(qs.get('sample') === '1' ? undefined : qs.get('sample'),
                     { auto: !qs.has('inspect') });
  }
  if ((qs.has('stale') || qs.has('history')) && state.phase !== 'result') {
    await new Promise((resolve) => {
      const t = setInterval(() => {
        if (state.phase === 'result') { clearInterval(t); resolve(); }
      }, 200);
      setTimeout(() => { clearInterval(t); resolve(); }, 60000);
    });
  }
  if (qs.has('stale') && state.phase === 'result') {
    $$('#preset-group button').forEach((b) => b.classList.toggle('on', b.dataset.preset === 'quality'));
    applyPreset('quality');
    markParamsChanged();
  }
  if (qs.has('history')) openDrawer();
}

init().catch((err) => toast(`初始化失败：${err.message}`, 'bad', 6000));
