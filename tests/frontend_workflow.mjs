import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import vm from 'node:vm';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const source = (await fs.readFile(path.join(root, 'app/static/app.js'), 'utf8'))
  .replace(/init\(\)\.catch\([^\n]+\);\s*$/, '');
class TestForm {
  values = new Map();
  append(key, value) { this.values.set(key, value); }
  get(key) { return this.values.get(key); }
}
const context = vm.createContext({ assert, console, FormData: TestForm, Blob,
  TextEncoder, TextDecoder, performance, setTimeout, clearTimeout, Uint8Array,
  Uint32Array, ArrayBuffer, DataView });
vm.runInContext(source, context);
const archive = await vm.runInContext(`(async () => {
  assert.equal(escapeHtml('<img src=x onerror="x">&'), '&lt;img src=x onerror=&quot;x&quot;&gt;&amp;');
  const paletteHtml = renderSourceColors({scope:'foreground', colors:[{hex:'#ffdc00'}, {hex:'#000000'}, {hex:'<img src=x>'}]});
  assert(paletteHtml.includes('原图主色'));
  assert(paletteHtml.includes('#FFDC00'));
  assert(paletteHtml.includes('RGB 255, 220, 0'));
  assert(paletteHtml.includes('2 种代表色'));
  assert(!paletteHtml.includes('<img'));
  assert(renderSourceColors().includes('重新转换'));
  assert(renderSourceColors({colors:[]}).includes('未识别到可见颜色'));
  const fields = Object.fromEntries(['spot_colors', 'spot_matching', 'strict_input', 'soft_effects', 'glyph_count', 'auto_verify', ...PARAM_IDS]
    .map(id => ['#'+id, {value:'', off:true, classList:{contains(){return fields['#'+id].off;}}}]));
  document = {querySelector: selector => fields[selector]};
  state.preset = 'fidelity';
  fields['#spot_colors'].value = '#ff0000,#00ff00';
  assert.equal(readParams().spot_matching, false);
  assert.equal(readParams().spot_colors, undefined);
  fields['#spot_matching'].off = false;
  assert.equal(readParams().spot_matching, true);
  assert.equal(buildForm().get('spot_colors'), '#ff0000,#00ff00');
  state.preset = 'pixel_exact';
  assert.equal(readParams().spot_matching, false);
  assert.equal(readParams().spot_colors, undefined);
  state.preset = 'balanced';
  assert.equal(readParams().spot_matching, true);
  assert.equal(paramsEqual({preset:'fidelity',spot_matching:false},{preset:'fidelity',spot_matching:true}),false);
  assert.equal(paramsEqual({spot_colors:'#ff0000'},{spot_colors:'#00ff00'}),false);
  renderQueue = () => {};
  toast = () => {};
  let current = { preset: 'balanced', tolerance: .5, max_nodes: 500, spot_matching: true, spot_colors: '#ff0000' };
  readParams = () => ({ ...current });
  const requests = [];
  const names = ['same', 'same', 'Same', '_batch_summary', '中文'];
  state.queue = names.map((name, i) => ({ file: { name: name+'.png', webkitRelativePath: 'folder'+i+'/'+name+'.png' }, status: 'idle' }));
  fetch = async (url, request) => {
    requests.push(request.body.get('tolerance'));
    assert.equal(request.body.get('spot_matching'), true);
    assert.equal(request.body.get('spot_colors'), '#ff0000');
    current.tolerance = 9;
    current.spot_colors = '#0000ff';
    await new Promise(resolve => setTimeout(resolve, 1));
    return { ok: true, json: async () => ({ stem: request.body.get('file').name.slice(0,-4), svg: '<svg>'+('x'.repeat(30000))+'</svg>', metrics: {}, elapsed_ms: 1 }) };
  };
  await runBatch();
  assert.deepEqual(Array.from(requests), [.5,.5,.5,.5,.5]);
  assert(state.queue.every(item => item.params.tolerance === .5));
  assert.equal(paramsEqual({preset:'balanced',glyph_count:4},{preset:'balanced',glyph_count:5}),false);
  assert.equal(paramsEqual({preset:'fidelity',soft_effects:true},{preset:'fidelity',soft_effects:false}),false);
  let captured;
  download = (blob, filename) => { captured = blob; };
  state.fmt = 'svg';
  await buildZip();
  return new Uint8Array(await captured.arrayBuffer());
})()`, context);
if (process.argv[2]) await fs.writeFile(process.argv[2], archive);
console.log(JSON.stringify({batch_parameter_snapshot:true, duplicate_and_reserved_names:true,
  archive_bytes:archive.length, archive_has_large_central_offset:archive.length>65535}));
