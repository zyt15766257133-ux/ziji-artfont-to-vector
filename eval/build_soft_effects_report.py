"""Build a local, inspectable report from saved conversions, without rerunning them."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'eval/results/soft_effects'
before = {r['file']: r for r in json.loads((OUT/'before/metrics.json').read_text(encoding='utf-8'))}
after_label = 'after_final' if (OUT/'after_final/metrics.json').is_file() else 'after'
after = json.loads((OUT/after_label/'metrics.json').read_text(encoding='utf-8'))
(OUT/'source').mkdir(exist_ok=True)
cases = []
for row in after:
    name = row['file']
    stem = Path(name).stem
    source = ROOT/'samples/images'/name
    assert hashlib.sha256(source.read_bytes()).hexdigest() == row['source_sha256'] == before[name]['source_sha256']
    shutil.copyfile(source, OUT/'source'/name)
    cases.append({'id': stem, 'label': name, 'source': f'source/{name}',
                  'before_png': f'before/{stem}/render.png', 'after_png': f'{after_label}/{stem}/render.png',
                  'svg': f'{after_label}/{stem}/result.svg', 'before': before[name], 'after': row})
refs = OUT/'references/metrics.json'
if refs.is_file():
    for row in json.loads(refs.read_text(encoding='utf-8')):
        stem = row['case']
        cases.append({'id': stem, 'label': '独立 SVG 参考 · '+stem,
                      'source': f'references/{stem}/source.png',
                      'before_png': f'references/{stem}/before.png',
                      'after_png': f'references/{stem}/after.png',
                      'svg': f'references/{stem}/after.svg', **row['runs']})
payload = json.dumps(cases, ensure_ascii=False).replace('</', '<\\/')
page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>柔光补偿 · 原图与前后结果</title><style>
body{font:15px/1.6 "Microsoft YaHei",sans-serif;margin:24px;background:#f3f5f8;color:#202b3d}
h1{font-size:24px;margin-bottom:8px}p{max-width:1000px}.tools{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin:16px 0}
select,button,a{font:inherit}select{padding:8px;min-width:260px}.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
figure{margin:0;background:white;border-radius:10px;overflow:hidden}figcaption{padding:10px 14px;font-weight:600}
.art{background:repeating-conic-gradient(#e4e7ed 0% 25%,#fff 0% 50%) 0/20px 20px;min-height:200px;display:flex;align-items:center}
img{width:100%;display:block}table{border-collapse:collapse;background:white;margin-top:16px;min-width:540px}
td,th{padding:8px 16px;border:1px solid #d9dfeb;text-align:left}pre{white-space:pre-wrap;word-break:break-word;background:white;padding:16px}
.status{padding:10px 14px;background:#e7edf9;border-radius:6px}.small{font-size:13px;color:#56647c}
@media(max-width:850px){.panels{grid-template-columns:1fr}table{min-width:0;width:100%}}
</style><h1>光晕与软阴影补偿 · 前后对照</h1>
<p>2026-10-18。已移除复杂背景样本，保留的内置图原图哈希未变；另含从独立 SVG 渲染得到的透明、偏移、多色和弱光参考。只有通过字身、光晕及弱光分区检查的候选才被采用。此页为历史诊断，不作为纯白底业务素材的验收结果。</p>
<p class="small">速度供后续评估：记录包含解码、候选、回渲和预览的服务调用，不包含网络、浏览器显示或其他格式导出。本轮存在超过五秒的调用，未作五秒验收。内置合成图不代表真实业务通过率。</p>
<div class="tools"><label>素材 <select id="case"></select></label><label><input type="checkbox" id="native" checked>直接显示最终 SVG</label><a id="download" download>下载本次 SVG</a></div>
<p class="status" id="status"></p><div class="panels">
<figure><figcaption>原图</figcaption><div class="art"><img id="source" alt="原始输入"></div></figure>
<figure><figcaption>补偿前 · 保存的回渲结果</figcaption><div class="art"><img id="before" alt="原流程回渲"></div></figure>
<figure><figcaption>当前输出 · <span id="renderer"></span></figcaption><div class="art"><img id="after" alt="当前输出"></div></figure>
</div><table><thead><tr><th>指标</th><th>补偿前</th><th>当前输出</th></tr></thead><tbody id="metrics"></tbody></table>
<p class="small">明暗偏差：输出减原图的 sRGB 加权通道均值，正数偏亮、负数偏暗；各指标单位为 /255，不是还原率。光晕指标只在采用补偿后显示，前后使用同一个源图区域。</p>
<details><summary>参数、阶段耗时与未采用原因</summary><pre id="details"></pre></details><script>
const cases = __DATA__;
const select=document.querySelector('#case');
for(const item of cases){const option=document.createElement('option');option.value=item.id;option.textContent=item.label;select.append(option)}
select.value=new URLSearchParams(location.search).get('case')||'21_neon';
if(!select.value)select.selectedIndex=0;
function update(){
 const item=cases.find(c=>c.id===select.value), a=item.after,b=item.before;
 const effect=a.stats.effect_compensation||{};
 for(const [id,url] of [['source',item.source],['before',item.before_png],['after',document.querySelector('#native').checked?item.svg:item.after_png]])document.getElementById(id).src=url;
 document.querySelector('#renderer').textContent=document.querySelector('#native').checked?'浏览器 SVG':'核验 PNG';
 document.querySelector('#download').href=item.svg;
 document.querySelector('#status').textContent=effect.selected?'已采用独立柔光效果，替换原色块光晕。':'保留原流程输出。拒绝或跳过原因：'+(effect.reason||a.stats.fallback_reason||'不适用');
 const rows=[['SVG 总锚点',b.stats.node_total,a.stats.node_total],['服务调用（ms）',b.wall_ms.toFixed(1),a.wall_ms.toFixed(1)],
 ['主体平均误差',b.metrics.source_fidelity.active_mae.toFixed(3),a.metrics.source_fidelity.active_mae.toFixed(3)]];
 if(effect.selected){const x=effect.comparison.baseline_local,y=effect.comparison.candidate_local;
 rows.push(['光晕平均误差',x.halo_mae.toFixed(3),y.halo_mae.toFixed(3)],['光晕明暗偏差',x.halo_luma_bias.toFixed(3),y.halo_luma_bias.toFixed(3)],['弱光晕平均误差',x.weak_halo_mae.toFixed(3),y.weak_halo_mae.toFixed(3)])}
 const tbody=document.querySelector('#metrics');tbody.replaceChildren();
 for(const row of rows){const tr=document.createElement('tr');for(const cell of row){const td=document.createElement('td');td.textContent=cell;tr.append(td)}tbody.append(tr)}
 document.querySelector('#details').textContent=JSON.stringify({timings:a.timings,effects:a.stats.effects,complexity:a.stats.complexity,diagnostic:effect},null,2);
}
select.addEventListener('change',update);document.querySelector('#native').addEventListener('change',update);update();
</script></html>'''.replace('__DATA__', payload)
(OUT/'report.html').write_text(page, encoding='utf-8')
print(OUT/'report.html')
