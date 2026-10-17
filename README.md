# 字迹 · 艺术字转矢量

本地运行的艺术字位图转矢量工具，提供 Web 界面、API 和 CLI。支持 PNG/JPG/WEBP 输入，以及 SVG、PDF、PDF 兼容 AI 和部分 EPS 输出。

当前开发版本为 `1.2.0-dev`。原始需求作为验收依据，未通过项继续记录。开发服务的验证入口为 `/validation`，原始证据由启动时的 `ZIJI_EVIDENCE_DIR` 指定。

## 启动

需要 Python 3.10～3.12 和 Node.js 22+。SVG 渲染使用常驻 resvg 进程。

```bash
pip install -r requirements.txt
npm ci --prefix eval/tools
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

访问 `http://127.0.0.1:8765/`。原有 `python -m app.main` 入口仍使用 8000 端口。

## 使用

上传图片后检查素材提示，填写实际字数可以核对节点预算。选择档位后转换，在滑块、并排或差异视图中检查细节，再选择格式下载。批量模式支持文件夹导入，批次开始时固定参数，ZIP 保存原文件映射，重名输出自动加后缀。

内置的 25 张历史合成图片用于诊断，其中原图主体占比不足 60%，会保留输入异常提示。开发验证另有 100 张不同的合规程序排版素材；真实客户素材仍需单独验收。

## 当前改进

2026-10-17 新增网页默认“保真曲线”和“原像素路径”。两者保留原背景、画布及透明度，不进行节点压缩；前者采用隔离的 VTracer 1.0 共享边界曲线，后者输出像素网格路径，放大仍有像素格。曲线仍为近似，不能保证任意放大完全无失真。全部 25 张的逐图证据、安装步骤和方法调研见 [全量保真验证](docs/全量保真验证_2026-10-17.md)；服务内入口为 `/fidelity-results/report.html`。

清洁素材直接保留原始像素边界，避免增强、去噪和误判倾斜损伤细字。单色字优先采用受节点预算及局部拓扑检查约束的三次曲线拟合。复杂图片继续使用轮廓引擎及本地 vtracer。

预览和指标使用 resvg 渲染，正确处理透明度、绘制顺序、渐变和滤镜。PDF 中的纯色柔边效果展开为可编辑的透明度矢量层，再用 MuPDF 对照回渲；这会增加导出节点及导出时间。

## 已知缺口

飞白纹理在当前节点预算下仍有失败。部分立体字特征和局部小孔诊断也未通过。单字节点必须结合真实字数和逐字分布检查，整体平均值仅供诊断。

EPS 无法保留当前文件中的透明度与滤镜，会明确报错。AI 为 PDF 兼容形式，Illustrator 2020+ 实际打开编辑和 Safari 实机仍待验证。WebKit 测试引擎的结果单独记录。

单张 ≤5 秒按完整服务调用记录，包含指标和启用的预览；HTTP、排队及文件格式导出另外计时。样本缓存响应标注 `cached`，不计入转换性能证据。

## 可选第三方服务

已提供官方 Vectorizer.AI 适配器，环境变量为 `VECTORIZER_API_ID`、`VECTORIZER_API_SECRET` 和 `VECTORIZER_API_MODE`。默认 `test` 模式会带水印。`production` 模式会使用账户额度。凭据只在服务环境读取，状态接口不返回凭据。

`GET /api/providers` 查看配置状态，`POST /api/provider/vectorizer_ai/convert` 使用表单 `file` 上传。当前本地转换不会自动把图片发送给第三方。官方服务需要账户凭据；公开网页演示只能预览，下载会跳转订阅页。

## 验证

```bash
python -m pytest -q -p no:cacheprovider tests
node --check app/static/app.js
node tests/frontend_workflow.mjs
```

原始需求见项目上一级 `docs/艺术字图片转矢量图-需求.docx`，验收规则见 `docs/艺术字矢量化验收标准.md`，接口与部署说明见 `docs/部署文档.md`。

## 字形模型实验

`ml/` 提供基于公开 OFL 中文字体的配对数据生成、CPU 前景模型训练、独立评估和单色 SVG 实验入口。具体范围与运行结果见 [ml/README.md](ml/README.md)，已有方案与数据来源见 [调研记录](ml/research/已有方案与公开数据集.md)。此模块独立运行，当前 Web/API 不默认启用学习模型。
