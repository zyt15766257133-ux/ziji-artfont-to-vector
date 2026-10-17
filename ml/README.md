# 中文字形前景模型实验

2026-10-17 已完成一次实际 CPU 训练与独立评估。已有方案及公开数据的结论见 [调研记录](research/已有方案与公开数据集.md)。

## 本次实际做了什么

从 Google Fonts 官方仓库下载 8 套 OFL 1.1 中文字体，保留每套 OFL.txt、来源 URL、提交号和 SHA256。用字体轮廓生成 SVG，再用 resvg 渲染出准确覆盖率真值，合成清洁、噪声、纸纹、不均匀光照和 JPEG 压缩图片。

实际数据共 3850 张 128×128 单字图片，其中训练 2800 张、验证 350 张、普通测试 350 张、未见字体测试 350 张。汉字分成互不重叠的 840/180/180 个；Long Cang、Liu Jian Mao Cao 两套字体仅在未见字体测试中出现。验证和测试集图片 SHA256 无交集。原图、软掩膜、输入特征、真值 SVG 及逐张来源均已保存。

模型是 29,905 参数的小型残差 U-Net，CPU 训练 16 轮，用时 5.03 分钟。固定随机种子 20261017；训练使用边界加权 BCE 与 Dice；仅用验证集 IoU 选择 best.pt，测试集没有参与权重选择。初始验证 IoU 为 70.05%，最佳验证 IoU 为 97.60%。

本次恢复情况为 第 10 轮后，从第 8 轮权重重新初始化优化器。详细起点与代码哈希见 resume_after_epoch_10.json。优化器重置的恢复不等同于连续无中断训练。每轮另存 last.pt，保留优化器、学习率调度器和 Torch 随机状态。上面用时为已完成各轮训练及验证的累计时间，未计中断和等待。

## 独立测试

下表均为同一准确字形真值下的前景掩膜 IoU。它衡量像素区域交并比，不等于项目全部“艺术字还原度”验收。

| 测试集 | 张数 | 固定色差阈值 | 当前项目预处理 | 训练模型 | 模型第10百分位 |
|---|---:|---:|---:|---:|---:|
| 未见汉字，已见字体 | 350 | 69.86% | 64.84% | 97.54% | 94.85% |
| 未见汉字及字体 | 350 | 64.00% | 55.05% | 97.34% | 95.89% |

每类样本的分数、最低值、≥90% 占比、逐张结果均在 [evaluation.json](runs/baseline_v1/evaluation.json)、test_per_image.json 和 unseen_font_per_image.json 中。

再取两个测试集固定的前 20 张，分别将旧掩膜和预测掩膜交给项目已有像素边界三次曲线拟合，导出 SVG，并用 resvg 独立回渲后与字体真值比较。输出仅包含可编辑路径，不嵌入位图。

| 测试集 | 张数 | 旧掩膜导出 SVG IoU | 模型导出 SVG IoU | 模型平均节点 | 单字50～500节点通过 |
|---|---:|---:|---:|---:|---:|
| 未见汉字 | 20 | 55.06% | 96.11% | 227.7 | 20/20 |
| 未见字体 | 20 | 60.30% | 96.51% | 274.3 | 20/20 |

这里比较的是同一个单色路径导出器输入不同掩膜后的结果，不是原项目完整多色转换服务的端到端比较。不能用此表宣称模型在所有艺术字上超过现有引擎或商业产品。

## 分布外检查与边界

另外把项目原有 25 张合成艺术字合成在白底上，保持原分辨率做分块推理。与其原有 truth 图片相比，模型平均掩膜 IoU 为 90.28%，现有预处理为 99.73%。这些素材包含本次未训练的阴影、霓虹、渐变、多色、飞白和多字排版，而且原项目真值采用不同覆盖率规则，因此是分布外压力检查，不能混入字体合成集测试。

这版权重在原项目压力集上低于现有预处理，阴影和霓虹的损失尤其明显。因此，本次结果支持把它继续作为噪声/纸纹背景中的单色字提取实验，不支持替换默认艺术字转换管线。不同数据分布的分数需要分别阅读。

当前权重适合继续研究单色中文前景提取。它没有完成渐变/描边/阴影分层、飞白保真、真实客户图验收或完整 1000×1000 图片 5 秒承诺。SVG 实验默认输出黑色字形，foreground.png 保留原 RGB 与预测透明度；颜色不是网络本次的训练目标。对已有真实透明通道，命令行优先保留输入 alpha。

现有 Web/API 默认流程没有接入此权重。源码和测试保留；当前依赖环境的原项目测试为 94 通过、1 失败，详情见 [基线测试记录](research/原项目测试记录.md)。

## 文件与运行

- 权重 [best.pt](runs/baseline_v1/best.pt)
- 实际日志 [history.json](runs/baseline_v1/history.json)、config.json、training_summary.json
- 对比图 [测试集](runs/baseline_v1/test_comparison.png)、[未见字体](runs/baseline_v1/unseen_font_comparison.png)、[原项目压力检查](runs/baseline_v1/project_stress_comparison.png)
- 训练曲线 [training_curves.png](runs/baseline_v1/training_curves.png)，测试分数图 [test_scores.png](runs/baseline_v1/test_scores.png)
- 示例原图、真值、模型掩膜、旧 SVG 与模型 SVG 在 runs/baseline_v1/examples/ 下
- 数据清单 data/paired/manifest.json，字体与授权 data/fonts/，完整环境锁 environment-lock.txt

在项目根目录用 PowerShell 执行。已有 `.venv` 和数据可直接运行推理；重新训练要换 run 名，脚本拒绝覆盖已有 best.pt。

```powershell
# 使用已训练权重，单色 SVG 实验
.\.venv\Scripts\python.exe -m ml.infer "你的图片.png" --output "out/learned/result.svg" --glyph-count 1

# 重复训练与独立评估
.\.venv\Scripts\python.exe -u -m ml.train --epochs 16 --run experiment_02
.\.venv\Scripts\python.exe -u -m ml.evaluate --run experiment_02
.\.venv\Scripts\python.exe -m ml.write_report --run experiment_02

# 新环境安装；CPU PyTorch 从官方索引安装
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
npm ci --prefix eval/tools

# 首次下载与生成，仅在数据目录不存在时运行
.\.venv\Scripts\python.exe -m ml.download_sources --commit 47d54ca1ebb14fd864695f1d39dc734f39017c33
.\.venv\Scripts\python.exe -m ml.make_dataset
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider ml/test_ml.py
```

数据与权重留在本机且被 .gitignore 排除；复制项目交付时需显式包含 ml/data/、ml/runs/。本次无付费 API、无云训练、无用户图片上传。
