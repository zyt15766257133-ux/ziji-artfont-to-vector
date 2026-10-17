"""从实际日志生成中文结果说明与标准训练曲线。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/"runs/.matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="baseline_v1")
    args = parser.parse_args()
    run = ROOT / "runs" / args.run
    config = json.loads((run / "config.json").read_text())
    summary = json.loads((run / "training_summary.json").read_text())
    evaluation = json.loads((run / "evaluation.json").read_text())
    history = json.loads((run / "history.json").read_text())
    manifest = json.loads((ROOT / "data/paired/manifest.json").read_text(encoding="utf-8"))
    resume_details = []
    restart_epochs = []
    for name in summary.get("resume_records", []):
        event = json.loads((run/name).read_text())
        method = "重新初始化优化器" if event["optimizer_reset"] else "恢复优化器及随机状态"
        resume_details.append(f"第 {event['resume_from_logged_epoch']} 轮后，从第 {event['resume_from_checkpoint_epoch']} 轮权重{method}")
        if event["optimizer_reset"]:
            restart_epochs.append(event["resume_from_logged_epoch"])
    resume_note = "；".join(resume_details) if resume_details else "无中断恢复"
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="Train: edge BCE + Dice")
    axes[0].plot(epochs, [r["val_loss"] for r in history], label="Val: plain BCE")
    axes[0].set(xlabel="Epoch", ylabel="Loss (different definitions)", title="Actual optimization history")
    axes[0].legend(fontsize=8)
    axes[1].plot(epochs, [r["val_iou"]*100 for r in history], marker=".", label="Validation IoU")
    axes[1].axhline(summary["initial_val_iou"]*100, color="#999999", linestyle="--", label="Initial fixed prior")
    axes[1].set(xlabel="Epoch", ylabel="Foreground IoU (%)", title="Validation only; test not used for selection", ylim=(0, 100))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
        for epoch in restart_epochs:
            ax.axvline(epoch+.5, color="#c46b42", linestyle=":", alpha=.6)
    fig.savefig(run / "training_curves.png", dpi=180)
    plt.close(fig)
    labels, vals = ["Unseen characters", "Unseen characters + fonts"], []
    for key in ["fixed_threshold_iou", "old_preprocess_iou", "model_iou"]:
        vals.append([evaluation["splits"][s]["metrics"][key]["mean"]*100 for s in ["test", "unseen_font"]])
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    x = np.arange(2)
    for i, (label, v) in enumerate(zip(["Fixed color threshold", "Current preprocessing", "Trained model"], vals)):
        bars = ax.bar(x + (i-1)*.24, v, width=.24, label=label)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 108), ylabel="Mean foreground IoU (%)", title="Independent synthetic test sets, 350 images each")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(run / "test_scores.png", dpi=180)
    plt.close(fig)
    table = []
    for split, label in [("test", "未见汉字，已见字体"), ("unseen_font", "未见汉字及字体")]:
        part = evaluation["splits"][split]
        m = part["metrics"]
        table.append(f"| {label} | {part['count']} | {m['fixed_threshold_iou']['mean']:.2%} | {m['old_preprocess_iou']['mean']:.2%} | {m['model_iou']['mean']:.2%} | {m['model_iou']['p10']:.2%} |")
    svg_table = []
    for split, label in [("test", "未见汉字"), ("unseen_font", "未见字体")]:
        m = evaluation["splits"][split]["svg_first20"]
        part = [r for r in evaluation["svg_examples"] if r["split"] == split]
        passed = sum(50 <= r["model_nodes"] <= 500 for r in part)
        svg_table.append(f"| {label} | {m['count']} | {m['old_iou_mean']:.2%} | {m['model_iou_mean']:.2%} | {m['model_nodes_mean']:.1f} | {passed}/{len(part)} |")
    stress = evaluation["project_stress_test"]
    readme = f'''# 中文字形前景模型实验

2026-10-17 已完成一次实际 CPU 训练与独立评估。已有方案及公开数据的结论见 [调研记录](research/已有方案与公开数据集.md)。

## 本次实际做了什么

从 Google Fonts 官方仓库下载 8 套 OFL 1.1 中文字体，保留每套 OFL.txt、来源 URL、提交号和 SHA256。用字体轮廓生成 SVG，再用 resvg 渲染出准确覆盖率真值，合成清洁、噪声、纸纹、不均匀光照和 JPEG 压缩图片。

实际数据共 {sum(s['count'] for s in manifest['splits'].values())} 张 128×128 单字图片，其中训练 {manifest['splits']['train']['count']} 张、验证 {manifest['splits']['val']['count']} 张、普通测试 {manifest['splits']['test']['count']} 张、未见字体测试 {manifest['splits']['unseen_font']['count']} 张。汉字分成互不重叠的 {len(manifest['char_splits']['train'])}/{len(manifest['char_splits']['val'])}/{len(manifest['char_splits']['test'])} 个；Long Cang、Liu Jian Mao Cao 两套字体仅在未见字体测试中出现。验证和测试集图片 SHA256 无交集。原图、软掩膜、输入特征、真值 SVG 及逐张来源均已保存。

模型是 {config['parameters']:,} 参数的小型残差 U-Net，CPU 训练 {config['epochs']} 轮，用时 {summary['elapsed_seconds']/60:.2f} 分钟。固定随机种子 {config['seed']}；训练使用边界加权 BCE 与 Dice；仅用验证集 IoU 选择 best.pt，测试集没有参与权重选择。初始验证 IoU 为 {summary['initial_val_iou']:.2%}，最佳验证 IoU 为 {summary['best_val_iou']:.2%}。

本次恢复情况为 {resume_note}。详细起点与代码哈希见 {', '.join(summary.get('resume_records', [])) or 'config.json'}。优化器重置的恢复不等同于连续无中断训练。每轮另存 last.pt，保留优化器、学习率调度器和 Torch 随机状态。上面用时为已完成各轮训练及验证的累计时间，未计中断和等待。

## 独立测试

下表均为同一准确字形真值下的前景掩膜 IoU。它衡量像素区域交并比，不等于项目全部“艺术字还原度”验收。

| 测试集 | 张数 | 固定色差阈值 | 当前项目预处理 | 训练模型 | 模型第10百分位 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

每类样本的分数、最低值、≥90% 占比、逐张结果均在 [evaluation.json](runs/{args.run}/evaluation.json)、test_per_image.json 和 unseen_font_per_image.json 中。

再取两个测试集固定的前 20 张，分别将旧掩膜和预测掩膜交给项目已有像素边界三次曲线拟合，导出 SVG，并用 resvg 独立回渲后与字体真值比较。输出仅包含可编辑路径，不嵌入位图。

| 测试集 | 张数 | 旧掩膜导出 SVG IoU | 模型导出 SVG IoU | 模型平均节点 | 单字50～500节点通过 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(svg_table)}

这里比较的是同一个单色路径导出器输入不同掩膜后的结果，不是原项目完整多色转换服务的端到端比较。不能用此表宣称模型在所有艺术字上超过现有引擎或商业产品。

## 分布外检查与边界

另外把项目原有 {stress['count']} 张合成艺术字合成在白底上，保持原分辨率做分块推理。与其原有 truth 图片相比，模型平均掩膜 IoU 为 {stress['model_mean']:.2%}，现有预处理为 {stress['old_mean']:.2%}。这些素材包含本次未训练的阴影、霓虹、渐变、多色、飞白和多字排版，而且原项目真值采用不同覆盖率规则，因此是分布外压力检查，不能混入字体合成集测试。

这版权重在原项目压力集上低于现有预处理，阴影和霓虹的损失尤其明显。因此，本次结果支持把它继续作为噪声/纸纹背景中的单色字提取实验，不支持替换默认艺术字转换管线。不同数据分布的分数需要分别阅读。

当前权重适合继续研究单色中文前景提取。它没有完成渐变/描边/阴影分层、飞白保真、真实客户图验收或完整 1000×1000 图片 5 秒承诺。SVG 实验默认输出黑色字形，foreground.png 保留原 RGB 与预测透明度；颜色不是网络本次的训练目标。对已有真实透明通道，命令行优先保留输入 alpha。

现有 Web/API 默认流程没有接入此权重。源码和测试保留；当前依赖环境的原项目测试为 94 通过、1 失败，详情见 [基线测试记录](research/原项目测试记录.md)。

## 文件与运行

- 权重 [best.pt](runs/{args.run}/best.pt)
- 实际日志 [history.json](runs/{args.run}/history.json)、config.json、training_summary.json
- 对比图 [测试集](runs/{args.run}/test_comparison.png)、[未见字体](runs/{args.run}/unseen_font_comparison.png)、[原项目压力检查](runs/{args.run}/project_stress_comparison.png)
- 训练曲线 [training_curves.png](runs/{args.run}/training_curves.png)，测试分数图 [test_scores.png](runs/{args.run}/test_scores.png)
- 示例原图、真值、模型掩膜、旧 SVG 与模型 SVG 在 runs/{args.run}/examples/ 下
- 数据清单 data/paired/manifest.json，字体与授权 data/fonts/，完整环境锁 environment-lock.txt

在项目根目录用 PowerShell 执行。已有 `.venv` 和数据可直接运行推理；重新训练要换 run 名，脚本拒绝覆盖已有 best.pt。

```powershell
# 使用已训练权重，单色 SVG 实验
.\\.venv\\Scripts\\python.exe -m ml.infer "你的图片.png" --output "out/learned/result.svg" --glyph-count 1

# 重复训练与独立评估
.\\.venv\\Scripts\\python.exe -u -m ml.train --epochs 16 --run experiment_02
.\\.venv\\Scripts\\python.exe -u -m ml.evaluate --run experiment_02
.\\.venv\\Scripts\\python.exe -m ml.write_report --run experiment_02

# 新环境安装；CPU PyTorch 从官方索引安装
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\\.venv\\Scripts\\python.exe -m pip install -r requirements-ml.txt
npm ci --prefix eval/tools

# 首次下载与生成，仅在数据目录不存在时运行
.\\.venv\\Scripts\\python.exe -m ml.download_sources --commit 47d54ca1ebb14fd864695f1d39dc734f39017c33
.\\.venv\\Scripts\\python.exe -m ml.make_dataset
.\\.venv\\Scripts\\python.exe -m pytest -q -p no:cacheprovider ml/test_ml.py
```

数据与权重留在本机且被 .gitignore 排除；复制项目交付时需显式包含 ml/data/、ml/runs/。本次无付费 API、无云训练、无用户图片上传。
'''
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    print(f"Report saved: {ROOT/'README.md'}")


if __name__ == "__main__":
    main()
