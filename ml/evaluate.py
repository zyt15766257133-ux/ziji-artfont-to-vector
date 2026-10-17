"""一次独立测试。比较同一真值下的旧预处理、固定色差基线与训练模型。"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from app.pipeline.native_render import render_svg
from app.pipeline.preprocess import preprocess
from .infer import mask_to_svg
from .model import features, load_model, predict

ROOT = Path(__file__).resolve().parent


def iou(pred, truth):
    p, t = np.asarray(pred) > .5, np.asarray(truth) > .5
    return float((p & t).sum() / max(1, (p | t).sum()))


def old_mask(rgb):
    p = preprocess(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), deskew=False, max_side=max(rgb.shape[:2]))
    output = np.zeros(rgb.shape[:2], np.float32)
    if p.crop_box:
        x0, y0, x1, y1 = p.crop_box
        output[y0:y1, x0:x1] = p.mask / 255
    else:
        output[:] = p.mask / 255
    return output


def summary(rows, key):
    values = np.array([r[key] for r in rows])
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)), "min": float(values.min()),
            "at_least_90_percent": float((values >= .9).mean())}


def montage(rows, path, headings):
    size, height = 128, 152
    header = 32
    image = Image.new("RGB", (size*len(headings), header+height*len(rows)), "#eef0f2")
    draw = ImageDraw.Draw(image)
    for c, title in enumerate(headings):
        draw.text((c*size+5, 8), title, fill="black")
    for r, (label, panels) in enumerate(rows):
        for c, panel in enumerate(panels):
            a = np.asarray(panel)
            if a.ndim == 2:
                a = np.repeat((255*(1-np.clip(a, 0, 1)))[:, :, None], 3, axis=2).astype(np.uint8)
            thumb = Image.fromarray(a).convert("RGB")
            thumb.thumbnail((128, 128))
            image.paste(thumb, (c*size, header+r*height))
        draw.text((4, header+r*height+130), label, fill="black")
    image.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="baseline_v1")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    run = ROOT / "runs" / args.run
    checkpoint = run / "best.pt"
    model = load_model(checkpoint)
    report = {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "scope": "128px synthetic OFL glyphs; not full art-text vector acceptance",
              "selection": "checkpoint chosen only on val; svg examples fixed first 20 per split",
              "splits": {}, "svg_examples": []}
    for split in ["test", "unseen_font"]:
        data = np.load(ROOT / f"data/paired/{split}.npz")
        records = json.loads((ROOT / f"data/paired/{split}.json").read_text(encoding="utf-8"))
        xx = torch.from_numpy(data["features"].astype(np.float32))
        t0 = time.perf_counter()
        with torch.inference_mode():
            preds = np.concatenate([model(batch).sigmoid().numpy()[:, 0] for batch in xx.split(32)])
        inference_s = time.perf_counter()-t0
        rows, previews = [], []
        for i, (rgb, mask, pred, record) in enumerate(zip(data["images"], data["masks"], preds, records)):
            truth = mask.astype(np.float32)/255
            old = old_mask(rgb)
            fixed = features(rgb)[4] > .1
            row = record | {"model_iou": iou(pred, truth), "old_preprocess_iou": iou(old, truth),
                            "fixed_threshold_iou": iou(fixed, truth),
                            "alpha_mae": float(np.abs(pred-truth).mean())}
            rows.append(row)
            if i < 20:
                examples = run / "examples" / split
                examples.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb).save(examples / f"{i:03}_input.png")
                Image.fromarray(mask).save(examples / f"{i:03}_truth.png")
                Image.fromarray((pred*255).astype(np.uint8)).save(examples / f"{i:03}_pred.png")
                svg, stats = mask_to_svg(pred)
                (examples / f"{i:03}_model.svg").write_text(svg, encoding="utf-8")
                svg_alpha = render_svg(svg)[:, :, 3].astype(np.float32)/255
                old_svg, old_stats = mask_to_svg(old)
                (examples / f"{i:03}_old.svg").write_text(old_svg, encoding="utf-8")
                old_alpha = render_svg(old_svg)[:, :, 3].astype(np.float32)/255
                report["svg_examples"].append({"split": split, "id": i,
                    "model_svg_iou": iou(svg_alpha, truth), "old_svg_iou": iou(old_alpha, truth),
                    "model_nodes": stats.get("node_total"), "old_nodes": old_stats.get("node_total")})
                if i < 10:
                    previews.append((f"{split}/{i} {record['style']}", [rgb, truth, old, pred, svg_alpha]))
            if i % 100 == 0:
                print(f"evaluate {split}: {i}/{len(records)}", flush=True)
        grouped = {}
        for style in sorted({r["style"] for r in rows}):
            part = [r for r in rows if r["style"] == style]
            grouped[style] = {key: summary(part, key) for key in ["model_iou", "old_preprocess_iou", "fixed_threshold_iou"]}
        report["splits"][split] = {"count": len(rows), "inference_ms_per_image_batch32": inference_s*1000/len(rows),
            "metrics": {key: summary(rows, key) for key in ["model_iou", "old_preprocess_iou", "fixed_threshold_iou"]},
            "mean_alpha_mae": float(np.mean([r["alpha_mae"] for r in rows])), "by_style": grouped}
        (run / f"{split}_per_image.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        montage(previews, run / f"{split}_comparison.png", ["INPUT", "TRUTH", "CURRENT", "TRAINED MASK", "TRAINED SVG"])
        worst = []
        for i in sorted(range(len(rows)), key=lambda j: rows[j]["model_iou"])[:6]:
            rgb = data["images"][i]
            truth = data["masks"][i].astype(np.float32)/255
            worst.append((f"{split}/{i}: IoU {rows[i]['model_iou']:.3f}",
                          [rgb, truth, old_mask(rgb), preds[i]]))
        montage(worst, run / f"{split}_worst.png", ["WORST INPUT", "TRUTH", "CURRENT", "TRAINED MASK"])
    for split in ["test", "unseen_font"]:
        part = [r for r in report["svg_examples"] if r["split"] == split]
        report["splits"][split]["svg_first20"] = {"count": len(part),
            "model_iou_mean": float(np.mean([r["model_svg_iou"] for r in part])),
            "old_iou_mean": float(np.mean([r["old_svg_iou"] for r in part])),
            "model_nodes_mean": float(np.mean([r["model_nodes"] for r in part]))}
    # 独立于字体合成集的已有项目样本，只做分布外压力检查，不用它调模型。
    project = ROOT.parent
    manifest = json.loads((project / "samples/manifest.json").read_text(encoding="utf-8"))
    stress, preview = [], []
    for record in manifest:
        name = record["file"]
        image = Image.open(project / "samples/images" / name).convert("RGBA")
        rgb = np.array(Image.alpha_composite(Image.new("RGBA", image.size, "white"), image).convert("RGB"))
        truth = np.array(Image.open(project / "samples/truth" / name).convert("L"), dtype=np.float32)/255
        if truth.shape != rgb.shape[:2]:
            raise ValueError(f"Truth shape mismatch: {name}")
        pred = predict(model, rgb)
        old = old_mask(rgb)
        stress.append({"file": name, "model_iou": iou(pred, truth), "old_iou": iou(old, truth)})
        if name.startswith(("01_", "07_", "09_", "12_", "14_", "24_")):
            preview.append((name, [rgb, truth, old, pred]))
        print(f"stress {name}", flush=True)
    report["project_stress_test"] = {"count": len(stress), "rows": stress,
        "model_mean": float(np.mean([r["model_iou"] for r in stress])),
        "old_mean": float(np.mean([r["old_iou"] for r in stress])),
        "note": "Existing synthetic art samples composited onto white, different truth convention; includes unsupported shadow/feibai/color. Not real-customer acceptance."}
    montage(preview, run / "project_stress_comparison.png", ["PROJECT INPUT", "PROJECT TRUTH", "CURRENT", "TRAINED MASK"])
    (run / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({split: report["splits"][split]["metrics"] for split in report["splits"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
