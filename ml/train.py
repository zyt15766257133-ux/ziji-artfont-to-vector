"""CPU 可复现训练；仅验证集选模型，测试集由独立 evaluate 命令读取。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .model import GlyphUNet

ROOT = Path(__file__).resolve().parent


def load_split(name):
    with np.load(ROOT / f"data/paired/{name}.npz") as data:
        return TensorDataset(torch.from_numpy(data["features"].astype(np.float32)),
                             torch.from_numpy(data["masks"].astype(np.float32)[:, None] / 255.0))


@torch.inference_mode()
def validation(model, loader):
    ious, losses = [], []
    for x, y in loader:
        logits = model(x)
        pred, truth = logits > 0, y > .5
        intersection = (pred & truth).sum((1, 2, 3)).float()
        union = (pred | truth).sum((1, 2, 3)).clamp(min=1).float()
        ious.extend((intersection / union).tolist())
        losses.append(float(F.binary_cross_entropy_with_logits(logits, y)))
    return float(np.mean(ious)), float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--run", default="baseline_v1")
    parser.add_argument("--resume", action="store_true", help="Continue existing run; record exact resume or best-weight warm restart")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    run = ROOT / "runs" / args.run
    run.mkdir(parents=True, exist_ok=True)
    if (run / "best.pt").exists() and not args.resume:
        raise FileExistsError("Run already exists. Use a new --run name to preserve results.")
    if args.resume and not (run / "best.pt").exists():
        raise FileNotFoundError("No checkpoint to resume")
    train_loader = DataLoader(load_split("train"), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(load_split("val"), batch_size=args.batch_size, num_workers=0)
    model = GlyphUNet()
    initial_iou, _ = validation(model, val_loader)
    config = vars(args) | {"device": "cpu", "torch": torch.__version__, "python": platform.python_version(),
                          "parameters": sum(p.numel() for p in model.parameters()),
                          "manifest_sha256": hashlib.sha256((ROOT / "data/paired/manifest.json").read_bytes()).hexdigest(),
                          "code_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                          for name in ["model.py", "make_dataset.py", "train.py"]}}
    history, best, start_epoch = [], -1.0, 0
    if args.resume:
        original = json.loads((run/"config.json").read_text())
        for key in ["epochs", "batch_size", "seed", "manifest_sha256"]:
            if config[key] != original[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        if config["code_sha256"]["model.py"] != original["code_sha256"]["model.py"]:
            raise ValueError("Model preprocessing/architecture changed")
        history = json.loads((run/"history.json").read_text())
        start_epoch = history[-1]["epoch"]
        if start_epoch >= args.epochs:
            raise ValueError("Run already completed all requested epochs")
        saved_best = torch.load(run/"best.pt", map_location="cpu", weights_only=True)
        best = saved_best["val_iou"]
        model.load_state_dict(saved_best["state_dict"])
        config["original_code_sha256"] = original["code_sha256"]
        config["resume_from_logged_epoch"] = start_epoch
        config["resume_from_checkpoint_epoch"] = saved_best["epoch"]
        config["optimizer_reset"] = True
        (run/f"history_before_resume_{start_epoch}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        torch.save(saved_best, run/f"best_before_resume_{start_epoch}.pt")
        (run/f"resume_after_epoch_{start_epoch}.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    else:
        (run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    lr = .00015 + (.002-.00015)*(1+math.cos(math.pi*start_epoch/args.epochs))/2
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-start_epoch, eta_min=.00015)
    if args.resume and (run/"last.pt").exists():
        last = torch.load(run/"last.pt", map_location="cpu", weights_only=True)
        if last["epoch"] != start_epoch:
            raise ValueError("last.pt and history do not match; preserve artifacts before recovery")
        model.load_state_dict(last["state_dict"])
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        torch.set_rng_state(last["rng_state"])
        config["resume_from_checkpoint_epoch"] = last["epoch"]
        config["optimizer_reset"] = False
        (run/f"resume_after_epoch_{start_epoch}.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({"initial_val_iou": initial_iou, **config}), flush=True)
    for epoch in range(start_epoch+1, args.epochs + 1):
        model.train()
        epoch_start, total = time.perf_counter(), 0.0
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            p = logits.sigmoid()
            edge = F.max_pool2d(y, 3, 1, 1) - (-F.max_pool2d(-y, 3, 1, 1))
            weight = 1 + 2 * edge
            bce = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * weight).mean()
            dice = 1 - ((2*(p*y).sum((1, 2, 3)) + 1) / ((p+y).sum((1, 2, 3)) + 1)).mean()
            loss = bce + dice
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += float(loss.detach()) * len(x)
        model.eval()
        iou, val_loss = validation(model, val_loader)
        row = {"epoch": epoch, "train_loss": total/len(train_loader.dataset), "val_loss": val_loss,
               "val_iou": iou, "seconds": time.perf_counter()-epoch_start, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        if iou > best:
            best = iou
            torch.save({"state_dict": model.state_dict(), "width": 8, "epoch": epoch,
                        "val_iou": iou, "seed": args.seed}, run / "best.pt")
        scheduler.step()
        torch.save({"state_dict": model.state_dict(), "width": 8, "epoch": epoch,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "rng_state": torch.get_rng_state()}, run/"last.pt")
        (run / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run / "training_summary.json").write_text(json.dumps({"initial_val_iou": initial_iou,
          "best_val_iou": best, "elapsed_seconds": sum(row["seconds"] for row in history),
          "completed_epochs": len(history), "resume_records": [p.name for p in run.glob("resume_after_epoch_*.json")],
          "checkpoint_sha256": hashlib.sha256((run/"best.pt").read_bytes()).hexdigest()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
