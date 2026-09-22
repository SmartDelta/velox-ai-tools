"""``velox-ai train``: train (or resume) a YOLO detector on a dataset.yaml,
or an RF-DETR model with ``--framework rfdetr``. Ultralytics writes
``<project>/<name>/results.csv`` and ``weights/best.pt``; the caller polls
those. The augmentation and schedule presets are the ones SmartCore AI's
Local Training tab used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import emit_result, resolve_weights


def add_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--data", default="", help="dataset.yaml")
    p.add_argument("--model", default="yolov8s.pt", help="start weights (file or asset name), or rfdetr-base/-large")
    p.add_argument("--framework", choices=["yolo", "rfdetr"], default="yolo")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="", help="'' = auto, '0' = first GPU, 'cpu'")
    p.add_argument("--project", required=True, help="runs folder")
    p.add_argument("--name", default="train", help="run name under --project")
    p.add_argument("--freeze", type=int, default=0, help="backbone layers to freeze")
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--optimizer", default="AdamW")
    p.add_argument("--lr0", type=float, default=0.001)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-amp", action="store_true", help="fp32 training (cuDNN stream-mismatch workaround)")
    p.add_argument("--aug", choices=["smartcore", "none"], default="smartcore")
    p.add_argument("--resume", default="", help="last.pt of an interrupted run; continues it")
    return p


def _train_yolo(args: argparse.Namespace) -> Path:
    from ultralytics import YOLO

    if args.resume:
        last = Path(args.resume)
        if not last.is_file():
            raise SystemExit(f"cannot resume: {last} not found")
        model = YOLO(str(last))
        kw = {"resume": True}
        if args.epochs:
            kw["epochs"] = args.epochs
        if args.device:
            kw["device"] = args.device
        model.train(**kw)
        return last.parent / "best.pt"
    if not args.data or not Path(args.data).is_file():
        raise SystemExit(f"dataset.yaml not found: {args.data}")
    model = YOLO(resolve_weights(args.model))
    kw = dict(data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
              project=str(args.project), name=args.name, exist_ok=True, workers=args.workers,
              amp=not args.no_amp, patience=args.patience, optimizer=args.optimizer,
              lr0=args.lr0, lrf=0.01, warmup_epochs=3, warmup_bias_lr=0.1, cos_lr=True,
              label_smoothing=args.label_smoothing)
    if args.device:
        kw["device"] = args.device
    if args.freeze > 0:
        kw["freeze"] = args.freeze
    if args.aug == "smartcore":
        kw.update(mosaic=1.0, mixup=0.15, copy_paste=0.1, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                  degrees=5.0, translate=0.1, scale=0.5, flipud=0.0, fliplr=0.5)
    model.train(**kw)
    return Path(args.project) / args.name / "weights" / "best.pt"


def _train_rfdetr(args: argparse.Namespace) -> Path:
    import yaml

    cfg = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    ds_dir = cfg.get("path", ".")
    from rfdetr import RFDETRBase, RFDETRLarge
    model = RFDETRLarge() if "large" in args.model.lower() else RFDETRBase()
    kw = dict(dataset_dir=ds_dir, epochs=args.epochs, batch_size=args.batch, imgsz=args.imgsz,
              grad_accumulation=max(1, 16 // max(1, args.batch)), project=str(args.project), name=args.name)
    if args.device:
        kw["device"] = args.device
    if args.lr0 != 0.001:
        kw["lr"] = args.lr0
    model.train(**kw)
    return Path(args.project) / args.name


def run(args: argparse.Namespace) -> int:
    rfdetr = args.framework == "rfdetr" or "rfdetr" in str(args.model).lower()
    out = _train_rfdetr(args) if rfdetr else _train_yolo(args)
    emit_result({"ok": True, "framework": "rfdetr" if rfdetr else "yolo", "run_dir": str(Path(args.project) / args.name),
                 "weights": str(out)})
    return 0
