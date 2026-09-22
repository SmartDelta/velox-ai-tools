"""``velox-ai prelabel``: propose YOLO boxes for the photos of a folder and
write them as label files a person then corrects.

Class ids follow the caller's class list (``--classes-file``, one name per
line); a model class that is not in it is appended, so the returned
``classes`` is the list the caller must keep. Photos that already have a
label file are skipped unless ``--overwrite`` (a reviewer who saved a photo
as empty must not get the proposals back). ``--only``/``--list`` restrict the
run to given file names; ``--no-write`` returns the boxes without touching
the label folder.

Result (``RESULT:`` line and ``--out``): ``{"ok", "model", "classes",
"n_images", "n_boxes", "skipped", "images": [{"name", "written", "boxes":
[{"cls_idx", "cls", "x", "y", "w", "h", "conf"}]}]}`` with x/y/w/h
normalised (YOLO txt convention).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from .common import (chunks, device_arg, emit_result, load_model, names_list, progress, read_class_file,
                     scan_images, write_json)


def default_label_dir(images_dir: Path) -> Path:
    """images/train -> labels/train (YOLO dataset layout), else <parent>/labels."""
    if images_dir.name in ("train", "val", "test") and images_dir.parent.name == "images":
        return images_dir.parent.parent / "labels" / images_dir.name
    return images_dir.parent / "labels"


def add_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--weights", required=True)
    p.add_argument("--images", required=True, help="folder with the photos")
    p.add_argument("--labels", default="", help="label folder (default: the YOLO layout next to --images)")
    p.add_argument("--classes-file", default="", help="class names, one per line (the dataset's classes.txt)")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--overwrite", action="store_true", help="also photos that already have a label file")
    p.add_argument("--only", nargs="*", default=None, help="file names to label (default: every photo)")
    p.add_argument("--list", default="", help="text file with one file name per line (long selections)")
    p.add_argument("--no-write", action="store_true", help="return the boxes, write nothing")
    p.add_argument("--out", default="", help="JSON result file")
    return p


def _write_label(label_dir: Path, name: str, boxes: List[Dict[str, Any]]) -> Path:
    label_dir.mkdir(parents=True, exist_ok=True)
    txt = label_dir / (Path(name).stem + ".txt")
    lines = [f"{int(b['cls_idx'])} {float(b['x']):.6f} {float(b['y']):.6f} {float(b['w']):.6f} {float(b['h']):.6f}"
             for b in boxes]
    txt.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")
    return txt


def run(args: argparse.Namespace) -> int:
    images_dir = Path(args.images)
    if not images_dir.is_dir():
        raise SystemExit(f"images folder not found: {images_dir}")
    label_dir = Path(args.labels) if args.labels else default_label_dir(images_dir)
    write = not args.no_write
    classes: List[str] = read_class_file(args.classes_file)

    names = scan_images(images_dir)
    wanted = list(args.only or [])
    if args.list:
        wanted += [ln.strip() for ln in Path(args.list).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if wanted:
        present = set(names)
        names = [n for n in wanted if n in present]
    if write and not args.overwrite:
        todo = [n for n in names if not (label_dir / (Path(n).stem + ".txt")).is_file()]
    else:
        todo = list(names)
    skipped = len(names) - len(todo)
    result: Dict[str, Any] = {"ok": True, "model": str(args.weights), "classes": classes, "n_images": len(todo),
                              "n_boxes": 0, "skipped": skipped, "images": []}
    if not todo:
        write_json(args.out, result)
        emit_result(result)
        return 0

    model = load_model(args.weights)
    model_names = names_list(model)
    done = 0
    for group in chunks(todo, args.batch):
        paths = [str(images_dir / n) for n in group]
        results = model.predict(paths, conf=args.conf, imgsz=args.imgsz, device=device_arg(args.device),
                                verbose=False)
        for name, res in zip(group, results):
            b = res.boxes
            boxes: List[Dict[str, Any]] = []
            if b is not None and len(b):
                xywhn = b.xywhn.cpu().numpy()
                cls = b.cls.cpu().numpy()
                conf = b.conf.cpu().numpy()
                for k in range(len(xywhn)):
                    cid = int(cls[k])
                    cname = model_names[cid] if cid < len(model_names) else f"class_{cid}"
                    if cname not in classes:
                        classes.append(cname)
                    boxes.append({"cls_idx": classes.index(cname), "cls": cname,
                                  "x": float(xywhn[k][0]), "y": float(xywhn[k][1]),
                                  "w": float(xywhn[k][2]), "h": float(xywhn[k][3]),
                                  "conf": round(float(conf[k]), 3)})
            if write:
                _write_label(label_dir, name, boxes)
            result["images"].append({"name": name, "written": write, "boxes": boxes})
            result["n_boxes"] += len(boxes)
            done += 1
            progress(done, len(todo), name)
    if write:
        label_dir.mkdir(parents=True, exist_ok=True)
        (label_dir / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    result["classes"] = classes
    write_json(args.out, result)
    summary = dict(result)
    summary.pop("images", None)
    summary["out"] = str(args.out) if args.out else ""
    emit_result(summary if args.out else result)
    return 0
