"""``velox-ai detect``: pixel detections for a list of frames, tiled.

Input ``--frames``: JSON list of ``{"id": ..., "img_path": ...}``.
Output ``--out``: ``{"weights", "names": {id: name}, "frames": [{"id", "width",
"height", "detections": [{"cls", "cls_id", "conf", "box": [x1,y1,x2,y2],
"poly": [[x,y],...] | null}]}], "seconds", ...}``. Boxes are in full-frame
pixels; nothing here knows about cameras or the ground.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

from .common import emit_result, load_model, names_list, progress
from .tiling import detect_image, ultralytics_predict


def add_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--weights", required=True, help="weight file, or an Ultralytics asset name (yolov8s.pt)")
    p.add_argument("--frames", required=True, help='JSON list of {"id", "img_path"}')
    p.add_argument("--out", required=True, help="JSON result file")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU inside one tile")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--tile", type=int, default=1280, help="tile size in px; 0 = whole frame at --imgsz")
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--classes", type=int, nargs="*", default=None, help="only these class ids")
    p.add_argument("--merge-iou", type=float, default=0.5, help="IoU at which boxes from neighbouring tiles merge")
    return p


def run(args: argparse.Namespace) -> int:
    import cv2

    frames = json.loads(Path(args.frames).read_text(encoding="utf-8"))
    model = load_model(args.weights)
    names = {i: n for i, n in enumerate(names_list(model))}
    predict = ultralytics_predict(model, imgsz=args.imgsz, conf=args.conf, device=args.device,
                                  classes=args.classes, iou=args.iou)
    out: Dict[str, Any] = {"weights": str(args.weights), "names": names, "frames": [],
                           "tile": args.tile, "overlap": args.overlap, "imgsz": args.imgsz, "conf": args.conf}
    t0 = time.time()
    n = len(frames)
    for i, fr in enumerate(frames):
        img = cv2.imread(str(fr["img_path"]))
        if img is None:
            out["frames"].append({"id": fr["id"], "error": "unreadable", "detections": []})
        else:
            dets = detect_image(predict, img, tile=args.tile, overlap=args.overlap,
                                batch=args.batch, names=names, iou_thr=args.merge_iou)
            out["frames"].append({"id": fr["id"], "width": int(img.shape[1]), "height": int(img.shape[0]),
                                  "detections": dets})
        progress(i + 1, n)
    out["seconds"] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    total = sum(len(f["detections"]) for f in out["frames"])
    emit_result({"ok": True, "frames": n, "detections": total, "seconds": out["seconds"], "out": str(args.out)})
    return 0
