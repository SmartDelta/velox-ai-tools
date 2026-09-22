"""``velox-ai detect``: pixel detections for a list of frames, tiled.

Input ``--frames``: JSON list of ``{"id": ..., "img_path": ...}``.
Output ``--out``: ``{"weights", "names": {id: name}, "frames": [{"id", "width",
"height", "detections": [{"cls", "cls_id", "conf", "box": [x1,y1,x2,y2],
"poly": [[x,y],...] | null}]}], "seconds", "resumed", ...}``. Boxes are in
full-frame pixels; nothing here knows about cameras or the ground.

Stop and resume: every finished frame is appended at once to
``<out>.partial.jsonl`` (one JSON record per line, flushed), so a killed run
loses at most the frame it was on. ``--resume`` keeps those records and only
runs the frames that are not in the file yet; the final ``--out`` is written
in the input order and the partial file is removed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from .common import emit_result, load_model, names_list, progress
from .tiling import crop_region, detect_image, ultralytics_predict


def partial_path(out: Any) -> Path:
    return Path(str(out) + ".partial.jsonl")


def read_partial(path: Path) -> Dict[str, Dict[str, Any]]:
    """{frame id: record} from a partial file; a torn last line is dropped."""
    done: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and "id" in rec:
            done[str(rec["id"])] = rec
    return done


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
    p.add_argument("--resume", action="store_true",
                   help="continue a stopped run: frames already in <out>.partial.jsonl are kept and skipped")
    p.add_argument("--region", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), default=None,
                   help="only this part of every frame (normalised 0..1, e.g. the road band of a 360 panorama); "
                        "boxes come back in full-frame pixels. X0 > X1 = a band across the seam of a panorama "
                        "(right part + left part stitched; x beyond the width = modulo the width)")
    return p


def run(args: argparse.Namespace) -> int:
    import cv2

    frames: List[Dict[str, Any]] = json.loads(Path(args.frames).read_text(encoding="utf-8"))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = partial_path(out_path)
    done = read_partial(partial) if args.resume else {}
    if not args.resume and partial.exists():
        partial.unlink()
    todo = [fr for fr in frames if str(fr["id"]) not in done]
    n = len(frames)
    resumed = n - len(todo)
    if resumed:
        print(f"resuming: {resumed}/{n} frames already done", flush=True)
    t0 = time.time()
    model = load_model(args.weights)
    names = {i: name for i, name in enumerate(names_list(model))}
    predict = ultralytics_predict(model, imgsz=args.imgsz, conf=args.conf, device=args.device,
                                  classes=args.classes, iou=args.iou)
    with partial.open("a", encoding="utf-8") as fh:
        for fr in todo:
            img = cv2.imread(str(fr["img_path"]))
            if img is None:
                rec: Dict[str, Any] = {"id": fr["id"], "error": "unreadable", "detections": []}
            else:
                full_h, full_w = img.shape[:2]
                ox = oy = 0
                if args.region:
                    img, ox, oy = crop_region(img, args.region)
                dets = detect_image(predict, img, tile=args.tile, overlap=args.overlap,
                                    batch=args.batch, names=names, iou_thr=args.merge_iou)
                if ox or oy:
                    for d in dets:
                        b = d["box"]
                        d["box"] = [b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy]
                        if d.get("poly"):
                            d["poly"] = [[u + ox, v + oy] for u, v in d["poly"]]
                rec = {"id": fr["id"], "width": int(full_w), "height": int(full_h), "detections": dets}
                if args.region:
                    rec["region"] = [float(v) for v in args.region]
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done[str(fr["id"])] = rec
            progress(len(done), n)
    out: Dict[str, Any] = {"weights": str(args.weights), "names": names,
                           "frames": [done[str(fr["id"])] for fr in frames if str(fr["id"]) in done],
                           "tile": args.tile, "overlap": args.overlap, "imgsz": args.imgsz, "conf": args.conf,
                           "resumed": resumed, "seconds": round(time.time() - t0, 1)}
    out_path.write_text(json.dumps(out), encoding="utf-8")
    partial.unlink(missing_ok=True)
    total = sum(len(f["detections"]) for f in out["frames"])
    emit_result({"ok": True, "frames": n, "detections": total, "seconds": out["seconds"], "resumed": resumed,
                 "out": str(out_path)})
    return 0
