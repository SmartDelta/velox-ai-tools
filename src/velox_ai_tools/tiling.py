"""Tiled inference: cut a large frame into overlapping windows, detect on
each, map the boxes back and merge them with class-wise NMS. A 24 MP road
photo is inferred at full resolution this way instead of squashed to 640 px.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common import chunks, device_arg

Box = Tuple[float, float, float, float]

# predict(crops: list[np.ndarray BGR]) -> list of per-crop results, each a
# dict {boxes: (n,4) float xyxy, cls: (n,) int, conf: (n,) float,
#       polys: list[np.ndarray (k,2)] | None}
PredictFn = Callable[[List[np.ndarray]], List[Dict[str, Any]]]


def tile_grid(width: int, height: int, tile: int, overlap: float = 0.2) -> List[Tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) windows of `tile` px that cover the image with the
    given overlap fraction; the last row/column is shifted inwards so every
    tile is full size and nothing is left uncovered. tile <= 0 = whole image."""
    tile = int(tile)
    if tile <= 0 or (width <= tile and height <= tile):
        return [(0, 0, int(width), int(height))]
    step = max(1, int(round(tile * (1.0 - float(overlap)))))

    def _starts(size: int) -> List[int]:
        if size <= tile:
            return [0]
        s = list(range(0, size - tile, step))
        s.append(size - tile)
        return sorted(set(s))

    return [(x, y, min(x + tile, width), min(y + tile, height))
            for y in _starts(height) for x in _starts(width)]


def crop_bounds(width: int, height: int, region: Optional[Sequence[float]]) -> Tuple[int, int, int, int]:
    """Pixel bounds (x0, y0, x1, y1) of a normalised region (x0 y0 x1 y1 in
    0..1) in a frame; the whole frame without a region. A road in a 360
    panorama, say, is the band between the horizon and the vehicle body.
    x0 > x1 is a band across the seam of a panorama: x1 then lies beyond the
    width (unwrapped), x is taken modulo the width."""
    if not region:
        return 0, 0, int(width), int(height)
    rx0, ry0, rx1, ry1 = (min(1.0, max(0.0, float(v))) for v in region[:4])
    x0 = int(round(rx0 * width))
    y0 = int(round(ry0 * height))
    x1 = int(round(rx1 * width)) + (int(width) if rx0 > rx1 else 0)
    x1 = max(x0 + 1, x1)
    y1 = max(y0 + 1, min(int(round(ry1 * height)), int(height)))
    return x0, y0, x1, y1


def crop_region(img: np.ndarray, region: Optional[Sequence[float]]) -> Tuple[np.ndarray, int, int]:
    """The band of a frame as one image plus the offset (ox, oy) from crop to
    frame pixels; across the seam the right and left parts are stitched and
    x offsets may exceed the width (modulo the width in the frame)."""
    if not region:
        return img, 0, 0
    h, w = img.shape[:2]
    x0, y0, x1, y1 = crop_bounds(w, h, region)
    band = img[y0:y1]
    if x1 <= w:
        return band[:, x0:x1], x0, y0
    return np.concatenate([band[:, x0:w], band[:, 0:x1 - w]], axis=1), x0, y0


def _iou(a: Box, b: Box) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0.0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def merge_detections(dets: Sequence[Dict[str, Any]], iou_thr: float = 0.5) -> List[Dict[str, Any]]:
    """Class-wise non-maximum suppression over detections that may come
    from overlapping tiles. Each det: {cls, conf, box:[x1,y1,x2,y2], ...}."""
    keep: List[Dict[str, Any]] = []
    for d in sorted(dets, key=lambda d: -float(d.get("conf", 0.0))):
        box = tuple(float(v) for v in d["box"])
        if any(k["cls"] == d["cls"] and _iou(box, tuple(k["box"])) >= iou_thr for k in keep):
            continue
        keep.append(dict(d, box=list(box)))
    return keep


def detect_image(predict: PredictFn, img: np.ndarray, *, tile: int, overlap: float,
                 batch: int, names: Dict[int, str], iou_thr: float = 0.5) -> List[Dict[str, Any]]:
    """Detections in FULL-image pixel coordinates."""
    h, w = img.shape[:2]
    windows = tile_grid(w, h, tile, overlap)
    dets: List[Dict[str, Any]] = []
    for group in chunks(windows, batch):
        crops = [img[y0:y1, x0:x1] for (x0, y0, x1, y1) in group]
        results = predict(crops)
        for (x0, y0, _x1, _y1), res in zip(group, results):
            boxes = np.asarray(res.get("boxes", np.zeros((0, 4))), dtype=float).reshape(-1, 4)
            cls = np.asarray(res.get("cls", np.zeros((0,))), dtype=int).reshape(-1)
            conf = np.asarray(res.get("conf", np.zeros((0,))), dtype=float).reshape(-1)
            polys = res.get("polys")
            for i in range(len(boxes)):
                bx = boxes[i] + np.array([x0, y0, x0, y0], dtype=float)
                d: Dict[str, Any] = {"cls": names.get(int(cls[i]), str(int(cls[i]))),
                                     "cls_id": int(cls[i]), "conf": float(conf[i]),
                                     "box": [float(v) for v in bx], "poly": None}
                if polys is not None and i < len(polys) and polys[i] is not None and len(polys[i]) >= 3:
                    p = np.asarray(polys[i], dtype=float) + np.array([x0, y0], dtype=float)
                    d["poly"] = [[float(u), float(v)] for u, v in p]
                dets.append(d)
    return merge_detections(dets, iou_thr=iou_thr) if len(windows) > 1 else dets


def ultralytics_predict(model: Any, *, imgsz: int, conf: float, device: str,
                        classes: Optional[Sequence[int]] = None, iou: float = 0.45) -> PredictFn:
    """Adapter from ultralytics results to the plain dicts detect_image wants."""
    def _predict(crops: List[np.ndarray]) -> List[Dict[str, Any]]:
        results = model.predict(crops, imgsz=imgsz, conf=conf, iou=iou, device=device_arg(device),
                                classes=list(classes) if classes else None, verbose=False)
        out = []
        for r in results:
            b = r.boxes
            polys = None
            if getattr(r, "masks", None) is not None and r.masks is not None:
                try:
                    polys = [np.asarray(p, dtype=float) for p in r.masks.xy]
                except Exception:                     # noqa: BLE001 - masks are optional
                    polys = None
            out.append({"boxes": b.xyxy.cpu().numpy() if b is not None else np.zeros((0, 4)),
                        "cls": b.cls.cpu().numpy() if b is not None else np.zeros((0,)),
                        "conf": b.conf.cpu().numpy() if b is not None else np.zeros((0,)),
                        "polys": polys})
        return out
    return _predict
