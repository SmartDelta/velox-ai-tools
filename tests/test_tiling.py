"""Tiles cover a big frame, boxes map back to frame pixels, neighbours merge."""
from __future__ import annotations

import numpy as np

from velox_ai_tools import tiling


def test_tiles_cover_the_frame_with_overlap_and_full_size():
    tiles = tiling.tile_grid(5328, 4608, 1280, 0.2)
    assert all(x1 - x0 == 1280 and y1 - y0 == 1280 for x0, y0, x1, y1 in tiles)
    assert min(t[0] for t in tiles) == 0 and max(t[2] for t in tiles) == 5328
    assert min(t[1] for t in tiles) == 0 and max(t[3] for t in tiles) == 4608
    xs = sorted({t[0] for t in tiles})
    assert xs[1] - xs[0] == 1024, "20 % overlap = 1024 px step"
    assert tiling.tile_grid(640, 480, 1280, 0.2) == [(0, 0, 640, 480)], "small frames are one tile"
    assert tiling.tile_grid(5328, 4608, 0, 0.2) == [(0, 0, 5328, 4608)], "tile 0 = whole frame"


def test_nms_merges_the_same_box_seen_in_two_tiles():
    dets = [{"cls": "D40", "conf": 0.9, "box": [100, 100, 200, 200]},
            {"cls": "D40", "conf": 0.7, "box": [105, 102, 203, 199]},
            {"cls": "D00", "conf": 0.8, "box": [100, 100, 200, 200]},
            {"cls": "D40", "conf": 0.6, "box": [500, 500, 600, 600]}]
    kept = tiling.merge_detections(dets, iou_thr=0.5)
    assert len(kept) == 3
    assert [k["conf"] for k in kept] == [0.9, 0.8, 0.6]


def test_detect_image_maps_tile_detections_back_to_frame_pixels():
    img = np.zeros((2000, 3000, 3), dtype=np.uint8)
    calls = []

    def fake_predict(crops):
        out = []
        for c in crops:
            calls.append(c.shape)
            out.append({"boxes": np.array([[10.0, 10.0, 60.0, 60.0]]), "cls": np.array([3]),
                        "conf": np.array([0.8]), "polys": [np.array([[10.0, 10.0], [60.0, 10.0], [60.0, 60.0]])]})
        return out

    dets = tiling.detect_image(fake_predict, img, tile=1280, overlap=0.2, batch=4, names={3: "D40"})
    assert calls and all(s[0] == 1280 and s[1] == 1280 for s in calls)
    assert all(d["cls"] == "D40" for d in dets)
    xs = sorted(d["box"][0] for d in dets)
    assert xs[0] == 10.0 and xs[-1] > 1280, "boxes were offset by their tile origin"
    assert dets[0]["poly"] is not None and len(dets[0]["poly"]) == 3


def test_the_ultralytics_adapter_hands_over_plain_arrays(fake_ultralytics):
    model = fake_ultralytics("w.pt")
    predict = tiling.ultralytics_predict(model, imgsz=640, conf=0.3, device="auto", classes=[3])
    out = predict([np.zeros((100, 100, 3), dtype=np.uint8)])
    assert out[0]["boxes"].shape == (1, 4) and int(out[0]["cls"][0]) == 3 and out[0]["polys"] is None
