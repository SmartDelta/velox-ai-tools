"""Offline test doubles: a fake `ultralytics` (and `torch`) so every command
can be exercised without a GPU, a download or a real model."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest


class T:
    """The little tensor the code touches: .cpu().numpy(), [i], .tolist(), .item()."""

    def __init__(self, v):
        self.v = np.asarray(v, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self.v

    def tolist(self):
        return self.v.tolist()

    def item(self):
        return self.v.item()

    def __getitem__(self, i):
        return T(self.v[i])

    def __len__(self):
        return len(self.v)


class FakeBoxes:
    def __init__(self, rows, w, h):
        rows = list(rows)
        self._n = len(rows)
        self.xyxy = T([[r[2], r[3], r[4], r[5]] for r in rows] if rows else np.zeros((0, 4)))
        self.cls = T([r[0] for r in rows])
        self.conf = T([r[1] for r in rows])
        self.xywhn = T([[(r[2] + r[4]) / 2 / w, (r[3] + r[5]) / 2 / h, (r[4] - r[2]) / w, (r[5] - r[3]) / h]
                        for r in rows] if rows else np.zeros((0, 4)))

    def __len__(self):
        return self._n


class FakeResult:
    def __init__(self, rows, w, h, names):
        self.boxes = FakeBoxes(rows, w, h)
        self.names = names
        self.masks = None


def _default_scenario(item, i, kw):
    return [(3, 0.8, 10.0, 10.0, 60.0, 60.0)]


class FakeYOLO:
    """Records every call; `scenario(item, index, kw)` decides the boxes
    (cls, conf, x1, y1, x2, y2 in pixels) each input gets."""
    calls: List[Any] = []
    names = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
    task = "detect"
    scenario = staticmethod(_default_scenario)

    def __init__(self, weights):
        FakeYOLO.calls.append(("load", str(weights)))

    def predict(self, source, **kw):
        items = source if isinstance(source, list) else [source]
        FakeYOLO.calls.append(("predict", len(items), kw.get("imgsz")))
        out = []
        for i, it in enumerate(items):
            if isinstance(it, np.ndarray):
                h, w = it.shape[:2]
            else:
                from PIL import Image
                with Image.open(it) as im:
                    w, h = im.size
            out.append(FakeResult(FakeYOLO.scenario(it, i, kw), w, h, dict(self.names)))
        return out

    def train(self, **kw):
        FakeYOLO.calls.append(("train", kw))
        wdir = Path(kw["project"]) / kw["name"] / "weights"
        wdir.mkdir(parents=True, exist_ok=True)
        (wdir / "best.pt").write_bytes(b"best")
        (Path(kw["project"]) / kw["name"] / "results.csv").write_text(
            "epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n1,0.5,0.3\n", encoding="utf-8")

    def val(self, **kw):
        box = types.SimpleNamespace(map50=0.5, map=0.3, ap50=[0.5, 0.4, 0.6, 0.5])
        return types.SimpleNamespace(box=box)


@pytest.fixture
def fake_ultralytics(monkeypatch):
    mod = types.ModuleType("ultralytics")
    mod.YOLO = FakeYOLO
    mod.__version__ = "0.0-fake"
    monkeypatch.setitem(sys.modules, "ultralytics", mod)
    torch = types.ModuleType("torch")
    torch.__version__ = "0.0-fake"
    torch.version = types.SimpleNamespace(cuda=None)
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, get_device_name=lambda i: "none")
    monkeypatch.setitem(sys.modules, "torch", torch)
    FakeYOLO.calls.clear()
    FakeYOLO.scenario = staticmethod(_default_scenario)
    return FakeYOLO


@pytest.fixture
def tool_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("VELOX_AI_TOOLS_HOME", str(home))
    return home


def last_result(capsys) -> Dict[str, Any]:
    """The RESULT: line a command printed."""
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("RESULT: ")]
    assert lines, f"no RESULT line in:\n{out}"
    return json.loads(lines[-1][len("RESULT: "):])


def make_image(path: Path, w: int = 200, h: int = 150) -> Path:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), (90, 90, 90)).save(path)
    return path
