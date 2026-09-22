"""Shared helpers: where the tool keeps its files, how weight names resolve,
and the two output conventions every command follows.

Output conventions (the contract Velox reads):
  * ``PROGRESS: i/n`` lines on stdout while a command runs;
  * one ``RESULT: {json}`` line at the end (also written to ``--out`` when a
    command takes that flag). Non-zero exit = failure; the last stderr/stdout
    lines carry the reason.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

HOME_ENV = "VELOX_AI_TOOLS_HOME"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# Ultralytics' published COCO checkpoints; a bare name is downloaded by
# ultralytics on first use. "yolov11s.pt" is a common misspelling of the
# YOLO11 assets, which are called "yolo11s.pt".
_ASSET_RE = re.compile(r"^yolo(v?)(5|8|9|10|11|12)([nsmlx]|[tce])?(-(seg|pose|obb|cls))?\.pt$", re.IGNORECASE)


def home_dir() -> Path:
    """Where downloaded checkpoints and scratch files live: $VELOX_AI_TOOLS_HOME,
    else %LOCALAPPDATA%/VeloxAITools (Windows) or ~/.velox_ai_tools."""
    env = os.environ.get(HOME_ENV, "").strip()
    if env:
        return Path(env)
    local = os.environ.get("LOCALAPPDATA", "").strip()
    if local:
        return Path(local) / "VeloxAITools"
    return Path.home() / ".velox_ai_tools"


def weights_cache_dir() -> Path:
    d = home_dir() / "weights"
    d.mkdir(parents=True, exist_ok=True)
    return d


def normalise_asset_name(name: str) -> str:
    """'yolov11s.pt' -> 'yolo11s.pt' (the real asset name); other names unchanged."""
    m = _ASSET_RE.match(name.strip())
    if not m:
        return name.strip()
    version = m.group(2)
    if version in ("11", "12") and m.group(1):
        return "yolo" + name.strip()[5:]
    return name.strip()


def is_asset_name(name: str) -> bool:
    return bool(_ASSET_RE.match(str(name).strip())) and not Path(name).is_file()


def resolve_weights(weights: str) -> str:
    """An existing file is used as given. A bare Ultralytics asset name is
    placed in the tool's own weights cache, where ultralytics downloads it on
    first use, so no checkpoint ever lands in the caller's working directory.
    Anything else is handed to ultralytics unchanged (it raises)."""
    w = str(weights or "").strip()
    if not w:
        raise SystemExit("no weights given")
    p = Path(w)
    if p.is_file():
        return str(p.resolve())
    if is_asset_name(w) and not p.parent.name:
        return str(weights_cache_dir() / normalise_asset_name(w))
    return w


def load_model(weights: str):
    """The Ultralytics model for a weight file or asset name."""
    from ultralytics import YOLO
    return YOLO(resolve_weights(weights))


def names_list(model: Any) -> List[str]:
    names = getattr(model, "names", None) or {}
    if isinstance(names, dict):
        if not names:
            return []
        return [str(names.get(i, f"class_{i}")) for i in range(max(int(k) for k in names) + 1)]
    return [str(n) for n in names]


def device_arg(device: str) -> Optional[str]:
    d = str(device or "auto").strip()
    return None if d in ("", "auto") else d


def scan_images(folder: Path) -> List[str]:
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def read_class_file(path: Optional[str]) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def progress(i: int, n: int, tail: str = "") -> None:
    print(f"PROGRESS: {i}/{n}" + (f" - {tail}" if tail else ""), flush=True)


def emit_result(result: Dict[str, Any]) -> None:
    print("RESULT: " + json.dumps(result, ensure_ascii=False), flush=True)


def write_json(path: Optional[str], data: Dict[str, Any]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def chunks(seq: Sequence[Any], n: int) -> Iterable[Sequence[Any]]:
    n = max(1, int(n))
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def python_version() -> str:
    return sys.version.split()[0]
