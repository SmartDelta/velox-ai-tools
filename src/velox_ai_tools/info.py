"""``velox-ai version`` (what is installed, is CUDA there) and
``velox-ai info --weights`` (the class names inside a weight file)."""
from __future__ import annotations

import argparse
from typing import Any, Dict, List

from . import __version__
from .common import emit_result, load_model, names_list, python_version, resolve_weights


def environment() -> Dict[str, Any]:
    d: Dict[str, Any] = {"ok": False, "tool": __version__, "python": python_version(), "ultralytics": None,
                         "torch": None, "cuda": None, "cuda_available": False, "device": None, "missing": []}
    missing: List[str] = []
    try:
        import ultralytics
        d["ultralytics"] = str(ultralytics.__version__)
    except ImportError:
        missing.append("ultralytics")
    try:
        import torch
        d["torch"] = str(torch.__version__)
        d["cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
        try:
            d["cuda_available"] = bool(torch.cuda.is_available())
            if d["cuda_available"]:
                d["device"] = str(torch.cuda.get_device_name(0))
        except Exception as exc:                      # noqa: BLE001 - a broken CUDA is reported, not fatal
            d["cuda_error"] = str(exc)
    except ImportError:
        missing.append("torch")
    d["missing"] = missing
    d["ok"] = not missing
    return d


def run_version(args: argparse.Namespace) -> int:
    d = environment()
    emit_result(d)
    return 0 if d["ok"] else 1


def run_info(args: argparse.Namespace) -> int:
    model = load_model(args.weights)
    names = names_list(model)
    emit_result({"ok": True, "weights": resolve_weights(args.weights), "names": names, "nc": len(names),
                 "task": getattr(model, "task", None)})
    return 0
