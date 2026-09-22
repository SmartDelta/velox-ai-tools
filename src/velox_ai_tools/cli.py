"""The ``velox-ai`` command line.

    velox-ai version                      what is installed (ultralytics, torch, CUDA)
    velox-ai info --weights W             class names in a weight file
    velox-ai detect ...                   tiled detections for a list of frames -> JSON
    velox-ai prelabel ...                 propose YOLO labels for a photo folder
    velox-ai infer ...                    SmartCore AI: detect on a folder, write labels, reports, GeoJSON
    velox-ai train ...                    train/resume a YOLO (or RF-DETR) detector
    velox-ai train-rdd2022 ...            the road damage model: RDD2022 (+ reviewed tiles) -> weight file
    velox-ai two-stage {classify,train,build-dataset}

Every command prints ``PROGRESS: i/n`` lines while it runs and one
``RESULT: {json}`` line at the end. Exit code 0 = success.
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from . import __version__, detect, infer, info, prelabel, train, train_rdd2022, two_stage


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="velox-ai", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"velox-ai {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("version", help="what is installed, and is CUDA usable")
    s.set_defaults(func=info.run_version)

    s = sub.add_parser("info", help="class names inside a weight file")
    s.add_argument("--weights", required=True)
    s.set_defaults(func=info.run_info)

    s = sub.add_parser("detect", help="tiled detections for a list of frames (JSON in, JSON out)")
    detect.add_arguments(s)
    s.set_defaults(func=detect.run)

    s = sub.add_parser("prelabel", help="propose YOLO labels for the photos of a folder")
    prelabel.add_arguments(s)
    s.set_defaults(func=prelabel.run)

    s = sub.add_parser("infer", help="SmartCore AI inference on an image folder (labels, reports, GeoJSON)")
    infer.add_arguments(s)
    s.set_defaults(func=infer.run)

    s = sub.add_parser("train", help="train or resume a YOLO / RF-DETR detector")
    train.add_arguments(s)
    s.set_defaults(func=train.run)

    s = sub.add_parser("train-rdd2022", help="road damage model from RDD2022 (+ reviewed datasets)")
    train_rdd2022.add_arguments(s)
    s.set_defaults(func=train_rdd2022.run)

    s = sub.add_parser("two-stage", help="detect -> crop -> classify pipeline and its classifier")
    two_stage.add_arguments(s)
    s.set_defaults(func=two_stage.run)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
