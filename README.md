# Velox AI Tools

The detector, pre-labelling and training commands that
[Velox SmartDelta Player](https://smartdelta.nl) uses for SmartCore AI and
Road Damage, packaged as a **separate, open-source program** (`velox-ai`).

It is built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
and is therefore licensed under the **GNU Affero General Public License
v3.0** (see `LICENSE`). Velox itself never imports this package: it starts
`velox-ai` as a child process and exchanges files and JSON with it, the way
any program calls `git` or `ffmpeg`. Velox works without it; the AI
functions then say the tools are not installed.

Weight files trained with this tool (for example the road damage model
`road_damage_rdd2022_yolo11s.pt`, classes D00/D10/D20/D40 from RDD2022) are
published under the same licence as GitHub releases.

## Install (Windows)

```powershell
git clone https://github.com/SmartDelta/velox-ai-tools.git velox_ai_tools
cd velox_ai_tools
.\install.ps1                 # %LOCALAPPDATA%\VeloxAITools\.venv, PyTorch cu118
.\install.ps1 -Cuda cpu       # no NVIDIA GPU
.\install.ps1 -Cuda cu126     # newer driver
```

`install.ps1` needs Python 3.10 - 3.12 and creates its own virtual
environment; nothing is installed into Velox. Velox looks for
`velox-ai.exe` in that default folder, on `PATH`, or where you point it
under *SmartCore AI > AI tools* (also the environment variable
`VELOX_AI_TOOLS`).

Manual install into any environment:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install .
velox-ai version
```

Optional extras: `pip install .[rfdetr]` (RF-DETR detector/training),
`.[onnx]` (ONNX classifiers).

## Commands

Every command prints `PROGRESS: i/n` lines while it runs and one
`RESULT: {json}` line at the end. Exit code 0 means success; on failure the
last lines of output say why.

| Command | Purpose |
| --- | --- |
| `velox-ai version` | What is installed: tool, ultralytics, torch, CUDA, GPU name. Exit 1 when ultralytics or torch is missing. |
| `velox-ai info --weights W` | Class names inside a weight file: `{"names": [...], "nc", "task"}`. |
| `velox-ai detect --weights W --frames frames.json --out dets.json [--tile 1280 --overlap 0.2 --imgsz 1280 --conf 0.25 --batch 4 --device auto --classes 0 2]` | Tiled detection for a list of frames. Input: `[{"id", "img_path"}]`. Output per frame: width, height and detections `{"cls", "cls_id", "conf", "box": [x1,y1,x2,y2] px, "poly"}`. `--tile 0` = whole frame. |
| `velox-ai prelabel --weights W --images DIR [--labels DIR --classes-file classes.txt --conf --imgsz --overwrite --only NAME... --list FILE --no-write --out r.json]` | Propose YOLO labels for a photo folder. Skips photos that already have a label file unless `--overwrite`; appends unknown model classes to the class list; `--no-write` only returns boxes (normalised x, y, w, h). |
| `velox-ai infer --yolo-version yolov8 --weights W --input-dir DIR --output-dir DIR [...]` | SmartCore AI inference: annotated images, `labels/*.txt`, `detection_results.csv`, `detection_summary.csv`, `labels_with_header.csv`, `detections.geojson`. Same flags as before (`--conf --iou --imgsz --device --save-txt --save-conf --classes --crop-top --crop-bottom --photo-csv --extra-dirs --area-bbox --augment --resume --batch-size --check-deps`). |
| `velox-ai train --data dataset.yaml --model yolov8s.pt --project RUNS --name train [--epochs --imgsz --batch --device --freeze --patience --optimizer --lr0 --label-smoothing --no-amp --resume last.pt --framework rfdetr]` | Train or resume a detector. Ultralytics writes `RUNS/name/results.csv` and `weights/best.pt`. |
| `velox-ai train-rdd2022 --root DIR --model M --weights-dir DIR [--epochs --imgsz --batch --device --no-amp --extra-dataset DS --extra-repeat 5 --no-rdd --no-download --out-name NAME --skip-train]` | The road damage model: downloads and prepares RDD2022 (13 GB) once, merges reviewed Velox datasets (tiles of one frame stay in one split), trains, and copies `best.pt` + a `.json` sidecar (classes, metrics, provenance) into `--weights-dir`. |
| `velox-ai two-stage classify --detector W --classifier C --input-dir DIR --output-dir DIR` | Detect, crop, classify (EfficientNet/ResNet). Also `two-stage train --dataset-dir --output-dir` and `two-stage build-dataset --detection-dir --output-dir`. |

Bare Ultralytics checkpoint names (`yolov8s.pt`, `yolo11n.pt`, also the
common misspelling `yolov11n.pt`) are downloaded on first use into the
tool's own folder (`%LOCALAPPDATA%\VeloxAITools\weights`, or
`$VELOX_AI_TOOLS_HOME/weights`), never into the caller's working directory.

## Development

```
python -m venv .venv
.venv\Scripts\python -m pip install -e .[test]
.venv\Scripts\python -m pytest
```

The tests run offline with a fake `ultralytics`; nothing is downloaded.

## Licence

Copyright (C) 2026 SmartDelta. This program is free software: you can
redistribute it and/or modify it under the terms of the GNU Affero General
Public License as published by the Free Software Foundation, either version
3 of the License, or (at your option) any later version. See `LICENSE`.
