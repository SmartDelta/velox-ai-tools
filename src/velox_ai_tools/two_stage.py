"""``velox-ai two-stage``: two-stage detection + classification pipeline.

Stage 1: RF-DETR or YOLO detects objects by shape (e.g. "round sign", "triangle sign")
Stage 2: EfficientNet classifier identifies the specific type (e.g. "RVV C1 30km/h")

This decouples detection from classification:
- Stage 1 is easy (few classes, high recall)
- Stage 2 trains on small crops (128x128), needs less data per class
- Adding new types = retrain classifier only, not detector

Sub-commands: ``classify`` (run the pipeline), ``train`` (the classifier),
``build-dataset`` (crops from a detection output). Each ends with a
``RESULT: {json}`` line.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .common import emit_result, resolve_weights


def crop_detection(img: Image.Image, bbox_norm: Tuple[float, float, float, float],
                   pad_factor: float = 0.15, crop_size: int = 128) -> Image.Image:
    """Crop a detection from an image with padding, resize to crop_size."""
    w, h = img.size
    cx, cy, bw, bh = bbox_norm
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    pw = (x2 - x1) * pad_factor
    ph = (y2 - y1) * pad_factor
    x1 = max(0, x1 - pw)
    y1 = max(0, y1 - ph)
    x2 = min(w, x2 + pw)
    y2 = min(h, y2 + ph)
    crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
    crop = crop.resize((crop_size, crop_size), Image.LANCZOS)
    return crop


class SignClassifier:
    """Stage 2: classify cropped sign patches into specific RVV types."""

    def __init__(self, model_path: str, class_names: Optional[List[str]] = None):
        self.model_path = model_path
        self.class_names = class_names or []
        self.model = None
        self._load_model()

    def _load_model(self):
        p = Path(self.model_path)
        if not p.is_file():
            print(f"WARNING: Classifier model not found: {self.model_path}")
            return

        suffix = p.suffix.lower()
        if suffix == '.onnx':
            import onnxruntime as ort
            self.model = ort.InferenceSession(str(p))
            self._backend = 'onnx'
            if not self.class_names:
                meta = self.model.get_modelmeta()
                if meta and meta.custom_metadata_map and 'class_names' in meta.custom_metadata_map:
                    self.class_names = json.loads(meta.custom_metadata_map['class_names'])
        elif suffix in ('.pt', '.pth'):
            import torch
            self.model = torch.jit.load(str(p), map_location='cpu')
            self.model.eval()
            self._backend = 'torchscript'
        else:
            print(f"WARNING: Unknown classifier format: {suffix}")

    def predict(self, crop: Image.Image) -> Tuple[str, float]:
        """Classify a single crop. Returns (class_name, confidence)."""
        if self.model is None:
            return ("unknown", 0.0)

        arr = np.array(crop.convert('RGB')).astype(np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        arr = arr.transpose(2, 0, 1)  # HWC → CHW
        batch = arr[np.newaxis, ...]    # add batch dim

        if self._backend == 'onnx':
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: batch.astype(np.float32)})
            logits = outputs[0][0]
        elif self._backend == 'torchscript':
            import torch
            with torch.no_grad():
                logits = self.model(torch.from_numpy(batch)).numpy()[0]
        else:
            return ("unknown", 0.0)

        probs = _softmax(logits)
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        name = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
        return (name, conf)

    def predict_batch(self, crops: List[Image.Image]) -> List[Tuple[str, float]]:
        """Classify a batch of crops."""
        return [self.predict(c) for c in crops]


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def train_classifier(dataset_dir: str, output_dir: str, epochs: int = 30,
                     imgsz: int = 128, batch_size: int = 32,
                     model_arch: str = "efficientnet_b0",
                     device: str = "auto") -> Dict:
    """Train a classifier on cropped sign patches.

    Expected dataset structure:
        dataset_dir/
            train/
                class_a/  img1.jpg, img2.jpg, ...
                class_b/  ...
            val/
                class_a/  ...
                class_b/  ...
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms, models

    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    train_dir = Path(dataset_dir) / "train"
    val_dir = Path(dataset_dir) / "val"
    if not train_dir.is_dir():
        return {"ok": False, "error": f"Train dir not found: {train_dir}"}

    transform_train = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(str(train_dir), transform=transform_train)
    val_ds = datasets.ImageFolder(str(val_dir), transform=transform_val) if val_dir.is_dir() else None
    class_names = train_ds.classes
    n_classes = len(class_names)

    print(f"Classifier training: {n_classes} classes, {len(train_ds)} train images")
    print(f"  Classes: {', '.join(class_names[:20])}{'...' if n_classes > 20 else ''}")
    print(f"  Device: {device}, Arch: {model_arch}, Epochs: {epochs}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0) if val_ds else None

    # Build model
    if model_arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    elif model_arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    else:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc = 0.0
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        scheduler.step()
        train_acc = correct / total if total else 0
        print(f"  Epoch {epoch+1}/{epochs}: loss={running_loss/len(train_loader):.4f} acc={train_acc:.3f}")
        sys.stdout.flush()

        # Validation
        if val_loader:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            val_acc = val_correct / val_total if val_total else 0
            print(f"    val_acc={val_acc:.3f}")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), str(out_path / "best_classifier.pt"))

    # Save final model as TorchScript for portable inference
    model.eval()
    model_cpu = model.to('cpu')
    scripted = torch.jit.trace(model_cpu, torch.randn(1, 3, imgsz, imgsz))
    scripted.save(str(out_path / "classifier.pt"))

    # Save class names
    meta = {"class_names": class_names, "n_classes": n_classes,
            "imgsz": imgsz, "arch": model_arch, "best_val_acc": best_acc}
    (out_path / "classifier_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Export to ONNX
    try:
        import torch.onnx
        dummy = torch.randn(1, 3, imgsz, imgsz)
        onnx_path = str(out_path / "classifier.onnx")
        torch.onnx.export(model_cpu, dummy, onnx_path,
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})
        print(f"  ONNX exported: {onnx_path}")
    except Exception as e:
        print(f"  ONNX export failed: {e}")

    print(f"\nClassifier training complete: {n_classes} classes, best_val_acc={best_acc:.3f}")
    print(f"  Model: {out_path / 'classifier.pt'}")
    return {"ok": True, "class_names": class_names, "best_val_acc": best_acc,
            "model_path": str(out_path / "classifier.pt")}


def run_two_stage(detector_weights: str, classifier_path: str,
                  input_dir: str, output_dir: str,
                  det_conf: float = 0.3, cls_conf: float = 0.5,
                  imgsz: int = 640, crop_size: int = 128,
                  device: str = "auto") -> Dict:
    """Run two-stage pipeline: detect → crop → classify → output."""

    # Load classifier
    meta_path = Path(classifier_path).parent / "classifier_meta.json"
    class_names = []
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        class_names = meta.get("class_names", [])

    classifier = SignClassifier(classifier_path, class_names)
    if classifier.model is None:
        return {"ok": False, "error": f"Cannot load classifier: {classifier_path}"}

    # Load detector
    is_rfdetr = 'rfdetr' in detector_weights.lower()
    if is_rfdetr:
        from rfdetr import RFDETRBase, RFDETRLarge
        if 'large' in detector_weights.lower():
            detector = RFDETRLarge()
        else:
            detector = RFDETRBase()
        det_type = 'rfdetr'
    else:
        from ultralytics import YOLO
        detector = YOLO(resolve_weights(detector_weights))
        det_type = 'yolo'

    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    labels_dir = out_path / "labels"
    labels_dir.mkdir(exist_ok=True)
    crops_dir = out_path / "crops"
    crops_dir.mkdir(exist_ok=True)

    images = sorted([p for p in in_path.rglob("*")
                     if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])
    total = len(images)
    print(f"Two-stage pipeline: {total} images")
    print(f"  Detector: {det_type} ({detector_weights})")
    print(f"  Classifier: {classifier_path} ({len(class_names)} classes)")

    all_results = []
    n_detections = 0
    n_classified = 0

    for i, img_path in enumerate(images):
        print(f"PROGRESS: {i+1}/{total} — {img_path.name}")
        sys.stdout.flush()

        img = Image.open(str(img_path)).convert('RGB')
        w, h = img.size

        # Stage 1: detect
        if det_type == 'rfdetr':
            dets = detector.predict(img, threshold=det_conf)
            boxes = []
            for j in range(len(dets.xyxy)):
                x1, y1, x2, y2 = dets.xyxy[j]
                det_cls = int(dets.class_id[j]) if dets.class_id is not None else 0
                det_conf_val = float(dets.confidence[j])
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                boxes.append((cx, cy, bw, bh, det_cls, det_conf_val))
        else:
            results = detector.predict(str(img_path), conf=det_conf, imgsz=imgsz,
                                       verbose=False, device=device if device != 'auto' else None)
            boxes = []
            for r in results:
                for b in r.boxes:
                    cx, cy, bw, bh = b.xywhn[0].tolist()
                    boxes.append((cx, cy, bw, bh, int(b.cls), float(b.conf)))

        n_detections += len(boxes)

        # Stage 2: crop + classify each detection
        label_lines = []
        for j, (cx, cy, bw, bh, det_cls, det_conf_val) in enumerate(boxes):
            crop = crop_detection(img, (cx, cy, bw, bh), crop_size=crop_size)
            cls_name, cls_conf_val = classifier.predict(crop)

            if cls_conf_val < cls_conf:
                cls_name = f"det_class_{det_cls}"
                cls_conf_val = det_conf_val

            # Map class name to index
            if cls_name in class_names:
                cls_id = class_names.index(cls_name)
            else:
                cls_id = det_cls

            label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {cls_conf_val:.4f}")
            n_classified += 1

            # Save crop for review
            crop.save(str(crops_dir / f"{img_path.stem}_det{j}_{cls_name}_{cls_conf_val:.2f}.jpg"))

            all_results.append({
                "image": img_path.name,
                "det_class": det_cls, "det_conf": det_conf_val,
                "cls_name": cls_name, "cls_conf": cls_conf_val,
                "bbox": [cx, cy, bw, bh],
            })

        if label_lines:
            (labels_dir / (img_path.stem + ".txt")).write_text(
                "\n".join(label_lines) + "\n", encoding="utf-8")

        # Save annotated image
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        colors = {'round': '#ff3333', 'triangle': '#ffcc00', 'square': '#3399ff',
                  'rectangle': '#33aa33', 'diamond': '#ff8800'}
        for j, (cx, cy, bw, bh, det_cls, det_conf_val) in enumerate(boxes):
            x1 = (cx - bw/2) * w
            y1 = (cy - bh/2) * h
            x2 = (cx + bw/2) * w
            y2 = (cy + bh/2) * h
            result = all_results[-(len(boxes) - j)]
            col = colors.get(result['cls_name'].split('_')[0], '#ffffff')
            draw.rectangle([x1, y1, x2, y2], outline=col, width=3)
            draw.text((x1+2, y1-12), f"{result['cls_name']} {result['cls_conf']:.2f}", fill=col)
        img.save(str(out_path / img_path.name))

    # Summary
    summary = {
        "total_images": total,
        "total_detections": n_detections,
        "total_classified": n_classified,
        "class_distribution": {},
    }
    for r in all_results:
        cn = r['cls_name']
        summary["class_distribution"][cn] = summary["class_distribution"].get(cn, 0) + 1

    (out_path / "two_stage_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nTwo-stage complete: {n_detections} detections, {n_classified} classified")
    for cn, cnt in sorted(summary["class_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {cn}: {cnt}")
    return {"ok": True, **summary}


def build_classifier_dataset(detection_output_dir: str, dest_dir: str,
                              crop_size: int = 128, val_ratio: float = 0.15) -> Dict:
    """Build a classifier training dataset from detection results.

    Takes YOLO detection output (images + labels), crops each detection,
    and organizes into class_name/image.jpg folder structure.
    """
    import random
    src = Path(detection_output_dir)
    dst = Path(dest_dir)

    labels_dir = src / "labels"
    if not labels_dir.is_dir():
        return {"ok": False, "error": "No labels directory found"}

    class_counts = {}
    crops = []

    for txt in sorted(labels_dir.glob("*.txt")):
        stem = txt.stem
        img_path = None
        for ext in ('.jpg', '.jpeg', '.png'):
            cand = src / (stem + ext)
            if cand.is_file():
                img_path = cand
                break
            for parent in [src / "images" / "train", src / "images" / "val", src]:
                cand = parent / (stem + ext)
                if cand.is_file():
                    img_path = cand
                    break
            if img_path:
                break
        if not img_path:
            continue

        img = Image.open(str(img_path)).convert('RGB')
        for line in txt.read_text(encoding="utf-8").strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = parts[0]
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            crop = crop_detection(img, (cx, cy, bw, bh), crop_size=crop_size)
            crops.append((cls_id, crop, stem))
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

    if not crops:
        return {"ok": False, "error": "No crops extracted"}

    # Split and save
    random.seed(42)
    random.shuffle(crops)
    n_val = max(1, int(len(crops) * val_ratio))

    for split, items in [("val", crops[:n_val]), ("train", crops[n_val:])]:
        for cls_id, crop, stem in items:
            cls_dir = dst / split / f"class_{cls_id}"
            cls_dir.mkdir(parents=True, exist_ok=True)
            idx = len(list(cls_dir.glob("*.jpg")))
            crop.save(str(cls_dir / f"{stem}_{idx}.jpg"))

    print(f"Classifier dataset: {len(crops)} crops, {len(class_counts)} classes")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  class_{cls}: {cnt}")

    return {"ok": True, "n_crops": len(crops), "n_classes": len(class_counts),
            "class_counts": class_counts}


def add_arguments(parser):
    sub = parser.add_subparsers(dest="two_stage_command", required=True)

    # Classify command
    p_cls = sub.add_parser("classify", help="Run two-stage pipeline")
    p_cls.add_argument("--detector", required=True, help="Detector weights (.pt or rfdetr-base)")
    p_cls.add_argument("--classifier", required=True, help="Classifier model (.pt or .onnx)")
    p_cls.add_argument("--input-dir", required=True)
    p_cls.add_argument("--output-dir", required=True)
    p_cls.add_argument("--det-conf", type=float, default=0.3)
    p_cls.add_argument("--cls-conf", type=float, default=0.5)
    p_cls.add_argument("--imgsz", type=int, default=640)
    p_cls.add_argument("--crop-size", type=int, default=128)
    p_cls.add_argument("--device", default="auto")

    # Train classifier command
    p_train = sub.add_parser("train", help="Train sign classifier")
    p_train.add_argument("--dataset-dir", required=True)
    p_train.add_argument("--output-dir", required=True)
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--imgsz", type=int, default=128)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--arch", default="efficientnet_b0", choices=["efficientnet_b0", "resnet18"])
    p_train.add_argument("--device", default="auto")

    # Build dataset command
    p_build = sub.add_parser("build-dataset", help="Build classifier dataset from detections")
    p_build.add_argument("--detection-dir", required=True)
    p_build.add_argument("--output-dir", required=True)
    p_build.add_argument("--crop-size", type=int, default=128)
    p_build.add_argument("--val-ratio", type=float, default=0.15)

    return parser


def run(args) -> int:
    cmd = args.two_stage_command
    if cmd == "classify":
        result = run_two_stage(args.detector, args.classifier, args.input_dir, args.output_dir,
                               args.det_conf, args.cls_conf, args.imgsz, args.crop_size, args.device)
    elif cmd == "train":
        result = train_classifier(args.dataset_dir, args.output_dir, args.epochs,
                                  args.imgsz, args.batch_size, args.arch, args.device)
    elif cmd == "build-dataset":
        result = build_classifier_dataset(args.detection_dir, args.output_dir,
                                          args.crop_size, args.val_ratio)
    else:
        raise SystemExit(f"unknown two-stage command {cmd!r}")
    result = dict(result or {})
    result.setdefault("ok", False)
    emit_result(result)
    return 0 if result.get("ok") else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Two-stage detection + classification pipeline")
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
