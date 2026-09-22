"""``velox-ai infer``: object detection on a directory of images, the way
SmartCore AI's inference tab runs it.

Supports YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLO11 via the ultralytics or
yolov5 packages, and RF-DETR. Writes annotated images, YOLO ``labels/*.txt``,
``detection_results.csv``, ``detection_summary.csv``, ``labels_with_header.csv``
and ``detections.geojson`` (camera position + bearing per detection when a
photo CSV is found) into ``--output-dir``.

    velox-ai infer --yolo-version yolov8 --weights best.pt --input-dir ./images \
        --output-dir ./output --conf 0.25 --iou 0.45 --imgsz 640 \
        --device auto --save-txt --save-conf
"""

import argparse
import sys
import os
from pathlib import Path

from .common import emit_result, is_asset_name, resolve_weights


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def find_images(input_dir: Path) -> list:
    """Find image files in input_dir (NOT recursive — only direct children)."""
    images = []
    for f in sorted(input_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(f)
    return images


def find_images_multi(dirs: list) -> list:
    """Find images across multiple directories."""
    images = []
    for d in dirs:
        p = Path(d)
        if p.is_dir():
            images.extend(find_images(p))
    return images


def filter_images_by_area(images: list, photo_meta: dict, bbox: dict) -> list:
    """Filter images to only those whose camera position falls within the bbox.

    bbox: {west, south, east, north} in lon/lat (WGS84 or projected coords)
    photo_meta: {stem: {latitude, longitude, ...}}
    """
    west, south = bbox['west'], bbox['south']
    east, north = bbox['east'], bbox['north']

    filtered = []
    skipped = 0
    for img_path in images:
        stem = img_path.stem
        meta = photo_meta.get(stem, {})
        lon = meta.get('longitude', 0)
        lat = meta.get('latitude', 0)
        if not lon and not lat:
            # No GPS — include by default (can't filter)
            filtered.append(img_path)
            continue
        if west <= lon <= east and south <= lat <= north:
            filtered.append(img_path)
        else:
            skipped += 1

    print(f"Area filter: {len(filtered)} images inside bbox, {skipped} outside")
    return filtered


def crop_equirect(img_path: Path, crop_top_pct: int, crop_bottom_pct: int, out_dir: Path) -> Path:
    """Crop top/bottom percentages from an equirectangular image.

    Returns path to the cropped temporary image.
    """
    from PIL import Image
    img = Image.open(img_path)
    w, h = img.size
    top_px = int(h * crop_top_pct / 100)
    bottom_px = int(h * crop_bottom_pct / 100)
    cropped = img.crop((0, top_px, w, h - bottom_px))
    crop_dir = out_dir / "_cropped"
    crop_dir.mkdir(exist_ok=True)
    out_path = crop_dir / img_path.name
    cropped.save(out_path, quality=95)
    return out_path


def check_dependencies(yolo_version: str) -> bool:
    """Check if required packages are installed. Returns True if OK."""
    ok = True
    if yolo_version == 'yolov5':
        try:
            import torch
            print(f"  torch {torch.__version__} — OK")
        except ImportError:
            print("MISSING: torch — install with: pip install torch")
            ok = False
        try:
            import yolov5
            print(f"  yolov5 — OK")
        except ImportError:
            print("  yolov5 package not found (will try torch.hub fallback)")
    else:
        try:
            from ultralytics import YOLO
            import ultralytics
            print(f"  ultralytics {ultralytics.__version__} — OK")
        except ImportError:
            print("MISSING: ultralytics — install with: pip install ultralytics")
            ok = False

    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
        else:
            print("  GPU: not available (will use CPU)")
    except ImportError:
        pass

    return ok


def run_ultralytics(args):
    """Run inference using the ultralytics package (YOLOv8/v9/v10/v11)."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics package not installed. Install with: pip install ultralytics")
        sys.exit(1)

    print(f"Loading model: {args.weights}")
    model = YOLO(resolve_weights(args.weights))

    # Extract class names from the model for report generation
    if hasattr(model, 'names') and model.names:
        names_dict = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}
        names_list = [names_dict.get(i, f'class_{i}') for i in range(max(names_dict.keys()) + 1)]
        args._model_names = names_list
        print(f"  Model classes: {len(names_list)} ({', '.join(names_list[:10])}{'...' if len(names_list) > 10 else ''})")

    device = args.device
    if device == 'auto':
        device = None  # ultralytics auto-detects

    images = getattr(args, '_filtered_images', None) or find_images(Path(args.input_dir))
    total = len(images)
    do_crop = (args.crop_top > 0 or args.crop_bottom > 0)
    if args.classes:
        print(f"  class filter: {args.classes}")
    if do_crop:
        print(f"  equirect crop: top={args.crop_top}%, bottom={args.crop_bottom}%")
    print(f"Running inference on: {args.input_dir} ({total} images)")
    print(f"  conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}, device={device or 'auto'}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resume = getattr(args, 'resume', False)
    batch_size = max(1, getattr(args, 'batch_size', 1))

    def _is_already_processed(p: Path) -> bool:
        if args.save_txt:
            lbl = output_dir / "labels" / f"{p.stem}.txt"
            if lbl.is_file():
                return True
        out_img = output_dir / p.name
        return out_img.is_file()

    if batch_size > 1 and not do_crop:
        i = 0
        while i < total:
            batch_slice = images[i:i + batch_size]
            to_process = []
            for img_p in batch_slice:
                if resume and _is_already_processed(img_p):
                    print(f"PROGRESS: {images.index(img_p) + 1}/{total} — {img_p.name} (skipped)")
                else:
                    to_process.append(img_p)

            if to_process:
                model.predict(
                    source=[str(p) for p in to_process],
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    device=device,
                    classes=args.classes,
                    save=True,
                    save_txt=args.save_txt,
                    save_conf=args.save_conf,
                    augment=getattr(args, 'augment', False),
                    project=str(output_dir),
                    name=".",
                    exist_ok=True,
                    verbose=False,
                )
                for img_p in to_process:
                    print(f"PROGRESS: {images.index(img_p) + 1}/{total} — {img_p.name}")
            i += len(batch_slice)
            sys.stdout.flush()
    else:
        for i, img_path in enumerate(images):
            if resume and _is_already_processed(img_path):
                print(f"PROGRESS: {i+1}/{total} — {img_path.name} (skipped)")
                sys.stdout.flush()
                continue

            print(f"PROGRESS: {i+1}/{total} — {img_path.name}")
            sys.stdout.flush()

            source_path = img_path
            if do_crop:
                source_path = crop_equirect(img_path, args.crop_top, args.crop_bottom, output_dir)

            model.predict(
                source=str(source_path),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=device,
                classes=args.classes,
                save=True,
                save_txt=args.save_txt,
                save_conf=args.save_conf,
                augment=getattr(args, 'augment', False),
                project=str(output_dir),
                name=".",
                exist_ok=True,
                verbose=False,
            )

            # Remap label coordinates back to original image if cropped
            if do_crop and args.save_txt:
                _remap_cropped_labels(output_dir, img_path.stem, args.crop_top, args.crop_bottom)

    # Clean up cropped temp dir
    if do_crop:
        import shutil
        crop_dir = output_dir / "_cropped"
        if crop_dir.is_dir():
            shutil.rmtree(crop_dir, ignore_errors=True)
            print("Cleaned up temporary cropped images")

    print(f"PROGRESS: {total}/{total} — complete")
    print(f"Inference complete: {total} images processed")
    print(f"Results saved to: {args.output_dir}")


def _remap_cropped_labels(output_dir: Path, stem: str, crop_top_pct: int, crop_bottom_pct: int):
    """Remap YOLO labels from cropped image coordinates back to original.

    In the cropped image, y=0 corresponds to crop_top_pct of the original.
    The visible band is (100 - crop_top - crop_bottom)% of the original height.
    """
    label_file = output_dir / "labels" / f"{stem}.txt"
    if not label_file.is_file():
        return

    top_frac = crop_top_pct / 100.0
    visible_frac = 1.0 - top_frac - (crop_bottom_pct / 100.0)
    if visible_frac <= 0:
        return

    lines = label_file.read_text(encoding="utf-8").strip().splitlines()
    remapped = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            remapped.append(line)
            continue
        cls_id = parts[0]
        cx = parts[1]
        cy_crop = float(parts[2])  # normalised in cropped image
        w = parts[3]
        h_crop = float(parts[4])   # normalised in cropped image

        # Remap to original image coordinates
        cy_orig = top_frac + cy_crop * visible_frac
        h_orig = h_crop * visible_frac

        new_parts = [cls_id, cx, f"{cy_orig:.6f}", w, f"{h_orig:.6f}"] + parts[5:]
        remapped.append(' '.join(new_parts))

    label_file.write_text('\n'.join(remapped), encoding="utf-8")


def run_rfdetr(args):
    """Run inference using RF-DETR (transformer-based, no NMS, Apache 2.0)."""
    try:
        from rfdetr import RFDETRBase, RFDETRLarge
    except ImportError:
        print("ERROR: rfdetr package not installed. Install with: pip install rfdetr")
        sys.exit(1)

    weights = args.weights
    print(f"Loading RF-DETR model: {weights}")
    if 'large' in weights.lower():
        model = RFDETRLarge()
    else:
        model = RFDETRBase()

    if Path(weights).is_file():
        import torch
        model.load_state_dict(torch.load(weights, map_location='cpu'), strict=False)
        print("  Loaded custom weights")

    images = getattr(args, '_filtered_images', None) or find_images(Path(args.input_dir))
    total = len(images)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    import supervision as sv

    resume = getattr(args, 'resume', False)
    print(f"Running RF-DETR inference on {total} images (conf={args.conf})")
    for i, img_path in enumerate(images):
        if resume and (labels_dir / f"{img_path.stem}.txt").is_file():
            print(f"PROGRESS: {i+1}/{total} — {img_path.name} (skipped)")
            sys.stdout.flush()
            continue

        print(f"PROGRESS: {i+1}/{total} — {img_path.name}")
        sys.stdout.flush()

        source_path = img_path
        do_crop = (args.crop_top > 0 or args.crop_bottom > 0)
        if do_crop:
            source_path = crop_equirect(img_path, args.crop_top, args.crop_bottom, output_dir)

        from PIL import Image
        img = Image.open(str(source_path))
        detections = model.predict(img, threshold=args.conf)

        # Save annotated image
        import numpy as np
        img_np = np.array(img)
        annotated = img_np.copy()
        if len(detections.xyxy) > 0:
            annotator = sv.BoxAnnotator()
            labels = [f"{detections.data.get('class_name', [''])[j] if 'class_name' in (detections.data or {}) else ''} {detections.confidence[j]:.2f}"
                      for j in range(len(detections.xyxy))]
            annotated = annotator.annotate(annotated, detections, labels=labels)
        Image.fromarray(annotated).save(str(output_dir / img_path.name))

        # Save YOLO-format labels
        w, h = img.size
        label_lines = []
        for j in range(len(detections.xyxy)):
            x1, y1, x2, y2 = detections.xyxy[j]
            cls_id = int(detections.class_id[j]) if detections.class_id is not None else 0
            conf = float(detections.confidence[j])
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            line = f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            if args.save_conf:
                line += f" {conf:.4f}"
            label_lines.append(line)
        if label_lines:
            (labels_dir / (img_path.stem + ".txt")).write_text(
                "\n".join(label_lines) + "\n", encoding="utf-8")

    # Cleanup cropped images
    if do_crop:
        import shutil
        crop_dir = output_dir / "_cropped"
        if crop_dir.is_dir():
            shutil.rmtree(crop_dir, ignore_errors=True)

    print(f"PROGRESS: {total}/{total} — complete")
    print(f"RF-DETR inference complete: {total} images processed")


def run_yolov5(args):
    """Run inference using the yolov5 package or torch.hub."""
    try:
        import torch
    except ImportError:
        print("ERROR: torch not installed. Install with: pip install torch")
        sys.exit(1)

    device = args.device
    if device == 'auto':
        device = '0' if torch.cuda.is_available() else 'cpu'

    print(f"Loading YOLOv5 model: {args.weights}")

    try:
        import yolov5
        model = yolov5.load(args.weights, device=device)
    except ImportError:
        print("yolov5 package not found, trying torch.hub...")
        model = torch.hub.load('ultralytics/yolov5', 'custom',
                               path=args.weights, device=device)

    model.conf = args.conf
    model.iou = args.iou
    if args.classes:
        model.classes = args.classes

    images = getattr(args, '_filtered_images', None) or find_images(Path(args.input_dir))
    if not images:
        print("ERROR: No images found in input directory")
        sys.exit(1)

    total = len(images)
    do_crop = (args.crop_top > 0 or args.crop_bottom > 0)
    if args.classes:
        print(f"  class filter: {args.classes}")
    if do_crop:
        print(f"  equirect crop: top={args.crop_top}%, bottom={args.crop_bottom}%")
    print(f"Running inference on {total} images...")
    print(f"  conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}, device={device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resume = getattr(args, 'resume', False)
    labels_dir = output_dir / 'labels'

    for i, img_path in enumerate(images):
        if resume and (labels_dir / f"{img_path.stem}.txt").is_file():
            print(f"PROGRESS: {i+1}/{total} — {img_path.name} (skipped)")
            sys.stdout.flush()
            continue

        print(f"PROGRESS: {i+1}/{total} — {img_path.name}")
        sys.stdout.flush()

        source_path = img_path
        if do_crop:
            source_path = crop_equirect(img_path, args.crop_top, args.crop_bottom, output_dir)

        results = model(str(source_path), size=args.imgsz)
        results.save(save_dir=str(output_dir), exist_ok=True)

        if args.save_txt:
            labels_dir = output_dir / 'labels'
            labels_dir.mkdir(exist_ok=True)
            txt_path = labels_dir / (img_path.stem + '.txt')
            preds = results.pandas().xyxyn[0]
            lines = []
            for _, row in preds.iterrows():
                cx = (row['xmin'] + row['xmax']) / 2
                cy = (row['ymin'] + row['ymax']) / 2
                w = row['xmax'] - row['xmin']
                h = row['ymax'] - row['ymin']
                cls_id = int(row['class'])
                line = f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                if args.save_conf:
                    line += f" {row['confidence']:.4f}"
                lines.append(line)
            txt_path.write_text('\n'.join(lines), encoding='utf-8')

        if do_crop and args.save_txt:
            _remap_cropped_labels(output_dir, img_path.stem, args.crop_top, args.crop_bottom)

    if do_crop:
        import shutil
        crop_dir = output_dir / "_cropped"
        if crop_dir.is_dir():
            shutil.rmtree(crop_dir, ignore_errors=True)

    print(f"PROGRESS: {total}/{total} — complete")
    print(f"Inference complete: {total} images processed")
    print(f"Results saved to: {output_dir}")


def add_arguments(parser):
    parser.add_argument('--yolo-version', default='yolov8',
                        choices=['yolov5', 'yolov8', 'yolov9', 'yolov10', 'yolov11', 'rfdetr'],
                        help='Model framework to use')
    parser.add_argument('--weights', required=True, help='Path to .pt weight file')
    parser.add_argument('--input-dir', required=True, help='Directory with input images')
    parser.add_argument('--output-dir', required=True, help='Directory for output results')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference image size')
    parser.add_argument('--device', default='auto', help='Device: auto, cpu, 0, 1, ...')
    parser.add_argument('--save-txt', action='store_true', help='Save labels as .txt')
    parser.add_argument('--save-conf', action='store_true', help='Include confidence in labels')
    parser.add_argument('--classes', nargs='+', type=int, default=None, help='Filter: only detect these class IDs')
    parser.add_argument('--crop-top', type=int, default=0, help='Crop top N%% of equirectangular image')
    parser.add_argument('--crop-bottom', type=int, default=0, help='Crop bottom N%% of equirectangular image')
    parser.add_argument('--photo-csv', default='', help='Photo CSV with GPS/heading per frame (auto-detected if in input dir)')
    parser.add_argument('--extra-dirs', nargs='+', default=[], help='Additional input directories (multi-dir mode)')
    parser.add_argument('--area-bbox', nargs=4, type=float, default=None, metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'),
                        help='Only process photos within this bounding box (lon/lat)')
    parser.add_argument('--augment', action='store_true', help='Test-time augmentation (TTA) for higher accuracy')
    parser.add_argument('--resume', action='store_true', help='Skip images that already have outputs in output directory')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for inference (default: 1)')
    parser.add_argument('--check-deps', action='store_true', help='Check dependencies and exit')
    return parser


def run(args):
    # Dependency check mode
    if args.check_deps:
        print(f"Checking dependencies for {args.yolo_version}...")
        ok = check_dependencies(args.yolo_version)
        sys.exit(0 if ok else 1)

    # Validate inputs — COCO pretrained models are auto-downloaded by ultralytics
    # into the tool's own weights cache (see common.resolve_weights)
    if is_asset_name(args.weights):
        print(f"Using COCO pretrained model: {args.weights} (will auto-download if needed)")
    elif not Path(args.weights).is_file():
        print(f"ERROR: Weight file not found: {args.weights}")
        sys.exit(1)

    if not Path(args.input_dir).is_dir():
        print(f"ERROR: Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Collect images from all directories
    all_dirs = [args.input_dir] + list(args.extra_dirs)
    if len(all_dirs) > 1:
        print(f"Multi-directory mode: {len(all_dirs)} directories")
        for d in all_dirs:
            print(f"  {d}")
        images = find_images_multi(all_dirs)
    else:
        images = find_images(Path(args.input_dir))
    print(f"Found {len(images)} images total")

    # Area filter: only process photos within bounding box
    if args.area_bbox:
        bbox = {'west': args.area_bbox[0], 'south': args.area_bbox[1],
                'east': args.area_bbox[2], 'north': args.area_bbox[3]}
        print(f"Area filter: W={bbox['west']:.5f} S={bbox['south']:.5f} "
              f"E={bbox['east']:.5f} N={bbox['north']:.5f}")
        # Load photo metadata to get GPS per frame
        photo_csv = args.photo_csv or _find_photo_csv(Path(args.input_dir))
        if photo_csv:
            photo_meta = _load_photo_csv(photo_csv)
            images = filter_images_by_area(images, photo_meta, bbox)
        else:
            # Try finding CSV near each directory
            combined_meta = {}
            for d in all_dirs:
                csv_path = _find_photo_csv(Path(d))
                if csv_path:
                    combined_meta.update(_load_photo_csv(csv_path))
            if combined_meta:
                images = filter_images_by_area(images, combined_meta, bbox)
            else:
                print("WARNING: No photo CSV found — cannot filter by area (processing all images)")

    if not images:
        print("ERROR: No images found (check input directory and area filter)")
        sys.exit(1)

    # Store filtered image list for the runner to use
    args._filtered_images = images

    # Create output dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Clear stale labels/crops from a previous run into the SAME folder. Without
    # this, the equirectangular-crop remap re-applies to already-remapped labels
    # every run (predict does not overwrite an existing .txt), compounding until
    # cy collapses to 0.5 and h shrinks to ~0 — detections become invisible
    # slivers ("YOLO finds nothing"). See JOURNAL 2026-06-03.
    import shutil as _sh
    for _sub in ("labels", "_cropped"):
        _stale = Path(args.output_dir) / _sub
        if _stale.is_dir():
            _sh.rmtree(_stale, ignore_errors=True)

    # Route to correct runner
    if args.yolo_version == 'rfdetr':
        run_rfdetr(args)
    elif args.yolo_version == 'yolov5':
        run_yolov5(args)
    else:
        run_ultralytics(args)

    # ── Generate report CSVs + detection GeoJSON ──
    generate_reports(args)

    print("Done.")
    out = Path(args.output_dir)
    emit_result({"ok": True, "output_dir": str(out), "images": len(images),
                 "geojson": str(out / "detections.geojson") if (out / "detections.geojson").is_file() else "",
                 "summary_csv": str(out / "detection_summary.csv") if (out / "detection_summary.csv").is_file() else ""})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="SmartCore AI inference runner")
    add_arguments(parser)
    return run(parser.parse_args(argv))


# ═══════════════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════════════

COCO_NAMES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush',
]


def _load_photo_csv(csv_path: str) -> dict:
    """Load photo CSV and return {imgname_stem: {lat, lon, yaw, heading, ...}}."""
    import csv as _csv
    p = Path(csv_path)
    if not p.is_file():
        return {}

    text = p.read_text(encoding='utf-8-sig')
    first_line = text.splitlines()[0]
    sep = ';' if ';' in first_line else (',' if ',' in first_line else '\t')

    reader = _csv.DictReader(text.splitlines(), delimiter=sep)
    lookup = {}
    for row in reader:
        # Find imgname column
        imgname = ''
        for key in ('imgname', 'image', 'filename', 'photo', 'file', 'Image'):
            if key in row:
                imgname = row[key].strip()
                break
        if not imgname:
            continue
        stem = Path(imgname).stem

        def _f(keys, default=0.0):
            for k in keys:
                if k in row:
                    try:
                        return float(row[k])
                    except (ValueError, TypeError):
                        pass
            return default

        lookup[stem] = {
            'imgname': imgname,
            'latitude': _f(['Latitude', 'latitude', 'lat']),
            'longitude': _f(['Longitude', 'longitude', 'lon', 'lng']),
            'altitude': _f(['altitude', 'Altitude', 'z', 'height']),
            'yaw': _f(['yaw', 'Yaw']),
            'heading': _f(['heading', 'Heading', 'azimuth']),
            'pitch': _f(['Pitch', 'pitch']),
            'roll': _f(['Roll', 'roll']),
            'stamp': row.get('Stamp', row.get('stamp', row.get('timestamp', ''))),
        }
    return lookup


def _find_photo_csv(input_dir: Path) -> str:
    """Auto-detect photo CSV in or near the input directory."""
    for candidate in [
        input_dir / 'output.csv',
        input_dir / 'photos.csv',
        input_dir.parent / 'output.csv',
        input_dir.parent / 'photos.csv',
    ]:
        if candidate.is_file():
            return str(candidate)
    # Search for any CSV with 'imgname' or 'Longitude' header
    for csv_file in sorted(input_dir.glob('*.csv')) + sorted(input_dir.parent.glob('*.csv')):
        try:
            header = csv_file.read_text(encoding='utf-8-sig').splitlines()[0].lower()
            if 'imgname' in header or 'longitude' in header or 'latitude' in header:
                return str(csv_file)
        except Exception:
            pass
    return ''


def _read_labels(output_dir: Path) -> dict:
    """Read all YOLO label files. Returns {stem: [(cls_id, cx, cy, w, h, conf), ...]}."""
    result = {}
    label_dir = output_dir / 'labels'
    if not label_dir.is_dir():
        return result
    for txt in sorted(label_dir.glob('*.txt')):
        dets = []
        for line in txt.read_text(encoding='utf-8').strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(parts[0])
            except ValueError:
                continue  # skip non-numeric lines (headers, metadata)
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            conf = float(parts[5]) if len(parts) > 5 else 1.0
            dets.append((cls_id, cx, cy, w, h, conf))
        result[txt.stem] = dets
    return result


def _class_name(cls_id: int, model_names=None) -> str:
    if model_names and cls_id < len(model_names):
        return model_names[cls_id]
    if cls_id < len(COCO_NAMES):
        return COCO_NAMES[cls_id]
    return f'class_{cls_id}'


def generate_reports(args):
    """Generate per-frame detail CSV, totals summary CSV, and detection GeoJSON."""
    import csv as _csv
    import json

    output_dir = Path(args.output_dir)
    labels = _read_labels(output_dir)
    if not labels:
        print("No label files found — skipping report generation")
        return

    # Load photo metadata
    photo_csv = args.photo_csv or _find_photo_csv(Path(args.input_dir))
    photo_meta = {}
    if photo_csv:
        print(f"Loading photo metadata from: {photo_csv}")
        photo_meta = _load_photo_csv(photo_csv)
        print(f"  Loaded metadata for {len(photo_meta)} frames")
    else:
        print("No photo CSV found — reports will not include GPS/heading data")

    # Collect all class IDs used — prefer model-embedded names over COCO
    model_names = getattr(args, '_model_names', None)
    all_class_ids = sorted(set(cls for dets in labels.values() for cls, *_ in dets))
    class_names = {cid: _class_name(cid, model_names) for cid in all_class_ids}

    # ── 1. Per-frame detail CSV ──
    detail_path = output_dir / 'detection_results.csv'
    geojson_features = []

    # Accumulators for summary
    total_detections = 0
    total_per_class = {cid: 0 for cid in all_class_ids}
    conf_per_class = {cid: [] for cid in all_class_ids}
    frames_with_detections = 0
    frames_total = 0

    with open(detail_path, 'w', newline='', encoding='utf-8') as f:
        writer = _csv.writer(f, delimiter=';')
        # Header
        header = ['frame', 'total_objects']
        for cid in all_class_ids:
            header.append(f'{class_names[cid]}_count')
            header.append(f'{class_names[cid]}_avg_conf')
        header += ['latitude', 'longitude', 'altitude', 'yaw', 'heading', 'pitch', 'roll', 'timestamp']
        writer.writerow(header)

        for stem in sorted(labels.keys()):
            dets = labels[stem]
            meta = photo_meta.get(stem, {})
            frames_total += 1

            # Count per class
            counts = {cid: 0 for cid in all_class_ids}
            confs = {cid: [] for cid in all_class_ids}
            for cls_id, cx, cy, w, h, conf in dets:
                if cls_id in counts:
                    counts[cls_id] += 1
                    confs[cls_id].append(conf)

            n_objects = len(dets)
            total_detections += n_objects
            if n_objects > 0:
                frames_with_detections += 1

            for cid in all_class_ids:
                total_per_class[cid] += counts[cid]
                conf_per_class[cid].extend(confs[cid])

            # Write row
            row = [stem, n_objects]
            for cid in all_class_ids:
                avg_c = round(sum(confs[cid]) / len(confs[cid]), 3) if confs[cid] else 0
                row.append(counts[cid])
                row.append(avg_c)
            row += [
                meta.get('latitude', ''),
                meta.get('longitude', ''),
                meta.get('altitude', ''),
                meta.get('yaw', ''),
                meta.get('heading', ''),
                meta.get('pitch', ''),
                meta.get('roll', ''),
                meta.get('stamp', ''),
            ]
            writer.writerow(row)

            # GeoJSON features per detection
            lat = meta.get('latitude', 0)
            lon = meta.get('longitude', 0)
            yaw_val = meta.get('yaw', 0)
            heading_val = meta.get('heading', 0)

            for cls_id, cx, cy, w, h, conf in dets:
                # Compute bearing of detection in panorama (cx=0..1 → -180..+180 from camera heading)
                det_local_yaw = (cx * 360.0) - 180.0
                det_bearing = (yaw_val + det_local_yaw) % 360.0

                feat = {
                    "type": "Feature",
                    "geometry": {"type": "Point",
                                 "coordinates": [lon, lat] if (lat and lon) else [0, 0]},
                    "properties": {
                        "frame": stem,
                        "class_id": cls_id,
                        "class_name": class_names.get(cls_id, f'class_{cls_id}'),
                        "confidence": round(conf, 3),
                        "bbox_cx": round(cx, 4),
                        "bbox_cy": round(cy, 4),
                        "bbox_w": round(w, 4),
                        "bbox_h": round(h, 4),
                        "bearing_deg": round(det_bearing, 1),
                        "camera_yaw": yaw_val,
                        "camera_heading": heading_val,
                        "latitude": lat,
                        "longitude": lon,
                        "timestamp": meta.get('stamp', ''),
                    }
                }
                geojson_features.append(feat)

    print(f"Detail CSV:  {detail_path} ({frames_total} frames)")

    # ── 2. Detection GeoJSON ──
    geojson_path = output_dir / 'detections.geojson'
    geojson = {
        "type": "FeatureCollection",
        "features": geojson_features,
    }
    geojson_path.write_text(json.dumps(geojson, indent=2), encoding='utf-8')
    print(f"GeoJSON:     {geojson_path} ({len(geojson_features)} detections)")

    # ── 3. Summary totals CSV ──
    summary_path = output_dir / 'detection_summary.csv'
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = _csv.writer(f, delimiter=';')

        writer.writerow(['SmartCore AI — Detection Summary Report'])
        writer.writerow([])

        writer.writerow(['Parameter', 'Value'])
        writer.writerow(['Model', args.weights])
        writer.writerow(['YOLO version', args.yolo_version])
        writer.writerow(['Input directory', args.input_dir])
        writer.writerow(['Confidence threshold', args.conf])
        writer.writerow(['IoU threshold', args.iou])
        writer.writerow(['Image size', args.imgsz])
        writer.writerow(['Equirect crop top', f'{args.crop_top}%'])
        writer.writerow(['Equirect crop bottom', f'{args.crop_bottom}%'])
        if args.classes:
            writer.writerow(['Class filter', ', '.join(str(c) for c in args.classes)])
        writer.writerow([])

        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total frames analyzed', frames_total])
        writer.writerow(['Frames with detections', frames_with_detections])
        writer.writerow(['Frames without detections', frames_total - frames_with_detections])
        writer.writerow(['Detection rate', f'{round(frames_with_detections / max(frames_total, 1) * 100, 1)}%'])
        writer.writerow(['Total detections', total_detections])
        writer.writerow(['Avg detections per frame', round(total_detections / max(frames_total, 1), 2)])
        writer.writerow([])

        writer.writerow(['Class', 'Total count', 'Avg confidence', '% of all detections', 'Frames containing'])
        for cid in all_class_ids:
            cnt = total_per_class[cid]
            avg_conf = round(sum(conf_per_class[cid]) / len(conf_per_class[cid]), 3) if conf_per_class[cid] else 0
            pct = round(cnt / max(total_detections, 1) * 100, 1)
            # Count frames containing this class
            n_frames = sum(1 for dets in labels.values() if any(c == cid for c, *_ in dets))
            writer.writerow([class_names[cid], cnt, avg_conf, f'{pct}%', n_frames])

        writer.writerow([])
        writer.writerow(['Top 5 frames by detection count'])
        writer.writerow(['Frame', 'Total objects', 'Latitude', 'Longitude', 'Heading'])
        top_frames = sorted(labels.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        for stem, dets in top_frames:
            meta = photo_meta.get(stem, {})
            writer.writerow([stem, len(dets),
                             meta.get('latitude', ''),
                             meta.get('longitude', ''),
                             meta.get('heading', '')])

    print(f"Summary CSV: {summary_path}")

    # ── 4. Labels with header CSV (human-readable version of YOLO labels) ──
    _generate_labels_csv(output_dir, labels, photo_meta, class_names)


def _generate_labels_csv(output_dir: Path, labels: dict, photo_meta: dict, class_names: dict):
    """Generate a labels_with_header.csv and a labels_README.txt."""
    import csv as _csv

    label_csv_path = output_dir / 'labels_with_header.csv'
    with open(label_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = _csv.writer(f, delimiter=';')
        writer.writerow([
            'frame',              # source image filename (without extension)
            'class_id',           # YOLO class index (integer, 0-based)
            'class_name',         # human-readable class name
            'bbox_cx',            # bounding box center X (normalised 0-1, left=0 right=1)
            'bbox_cy',            # bounding box center Y (normalised 0-1, top=0 bottom=1)
            'bbox_w',             # bounding box width (normalised 0-1)
            'bbox_h',             # bounding box height (normalised 0-1)
            'confidence',         # detection confidence score (0-1)
            'bearing_deg',        # compass bearing of detection from camera (degrees, 0=N 90=E)
            'latitude',           # camera latitude (from photo CSV)
            'longitude',          # camera longitude (from photo CSV)
            'altitude',           # camera altitude (from photo CSV)
            'yaw',                # camera yaw in degrees
            'heading',            # camera heading in degrees (0=N)
            'pitch',              # camera pitch in degrees
            'roll',               # camera roll in degrees
            'timestamp',          # capture timestamp
        ])

        for stem in sorted(labels.keys()):
            meta = photo_meta.get(stem, {})
            yaw_val = meta.get('yaw', 0)
            for cls_id, cx, cy, w, h, conf in labels[stem]:
                det_local_yaw = (cx * 360.0) - 180.0
                bearing = (yaw_val + det_local_yaw) % 360.0 if yaw_val else ''

                writer.writerow([
                    stem,
                    cls_id,
                    class_names.get(cls_id, f'class_{cls_id}'),
                    round(cx, 6),
                    round(cy, 6),
                    round(w, 6),
                    round(h, 6),
                    round(conf, 4),
                    round(bearing, 1) if bearing != '' else '',
                    meta.get('latitude', ''),
                    meta.get('longitude', ''),
                    meta.get('altitude', ''),
                    meta.get('yaw', ''),
                    meta.get('heading', ''),
                    meta.get('pitch', ''),
                    meta.get('roll', ''),
                    meta.get('stamp', ''),
                ])

    print(f"Labels CSV:  {label_csv_path}")

    # README explaining the YOLO label format
    readme_path = output_dir / 'labels' / 'labels_README.txt'
    if (output_dir / 'labels').is_dir():
        readme_path.write_text(
            "YOLO Label Format — SmartCore AI\n"
            "=================================\n\n"
            "Each .txt file contains one line per detected object.\n"
            "Line format:  class_id  cx  cy  w  h  [confidence]\n\n"
            "Column descriptions:\n"
            "  class_id    - Integer class index (0-based). See class list below.\n"
            "  cx          - Bounding box center X, normalised 0-1 (0=left, 1=right)\n"
            "  cy          - Bounding box center Y, normalised 0-1 (0=top, 1=bottom)\n"
            "  w           - Bounding box width, normalised 0-1\n"
            "  h           - Bounding box height, normalised 0-1\n"
            "  confidence  - Detection confidence score 0-1 (optional, present if save_conf=True)\n\n"
            "All coordinates are relative to the ORIGINAL image dimensions\n"
            "(equirectangular crop offsets have been remapped back).\n\n"
            "For equirectangular panoramas, cx can be converted to a bearing:\n"
            "  local_yaw = (cx * 360) - 180  (degrees relative to camera forward)\n"
            "  compass_bearing = (camera_yaw + local_yaw) % 360\n\n"
            "Class list (COCO):\n" +
            '\n'.join(f'  {i}: {n}' for i, n in enumerate(COCO_NAMES)) + '\n',
            encoding='utf-8',
        )
        print(f"Labels README: {readme_path}")


if __name__ == '__main__':
    main()
