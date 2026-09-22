"""``velox-ai train-rdd2022``: the road damage weight file.

Data: RDD2022 (Arya et al., CRDDC'2022), figshare article 21431547, one zip
of 13.3 GB with per-country train/images + train/annotations/xmls (PASCAL
VOC). Classes kept by default: D00 longitudinal crack, D10 transverse crack,
D20 alligator crack, D40 pothole (RDD2020 merged D01/D11 into D00/D10).

    velox-ai train-rdd2022 --root D:/AI_training_data/RDD2022 \
        --model yolo11s.pt --epochs 50 --imgsz 640 --batch 32 --weights-dir <folder>

Steps (each skipped when its output exists): download, unzip, convert to
YOLO txt, split by country (90/10 by image, seeded), dataset.yaml, train,
copy best.pt + a sidecar JSON into ``--weights-dir`` (Velox passes its
Weight_files folder; default: the tool's own weights folder).

Fine-tune on a dataset reviewed in Velox' 2D label tool (pre-labelled tiles
of the Basler frames, wrong boxes deleted), starting from the road damage
weights instead of the base model:

    velox-ai train-rdd2022 --root D:/AI_training_data/RDD2022 \
        --model <Weight_files>/road_damage_rdd2022_yolo11s.pt \
        --extra-dataset "<project>/AI_training/road_damage_basler" \
        --extra-repeat 5 --epochs 20 --imgsz 1280 --batch 8 --no-amp \
        --weights-dir <Weight_files>

Result: <weights-dir>/road_damage_finetuned_<model>.pt + .json. The weight
file is trained with ultralytics and published under the same AGPL-3.0 as
this tool.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .common import emit_result, resolve_weights, weights_cache_dir

FIGSHARE_ZIP = "https://ndownloader.figshare.com/files/38030910"
ZIP_NAME = "RDD2022_released_through_CRDDC2022.zip"
DEFAULT_CLASSES = ["D00", "D10", "D20", "D40"]
EXTRA_CLASSES = ["D43", "D44", "D50"]


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def download(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    dst = root / ZIP_NAME
    if dst.is_file() and dst.stat().st_size > 13_000_000_000:
        log(f"zip present: {dst} ({dst.stat().st_size / 1e9:.2f} GB)")
        return dst
    tmp = dst.with_suffix(".part")
    done = tmp.stat().st_size if tmp.is_file() else 0
    req = urllib.request.Request(FIGSHARE_ZIP, headers={"User-Agent": "VeloxSmartDelta/1.0"})
    if done:
        req.add_header("Range", f"bytes={done}-")
    log(f"downloading RDD2022 ({'resume at %.2f GB' % (done / 1e9) if done else 'from start'})")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "ab" if done else "wb") as fh:
        total = done + int(r.headers.get("Content-Length") or 0)
        got = done
        t0 = time.time()
        while True:
            chunk = r.read(8 << 20)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            if int(time.time() - t0) % 30 == 0:
                log(f"  {got / 1e9:.2f} / {total / 1e9:.2f} GB")
    tmp.replace(dst)
    log(f"downloaded {dst.stat().st_size / 1e9:.2f} GB")
    return dst


def _wanted(member: str) -> bool:
    m = member.replace("\\", "/")
    return ("/train/" in m or m.startswith("train/")) and ("images/" in m or "annotations/" in m) \
        and not m.endswith("/")


def _extract_country(inner: zipfile.ZipFile, country: str, dst_root: Path) -> int:
    """Train images + xml annotations of one country zip into
    dst_root/<country>/train/..., whatever prefix the zip uses inside."""
    n = 0
    for m in inner.namelist():
        if not _wanted(m):
            continue
        rel = m.replace("\\", "/").lstrip("/")
        if rel.startswith(country + "/"):
            rel = rel[len(country) + 1:]
        elif rel.startswith("RDD2022/" + country + "/"):
            rel = rel[len("RDD2022/" + country) + 1:]
        target = dst_root / country / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with inner.open(m) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst, 8 << 20)
        n += 1
        if n % 5000 == 0:
            log(f"  {country}: {n}")
    return n


def unzip(zip_path: Path, root: Path) -> Path:
    """The figshare zip holds one zip per country (RDD2022/<Country>.zip);
    older mirrors hold the folders directly. Both land in root/RDD2022/."""
    dst_root = root / "RDD2022"
    marker = root / ".unzipped"
    if marker.is_file() and dst_root.is_dir() and any(p.is_dir() for p in dst_root.iterdir()):
        return dst_root
    log("unzipping (train images + xml annotations only)")
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        nested = [m for m in names if m.lower().endswith(".zip")]
        if nested:
            for m in nested:
                country = Path(m).stem
                log(f"  {country}.zip ...")
                with z.open(m) as fh:
                    with zipfile.ZipFile(fh) as inner:
                        n = _extract_country(inner, country, dst_root)
                log(f"  {country}: {n} files")
        else:
            members = [m for m in names if _wanted(m)]
            for i, m in enumerate(members):
                z.extract(m, root)
                if i % 5000 == 0:
                    log(f"  {i}/{len(members)}")
    if not dst_root.is_dir() or not any(p.is_dir() for p in dst_root.iterdir()):
        raise SystemExit(f"nothing usable extracted from {zip_path}: expected RDD2022/<Country>/train/...")
    marker.write_text("ok", encoding="utf-8")
    return dst_root


def voc_to_yolo(xml_path: Path, classes: Sequence[str]) -> Optional[List[str]]:
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return None
    r = tree.getroot()
    size = r.find("size")
    if size is None:
        return None
    w = float(size.findtext("width") or 0)
    h = float(size.findtext("height") or 0)
    if w <= 0 or h <= 0:
        return None
    lines: List[str] = []
    for obj in r.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in classes:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        x1, y1 = float(bb.findtext("xmin") or 0), float(bb.findtext("ymin") or 0)
        x2, y2 = float(bb.findtext("xmax") or 0), float(bb.findtext("ymax") or 0)
        x1, x2 = max(0.0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(h, max(y1, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{classes.index(name)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def build_dataset(src: Path, out: Path, classes: Sequence[str], val_fraction: float = 0.1,
                  seed: int = 42, keep_empty: bool = True) -> Path:
    yaml = out / "dataset.yaml"
    if yaml.is_file():
        log(f"dataset present: {yaml}")
        return yaml
    rng = random.Random(seed)
    counts: Dict[str, Dict[str, int]] = {}
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for country in sorted(p for p in src.iterdir() if p.is_dir()):
        xml_dir = country / "train" / "annotations" / "xmls"
        img_dir = country / "train" / "images"
        if not xml_dir.is_dir() or not img_dir.is_dir():
            continue
        xmls = sorted(xml_dir.glob("*.xml"))
        rng.shuffle(xmls)
        n_val = int(len(xmls) * val_fraction)
        c = counts.setdefault(country.name, {"train": 0, "val": 0, "labels": 0, "skipped": 0})
        for i, xp in enumerate(xmls):
            lines = voc_to_yolo(xp, classes)
            if lines is None or (not lines and not keep_empty):
                c["skipped"] += 1
                continue
            img = next((img_dir / (xp.stem + ext) for ext in (".jpg", ".jpeg", ".png")
                        if (img_dir / (xp.stem + ext)).is_file()), None)
            if img is None:
                c["skipped"] += 1
                continue
            split = "val" if i < n_val else "train"
            stem = f"{country.name}_{xp.stem}"
            link = out / "images" / split / (stem + img.suffix.lower())
            if not link.exists():
                try:
                    link.hardlink_to(img)
                except OSError:
                    shutil.copy2(img, link)
            (out / "labels" / split / (stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""),
                                                                   encoding="utf-8")
            c[split] += 1
            c["labels"] += len(lines)
        log(f"  {country.name}: {c}")
    yaml.write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(classes)), encoding="utf-8")
    (out / "build_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    return yaml


# ── fine-tuning on reviewed Velox datasets ───────────────────────────────────
# The base model is right often enough to pre-label and wrong often enough on
# our own frames (tree shadow = pothole) that a person reviews the proposals
# in Velox' 2D label tool. That reviewed dataset (<project>/AI_training/
# road_damage_basler: images/train + labels/train, classes.txt) is merged
# here with RDD2022 so the model learns our frames without forgetting the
# public ones, and training starts from the existing road damage weights.

_TILE_SUFFIX = re.compile(r"_r\d+c\d+$")


def read_names(dataset_dir: Path) -> List[str]:
    """Class names of a Velox/YOLO dataset: dataset.yaml `names:` first,
    else labels/classes.txt or labels/train/classes.txt."""
    yaml = dataset_dir / "dataset.yaml"
    if yaml.is_file():
        names: Dict[int, str] = {}
        in_names = False
        for line in yaml.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("names:"):
                in_names = True
                continue
            if in_names:
                m = re.match(r"\s+(\d+):\s*(.+?)\s*$", line)
                if m:
                    names[int(m.group(1))] = m.group(2).strip("'\"")
                elif line.strip() and not line.startswith(" "):
                    break
        if names:
            return [names[i] for i in sorted(names)]
    for p in (dataset_dir / "labels" / "classes.txt", dataset_dir / "labels" / "train" / "classes.txt"):
        if p.is_file():
            return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return []


def frame_key(stem: str) -> str:
    """Tiles of one frame (cam1_frame_12_r0c1) must land in the same split,
    or the model sees the neighbouring tile of a validation frame."""
    return _TILE_SUFFIX.sub("", stem)


def add_extra_dataset(combined: Path, extra: Path, classes: Sequence[str], *, val_fraction: float = 0.1,
                      seed: int = 42, repeat: int = 1, tag: str = "") -> Dict[str, int]:
    """Copy (hardlink) a reviewed dataset into the combined one, remapping
    class ids by NAME onto `classes`; unknown names are dropped, frames stay
    together across the split, and `repeat` copies of every training tile
    weigh the small reviewed set against the large public one."""
    names = read_names(extra)
    if not names:
        raise SystemExit(f"{extra}: no class names (dataset.yaml or classes.txt)")
    remap = {i: classes.index(n) for i, n in enumerate(names) if n in classes}
    rng = random.Random(seed)
    tag = tag or _TILE_SUFFIX.sub("", extra.name)
    counts = {"train": 0, "val": 0, "labels": 0, "dropped_labels": 0, "skipped": 0}
    label_files = sorted((extra / "labels" / "train").glob("*.txt")) + sorted((extra / "labels" / "val").glob("*.txt"))
    label_files = [p for p in label_files if p.name != "classes.txt"]
    groups = sorted({frame_key(p.stem) for p in label_files})
    rng.shuffle(groups)
    val_groups = set(groups[:int(len(groups) * val_fraction)])
    for split in ("train", "val"):
        (combined / "images" / split).mkdir(parents=True, exist_ok=True)
        (combined / "labels" / split).mkdir(parents=True, exist_ok=True)
    for lp in label_files:
        img_dir = extra / "images" / lp.parent.name
        img = next((img_dir / (lp.stem + ext) for ext in (".jpg", ".jpeg", ".png")
                    if (img_dir / (lp.stem + ext)).is_file()), None)
        if img is None:
            counts["skipped"] += 1
            continue
        lines = []
        for ln in lp.read_text(encoding="utf-8").splitlines():
            parts = ln.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
            except ValueError:
                continue
            if cid not in remap:
                counts["dropped_labels"] += 1
                continue
            lines.append(" ".join([str(remap[cid])] + parts[1:5]))
        split = "val" if frame_key(lp.stem) in val_groups else "train"
        copies = repeat if split == "train" else 1
        for k in range(max(1, copies)):
            stem = f"{tag}_{lp.stem}" + (f"_k{k}" if k else "")
            link = combined / "images" / split / (stem + img.suffix.lower())
            if not link.exists():
                try:
                    link.hardlink_to(img)
                except OSError:
                    shutil.copy2(img, link)
            (combined / "labels" / split / (stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            counts[split] += 1
            counts["labels"] += len(lines)
    return counts


def build_combined(rdd_yaml: Optional[Path], extras: Sequence[Path], out: Path, classes: Sequence[str],
                   repeat: int = 5) -> Path:
    """RDD2022 (already YOLO; None = reviewed data only) + every reviewed
    dataset, hardlinked into `out`."""
    if out.is_dir():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    if rdd_yaml is not None:
        rdd = rdd_yaml.parent
        for split in ("train", "val"):
            n = 0
            for img in (rdd / "images" / split).iterdir():
                link = out / "images" / split / img.name
                try:
                    link.hardlink_to(img)
                except OSError:
                    shutil.copy2(img, link)
                lbl = rdd / "labels" / split / (img.stem + ".txt")
                if lbl.is_file():
                    shutil.copy2(lbl, out / "labels" / split / lbl.name)
                n += 1
            log(f"  combined: {n} RDD2022 {split} images")
    summary = {}
    for extra in extras:
        c = add_extra_dataset(out, extra, classes, repeat=repeat)
        summary[extra.name] = c
        log(f"  combined: {extra.name}: {c}")
    yaml = out / "dataset.yaml"
    yaml.write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(classes)), encoding="utf-8")
    (out / "build_counts.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return yaml


def train(yaml: Path, model_name: str, epochs: int, imgsz: int, batch: int, device: str,
          project: Path, run_name: str, amp: bool = True, workers: int = 8) -> Path:
    from ultralytics import YOLO

    model = YOLO(resolve_weights(model_name))
    # amp=False: torch 2.7.1+cu118 raised CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH in
    # the first backward pass with the AMP GradScaler on the RTX 4070 SUPER
    # (2026-09-21); fp32 is slower but trains.
    model.train(data=str(yaml), epochs=epochs, imgsz=imgsz, batch=batch, device=device,
                project=str(project), name=run_name, exist_ok=True, patience=15,
                amp=amp, workers=workers,
                # the brief's shadow/lighting robustness: photometric jitter + blur/flip
                hsv_h=0.02, hsv_s=0.6, hsv_v=0.5, fliplr=0.5, degrees=5.0, scale=0.5,
                mosaic=1.0, close_mosaic=10, plots=True, verbose=True)
    best = project / run_name / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"training produced no best.pt in {best.parent}")
    return best


def export(best: Path, model_name: str, classes: Sequence[str], yaml: Path, run_dir: Path,
           out_name: str = "", extras: Sequence[Path] = (), weights_dir: Optional[Path] = None) -> Path:
    """Copy best.pt to `weights_dir` under its product name, validate it, and
    write the .json sidecar Velox shows (classes, trained_on, metrics)."""
    from ultralytics import YOLO

    wd = Path(weights_dir) if weights_dir else weights_cache_dir()
    wd.mkdir(parents=True, exist_ok=True)
    tag = Path(model_name).stem
    dst = wd / ((out_name.strip() or f"road_damage_rdd2022_{tag}") + ".pt")
    shutil.copy2(best, dst)
    metrics = {}
    try:
        res = YOLO(str(dst)).val(data=str(yaml), plots=False, verbose=False)
        metrics = {"map50": float(res.box.map50), "map50_95": float(res.box.map),
                   "per_class_map50": {classes[i]: float(v) for i, v in enumerate(res.box.ap50)}}
    except Exception as exc:                          # noqa: BLE001 - metrics are a courtesy
        metrics = {"error": str(exc)}
    trained_on = "RDD2022 (figshare 21431547)"
    if extras:
        trained_on += " + reviewed: " + ", ".join(Path(e).name for e in extras)
    dst.with_suffix(".json").write_text(json.dumps({
        "classes": list(classes), "trained_on": trained_on, "base_model": model_name,
        "extra_datasets": [str(e) for e in extras],
        "metrics": metrics, "run_dir": str(run_dir), "created": time.strftime("%Y-%m-%d %H:%M"),
        "licence_note": "trained with Velox AI Tools (ultralytics, AGPL-3.0); this weight file is AGPL-3.0",
        "tool": "velox-ai train-rdd2022",
    }, indent=2), encoding="utf-8")
    log(f"weights: {dst} metrics: {metrics}")
    return dst


def add_arguments(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--root", default="D:/AI_training_data/RDD2022")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0")
    ap.add_argument("--extra-classes", action="store_true", help="also D43/D44/D50")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--no-amp", action="store_true", help="fp32 training (cuDNN stream-mismatch workaround)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--extra-dataset", action="append", default=[],
                    help="reviewed Velox dataset dir (images/train + labels/train + classes); repeatable")
    ap.add_argument("--extra-repeat", type=int, default=5,
                    help="how many times each reviewed training tile is repeated against RDD2022")
    ap.add_argument("--out-name", default="", help="weights file stem (default road_damage_rdd2022_<model>, "
                                                   "or road_damage_finetuned_<model> with --extra-dataset)")
    ap.add_argument("--no-rdd", action="store_true",
                    help="fine-tune on the reviewed dataset(s) only, without RDD2022 (quick, forgets more)")
    ap.add_argument("--no-download", action="store_true",
                    help="refuse instead of downloading 13 GB when RDD2022 is not prepared under --root")
    ap.add_argument("--weights-dir", default="",
                    help="where the weight file and its .json sidecar land (Velox: its Weight_files folder)")
    return ap


def run(a: argparse.Namespace) -> int:
    root = Path(a.root)
    classes = DEFAULT_CLASSES + (EXTRA_CLASSES if a.extra_classes else [])
    extras = [Path(e) for e in a.extra_dataset]
    for e in extras:                                 # before the 13 GB download, not after
        if not (e / "labels" / "train").is_dir():
            raise SystemExit(f"--extra-dataset {e}: no labels/train (review it in Velox' 2D label tool first)")
    if a.no_rdd:
        if not extras:
            raise SystemExit("--no-rdd needs at least one --extra-dataset")
        log(f"fine-tune on {len(extras)} reviewed dataset(s) only (no RDD2022)")
        yaml = build_combined(None, extras, root / "yolo_ft", classes, repeat=a.extra_repeat)
    else:
        prepared = (root / "yolo" / "dataset.yaml").is_file()
        if a.no_download and not prepared and not (root / ZIP_NAME).is_file():
            raise SystemExit(f"RDD2022 is not prepared under {root} (no yolo/dataset.yaml, no {ZIP_NAME}); "
                             f"run once without --no-download to fetch it (13 GB), or use --no-rdd")
        zip_path = download(root) if not prepared else root / ZIP_NAME
        src = unzip(zip_path, root) if not prepared else root / "RDD2022"
        yaml = build_dataset(src, root / "yolo", classes)
        if extras:
            log(f"fine-tune: merging {len(extras)} reviewed dataset(s) x{a.extra_repeat} with RDD2022")
            yaml = build_combined(yaml, extras, root / "yolo_ft", classes, repeat=a.extra_repeat)
    if a.skip_train:
        emit_result({"ok": True, "dataset": str(yaml), "trained": False})
        return 0
    run_name = f"{'finetune' if extras else 'rdd2022'}_{Path(a.model).stem}_{a.imgsz}"
    best = train(yaml, a.model, a.epochs, a.imgsz, a.batch, a.device, root / "runs", run_name,
                 amp=not a.no_amp, workers=a.workers)
    out_name = a.out_name or (f"road_damage_finetuned_{Path(a.model).stem}" if extras else "")
    dst = export(best, a.model, classes, yaml, root / "runs" / run_name, out_name=out_name, extras=extras,
                 weights_dir=Path(a.weights_dir) if a.weights_dir else None)
    emit_result({"ok": True, "weights": str(dst), "sidecar": str(dst.with_suffix(".json")),
                 "run_dir": str(root / "runs" / run_name), "trained": True})
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
