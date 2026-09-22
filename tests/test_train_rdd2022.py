"""The road damage training command: VOC -> YOLO conversion, the fine-tune
merge of a dataset reviewed in Velox' 2D label tool with RDD2022, and the
export into a given weights folder. Offline, on files the tests write."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from velox_ai_tools import train_rdd2022 as tr

from conftest import last_result, make_image


def _reviewed_dataset(root: Path, name="road_damage_basler", classes=("D00", "D10", "D20", "D40"),
                      frames=6, tiles=2) -> Path:
    """What the label tool leaves behind: tiles per frame + YOLO txt + classes."""
    ds = root / "AI_training" / name
    pothole = classes.index("D40")                    # the id D40 has IN THIS dataset's order
    for f in range(1, frames + 1):
        for t in range(tiles):
            stem = f"cam1_frame_{f}_r0c{t}"
            make_image(ds / "images" / "train" / (stem + ".jpg"), 40, 30)
            body = f"{pothole} 0.5 0.5 0.2 0.2\n" if t == 0 else ""   # a pothole on the first tile only
            if f == 1 and t == 0:
                body += "9 0.1 0.1 0.05 0.05\n"                      # unknown class id -> dropped
            (ds / "labels" / "train").mkdir(parents=True, exist_ok=True)
            (ds / "labels" / "train" / (stem + ".txt")).write_text(body, encoding="utf-8")
    (ds / "labels" / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    (ds / "dataset.yaml").write_text("path: x\ntrain: images/train\nval: images/val\nnames:\n"
                                     + "".join(f"  {i}: {c}\n" for i, c in enumerate(classes)), encoding="utf-8")
    return ds


def _rdd_yolo(root: Path, n=10) -> Path:
    yolo = root / "yolo"
    for split, k in (("train", n), ("val", 2)):
        for i in range(k):
            make_image(yolo / "images" / split / f"Japan_{split}_{i}.jpg", 40, 30)
            (yolo / "labels" / split).mkdir(parents=True, exist_ok=True)
            (yolo / "labels" / split / f"Japan_{split}_{i}.txt").write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    (yolo / "dataset.yaml").write_text("path: x\ntrain: images/train\nval: images/val\nnames:\n  0: D00\n  1: D10\n  2: D20\n  3: D40\n",
                                       encoding="utf-8")
    return yolo / "dataset.yaml"


def test_class_names_come_from_the_yaml_or_the_classes_file(tmp_path):
    ds = _reviewed_dataset(tmp_path)
    assert tr.read_names(ds) == ["D00", "D10", "D20", "D40"]
    (ds / "dataset.yaml").unlink()
    assert tr.read_names(ds) == ["D00", "D10", "D20", "D40"]


def test_tiles_of_one_frame_stay_on_the_same_side_of_the_split():
    assert tr.frame_key("cam1_frame_12_r0c1") == "cam1_frame_12"
    assert tr.frame_key("Japan_000123") == "Japan_000123"


def test_a_reviewed_dataset_is_merged_with_remapped_classes_and_repeats(tmp_path):
    # the reviewed dataset lists its classes in another order than the target
    ds = _reviewed_dataset(tmp_path, classes=("D40", "D00", "D10", "D20"))
    out = tmp_path / "combined"
    counts = tr.add_extra_dataset(out, ds, ["D00", "D10", "D20", "D40"], val_fraction=0.34, seed=1,
                                  repeat=3, tag="basler")
    assert counts["dropped_labels"] == 1, "the unknown class id 9 is dropped"
    assert counts["val"] == 4 and counts["train"] == 24, counts     # 12 tiles: 4 val, 8 train x 3 repeats
    train_lbls = sorted((out / "labels" / "train").glob("*.txt"))
    val_lbls = sorted((out / "labels" / "val").glob("*.txt"))
    assert len(val_lbls) == 4 and len(train_lbls) == 24
    assert all(len(list((out / "images" / s).glob("*.jpg"))) == len(list((out / "labels" / s).glob("*.txt")))
               for s in ("train", "val"))
    val_frames = {tr.frame_key(p.stem.replace("basler_", "")) for p in val_lbls}
    train_frames = {tr.frame_key(p.stem.replace("basler_", "").rsplit("_k", 1)[0]) for p in train_lbls}
    assert not (val_frames & train_frames), "frames never straddle the split"
    first = next(p for p in train_lbls + val_lbls if p.read_text(encoding="utf-8").strip())
    assert first.read_text(encoding="utf-8").split()[0] == "3", "class 0 in the reviewed file meant D40 -> id 3"


def test_combined_dataset_holds_rdd_and_the_reviewed_tiles(tmp_path):
    rdd = _rdd_yolo(tmp_path)
    ds = _reviewed_dataset(tmp_path)
    yaml = tr.build_combined(rdd, [ds], tmp_path / "yolo_ft", ["D00", "D10", "D20", "D40"], repeat=2)
    assert yaml.is_file() and "3: D40" in yaml.read_text(encoding="utf-8")
    imgs = list((tmp_path / "yolo_ft" / "images" / "train").glob("*.jpg"))
    assert sum(1 for p in imgs if p.name.startswith("Japan_")) == 10
    assert sum(1 for p in imgs if p.name.startswith("road_damage_basler_")) > 10
    counts = json.loads((tmp_path / "yolo_ft" / "build_counts.json").read_text(encoding="utf-8"))
    assert "road_damage_basler" in counts


def test_voc_boxes_become_normalised_yolo_lines(tmp_path):
    xml = tmp_path / "a.xml"
    xml.write_text("""<annotation><size><width>200</width><height>100</height></size>
<object><name>D40</name><bndbox><xmin>50</xmin><ymin>20</ymin><xmax>150</xmax><ymax>60</ymax></bndbox></object>
<object><name>D43</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>10</xmax><ymax>10</ymax></bndbox></object>
</annotation>""", encoding="utf-8")
    lines = tr.voc_to_yolo(xml, ["D00", "D10", "D20", "D40"])
    assert lines == ["3 0.500000 0.400000 0.500000 0.400000"]


def test_no_rdd_builds_the_reviewed_dataset_alone_and_no_download_refuses(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tr, "download", lambda root: (_ for _ in ()).throw(AssertionError("download must not run")))
    ds = _reviewed_dataset(tmp_path)
    root = tmp_path / "rdd"
    assert tr.main(["--root", str(root), "--skip-train", "--no-rdd", "--extra-dataset", str(ds)]) == 0
    yaml = root / "yolo_ft" / "dataset.yaml"
    assert yaml.is_file()
    assert all(p.name.startswith("road_damage_basler_") for p in (root / "yolo_ft" / "images" / "train").iterdir())
    assert last_result(capsys)["trained"] is False
    with pytest.raises(SystemExit):
        tr.main(["--root", str(root), "--skip-train", "--no-download", "--extra-dataset", str(ds)])


def test_cli_refuses_an_unreviewed_extra_dataset_before_downloading(tmp_path, monkeypatch):
    """The check must come BEFORE the 13 GB download: an earlier version of
    this test started a real figshare download from a tmp root."""
    monkeypatch.setattr(tr, "download", lambda root: (_ for _ in ()).throw(AssertionError("download must not run")))
    with pytest.raises(SystemExit):
        tr.main(["--root", str(tmp_path / "nowhere"), "--skip-train", "--extra-dataset", str(tmp_path / "empty")])


def test_the_whole_fine_tune_lands_in_the_given_weights_folder(fake_ultralytics, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tr, "download", lambda root: (_ for _ in ()).throw(AssertionError("download must not run")))
    ds = _reviewed_dataset(tmp_path)
    root = tmp_path / "rdd"
    start = tmp_path / "start.pt"
    start.write_bytes(b"s")
    wf = tmp_path / "Weight_files"
    assert tr.main(["--root", str(root), "--no-rdd", "--extra-dataset", str(ds), "--model", str(start),
                    "--epochs", "1", "--imgsz", "64", "--batch", "2", "--no-amp", "--no-download",
                    "--out-name", "road_damage_finetuned_test", "--weights-dir", str(wf)]) == 0
    r = last_result(capsys)
    assert r["ok"] and r["trained"] and r["weights"] == str(wf / "road_damage_finetuned_test.pt")
    assert (wf / "road_damage_finetuned_test.pt").read_bytes() == b"best"
    side = json.loads((wf / "road_damage_finetuned_test.json").read_text(encoding="utf-8"))
    assert side["classes"] == ["D00", "D10", "D20", "D40"] and side["metrics"]["map50"] == 0.5
    assert "reviewed: road_damage_basler" in side["trained_on"] and "AGPL" in side["licence_note"]
    train_call = next(c for c in fake_ultralytics.calls if c[0] == "train")
    assert train_call[1]["amp"] is False and train_call[1]["epochs"] == 1
