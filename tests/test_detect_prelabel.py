"""`detect` and `prelabel` end to end, with the fake model."""
from __future__ import annotations

import json
from pathlib import Path

from velox_ai_tools import cli

from conftest import last_result, make_image


def _weights(tmp_path: Path) -> str:
    w = tmp_path / "w.pt"
    w.write_bytes(b"w")
    return str(w)


def test_detect_writes_full_frame_pixel_boxes_per_frame(fake_ultralytics, tmp_path, capsys):
    frames = [{"id": f"cam1:{i}", "img_path": str(make_image(tmp_path / f"f{i}.jpg"))} for i in range(2)]
    fj = tmp_path / "frames.json"
    fj.write_text(json.dumps(frames), encoding="utf-8")
    out = tmp_path / "out" / "dets.json"
    fake_ultralytics.scenario = staticmethod(lambda item, i, kw: [(3, 0.9, 10.0, 10.0, 60.0, 50.0)])
    assert cli.main(["detect", "--weights", _weights(tmp_path), "--frames", str(fj), "--out", str(out),
                     "--tile", "0", "--device", "cpu"]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["names"] == {"0": "D00", "1": "D10", "2": "D20", "3": "D40"}
    f0 = data["frames"][0]
    assert f0["id"] == "cam1:0" and f0["width"] == 200 and f0["height"] == 150
    assert f0["detections"][0]["cls"] == "D40" and f0["detections"][0]["box"] == [10.0, 10.0, 60.0, 50.0]
    r = last_result(capsys)
    assert r["ok"] and r["frames"] == 2 and r["detections"] == 2
    assert ("predict", 1, 1280) in fake_ultralytics.calls


def test_detect_tiles_a_big_frame(fake_ultralytics, tmp_path, capsys):
    img = make_image(tmp_path / "big.jpg", 3000, 2000)
    fj = tmp_path / "frames.json"
    fj.write_text(json.dumps([{"id": "a", "img_path": str(img)}]), encoding="utf-8")
    out = tmp_path / "dets.json"
    assert cli.main(["detect", "--weights", _weights(tmp_path), "--frames", str(fj), "--out", str(out),
                     "--tile", "1280", "--overlap", "0.2", "--batch", "4", "--device", "cpu"]) == 0
    dets = json.loads(out.read_text(encoding="utf-8"))["frames"][0]["detections"]
    assert len(dets) == 6, "3 x 2 tiles, one box each, none overlapping after the offset"
    assert max(d["box"][0] for d in dets) > 1280


def test_detect_resumes_from_the_partial_file_and_a_fresh_run_discards_it(fake_ultralytics, tmp_path, capsys):
    """A killed run leaves <out>.partial.jsonl; --resume keeps those frames
    and only runs the rest; the final JSON comes out in the input order."""
    frames = [{"id": f"cam1:{i}", "img_path": str(make_image(tmp_path / f"f{i}.jpg"))} for i in range(3)]
    fj = tmp_path / "frames.json"
    fj.write_text(json.dumps(frames), encoding="utf-8")
    out = tmp_path / "dets.json"
    partial = tmp_path / "dets.json.partial.jsonl"
    partial.write_text(json.dumps({"id": "cam1:0", "width": 1, "height": 1, "detections": []}) + "\n"
                       + '{"torn line', encoding="utf-8")
    assert cli.main(["detect", "--weights", _weights(tmp_path), "--frames", str(fj), "--out", str(out),
                     "--tile", "0", "--device", "cpu", "--resume"]) == 0
    r = last_result(capsys)
    assert r["resumed"] == 1 and r["frames"] == 3
    assert sum(1 for c in fake_ultralytics.calls if c[0] == "predict") == 2, "only the two missing frames ran"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert [f["id"] for f in data["frames"]] == ["cam1:0", "cam1:1", "cam1:2"]
    assert data["frames"][0]["width"] == 1, "the kept record was not recomputed"
    assert data["frames"][1]["width"] == 200 and data["resumed"] == 1
    assert not partial.exists(), "the partial file goes once the final JSON is written"
    # a fresh run (no --resume) discards a stale partial file and does every frame
    partial.write_text(json.dumps({"id": "cam1:0", "width": 1, "height": 1, "detections": []}) + "\n", encoding="utf-8")
    fake_ultralytics.calls.clear()
    assert cli.main(["detect", "--weights", _weights(tmp_path), "--frames", str(fj), "--out", str(out),
                     "--tile", "0", "--device", "cpu"]) == 0
    assert sum(1 for c in fake_ultralytics.calls if c[0] == "predict") == 3
    assert json.loads(out.read_text(encoding="utf-8"))["frames"][0]["width"] == 200 and not partial.exists()


def _photos(tmp_path: Path):
    img_dir = tmp_path / "ds" / "images" / "train"
    lbl_dir = tmp_path / "ds" / "labels" / "train"
    for i in range(3):
        make_image(img_dir / f"frame_{i:03d}.jpg", 64, 48)
    lbl_dir.mkdir(parents=True)
    (lbl_dir / "frame_002.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")   # already reviewed
    cf = tmp_path / "classes.txt"
    cf.write_text("car\n", encoding="utf-8")
    return img_dir, lbl_dir, cf


def test_prelabel_labels_only_unlabelled_photos_and_appends_new_classes(fake_ultralytics, tmp_path, capsys):
    img_dir, lbl_dir, cf = _photos(tmp_path)
    out = tmp_path / "pl.json"
    args = ["prelabel", "--weights", _weights(tmp_path), "--images", str(img_dir), "--classes-file", str(cf),
            "--conf", "0.3", "--imgsz", "1280", "--out", str(out), "--device", "cpu"]
    assert cli.main(args) == 0
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["ok"] and r["n_images"] == 2 and r["skipped"] == 1 and r["n_boxes"] == 2
    assert r["classes"] == ["car", "D40"], "the model's class is appended to the dataset's list"
    assert (lbl_dir / "frame_000.txt").read_text(encoding="utf-8").split()[0] == "1"
    assert (lbl_dir / "frame_002.txt").read_text(encoding="utf-8").startswith("0 "), "reviewed file untouched"
    assert (lbl_dir / "classes.txt").read_text(encoding="utf-8").splitlines() == ["car", "D40"]
    assert any(c[0] == "predict" and c[2] == 1280 for c in fake_ultralytics.calls)
    summary = last_result(capsys)
    assert summary["n_images"] == 2 and "images" not in summary and summary["out"] == str(out)
    # second run: nothing left
    assert cli.main(args) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["n_images"] == 0


def test_prelabel_only_with_no_write_returns_boxes_and_touches_nothing(fake_ultralytics, tmp_path, capsys):
    img_dir, lbl_dir, cf = _photos(tmp_path)
    assert cli.main(["prelabel", "--weights", _weights(tmp_path), "--images", str(img_dir), "--labels", str(lbl_dir),
                     "--classes-file", str(cf), "--only", "frame_001.jpg", "--no-write", "--device", "cpu"]) == 0
    r = last_result(capsys)
    assert r["n_images"] == 1 and r["images"][0]["name"] == "frame_001.jpg" and r["images"][0]["written"] is False
    box = r["images"][0]["boxes"][0]
    assert box["cls"] == "D40" and box["cls_idx"] == 1 and 0 < box["x"] < 1 and 0 < box["w"] <= 1
    assert not (lbl_dir / "frame_001.txt").exists()


def test_prelabel_takes_a_list_file_and_overwrite(fake_ultralytics, tmp_path, capsys):
    img_dir, lbl_dir, cf = _photos(tmp_path)
    lst = tmp_path / "names.txt"
    lst.write_text("frame_002.jpg\nnot_there.jpg\n", encoding="utf-8")
    assert cli.main(["prelabel", "--weights", _weights(tmp_path), "--images", str(img_dir), "--classes-file", str(cf),
                     "--list", str(lst), "--overwrite", "--device", "cpu"]) == 0
    r = last_result(capsys)
    assert r["n_images"] == 1 and r["skipped"] == 0
    assert (lbl_dir / "frame_002.txt").read_text(encoding="utf-8").split()[0] == "1", "overwritten with the proposal"
