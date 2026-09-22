"""`two-stage build-dataset` needs only PIL: crops from a detection output
land in class folders, and the command ends with a RESULT line."""
from __future__ import annotations

from velox_ai_tools import cli

from conftest import last_result, make_image


def test_build_dataset_crops_every_labelled_detection(tmp_path, capsys):
    det = tmp_path / "det"
    make_image(det / "frame_1.jpg", 200, 150)
    (det / "labels").mkdir()
    (det / "labels" / "frame_1.txt").write_text("2 0.5 0.5 0.2 0.2 0.9\n2 0.2 0.2 0.1 0.1\n", encoding="utf-8")
    out = tmp_path / "cls_ds"
    assert cli.main(["two-stage", "build-dataset", "--detection-dir", str(det), "--output-dir", str(out)]) == 0
    r = last_result(capsys)
    assert r["ok"] and r["n_crops"] == 2 and r["n_classes"] == 1
    crops = list(out.rglob("*.jpg"))
    assert len(crops) == 2 and all(p.parent.name == "class_2" for p in crops)


def test_build_dataset_without_labels_reports_a_failure(tmp_path, capsys):
    det = tmp_path / "empty"
    det.mkdir()
    assert cli.main(["two-stage", "build-dataset", "--detection-dir", str(det), "--output-dir", str(tmp_path / "o")]) == 1
    assert last_result(capsys)["ok"] is False
