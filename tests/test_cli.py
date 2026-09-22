"""The command line: every command wired, version/info JSON, weight names."""
from __future__ import annotations

from velox_ai_tools import cli, common

from conftest import last_result


def test_every_command_is_wired():
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if hasattr(a, "choices") and a.choices and "detect" in a.choices)
    assert set(sub.choices) == {"version", "info", "detect", "prelabel", "infer", "train", "train-rdd2022", "two-stage"}


def test_version_reports_the_environment(fake_ultralytics, capsys):
    assert cli.main(["version"]) == 0
    r = last_result(capsys)
    assert r["ok"] and r["ultralytics"] == "0.0-fake" and r["torch"] == "0.0-fake"
    assert r["cuda_available"] is False and r["missing"] == [] and r["tool"]


def test_version_fails_when_ultralytics_is_missing(monkeypatch, capsys):
    import sys
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    assert cli.main(["version"]) == 1
    r = last_result(capsys)
    assert not r["ok"] and set(r["missing"]) == {"ultralytics", "torch"}


def test_info_lists_the_class_names_of_a_weight_file(fake_ultralytics, tmp_path, capsys):
    w = tmp_path / "w.pt"
    w.write_bytes(b"w")
    assert cli.main(["info", "--weights", str(w)]) == 0
    r = last_result(capsys)
    assert r["names"] == ["D00", "D10", "D20", "D40"] and r["nc"] == 4 and r["task"] == "detect"
    assert ("load", str(w.resolve())) in fake_ultralytics.calls


def test_asset_names_resolve_into_the_tools_own_cache(tool_home, tmp_path):
    assert common.resolve_weights("yolov11n.pt") == str(tool_home / "weights" / "yolo11n.pt")
    assert common.resolve_weights("yolov8s.pt") == str(tool_home / "weights" / "yolov8s.pt")
    assert common.resolve_weights("yolo11s-seg.pt").endswith("yolo11s-seg.pt")
    own = tmp_path / "road.pt"
    own.write_bytes(b"x")
    assert common.resolve_weights(str(own)) == str(own.resolve())
    assert common.is_asset_name("yolo11s.pt") and not common.is_asset_name("road_damage.pt")
    assert common.resolve_weights("somewhere/else.pt") == "somewhere/else.pt", "handed to ultralytics unchanged"
