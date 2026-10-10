"""Benchmark protocol tests, not real module performance measurements."""

import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scripts import week4_performance as perf


def test_statistics_use_linear_percentiles_and_explicit_units():
    result = perf.summarize([4, 1, 3, 2], "ms")
    assert result["count"] == 4
    assert result["mean"] == result["p50"] == 2.5
    assert result["p95"] == pytest.approx(3.85)
    assert result["min"] == 1 and result["max"] == 4
    assert result["unit"] == "ms"


@pytest.mark.parametrize("values", [[], [-1], [float("inf")], [float("nan")]])
def test_invalid_samples_are_rejected(values):
    with pytest.raises(ValueError):
        perf.summarize(values, "ms")


def test_warmup_is_recorded_but_excluded_and_batch_is_normalized(monkeypatch):
    clock = iter([0, 1_000_000, 2_000_000, 4_000_000, 5_000_000, 9_000_000])
    monkeypatch.setattr(perf.time, "perf_counter_ns", lambda: next(clock))
    calls, records = [], []
    stats = perf.measure(lambda: calls.append(1), warmup=1, repeats=2, batch_size=10,
                         unit="us", emit=records.append, describe=lambda _: {})
    assert len(calls) == 30
    assert [r["phase"] for r in records] == ["warmup", "measurement", "measurement"]
    assert [r["elapsed_ns"] for r in records] == [1_000_000, 2_000_000, 4_000_000]
    assert stats["count"] == 2 and stats["mean"] == 300


def test_operation_error_is_recorded_without_retry():
    calls, records = [], []

    def fail():
        calls.append(1)
        raise RuntimeError("broken operation")

    with pytest.raises(RuntimeError, match="broken operation"):
        perf.measure(fail, warmup=2, repeats=20, batch_size=1, unit="ms",
                     emit=records.append, describe=lambda _: {})
    assert len(calls) == len(records) == 1
    assert records[0]["status"] == "error"


def test_invalid_result_stops_without_counting_success():
    records = []

    def describe(_):
        raise ValueError("empty output")

    with pytest.raises(ValueError, match="empty output"):
        perf.measure(lambda: [], warmup=0, repeats=20, batch_size=1, unit="ms",
                     emit=records.append, describe=describe)
    assert len(records) == 1 and records[0]["status"] == "error"


def test_cached_reader_is_cpu_only_and_downloads_disabled(tmp_path, monkeypatch):
    for filename in ("craft_mlt_25k.pth", "zh_sim_g2.pth"):
        (tmp_path / filename).write_bytes(b"offline fixture")
    calls = []
    monkeypatch.setitem(sys.modules, "easyocr", SimpleNamespace(
        Reader=lambda *args, **kwargs: calls.append((args, kwargs)) or "reader"))
    assert perf.create_reader(tmp_path) == "reader"
    args, kwargs = calls[0]
    assert args == (["ch_sim", "en"],)
    assert kwargs["gpu"] is False and kwargs["download_enabled"] is False
    assert kwargs["model_storage_directory"] == str(tmp_path.resolve())


def test_missing_weights_stop_before_reader_import(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "easyocr", None)
    with pytest.raises(FileNotFoundError, match="downloads forbidden"):
        perf.create_reader(tmp_path)


def saved_case(tmp_path):
    image_path = tmp_path / "saved.png"
    cv2.imwrite(str(image_path), np.zeros((80, 100, 3), dtype=np.uint8))
    event = {"kind": "observation", "observation_id": "obs-1", "image_path": str(image_path),
             "screen_region": {"left": 0, "top": 0, "width": 200, "height": 200},
             "screen_size": [200, 200], "crop_origin": [10, 20],
             "context": {"observation_id": "obs-1", "image_size": [100, 80],
                         "allowed_box": [0, 0, 100, 80], "elements": [
                             {"target_id": "obs-1:text-1", "text": "Search query",
                              "confidence": 0.9, "bbox": [20, 30, 60, 50]}]}}
    path = tmp_path / "summary.json"
    return path, event


def test_saved_observation_uses_real_ids_geometry_and_hashes(tmp_path):
    path, event = saved_case(tmp_path)
    path.write_text(json.dumps({"runtime": {"events": [event]}}), encoding="utf-8")
    case = perf.load_case("search", path, "obs-1", "Search query")
    assert case["metadata"]["grounded_point"] == [50, 60]
    assert case["metadata"]["point_used_for_execution"] is False
    assert case["metadata"]["image_sha256"] == perf.digest(event["image_path"])
    assert case["observation"].targets[0].target_id == "obs-1:text-1"


@pytest.mark.parametrize("fault", ["dimensions", "duplicate_event", "id", "text", "duplicate_target", "confidence"])
def test_saved_input_mismatch_or_unsafe_target_stops(tmp_path, fault):
    path, event = saved_case(tmp_path)
    events = [event]
    element = event["context"]["elements"][0]
    if fault == "dimensions":
        event["context"]["image_size"] = [99, 80]
    elif fault == "duplicate_event":
        events.append(event)
    elif fault == "id":
        element["target_id"] = "stale:text-1"
    elif fault == "text":
        element["text"] = "Address bar"
    elif fault == "duplicate_target":
        event["context"]["elements"].append({**element, "target_id": "obs-1:text-2"})
    else:
        element["confidence"] = 0.1
    path.write_text(json.dumps({"events": events}), encoding="utf-8")
    with pytest.raises(ValueError):
        perf.load_case("search", path, "obs-1", "Search query")


@pytest.mark.parametrize("fault", ["repo_output", "existing_output", "counts", "case_count"])
def test_unapproved_protocol_or_output_is_rejected_before_ocr(tmp_path, fault, monkeypatch):
    args = perf.parser().parse_args(["offline", "--case", "qt", "missing", "obs-1", "CLICKED",
                                    "--case", "search", "missing", "obs-2", "Search query",
                                    "--model-dir", str(tmp_path), "--output-dir", str(tmp_path / "new")])
    if fault == "repo_output":
        args.output_dir = perf.ROOT / "evidence"
    elif fault == "existing_output":
        args.output_dir.mkdir()
    elif fault == "counts":
        args.ocr_repeat = 21
    else:
        args.case.pop()
    monkeypatch.setattr(perf, "create_reader", lambda _: pytest.fail("must not initialize OCR"))
    with pytest.raises((ValueError, FileExistsError)):
        perf.run(args)


def test_capture_mode_is_not_available():
    with pytest.raises(SystemExit):
        perf.parser().parse_args(["capture"])
