"""Offline evidence interpretation, not UIA/provider or live focus validation."""

from copy import deepcopy
from dataclasses import replace
import json
import sys
from types import SimpleNamespace

import pytest

from scripts import week4_focus_capability as diagnostic
from scripts.week4_focus_capability import classify, classify_saved_report


def evidence():
    focus = {
        "runtime_id": [1, 2], "control_type": "ControlType.Edit",
        "in_bound": True, "has_keyboard_focus": True, "enabled": True,
        "offscreen": False, "value_pattern": True, "value_read_only": False,
        "value_empty": True,
        "labeled_by": {"name": "Search query", "runtime_id": [1, 3], "in_bound": True,
                       "box": [10, 20, 120, 45]},
        "ancestors": [{"runtime_id": [1, 4]}],
    }
    return {
        "focused_present": True, "focused_in_bound": True,
        "focus_stable_during_read": True, "focused": focus,
        "documents": [{"runtime_id": [1, 4], "name": "Week4 Search", "offscreen": False}],
        "edits": [focus],
        "initial_ocr_scope_verified": True,
        "initial_ocr_label": {"text": "Search query", "confidence": 0.9,
                              "screen_box": [12, 22, 118, 43],
                              "grounding_point_read_only": [65, 32]},
    }


def test_capability_never_grants_input_or_claims_agent_binding():
    result = classify(evidence())
    assert result["uia_capability"] == "explicit_label_link_observed"
    assert result["input_permission"] is False
    assert result["agent_click_binding_verified"] is False


@pytest.mark.parametrize("key", ["focused_present", "focused_in_bound", "focus_stable_during_read"])
def test_missing_window_or_focus_evidence_is_unsafe(key):
    data = evidence()
    data[key] = False
    assert classify(data)["uia_capability"] == "unproven"


@pytest.mark.parametrize("name", ["Search query ", " Search query", "\tSearch query\n"])
def test_only_leading_and_trailing_label_whitespace_is_ignored(name):
    data = evidence()
    data["focused"]["labeled_by"]["name"] = name
    assert classify(data)["uia_capability"] == "explicit_label_link_observed"


def test_label_whitespace_normalization_applies_to_both_reads():
    data = evidence()
    data["focused"] = deepcopy(data["focused"])
    data["focused"]["labeled_by"]["name"] = "Search query "
    data["edits"][0]["labeled_by"]["name"] = " Search query"
    assert classify(data)["uia_capability"] == "explicit_label_link_observed"


def test_existing_in_memory_grounding_tuple_is_also_supported():
    data = evidence()
    data["initial_ocr_label"]["grounding_point_read_only"] = (65, 32)
    assert classify(data)["uia_capability"] == "explicit_label_link_observed"


@pytest.mark.parametrize("name", ["Search  query", "search query", "Search querx", "Search query extra"])
def test_label_matching_is_not_fuzzy_or_case_insensitive(name):
    data = evidence()
    data["focused"]["labeled_by"]["name"] = name
    assert classify(data)["uia_capability"] == "unproven"


def test_ocr_rectangle_need_not_be_fully_contained_when_existing_point_matches():
    data = evidence()
    data["initial_ocr_label"]["screen_box"] = [8, 18, 124, 48]
    assert classify(data)["uia_capability"] == "explicit_label_link_observed"


@pytest.mark.parametrize("point", [[9, 32], [120, 32], [65, 19], [65, 45], None, [float("nan"), 32]])
def test_existing_point_outside_label_or_missing_is_rejected(point):
    data = evidence()
    data["initial_ocr_label"]["screen_box"] = [0, 0, 200, 100]
    data["initial_ocr_label"]["grounding_point_read_only"] = point
    assert classify(data)["uia_capability"] == "unproven"


def test_point_must_also_belong_to_original_ocr_rectangle():
    data = evidence()
    data["initial_ocr_label"]["screen_box"] = [70, 22, 118, 43]
    assert classify(data)["uia_capability"] == "unproven"


@pytest.mark.parametrize("field", ["text", "actual_ocr_text"])
def test_ocr_target_mismatch_is_rejected(field):
    data = evidence()
    data["initial_ocr_label"][field] = "Search"
    assert classify(data)["uia_capability"] == "unproven"


@pytest.mark.parametrize("confidence", [0.49, None, float("nan"), float("inf"), True])
def test_confidence_missing_nonfinite_or_below_existing_threshold_is_rejected(confidence):
    data = evidence()
    data["initial_ocr_label"]["confidence"] = confidence
    assert classify(data)["uia_capability"] == "unproven"


def test_original_confidence_threshold_is_unchanged():
    data = evidence()
    data["initial_ocr_label"]["confidence"] = 0.5
    assert classify(data)["uia_capability"] == "explicit_label_link_observed"


def test_duplicate_association_after_trimming_is_still_rejected():
    data = evidence()
    data["focused"]["labeled_by"]["name"] = "Search query "
    other = deepcopy(data["edits"][0])
    other["runtime_id"] = [1, 7]
    other["labeled_by"]["name"] = " Search query"
    data["edits"].append(other)
    assert classify(data)["uia_capability"] == "unproven"


def test_missing_focused_element_identity_cannot_match_a_missing_candidate_identity():
    data = evidence()
    del data["focused"]["runtime_id"]
    assert classify(data)["uia_capability"] == "unproven"


@pytest.mark.parametrize("box", [None, [10, 20, 120], [10, 20, float("inf"), 45], [120, 20, 10, 45]])
def test_invalid_or_nonfinite_label_geometry_is_rejected(box):
    data = evidence()
    data["focused"]["labeled_by"]["box"] = box
    assert classify(data)["uia_capability"] == "unproven"


def test_labeled_by_relationship_must_match_actual_focused_element():
    data = evidence()
    data["focused"] = deepcopy(data["focused"])
    data["focused"]["labeled_by"]["runtime_id"] = [1, 99]
    assert classify(data)["uia_capability"] == "unproven"


def saved_report():
    data = evidence()
    return {
        "status": "evidence_collected_not_input_authorized", "error": None,
        "bound_window": {"handle": 7, "pid": 8, "title": "Week4 Search - Microsoft Edge", "box": [0, 0, 900, 700]},
        "bound_profile": {"pid": 8, "verified": True,
                          "test_profile": str(diagnostic.PREP / "edge-search-test-profile"),
                          "exe": r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"},
        "initial_ocr_label": deepcopy(data["initial_ocr_label"]),
        "snapshots": [{"phase": "search_input", "uia": data,
                       "assessment": {"uia_capability": "unproven", "input_permission": False}}],
    }


@pytest.mark.parametrize("change", ["window_missing", "pid_mismatch", "profile_missing", "profile_changed", "exe_changed", "unverified_profile"])
def test_saved_scope_identity_is_required(change):
    report = saved_report()
    if change == "window_missing":
        del report["bound_window"]
    elif change == "pid_mismatch":
        report["bound_profile"]["pid"] = 9
    elif change == "profile_missing":
        del report["bound_profile"]
    elif change == "profile_changed":
        report["bound_profile"]["test_profile"] = r"C:\personal\Edge"
    elif change == "exe_changed":
        report["bound_profile"]["exe"] = r"C:\other.exe"
    else:
        report["bound_profile"]["verified"] = False
    result = classify_saved_report(report)
    assert result["recorded_scope_validated"] is False
    assert result["snapshots"][0]["assessment"]["uia_capability"] == "unproven"


def test_snapshot_cannot_substitute_another_ocr_target():
    report = saved_report()
    report["snapshots"][0]["uia"]["initial_ocr_label"]["target_id"] = "different-target"
    result = classify_saved_report(report)
    assert result["snapshots"][0]["assessment"]["uia_capability"] == "unproven"


def test_no_saved_snapshots_is_not_valid_evidence():
    report = saved_report()
    report["snapshots"] = []
    assert classify_saved_report(report)["recorded_scope_validated"] is False


def test_saved_document_identity_change_is_rejected():
    report = saved_report()
    second = deepcopy(report["snapshots"][0])
    second["uia"]["documents"][0]["runtime_id"] = [1, 100]
    second["uia"]["focused"]["ancestors"][0]["runtime_id"] = [1, 100]
    report["snapshots"].append(second)
    result = classify_saved_report(report)
    assert result["snapshots"][1]["assessment"]["uia_capability"] == "unproven"


@pytest.fixture
def actual_saved_report():
    path = diagnostic.ROOT.parent / ".test-tmp" / "week4-focus-capability-once-20261008-v2" / "uia-capability.json"
    if not path.is_file():
        pytest.skip("Immutable local UIA evidence is not bundled with the repository")
    return json.loads(path.read_text(encoding="utf-8"))


def test_actual_saved_focus_states_are_distinguished_without_granting_input(actual_saved_report):
    before = deepcopy(actual_saved_report)
    result = classify_saved_report(actual_saved_report)
    assert result["recorded_scope_validated"] is True
    assert [s["assessment"]["uia_capability"] for s in result["snapshots"]] == [
        "explicit_label_link_observed", "unproven", "unproven",
    ]
    assert all(s["assessment"]["input_permission"] is False for s in result["snapshots"])
    assert result["agent_click_binding_verified"] is False
    assert result["real_input_safety_verified"] is False
    assert actual_saved_report == before


def test_actual_address_bar_focus_overrides_stale_search_focus_flag(actual_saved_report):
    data = actual_saved_report["snapshots"][1]["uia"]
    assert data["focused"]["has_keyboard_focus"] is True
    search = next(e for e in data["edits"] if e["name"] == "Search query")
    assert search["has_keyboard_focus"] is True
    assert search["runtime_id"] != data["focused"]["runtime_id"]
    assert "another control" in classify(data)["reason"]


@pytest.mark.parametrize("split", [False, True])
def test_local_diagnostic_profile_retains_original_identity_policy(split):
    from scripts import week4_search_e2e as search
    process = {"pid": 123, "exe": str(search.EDGE), "command_line": "offline process metadata"}
    args = ["--user-data-dir", str(search.PROFILE)] if split else ["--user-data-dir=" + str(search.PROFILE)]
    record = diagnostic.validate_diagnostic_profile(process, 123, args)
    assert record["verified"] is True
    assert record["test_profile"] == str(search.PROFILE)
    # Diagnostic scope did not originally require the execution mode's network flags.
    assert record["pid"] == 123


@pytest.mark.parametrize("fault", ["pid", "exe", "missing", "duplicate", "personal"])
def test_local_diagnostic_rejects_invalid_profile_metadata(fault):
    from scripts import week4_search_e2e as search
    process = {"pid": 123, "exe": str(search.EDGE), "command_line": "offline"}
    args = ["--user-data-dir=" + str(search.PROFILE)]
    if fault == "pid":
        process["pid"] = 124
    elif fault == "exe":
        process["exe"] = "other.exe"
    elif fault == "missing":
        args = []
    elif fault == "duplicate":
        args *= 2
    else:
        args = ["--user-data-dir=C:/personal"]
    with pytest.raises(ValueError):
        diagnostic.validate_diagnostic_profile(process, 123, args)


def test_diagnostic_process_reader_reuses_repository_metadata_only(monkeypatch):
    from scripts import week4_search_e2e as search
    process = {"pid": 123, "exe": str(search.EDGE), "command_line": "offline"}
    calls = []
    monkeypatch.setattr(search, "read_profile", lambda pid: calls.append(pid) or process)
    monkeypatch.setattr(search, "windows_arguments", lambda _: ["--user-data-dir=" + str(search.PROFILE)])
    assert diagnostic.read_diagnostic_profile(123)["verified"] is True
    assert calls == [123]
    process["command_line"] = ""
    with pytest.raises(ValueError, match="unavailable"):
        diagnostic.read_diagnostic_profile(123)


def diagnostic_scene(tmp_path):
    from gui_agent.capture import ScreenRegion
    from gui_agent.grounding import build_observation
    from gui_agent.perception import TextElement
    from gui_agent.runtime import SceneObservation, WindowInfo
    bound = WindowInfo(123, 456, "Week4 Search - Microsoft Edge", (0, 0, 300, 200))
    obs = build_observation("offline", tmp_path / "offline.png", [
        TextElement("Search query", 0.9, (10, 20, 120, 45)),
        TextElement("127.0.0.1:8765/search", 0.9, (10, 100, 250, 120)),
    ], screen_region=ScreenRegion(0, 0, 300, 200), screen_size=(300, 200))
    return SceneObservation(obs, bound, bound, 1.0)


@pytest.mark.parametrize("fault", ["unfocused", "wrong_url", "low_url_confidence", "duplicate_label", "low_label_confidence", "token", "results"])
def test_diagnostic_initial_checks_remain_fail_closed(tmp_path, fault):
    from gui_agent.grounding import GroundingError, build_observation
    from gui_agent.capture import ScreenRegion
    from gui_agent.perception import TextElement
    scene = diagnostic_scene(tmp_path)
    elements = [t.element for t in scene.observation.targets]
    if fault == "unfocused":
        scene = replace(scene, foreground=None)
    else:
        if fault == "wrong_url":
            elements[1] = replace(elements[1], text="https://external.invalid")
        elif fault == "low_url_confidence":
            elements[1] = replace(elements[1], confidence=0.49)
        elif fault == "duplicate_label":
            elements.append(TextElement("Search query", 0.9, (10, 140, 120, 165)))
        elif fault == "low_label_confidence":
            elements[0] = replace(elements[0], confidence=0.49)
        else:
            elements.append(TextElement("W4-E2E-SEARCH-001" if fault == "token" else "Search results", 0.9, (10, 140, 250, 165)))
        scene = replace(scene, observation=build_observation("offline", tmp_path / "offline.png", elements,
                        screen_region=ScreenRegion(0, 0, 300, 200), screen_size=(300, 200)))
    with pytest.raises(GroundingError):
        diagnostic.require_initial_scene(scene)


def test_live_diagnostic_mocked_without_external_python_dependency(tmp_path, monkeypatch):
    from gui_agent import runtime
    from scripts import week4_search_e2e as search
    scene = diagnostic_scene(tmp_path)
    history = tmp_path / "history"
    history.mkdir()
    (history / "run-summary.json").write_text('{"executed_actions": 0}', encoding="utf-8")
    prep = tmp_path / "external"
    prep.mkdir()
    (prep / "run_once.py").write_text("raise AssertionError('External Python must never execute')", encoding="utf-8")
    monkeypatch.setattr(diagnostic, "HISTORY", history)
    monkeypatch.setattr(diagnostic, "PREP", prep)
    monkeypatch.setattr(diagnostic, "read_records", lambda: {"searches": [], "messages": []})
    monkeypatch.setattr(diagnostic, "signal_done", lambda: None)
    monkeypatch.setattr(diagnostic.time, "sleep", lambda _: None)
    monkeypatch.setattr("builtins.input", lambda *args: "")
    monkeypatch.setattr(diagnostic, "read_diagnostic_profile", lambda _: {
        "pid": 456, "exe": str(search.EDGE), "test_profile": str(search.PROFILE), "verified": True})
    monkeypatch.setattr(runtime, "WindowsWindowProbe", lambda: SimpleNamespace(
        foreground=lambda: scene.window, read=lambda _: scene.window))
    monkeypatch.setattr(diagnostic, "EvidenceObserver", lambda *args, **kwargs: SimpleNamespace(observe=lambda: scene))
    monkeypatch.setitem(sys.modules, "easyocr", SimpleNamespace(Reader=lambda *args, **kwargs: object()))
    monkeypatch.setattr(search.demo, "EasyOcrRecognizer", lambda **kwargs: object())

    def forbidden(*args, **kwargs):
        pytest.fail("No real network/process/desktop/model boundary is permitted")

    monkeypatch.setattr(diagnostic.subprocess, "run", forbidden)
    monkeypatch.setattr(search, "OpenAICompatibleVisionClient", forbidden)
    reads = []

    def readonly_uia(handle):
        data = evidence()
        reads.append(handle)
        if len(reads) == 3:
            data["focused"] = {"runtime_id": [1, 99], "control_type": "ControlType.Edit", "has_keyboard_focus": True}
        elif len(reads) == 4:
            data["focused"] = {"runtime_id": [1, 98], "control_type": "ControlType.Button", "has_keyboard_focus": True}
        return data

    monkeypatch.setattr(diagnostic, "read_uia", readonly_uia)
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["diagnostic", "--output-dir", str(output)])
    assert diagnostic.main() == 0
    report = json.loads((output / "uia-capability.json").read_text(encoding="utf-8"))
    assert report["input_permission"] is False and report["automated_desktop_actions"] == 0
    assert report["records_unchanged"] is True
    assert [s["assessment"]["uia_capability"] for s in report["snapshots"]] == [
        "explicit_label_link_observed", "unproven", "unproven"]
    assert len(reads) == 4


def test_offline_entry_never_samples_or_accesses_services(tmp_path, monkeypatch):
    source = tmp_path / "saved.json"
    raw = json.dumps(saved_report()).encode("utf-8")
    source.write_bytes(raw)
    output = tmp_path / "new-analysis"

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline interpretation must not sample, access services or interact")

    for name in ("read_uia", "read_records", "signal_done", "build_opener"):
        monkeypatch.setattr(diagnostic, name, forbidden)
    monkeypatch.setattr(diagnostic.subprocess, "run", forbidden)
    monkeypatch.setattr(sys, "argv", ["diagnostic", "--saved-evidence", str(source), "--output-dir", str(output)])
    assert diagnostic.main() == 0
    result = json.loads((output / "offline-assessment.json").read_text(encoding="utf-8"))
    assert source.read_bytes() == raw
    assert result["uia_sample_calls"] == result["capture_calls"] == result["ocr_calls"] == 0
    assert result["api_calls"] == result["automated_desktop_actions"] == 0
    assert result["input_permission"] is False


@pytest.mark.parametrize("key,value", [
    ("control_type", "ControlType.Button"), ("has_keyboard_focus", False),
    ("enabled", False), ("offscreen", True), ("value_pattern", False),
    ("value_read_only", True), ("value_empty", False),
])
def test_control_properties_fail_closed(key, value):
    data = evidence()
    data["focused"][key] = value
    assert classify(data)["uia_capability"] == "unproven"


def test_address_bar_is_not_search_input_even_when_editable_and_focused():
    data = evidence()
    address = deepcopy(data["focused"])
    address.update(runtime_id=[1, 9], name="Address and search bar", labeled_by=None, ancestors=[])
    data["focused"] = address
    data["edits"].append(address)
    assert classify(data)["uia_capability"] == "unproven"


def test_duplicate_explicit_label_associations_are_unsafe():
    data = evidence()
    other = deepcopy(data["edits"][0])
    other["runtime_id"] = [1, 7]
    data["edits"].append(other)
    assert classify(data)["uia_capability"] == "unproven"


def test_name_without_label_relation_is_insufficient():
    data = evidence()
    data["focused"].update(name="Search query", labeled_by=None)
    assert classify(data)["uia_capability"] == "unproven"


def test_wrong_or_unknown_document_is_unsafe():
    data = evidence()
    data["documents"][0]["name"] = "Other page"
    assert classify(data)["uia_capability"] == "unproven"


def test_document_must_be_an_actual_ancestor_of_input():
    data = evidence()
    data["focused"]["ancestors"] = []
    assert classify(data)["uia_capability"] == "unproven"


@pytest.mark.parametrize("change", ["low_confidence", "moved_label", "missing_ocr", "missing_scope"])
def test_ocr_link_evidence_is_required(change):
    data = evidence()
    if change == "low_confidence":
        data["initial_ocr_label"]["confidence"] = 0.49
    elif change == "moved_label":
        data["initial_ocr_label"]["screen_box"] = [200, 22, 300, 43]
    elif change == "missing_ocr":
        del data["initial_ocr_label"]
    else:
        data["initial_ocr_scope_verified"] = False
    assert classify(data)["uia_capability"] == "unproven"
