"""Offline probe regressions; all desktop, UIA, OCR and service calls are fakes."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from gui_agent.grounding import GroundingError, KeyAction
from scripts import week4_search_focus_probe as probe
from test_week4_search_e2e import setup as setup_search


def setup_probe(tmp_path, monkeypatch):
    world, observer, base, raw, _ = setup_search(tmp_path, monkeypatch, approve=False)
    session = probe.ProbeSession(base.scope, observer, base.probe, base.log, base.stop_file,
                                 enabled=True, profile_reader=world.profile, uia_reader=world.uia,
                                 human_authorize=world.authorize, clock=lambda: world.now)
    return world, observer, session, raw


def execute(tmp_path, world, session, raw):
    return probe.execute_probe(session, raw, tmp_path, wait=world.wait, clock=lambda: world.now)


def test_full_mock_probe_reaches_both_focus_gates_and_mandatory_veto(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No API or live UIA/desktop calls permitted in offline tests")

    monkeypatch.setattr(probe.search, "OpenAICompatibleVisionClient", forbidden)
    monkeypatch.setattr(probe.search.uia, "read_uia", forbidden)
    world, observer, session, raw = setup_probe(tmp_path, monkeypatch)
    result, recorder, veto = execute(tmp_path, world, session, raw)
    assert (result.status, result.reason) == ("blocked", probe.VETO_REASON)
    assert result.executed_actions == 1 and result.attempted_actions == 2
    assert recorder.calls == 3
    assert veto.click_calls == veto.successful_clicks == veto.write_veto_calls == 1
    assert [call[0] for call in raw.calls] == ["click"]
    assert world.query == "" and world.phase == 1
    assert session.focus_results["focus_callback"]["passed"] is True
    assert session.focus_results["write_entry"]["passed"] is True
    assert session.focus_results["focus_callback"]["binding"] == session.focus_results["write_entry"]["binding"]
    assert raw.FAILSAFE is True
    assert session.typed is False and session.submitted is False
    assert len(world.declarations) == 1
    grant = world.declarations[0]
    assert grant["actual_click_limit"] == 1
    assert grant["real_keyboard_authorized"] is False
    assert grant["task"] == probe.TASK
    assert "no GLM/API" in grant["model"]
    assert grant["budgets"]["max_actions"] == 20
    assert grant["budgets"]["timeout_seconds"] == 300
    intercepts = [e for e in session.log.events if e["kind"] == "probe_write_intercepted"]
    assert len(intercepts) == 1
    assert intercepts[0]["source"] == "preset_probe_backend_veto"
    assert intercepts[0]["human_answer"] is None
    assert intercepts[0]["real_write_calls"] == 0
    field = probe.independent_field_check(session)
    assert field["checked"] is True and field["empty"] is True
    assert field["source"] == "independent_post_stop_uia_value"
    assert probe.probe_passed(result, recorder, veto, session, probe.EMPTY_RECORDS,
                              probe.EMPTY_RECORDS, field, None) is True
    assert sorted(p.name for p in tmp_path.glob("model-*.json")) == ["model-01.json", "model-02.json", "model-03.json"]
    for path in tmp_path.glob("model-*.json"):
        request = json.loads(path.read_text(encoding="utf-8"))["request"]
        assert "labeled_by" not in request["instruction"]
        assert "value_matches_expected" not in request["instruction"]


@pytest.mark.parametrize("text", [probe.search.TOKEN, "unexpected", ""])
def test_write_replacement_veto_is_unconditional_and_independent_of_uia(tmp_path, monkeypatch, text):
    world, observer, session, raw = setup_probe(tmp_path, monkeypatch)
    veto = probe.VetoBackend(raw, session.log)
    with pytest.raises(probe.PresetWriteVeto, match="PRESET_PROBE_WRITE_VETO"):
        veto.write(text, interval=0.02)
    assert veto.write_veto_calls == 1 and raw.calls == []
    assert world.query == ""
    assert session.focus_results == {}


@pytest.mark.parametrize("method,args", [("press", ("enter",)), ("hotkey", ("ctrl", "l")),
                                         ("scroll", (1,)), ("moveTo", (100, 100)), ("dragTo", (100, 100))])
def test_other_raw_desktop_actions_cannot_be_forwarded(tmp_path, monkeypatch, method, args):
    world, observer, session, raw = setup_probe(tmp_path, monkeypatch)
    veto = probe.VetoBackend(raw, session.log)
    with pytest.raises(GroundingError):
        getattr(veto, method)(*args)
    assert raw.calls == []


def test_second_click_has_independent_hard_backend_cap(tmp_path, monkeypatch):
    world, observer, session, raw = setup_probe(tmp_path, monkeypatch)
    result, recorder, veto = execute(tmp_path, world, session, raw)
    kwargs = raw.calls[0][1]
    with pytest.raises(GroundingError, match="at most one"):
        veto.click(**kwargs)
    assert veto.click_calls == 1 and len(raw.calls) == 1


def test_probe_cannot_disable_failsafe(tmp_path, monkeypatch):
    world, observer, session, raw = setup_probe(tmp_path, monkeypatch)
    veto = probe.VetoBackend(raw, session.log)
    veto.FAILSAFE = True
    with pytest.raises(GroundingError):
        veto.FAILSAFE = False
    assert raw.FAILSAFE is True


@pytest.mark.parametrize("focus", ["address", "button", "body"])
def test_last_focus_failure_is_not_a_successful_write_veto_probe(tmp_path, monkeypatch, focus):
    world, observer, session, raw = setup_probe(tmp_path, monkeypatch)
    original = session.focused_input

    def transfer_only_at_last_entry(scene, expected, phase):
        if phase == "write_entry":
            world.focus = focus
        return original(scene, expected, phase)

    session.focused_input = transfer_only_at_last_entry
    result, recorder, veto = execute(tmp_path, world, session, raw)
    assert result.status == "blocked" and result.reason != probe.VETO_REASON
    assert session.focus_results["focus_callback"]["passed"] is True
    assert session.focus_results["write_entry"]["passed"] is False
    assert veto.write_veto_calls == 0
    assert recorder.calls == 3 and [call[0] for call in raw.calls] == ["click"]
    assert world.query == ""
    assert probe.probe_passed(result, recorder, veto, session, probe.EMPTY_RECORDS,
                              probe.EMPTY_RECORDS, {"checked": False, "empty": None}, None) is False


@pytest.mark.parametrize("fault", ["profile_bad", "foreign", "moved", "ocr_duplicate", "uia_error"])
def test_scope_or_observation_failure_stops_without_input(tmp_path, monkeypatch, fault):
    world, observer, session, raw = setup_probe(tmp_path, monkeypatch)
    setattr(world, fault, True)
    result, recorder, veto = execute(tmp_path, world, session, raw)
    assert result.status == "blocked"
    assert raw.calls == [] and world.query == ""
    assert veto.write_veto_calls == 0


def test_no_plan_grant_no_click_or_later_mock_decision(tmp_path, monkeypatch):
    world, observer, session, raw = setup_probe(tmp_path, monkeypatch)
    session.human_authorize = lambda *args: False
    result, recorder, veto = execute(tmp_path, world, session, raw)
    assert result.status == "cancelled"
    assert recorder.calls == 1
    assert raw.calls == [] and veto.write_veto_calls == 0


def test_missing_focus_evidence_is_not_filled_in_by_mock_client(tmp_path, monkeypatch):
    world, observer, session, raw = setup_probe(tmp_path, monkeypatch)
    world.uia_override = lambda data: data.update(focused_present=False)
    result, recorder, veto = execute(tmp_path, world, session, raw)
    assert result.status == "blocked" and recorder.calls == 0
    assert raw.calls == [] and session.focus_results == {}


def test_stop_flag_after_click_prevents_further_mock_call_and_write(tmp_path, monkeypatch):
    world, observer, session, raw = setup_probe(tmp_path, monkeypatch)
    original = raw.click

    def click_then_stop(**kwargs):
        original(**kwargs)
        session.stop_file.touch()

    raw.click = click_then_stop
    result, recorder, veto = execute(tmp_path, world, session, raw)
    assert result.status == "blocked" and recorder.calls == 2
    assert [c[0] for c in raw.calls] == ["click"]
    assert veto.write_veto_calls == 0


@pytest.mark.parametrize("fault", ["field_unknown", "field_nonempty", "record_added", "missing_last_gate", "error"])
def test_probe_acceptance_requires_all_independent_evidence(tmp_path, monkeypatch, fault):
    world, observer, session, raw = setup_probe(tmp_path, monkeypatch)
    result, recorder, veto = execute(tmp_path, world, session, raw)
    field = probe.independent_field_check(session)
    after = deepcopy(probe.EMPTY_RECORDS)
    error = None
    if fault == "field_unknown":
        field = {"checked": False, "empty": None}
    elif fault == "field_nonempty":
        field["empty"] = False
    elif fault == "record_added":
        after["searches"].append("unexpected")
    elif fault == "missing_last_gate":
        session.focus_results.pop("write_entry")
    else:
        error = "Evidence write failed"
    assert probe.probe_passed(result, recorder, veto, session, probe.EMPTY_RECORDS, after, field, error) is False


def test_mock_reads_only_current_ocr_ids_without_coordinates_or_uia():
    client = probe.MockClient()
    planning = SimpleNamespace(instruction="plan with no UIA")
    assert json.loads(client.generate(planning).text)["status"] == "ready"
    observation = {"observation_id": "fresh-123", "elements": [{"target_id": "fresh-123:text-7", "text": "Search query"}]}
    request = SimpleNamespace(instruction="Current step_id: probe-focus\nCurrent observation: " + json.dumps(observation))
    click = json.loads(client.generate(request).text)
    assert click["action"] == {"type": "click", "target_id": "fresh-123:text-7", "target_text": "Search query"}
    observation["observation_id"] = "fresh-124"
    observation["elements"][0]["target_id"] = "fresh-124:text-9"
    request.instruction = "Current step_id: probe-focus\nCurrent observation: " + json.dumps(observation)
    assert json.loads(client.generate(request).text)["action"]["target_id"] == "fresh-124:text-9"
    with pytest.raises(GroundingError, match="further model decisions"):
        client.generate(request)


def test_cli_requires_explicit_flag_before_any_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "OUTPUT", tmp_path / "once")
    monkeypatch.setattr(probe.search.uia, "read_records", lambda: pytest.fail("Must not read services"))
    with pytest.raises(GroundingError):
        probe.run_once()
    assert not probe.OUTPUT.exists()


def test_full_fake_cli_records_veto_then_independent_checks_and_cannot_repeat(tmp_path, monkeypatch, capsys):
    world, observer, session, raw = setup_probe(tmp_path, monkeypatch)
    output = tmp_path / "one-attempt"
    monkeypatch.setattr(probe, "OUTPUT", output)
    order = []
    monkeypatch.setattr(probe.search.uia, "read_records", lambda: order.append("records") or deepcopy(probe.EMPTY_RECORDS))
    monkeypatch.setattr(probe.search.uia, "signal_done", lambda: order.append("tones"))
    monkeypatch.setattr("builtins.input", lambda *args: "READY")
    monkeypatch.setattr(probe, "load_recognizer", lambda: object())
    monkeypatch.setattr(probe.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(probe, "WindowsWindowProbe", lambda: session.probe)
    capture_scope = probe.search.validate_capture_scope
    monkeypatch.setattr(probe.search, "validate_capture_scope", lambda bound, p, log:
                        capture_scope(bound, p, log, profile_reader=world.profile))
    monkeypatch.setattr(probe.search, "SearchObserver", lambda *args, **kwargs: observer)
    monkeypatch.setattr(probe, "ProbeSession", lambda *args, **kwargs: session)
    monkeypatch.setattr(probe.search.demo, "DesktopController", lambda: SimpleNamespace(_backend=raw))
    execute_actual = probe.execute_probe
    monkeypatch.setattr(probe, "execute_probe", lambda s, r, path:
                        execute_actual(s, r, path, wait=world.wait, clock=lambda: world.now))
    monkeypatch.setattr(probe.search, "OpenAICompatibleVisionClient", lambda *args, **kwargs: pytest.fail("API is forbidden"))
    assert probe.run_once(enabled=True) == 0
    summary = json.loads((output / "probe-summary.json").read_text(encoding="utf-8"))
    assert summary["probe_passed"] is True
    assert summary["runtime"]["status"] == "blocked"
    assert summary["runtime_exit_code"] == 1 and summary["entry_exit_code"] == 0
    assert summary["api_calls"] == summary["real_write_calls"] == 0
    assert summary["click_calls"] == summary["successful_clicks"] == summary["write_veto_calls"] == 1
    assert summary["mock_calls"] == 3
    assert summary["accepted_search_e2e_success"] is False
    assert summary["field_after"]["checked"] is True and summary["field_after"]["empty"] is True
    assert summary["records_unchanged"] is True
    assert order == ["records", "records", "tones"]
    assert world.query == "" and [c[0] for c in raw.calls] == ["click"]
    assert len(world.declarations) == 1
    assert (output / "attempt-claimed.json").exists()
    assert "labeled_by" not in capsys.readouterr().out
    preserved = (output / "probe-summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        probe.run_once(enabled=True)
    assert (output / "probe-summary.json").read_bytes() == preserved
    assert order == ["records", "records", "tones"]


def test_nonempty_service_stops_before_ocr_or_desktop(tmp_path, monkeypatch):
    output = tmp_path / "nonempty"
    monkeypatch.setattr(probe, "OUTPUT", output)
    monkeypatch.setattr(probe.search.uia, "read_records", lambda: {"searches": ["previous"], "messages": []})
    monkeypatch.setattr(probe.search.uia, "signal_done", lambda: None)
    monkeypatch.setattr(probe, "load_recognizer", lambda: pytest.fail("Must not load OCR"))
    monkeypatch.setattr(probe, "WindowsWindowProbe", lambda: pytest.fail("Must not touch desktop"))
    assert probe.run_once(enabled=True) == 1
    result = json.loads((output / "probe-summary.json").read_text(encoding="utf-8"))
    assert result["probe_passed"] is False and result["field_after"]["empty"] is None
    assert result["click_calls"] == result["write_veto_calls"] == result["mock_calls"] == 0
