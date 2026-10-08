"""Mock desktop/model regression tests; never invokes UIA, GLM or real input."""

from copy import deepcopy
from dataclasses import replace
import base64
import json
from types import SimpleNamespace

import pytest

from gui_agent.agent import GuiActionAgent, GuiPlanningAgent, TaskPlan
from gui_agent.capture import ScreenRegion
from gui_agent.grounding import (
    ActionDecision, ClickAction, GroundedAction, GroundingError, KeyAction, TypeAction,
    build_observation, resolve_target,
)
from gui_agent.models import MultimodalResponse
from gui_agent.perception import TextElement
from gui_agent.runtime import GuiRuntime, SceneObservation, WindowInfo
from scripts import week4_search_e2e as mode


FLAGS = ["--disable-sync", "--disable-extensions", "--disable-background-networking",
         "--disable-component-update", "--disable-quic", "--proxy-server=http://127.0.0.1:9",
         "--proxy-bypass-list=127.0.0.1", "--user-data-dir=" + str(mode.PROFILE)]


class Log:
    def __init__(self):
        self.events = []

    def emit(self, kind, **fields):
        self.events.append({"kind": kind, **deepcopy(fields)})


class World:
    def __init__(self):
        self.window = WindowInfo(123, 456, "Week4 Search - Microsoft Edge", (20, 30, 420, 230))
        self.foreign = False
        self.moved = False
        self.phase = 0
        self.query = ""
        self.focus = "body"
        self.captures = 0
        self.now = 0.0
        self.requests = []
        self.profile_bad = False
        self.profile_unavailable = False
        self.uia_override = None
        self.uia_error = False
        self.ocr_duplicate = False
        self.ocr_shift = 0
        self.ocr_confidence = 0.99
        self.uia_reads = 0
        self.declarations = []

    def wait(self, seconds):
        self.now += seconds

    def profile(self, pid):
        if self.profile_unavailable:
            raise TimeoutError("Unavailable")
        return {"pid": pid, "exe": str(mode.EDGE),
                "command_line": "bad" if self.profile_bad else "good"}

    def authorize(self, title, declaration, handle):
        self.declarations.append(json.loads(declaration))
        return True

    def uia(self, handle, *, expected_value, allowed_urls):
        self.uia_reads += 1
        if self.uia_error:
            raise TimeoutError("Unavailable")
        doc = [42, 10 if self.phase < 3 else 11]
        input_element = {
            "runtime_id": [42, 20], "name": "Search query", "control_type": "ControlType.Edit",
            "in_bound": True, "enabled": True, "offscreen": False, "value_pattern": True,
            "value_read_only": False, "value_empty": self.query == "",
            "value_matches_expected": self.query == expected_value,
            "has_keyboard_focus": self.focus in ("input", "address"),
            "labeled_by": {"name": "Search query ", "runtime_id": [42, 21],
                           "box": [25, 35, 185, 75], "in_bound": True},
            "ancestors": [{"runtime_id": doc}],
        }
        address = {**input_element, "runtime_id": [42, 30], "name": "Address bar",
                   "labeled_by": None, "ancestors": [], "has_keyboard_focus": self.focus == "address"}
        button = {"runtime_id": [42, 40], "name": "Search", "control_type": "ControlType.Button",
                  "in_bound": True, "enabled": True, "offscreen": False, "box": [25, 85, 130, 130],
                  "has_keyboard_focus": self.focus == "button", "ancestors": [{"runtime_id": doc}]}
        body = {"runtime_id": [42, 50], "control_type": "ControlType.Pane", "in_bound": True}
        focused = {"input": input_element, "address": address, "button": button, "body": body}[self.focus]
        data = {"focused_present": True, "focused_in_bound": True, "focus_stable_during_read": True,
                "focused": deepcopy(focused), "edits": [input_element, address], "buttons": [button],
                "documents": [{"runtime_id": doc, "name": "Week4 Search", "offscreen": False,
                               "url_allowed": (mode.RESULT_URL if self.phase >= 3 else mode.uia.URL) in allowed_urls}]}
        if self.uia_override:
            self.uia_override(data)
        return data


class Probe:
    def __init__(self, world):
        self.world = world

    def read(self, handle):
        return replace(self.world.window, box=(30, 30, 430, 230)) if self.world.moved else self.world.window

    def foreground(self):
        if self.world.foreign:
            return WindowInfo(999, 888, "Other app", (20, 30, 420, 230))
        return self.read(self.world.window.handle)


class Observer:
    def __init__(self, world, probe, tmp_path):
        self.world, self.probe, self.tmp_path = world, probe, tmp_path
        self.latest = None

    def observe(self):
        w = self.world
        w.captures += 1
        shift = w.ocr_shift
        elements = [TextElement("Search query", w.ocr_confidence, (10 + shift, 10, 150 + shift, 40)),
                    TextElement("Search", 0.99, (10, 60, 100, 90)),
                    TextElement("127.0.0.1:8765/search", 0.99, (10, 105, 300, 120))]
        if w.ocr_duplicate:
            elements.append(TextElement("Search query", 0.99, (10, 125, 150, 145)))
        if w.query:
            elements.append(TextElement(w.query, 0.99, (180, 10, 390, 40)))
        if w.phase >= 3:
            elements.extend([TextElement(mode.FINAL_TEXTS[0], 0.99, (10, 125, 200, 145)),
                             TextElement(mode.FINAL_TEXTS[1], 0.99, (10, 155, 390, 180))])
        observation = build_observation(f"obs-{w.captures}", self.tmp_path / f"obs-{w.captures}.png", elements,
                                        screen_region=ScreenRegion(0, 0, 800, 600), screen_size=(800, 600),
                                        image_size=(400, 200), crop_origin=(20, 30))
        self.latest = SceneObservation(observation, self.probe.read(123), self.probe.foreground(), float(w.captures))
        return self.latest

    def assert_current(self, scene):
        if scene.window != self.probe.read(123) or scene.foreground != self.probe.foreground():
            raise GroundingError("Window changed or lost focus")


class Backend:
    FAILSAFE = False

    def __init__(self, world):
        self.world = world
        self.calls = []

    def size(self):
        return 800, 600

    def click(self, **kwargs):
        self.calls.append(("click", kwargs))
        if kwargs["y"] < 85:
            self.world.phase = max(1, self.world.phase)
            self.world.focus = "input"
        else:
            self.world.phase = 3
            self.world.focus = "button"

    def write(self, text, **kwargs):
        self.calls.append(("write", text))
        self.world.query += text
        self.world.phase = 2

    def press(self, key):
        self.calls.append(("press", key))
        self.world.phase = 3


def plan():
    return TaskPlan(goal=mode.TASK, status="ready", steps=[
        {"step_id": "act", "description": "Perform the local search", "success_criteria": "New results visible"},
        {"step_id": "verify", "description": "Verify the results", "depends_on": ["act"], "success_criteria": "Expected result visible"},
    ])


def setup(tmp_path, monkeypatch, *, approve=True, enabled=True):
    world = World()
    monkeypatch.setattr(mode, "windows_arguments", lambda command: FLAGS if command == "good" else FLAGS[:-1] + ["--user-data-dir=C:\\personal"])
    probe = Probe(world)
    observer = Observer(world, probe, tmp_path)
    initial = observer.observe()
    log = Log()
    session = mode.SearchSession(mode.Scope(world.window, mode.full_window_checks(initial)), observer, probe,
                                 log, tmp_path / "STOP", enabled=enabled, profile_reader=world.profile,
                                 uia_reader=world.uia, human_authorize=world.authorize, clock=lambda: world.now)
    backend = Backend(world)
    controller = mode.SearchController(backend, session)
    if enabled:
        session.start()
        if approve:
            assert session.confirm_plan(plan(), session.scope.checks) is True
    return world, observer, session, backend, controller


def authorize(session, observer, action):
    scene = observer.latest
    if isinstance(action, KeyAction):
        point = None
    else:
        point = resolve_target(scene.observation, action.target_id, action.target_text)
    decision = ActionDecision(status="action", step_id="act", observation_id=scene.observation.observation_id, action=action)
    grounded = GroundedAction(action, point)
    assert session.confirm_action(decision, grounded) is True
    return grounded


def click_input(observer, session, controller):
    target, _ = mode.ocr_target(observer.latest, "Search query")
    grounded = authorize(session, observer, ClickAction(type="click", target_id=target.target_id, target_text="Search query"))
    observer.observe()
    controller.click(grounded.point)
    observer.observe()
    return grounded


def prepare_input(observer, session, controller):
    clicked = click_input(observer, session, controller)
    target, _ = mode.ocr_target(observer.latest, "Search query")
    authorize(session, observer, TypeAction(type="type", target_id=target.target_id, target_text="Search query", text=mode.TOKEN, mode="append"))
    observer.observe()
    assert session.confirm_input_focus(clicked, observer.latest) is True
    observer.observe()
    return clicked


def test_two_independent_focus_checks_before_mock_write(tmp_path, monkeypatch):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch)
    prepare_input(observer, session, controller)
    assert not any(c[0] == "write" for c in backend.calls)
    controller.type_text(mode.TOKEN)
    assert backend.calls[-1] == ("write", mode.TOKEN)
    assert [e["phase"] for e in session.log.events if e["kind"] == "focus_check"] == ["focus_callback", "write_entry"]
    assert len(world.declarations) == 1
    assert backend.FAILSAFE is True


@pytest.mark.parametrize("focus", ["address", "button", "body"])
def test_focus_transfer_after_earlier_confirmation_blocks_last_write(tmp_path, monkeypatch, focus):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch)
    prepare_input(observer, session, controller)
    world.focus = focus
    with pytest.raises(GroundingError):
        controller.type_text(mode.TOKEN)
    assert not any(c[0] == "write" for c in backend.calls)
    assert world.query == ""
    assert session.log.events[-1]["kind"] == "focus_check"
    assert session.log.events[-1]["passed"] is False


@pytest.mark.parametrize("fault", ["foreign", "moved", "profile_bad", "profile_unavailable", "uia_error"])
def test_final_window_profile_or_evidence_failure_never_calls_write(tmp_path, monkeypatch, fault):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch)
    prepare_input(observer, session, controller)
    setattr(world, fault, True)
    with pytest.raises(GroundingError):
        controller.type_text(mode.TOKEN)
    assert not any(c[0] == "write" for c in backend.calls)


@pytest.mark.parametrize("fault", ["missing_focus", "missing_label", "duplicate_link", "unmatched_identity", "unknown_value", "external_url", "unstable", "read_only"])
def test_missing_ambiguous_or_mismatched_uia_evidence_blocks_write(tmp_path, monkeypatch, fault):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch)
    prepare_input(observer, session, controller)

    def change(data):
        if fault == "missing_focus":
            data["focused"] = None
        elif fault == "missing_label":
            data["focused"]["labeled_by"] = None
        elif fault == "duplicate_link":
            data["edits"].append(deepcopy(data["edits"][0]))
        elif fault == "unmatched_identity":
            data["focused"]["runtime_id"] = [42, 999]
        elif fault == "unknown_value":
            del data["focused"]["value_matches_expected"]
        elif fault == "external_url":
            data["documents"][0]["url_allowed"] = False
        elif fault == "unstable":
            data["focus_stable_during_read"] = False
        else:
            data["focused"]["value_read_only"] = True

    world.uia_override = change
    with pytest.raises(GroundingError):
        controller.type_text(mode.TOKEN)
    assert not any(c[0] == "write" for c in backend.calls)


@pytest.mark.parametrize("fault", ["ocr_duplicate", "ocr_shift", "ocr_confidence"])
def test_latest_ocr_still_requires_unique_unmoved_confident_target(tmp_path, monkeypatch, fault):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch)
    prepare_input(observer, session, controller)
    setattr(world, fault, {"ocr_duplicate": True, "ocr_shift": 2, "ocr_confidence": 0.49}[fault])
    observer.observe()
    with pytest.raises(GroundingError):
        controller.type_text(mode.TOKEN)
    assert not any(c[0] == "write" for c in backend.calls)


def test_type_candidate_cannot_be_substituted_or_faked(tmp_path, monkeypatch):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch)
    clicked = prepare_input(observer, session, controller)
    fake = GroundedAction(clicked.action, (clicked.point[0] + 1, clicked.point[1]))
    with pytest.raises(GroundingError):
        session.confirm_input_focus(fake, observer.latest)
    assert not any(c[0] == "write" for c in backend.calls)


def test_no_grant_is_not_automatic_authorization(tmp_path, monkeypatch):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch, approve=False)
    with pytest.raises(GroundingError):
        click_input(observer, session, controller)
    with pytest.raises(GroundingError):
        controller.type_text(mode.TOKEN)
    assert backend.calls == []


def test_mode_is_disabled_by_default(tmp_path, monkeypatch):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch, enabled=False)
    with pytest.raises(GroundingError):
        session.start()
    with pytest.raises(GroundingError):
        controller.type_text(mode.TOKEN)
    assert backend.calls == []


@pytest.mark.parametrize("text,mode_name", [("arbitrary", "append"), (mode.TOKEN, "replace")])
def test_only_frozen_word_append_is_authorized(tmp_path, monkeypatch, text, mode_name):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch)
    click_input(observer, session, controller)
    target, _ = mode.ocr_target(observer.latest, "Search query")
    with pytest.raises(GroundingError):
        authorize(session, observer, TypeAction(type="type", target_id=target.target_id,
                                               target_text="Search query", text=text, mode=mode_name))
    assert world.query == ""


def test_backend_content_cannot_differ_from_authorized_model_text(tmp_path, monkeypatch):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch)
    prepare_input(observer, session, controller)
    with pytest.raises(GroundingError):
        controller.type_text("other word")
    assert not any(c[0] == "write" for c in backend.calls)


def test_stop_and_original_time_budget_at_last_entry(tmp_path, monkeypatch):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch)
    prepare_input(observer, session, controller)
    session.stop_file.touch()
    with pytest.raises(GroundingError):
        controller.type_text(mode.TOKEN)
    assert not any(c[0] == "write" for c in backend.calls)
    session.stop_file.unlink()
    world.now = 300
    with pytest.raises(GroundingError):
        controller.type_text(mode.TOKEN)
    with pytest.raises(GroundingError):
        session.start()


def test_clipboard_and_navigation_hotkeys_never_reach_backend(tmp_path, monkeypatch):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch)
    with pytest.raises(GroundingError):
        controller.paste_text("anything")
    with pytest.raises(GroundingError):
        controller.hotkey("ctrl", "l")
    with pytest.raises(GroundingError):
        controller._backend.FAILSAFE = False
    assert backend.calls == []


@pytest.mark.parametrize("key", ["enter", "esc", "ctrl+l", "ctrl+o", "alt+f4"])
def test_keys_before_actual_input_are_outside_scope(tmp_path, monkeypatch, key):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch)
    with pytest.raises(GroundingError):
        authorize(session, observer, KeyAction(type="key", key=key, purpose="untrusted model purpose"))
    assert backend.calls == []


def test_early_submit_click_is_rejected(tmp_path, monkeypatch):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch)
    target, _ = mode.ocr_target(observer.latest, "Search")
    with pytest.raises(GroundingError):
        authorize(session, observer, ClickAction(type="click", target_id=target.target_id, target_text="Search"))
    assert backend.calls == []


@pytest.mark.parametrize("method", ["button", "enter"])
def test_submission_requires_actual_fixed_value_and_only_once(tmp_path, monkeypatch, method):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch)
    prepare_input(observer, session, controller)
    controller.type_text(mode.TOKEN)
    observer.observe()
    if method == "enter":
        authorize(session, observer, KeyAction(type="key", key="enter", purpose="Submit local search"))
        observer.observe()
        controller.press("enter")
    else:
        target, _ = mode.ocr_target(observer.latest, "Search")
        grounded = authorize(session, observer, ClickAction(type="click", target_id=target.target_id, target_text="Search"))
        observer.observe()
        controller.click(grounded.point)
    assert session.submitted is True
    assert len(backend.calls) == 3
    observer.observe()
    with pytest.raises(GroundingError):
        authorize(session, observer, KeyAction(type="key", key="enter", purpose="Duplicate submit"))


def test_submit_rejects_changed_value_or_address_bar_focus(tmp_path, monkeypatch):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch)
    prepare_input(observer, session, controller)
    controller.type_text(mode.TOKEN)
    observer.observe()
    authorize(session, observer, KeyAction(type="key", key="enter", purpose="Submit local search"))
    observer.observe()
    world.query = "different"
    world.focus = "address"
    with pytest.raises(GroundingError):
        controller.press("enter")
    assert not any(c[0] == "press" for c in backend.calls)


class Client:
    def __init__(self, world, session, *, final_fault=False):
        self.world, self.session, self.final_fault = world, session, final_fault

    def generate(self, request):
        w = self.world
        w.requests.append(request)
        if request.system_prompt.startswith("You are a GUI task planning"):
            return MultimodalResponse(text=plan().model_dump_json(), provider="mock", model="mock")
        step = request.instruction.split("Current step_id: ", 1)[1].splitlines()[0]
        context = json.loads(request.instruction.split("Current observation: ", 1)[1].splitlines()[0])
        data = {"step_id": step, "observation_id": context["observation_id"]}
        if w.phase >= 3:
            data.update(status="step_complete", checks=[c.model_dump() for c in self.session.scope.checks])
        elif w.phase == 2:
            data.update(status="action", action={"type": "key", "key": "enter", "purpose": "Submit local search"})
        else:
            target = next(t for t in context["elements"] if t["text"] == "Search query")
            action = {"type": "click" if w.phase == 0 else "type", "target_id": target["target_id"], "target_text": target["text"]}
            if w.phase == 1:
                action.update(text=mode.TOKEN, mode="append")
                if self.final_fault:
                    original = self.session.confirm_input_focus

                    def valid_callback_then_transfer(clicked, scene):
                        answer = original(clicked, scene)
                        w.focus = "address"
                        return answer

                    self.session.confirm_input_focus = valid_callback_then_transfer
            data.update(status="action", action=action)
        return MultimodalResponse(text=json.dumps(data), provider="mock", model="mock")


def test_runtime_integration_without_execution_time_dialogs_or_uia_model_context(tmp_path, monkeypatch):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch, approve=False)
    recorder = mode.ScopedRecorder(Client(world, session), tmp_path, session)
    runtime = GuiRuntime(GuiPlanningAgent(recorder), GuiActionAgent(recorder), observer, controller,
                         limits=mode.LIMITS, wait=world.wait, clock=lambda: world.now)
    result = runtime.run(mode.TASK, session.scope.checks, execute=True, confirm_plan=session.confirm_plan,
                         confirm_action=session.confirm_action, supervised_input_focus=True,
                         confirm_input_focus=session.confirm_input_focus)
    assert result.status == "completed"
    assert result.executed_actions == 3
    assert len(world.declarations) == 1
    assert [c[0] for c in backend.calls] == ["click", "write", "press"]
    declaration = world.declarations[0]
    assert declaration["task"] == mode.TASK
    assert declaration["fixed_search_word"] == mode.TOKEN
    assert declaration["budgets"]["max_actions"] == 20
    assert declaration["budgets"]["timeout_seconds"] == 300
    assert declaration["final_checks"] == [c.model_dump(mode="json") for c in session.scope.checks]
    assert declaration["window"] == {"handle": 123, "pid": 456, "title": world.window.title, "box": [20, 30, 420, 230]}
    for request in world.requests:
        assert "labeled_by" not in request.instruction
        assert "value_matches_expected" not in request.instruction
        assert "uia_readonly_probe" not in request.instruction
    assert all(e["selection_verified"] is False for e in result.events if "selection_verified" in e)
    assert mode.accepted_result(result, session, {"searches": [], "messages": []},
                                {"searches": [mode.TOKEN], "messages": []}) is True


def test_runtime_stops_without_another_model_call_if_last_write_check_fails(tmp_path, monkeypatch):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch, approve=False)
    client = Client(world, session)
    recorder = mode.ScopedRecorder(client, tmp_path, session)
    original = session.focused_input

    def fail_only_last_entry(scene, expected, phase):
        if phase == "write_entry":
            world.focus = "address"
        return original(scene, expected, phase)

    session.focused_input = fail_only_last_entry
    runtime = GuiRuntime(GuiPlanningAgent(recorder), GuiActionAgent(recorder), observer, controller,
                         limits=mode.LIMITS, wait=world.wait, clock=lambda: world.now)
    result = runtime.run(mode.TASK, session.scope.checks, execute=True, confirm_plan=session.confirm_plan,
                         confirm_action=session.confirm_action, supervised_input_focus=True,
                         confirm_input_focus=session.confirm_input_focus)
    assert result.status == "blocked"
    assert result.executed_actions == 1
    assert recorder.calls == 3
    assert [c[0] for c in backend.calls] == ["click"]
    assert world.query == ""


def test_default_runtime_still_blocks_text_even_with_controlled_controller(tmp_path, monkeypatch):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch, approve=False)
    recorder = mode.ScopedRecorder(Client(world, session), tmp_path, session)
    runtime = GuiRuntime(GuiPlanningAgent(recorder), GuiActionAgent(recorder), observer, controller,
                         limits=mode.LIMITS, wait=world.wait, clock=lambda: world.now)
    result = runtime.run(mode.TASK, session.scope.checks, execute=True, confirm_plan=session.confirm_plan,
                         confirm_action=session.confirm_action)
    assert result.status == "blocked"
    assert "Strict mode" in result.reason
    assert [c[0] for c in backend.calls] == ["click"]


def test_cli_without_all_explicit_flags_does_not_prepare_or_execute(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Must not touch API, UIA or services without explicit flags")

    monkeypatch.setattr(mode.uia, "read_records", forbidden)
    monkeypatch.setattr(mode, "OpenAICompatibleVisionClient", forbidden)
    for missing in ("controlled_local_search", "execute", "allow_screenshot_upload", "allow_keyboard_input", "allow_search_submit"):
        args = SimpleNamespace(controlled_local_search=True, execute=True, allow_screenshot_upload=True,
                               allow_keyboard_input=True, allow_search_submit=True,
                               output_dir=tmp_path / missing)
        setattr(args, missing, False)
        with pytest.raises(GroundingError):
            mode.run_once(args)
        assert not args.output_dir.exists()


@pytest.mark.parametrize("answer", [False, None, 1, "yes"])
def test_scope_grant_requires_literal_human_true(tmp_path, monkeypatch, answer):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch, approve=False)
    session.human_authorize = lambda *args: answer
    assert session.confirm_plan(plan(), session.scope.checks) is False
    with pytest.raises(GroundingError):
        click_input(observer, session, controller)
    assert backend.calls == []


def install_offline_entry(tmp_path, monkeypatch):
    """Exercise the live entry with every desktop/network/UIA/model boundary replaced."""
    world = World()
    probe = Probe(world)
    raw = Backend(world)
    holder = {"world": world, "raw": raw, "service_reads": 0, "api_constructors": []}
    monkeypatch.setattr(mode, "windows_arguments", lambda command: FLAGS if command == "good" else [])
    monkeypatch.setattr(mode, "load_recognizer", lambda: object())
    monkeypatch.setattr(mode.time, "sleep", world.wait)
    monkeypatch.setattr(mode, "WindowsWindowProbe", lambda: probe)
    original_capture_check = mode.validate_capture_scope
    monkeypatch.setattr(mode, "validate_capture_scope", lambda bound, p, log:
                        original_capture_check(bound, p, log, profile_reader=world.profile))
    monkeypatch.setattr(mode, "SearchObserver", lambda bound, output, **kwargs: Observer(world, probe, output))
    original_session = mode.SearchSession

    def create_session(*args, **kwargs):
        session = original_session(*args, **kwargs, profile_reader=world.profile, uia_reader=world.uia,
                                   human_authorize=world.authorize, clock=lambda: world.now)
        holder["session"] = session
        return session

    monkeypatch.setattr(mode, "SearchSession", create_session)
    monkeypatch.setattr(mode.demo, "DesktopController", lambda: SimpleNamespace(_backend=raw))

    class OfflineRuntime(GuiRuntime):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, wait=world.wait, clock=lambda: world.now)

    monkeypatch.setattr(mode, "GuiRuntime", OfflineRuntime)

    def model_constructor(*args, **kwargs):
        holder["api_constructors"].append((args, kwargs))
        return Client(world, holder["session"])

    def records():
        holder["service_reads"] += 1
        return {"searches": [mode.TOKEN] if world.phase >= 3 else [], "messages": []}

    monkeypatch.setattr(mode, "OpenAICompatibleVisionClient", model_constructor)
    monkeypatch.setattr(mode.uia, "read_records", records)
    monkeypatch.setattr(mode.uia, "signal_done", lambda: None)
    monkeypatch.setattr("builtins.input", lambda *args: "READY")
    monkeypatch.setenv("GUI_AGENT_API_BASE", "http://invalid.test/mock-only")
    monkeypatch.setenv("GUI_AGENT_API_KEY", "fake-offline-secret")
    monkeypatch.setenv("GUI_AGENT_API_MODEL", mode.MODEL)
    holder["args"] = SimpleNamespace(controlled_local_search=True, execute=True, allow_screenshot_upload=True,
                                     allow_keyboard_input=True, allow_search_submit=True, output_dir=tmp_path / "result")
    return holder


def test_complete_live_entry_offline_preserves_evidence_and_exit_codes(tmp_path, monkeypatch):
    case = install_offline_entry(tmp_path, monkeypatch)
    assert mode.run_once(case["args"]) == 0
    output = case["args"].output_dir
    summary = json.loads((output / "run-summary.json").read_text(encoding="utf-8"))
    assert summary["runtime"]["status"] == "completed"
    assert summary["runtime_exit_code"] == summary["entry_exit_code"] == 0
    assert summary["accepted_success"] is True
    assert summary["real_input_returned"] is summary["real_submit_returned"] is True
    assert summary["scope_granted"] is True
    assert summary["selection_verified"] is summary["os_level_zero_egress_guarantee"] is False
    assert case["api_constructors"][0][0][0] == "glm-4v-flash"
    assert case["api_constructors"][0][1]["max_retries"] == 0
    assert case["service_reads"] == 2
    assert [call[0] for call in case["raw"].calls] == ["click", "write", "press"]
    events = [json.loads(line) for line in (output / "safety.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["phase"] for e in events if e["kind"] == "focus_check"] == ["focus_callback", "write_entry", "submit_entry"]
    assert len(case["world"].declarations) == 1
    assert case["world"].declarations[0]["residual_focus_risk"] == mode.FOCUS_RISK
    assert json.loads((output / "service-after.json").read_text()) == {"searches": [mode.TOKEN], "messages": []}
    for name in ("attempt-claimed.json", "source-hashes.json", "service-before.json", "initial-empty-input.json",
                 "final-checks.json", "bound-window.json", "run-console.txt", "model-01.json"):
        assert (output / name).is_file()
    for file in output.glob("model-*.json"):
        data = json.loads(file.read_text(encoding="utf-8"))
        assert data["response_before_parsing"]["provider"] == "mock"
        for forbidden in ("labeled_by", "value_matches_expected", "uia_readonly_probe"):
            assert forbidden not in data["request"]["instruction"]
    assert "fake-offline-secret" not in (output / "run-console.txt").read_text()


@pytest.mark.parametrize("fault", ["nonempty_records", "consent_denied", "profile", "window", "duplicate_ocr", "nonempty_input", "api_config"])
def test_entry_preparation_failure_never_reaches_model_or_input(tmp_path, monkeypatch, fault):
    case = install_offline_entry(tmp_path, monkeypatch)
    world = case["world"]
    if fault == "nonempty_records":
        monkeypatch.setattr(mode.uia, "read_records", lambda: {"searches": ["prior"], "messages": []})
    elif fault == "consent_denied":
        monkeypatch.setattr("builtins.input", lambda *args: "NO")
    elif fault == "profile":
        world.profile_bad = True
    elif fault == "window":
        world.foreign = True
    elif fault == "duplicate_ocr":
        world.ocr_duplicate = True
    elif fault == "nonempty_input":
        world.query = "other"
    else:
        monkeypatch.delenv("GUI_AGENT_API_KEY")
    assert mode.run_once(case["args"]) == 1
    summary = json.loads((case["args"].output_dir / "run-summary.json").read_text())
    assert summary["accepted_success"] is False
    assert summary["runtime"] is None and summary["model_calls"] == 0
    assert summary["entry_exit_code"] == 1
    assert case["api_constructors"] == case["raw"].calls == []


def test_existing_once_directory_rejects_before_services_or_models(tmp_path, monkeypatch):
    case = install_offline_entry(tmp_path, monkeypatch)
    output = case["args"].output_dir
    output.mkdir()
    marker = output / "original.json"
    marker.write_text('{"keep": true}', encoding="utf-8")
    with pytest.raises(FileExistsError):
        mode.run_once(case["args"])
    assert marker.read_text() == '{"keep": true}'
    assert case["service_reads"] == 0
    assert case["api_constructors"] == case["raw"].calls == []


@pytest.mark.parametrize("fault", ["wrong_record", "duplicate_record", "unavailable"])
def test_entry_completed_but_independent_records_failure_is_not_success(tmp_path, monkeypatch, fault):
    case = install_offline_entry(tmp_path, monkeypatch)
    reads = []

    def records():
        reads.append(True)
        if len(reads) == 1:
            return {"searches": [], "messages": []}
        if fault == "unavailable":
            raise TimeoutError("Mock independent service unavailable")
        return {"searches": ["wrong"] if fault == "wrong_record" else [mode.TOKEN, mode.TOKEN], "messages": []}

    monkeypatch.setattr(mode.uia, "read_records", records)
    assert mode.run_once(case["args"]) == 1
    summary = json.loads((case["args"].output_dir / "run-summary.json").read_text())
    assert summary["runtime"]["status"] == "completed"
    assert summary["runtime_exit_code"] == 0 and summary["entry_exit_code"] == 1
    assert summary["accepted_success"] is False
    assert len(reads) == 2
    assert [call[0] for call in case["raw"].calls] == ["click", "write", "press"]


def test_entry_plan_refusal_stops_after_planning_without_action_calls(tmp_path, monkeypatch):
    case = install_offline_entry(tmp_path, monkeypatch)
    case["world"].authorize = lambda *args: False
    assert mode.run_once(case["args"]) == 1
    summary = json.loads((case["args"].output_dir / "run-summary.json").read_text())
    assert summary["runtime"]["status"] == "cancelled"
    assert summary["model_calls"] == 1 and summary["scope_granted"] is False
    assert case["raw"].calls == []
    assert summary["records_before"] == summary["records_after"] == {"searches": [], "messages": []}


def test_entry_premature_completion_bad_evidence_stops_without_real_actions(tmp_path, monkeypatch):
    case = install_offline_entry(tmp_path, monkeypatch)

    class PrematureClient(Client):
        def generate(self, request):
            if request.system_prompt.startswith("You are a GUI task planning"):
                return super().generate(request)
            self.world.requests.append(request)
            step = request.instruction.split("Current step_id: ", 1)[1].splitlines()[0]
            context = json.loads(request.instruction.split("Current observation: ", 1)[1].splitlines()[0])
            return MultimodalResponse(text=json.dumps({
                "status": "step_complete", "step_id": step, "observation_id": context["observation_id"],
                "checks": [{"type": "text_present", "text": "Missing result", "region": [0, 0, 400, 200]}],
            }), provider="mock", model="mock")

    monkeypatch.setattr(mode, "OpenAICompatibleVisionClient", lambda *args, **kwargs:
                        PrematureClient(case["world"], case["session"]))
    assert mode.run_once(case["args"]) == 1
    summary = json.loads((case["args"].output_dir / "run-summary.json").read_text())
    assert summary["runtime"]["status"] == "inconclusive"
    assert summary["model_calls"] == 2
    assert summary["runtime"]["executed_actions"] == 0 and case["raw"].calls == []
    assert summary["accepted_success"] is False


def test_fixed_once_cli_has_no_output_override(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(mode, "run_once", lambda args: calls.append(args) or 0)
    command = ["search", "--run-one-e2e", "--execute", "--allow-screenshot-upload",
               "--allow-keyboard-input", "--allow-search-submit"]
    monkeypatch.setattr(mode.sys, "argv", command)
    assert mode.main() == 0
    assert calls[0].output_dir == mode.ONCE_OUTPUT
    assert calls[0].controlled_local_search is True
    monkeypatch.setattr(mode.sys, "argv", command + ["--output-dir", str(tmp_path / "other")])
    with pytest.raises(SystemExit):
        mode.main()
    assert len(calls) == 1


@pytest.mark.parametrize("fault", ["status", "no_write", "no_submit", "no_actions", "prior_record", "missing_record", "wrong_word", "duplicate_record", "message_record"])
def test_success_requires_completed_real_actions_and_exact_independent_records(fault):
    result = SimpleNamespace(status="completed", executed_actions=3)
    session = SimpleNamespace(typed=True, submitted=True)
    before = {"searches": [], "messages": []}
    after = {"searches": [mode.TOKEN], "messages": []}
    if fault == "status":
        result.status = "inconclusive"
    elif fault == "no_write":
        session.typed = False
    elif fault == "no_submit":
        session.submitted = False
    elif fault == "no_actions":
        result.executed_actions = 0
    elif fault == "prior_record":
        before["searches"] = [mode.TOKEN]
    elif fault == "missing_record":
        after["searches"] = []
    elif fault == "wrong_word":
        after["searches"] = ["other"]
    elif fault == "duplicate_record":
        after["searches"] *= 2
    else:
        after["messages"] = ["unexpected"]
    assert mode.accepted_result(result, session, before, after) is False


def test_checks_and_step_scope_cannot_be_replaced(tmp_path, monkeypatch):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch, approve=False)
    with pytest.raises(GroundingError):
        session.confirm_plan(plan(), session.scope.checks[:1])
    assert world.declarations == []
    assert session.confirm_plan(plan(), session.scope.checks) is True
    with pytest.raises(GroundingError):
        session.confirm_plan(plan(), session.scope.checks)
    target, point = mode.ocr_target(observer.latest, "Search query")
    action = ClickAction(type="click", target_id=target.target_id, target_text="Search query")
    decision = ActionDecision(status="action", step_id="not-approved", observation_id=observer.latest.observation.observation_id, action=action)
    with pytest.raises(GroundingError):
        session.confirm_action(decision, GroundedAction(action, point))
    assert backend.calls == []


@pytest.mark.parametrize("fault", ["wrong_text", "wrong_point", "missing_latest"])
def test_target_or_latest_observation_outside_scope_is_blocked(tmp_path, monkeypatch, fault):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch)
    target, point = mode.ocr_target(observer.latest, "Search query")
    action = ClickAction(type="click", target_id=target.target_id,
                         target_text="Address bar" if fault == "wrong_text" else "Search query")
    decision = ActionDecision(status="action", step_id="act", observation_id=observer.latest.observation.observation_id, action=action)
    if fault == "wrong_point":
        point = (point[0] + 1, point[1])
    if fault == "missing_latest":
        observer.latest = None
    with pytest.raises(GroundingError):
        session.confirm_action(decision, GroundedAction(action, point))
    assert backend.calls == []


def test_earlier_callback_cannot_be_skipped(tmp_path, monkeypatch):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch)
    click_input(observer, session, controller)
    target, _ = mode.ocr_target(observer.latest, "Search query")
    authorize(session, observer, TypeAction(type="type", target_id=target.target_id,
                                          target_text="Search query", text=mode.TOKEN, mode="append"))
    observer.observe()
    with pytest.raises(GroundingError):
        controller.type_text(mode.TOKEN)
    assert not any(c[0] == "write" for c in backend.calls)


@pytest.mark.parametrize("key", ["documents", "edits", "buttons"])
def test_malformed_uia_collections_fail_closed(tmp_path, monkeypatch, key):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch)
    prepare_input(observer, session, controller)
    world.uia_override = lambda data: data.update({key: None})
    with pytest.raises(GroundingError):
        controller.type_text(mode.TOKEN)
    assert not any(c[0] == "write" for c in backend.calls)


@pytest.mark.parametrize("fault", ["duplicate", "outside", "missing"])
def test_submit_button_requires_unique_current_geometric_association(tmp_path, monkeypatch, fault):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch)
    prepare_input(observer, session, controller)
    controller.type_text(mode.TOKEN)
    observer.observe()
    target, _ = mode.ocr_target(observer.latest, "Search")
    grounded = authorize(session, observer, ClickAction(type="click", target_id=target.target_id, target_text="Search"))
    observer.observe()

    def change(data):
        if fault == "duplicate":
            data["buttons"].append(deepcopy(data["buttons"][0]))
        elif fault == "outside":
            data["buttons"][0]["box"] = [300, 300, 350, 350]
        else:
            data["buttons"] = []

    world.uia_override = change
    with pytest.raises(GroundingError):
        controller.click(grounded.point)
    assert session.submitted is False
    assert [c[0] for c in backend.calls] == ["click", "write"]


def test_original_runtime_action_budget_still_blocks(tmp_path, monkeypatch):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch, approve=False)
    recorder = mode.ScopedRecorder(Client(world, session), tmp_path, session)
    limits = replace(mode.LIMITS, max_actions=1)
    runtime = GuiRuntime(GuiPlanningAgent(recorder), GuiActionAgent(recorder), observer, controller,
                         limits=limits, wait=world.wait, clock=lambda: world.now)
    result = runtime.run(mode.TASK, session.scope.checks, execute=True, confirm_plan=session.confirm_plan,
                         confirm_action=session.confirm_action, supervised_input_focus=True,
                         confirm_input_focus=session.confirm_input_focus)
    assert result.status == "blocked"
    assert result.executed_actions == 1
    assert [c[0] for c in backend.calls] == ["click"]


def test_uia_worker_parameters_are_encoded_and_readonly_without_sampling(monkeypatch):
    commands = []

    def fake_worker(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout='{"mock": true}', stderr="")

    monkeypatch.setattr(mode.uia.subprocess, "run", fake_worker)
    assert mode.uia.read_uia(123, expected_value=mode.TOKEN, allowed_urls=(mode.uia.URL, mode.RESULT_URL)) == {"mock": True}
    source = base64.b64decode(commands[0][-1]).decode("utf-16-le")
    assert "__HANDLE__" not in source and "__EXPECTED_VALUE__" not in source and "__ALLOWED_URLS__" not in source
    assert "FromHandle([IntPtr]123)" in source
    assert base64.b64encode(mode.TOKEN.encode()).decode() in source
    for forbidden in (".SetValue(", ".SetFocus(", ".Invoke(", "SendKeys", "mouse_event", "keybd_event"):
        assert forbidden not in source


def test_allow_nonempty_diagnostic_assessment_does_not_grant_input(tmp_path, monkeypatch):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch)
    prepare_input(observer, session, controller)
    controller.type_text(mode.TOKEN)
    scene = observer.observe()
    data = session.inspect(scene, mode.TOKEN, "offline_test")
    assert mode.uia.classify(data)["uia_capability"] == "unproven"
    assessment = mode.uia.classify(data, require_empty=False)
    assert assessment["uia_capability"] == "explicit_label_link_observed"
    assert assessment["input_permission"] is False


@pytest.mark.parametrize("fault", ["none", "profile_bad", "profile_unavailable", "foreign", "moved", "valid"])
def test_profile_identity_and_current_window_checked_before_capture(tmp_path, monkeypatch, fault):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch)
    previous_captures = world.captures
    if fault == "valid":
        mode.validate_capture_scope(world.window, session.probe, session.log, profile_reader=world.profile)
        assert session.log.events[-1]["kind"] == "initial_capture_scope"
        assert session.log.events[-1]["passed"] is True
    else:
        if fault != "none":
            setattr(world, fault, True)
        with pytest.raises(GroundingError):
            mode.validate_capture_scope(None if fault == "none" else world.window, session.probe,
                                        session.log, profile_reader=world.profile)
    assert world.captures == previous_captures
    assert backend.calls == []


@pytest.mark.parametrize("size", [(1064, 1020), (1274, 1124), (800, 600)])
def test_initial_actual_crop_freezes_full_window_checks(tmp_path, size):
    w, h = size
    bound = WindowInfo(123, 456, "Week4 Search - Microsoft Edge", (20, 30, 20 + w, 30 + h))
    obs = build_observation("dynamic", tmp_path / "dynamic.png", [],
                            screen_region=ScreenRegion(0, 0, 2560, 1440), screen_size=(2560, 1440),
                            image_size=size, crop_origin=(20, 30))
    scene = SceneObservation(obs, bound, bound, 1.0)
    checks = mode.full_window_checks(scene)
    assert [c.type for c in checks] == ["text_present", "text_present"]
    assert tuple(c.text for c in checks) == mode.FINAL_TEXTS
    assert all(c.region == (0, 0, w, h) for c in checks)
    scope = mode.Scope(bound, checks)
    changed = replace(bound, box=(20, 30, 20 + w - 1, 30 + h))
    assert scope.window != changed
    assert scope.checks == checks and scope.checks[0].region == (0, 0, w, h)


@pytest.mark.parametrize("fault", ["origin", "size", "allowed_box", "unfocused"])
def test_initial_region_binding_rejects_inconsistent_geometry(tmp_path, monkeypatch, fault):
    _, observer, _, _, _ = setup(tmp_path, monkeypatch)
    scene = observer.latest
    if fault == "unfocused":
        scene = replace(scene, foreground=None)
    else:
        changes = {"origin": {"crop_origin": (21, 30)}, "size": {"image_size": (399, 200), "allowed_box": (0, 0, 399, 200)},
                   "allowed_box": {"allowed_box": (0, 0, 399, 200)}}[fault]
        scene = replace(scene, observation=replace(scene.observation, **changes))
    with pytest.raises(GroundingError):
        mode.full_window_checks(scene)


def test_frozen_checks_unchanged_and_resize_blocks_before_next_write(tmp_path, monkeypatch):
    world, observer, session, backend, controller = setup(tmp_path, monkeypatch)
    frozen = session.scope.checks
    prepare_input(observer, session, controller)
    world.moved = True
    with pytest.raises(GroundingError):
        controller.type_text(mode.TOKEN)
    assert session.scope.checks == frozen
    assert all(c.region == (0, 0, 400, 200) for c in frozen)
    assert not any(call[0] == "write" for call in backend.calls)
