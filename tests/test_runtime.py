from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
from pathlib import Path

import numpy as np
import pyautogui
import pytest

import gui_agent.runtime as runtime_module
from gui_agent.agent import GuiActionAgent, GuiPlanningAgent, TaskPlan
from gui_agent.capture import ScreenFrame, ScreenRegion
from gui_agent.control import DesktopController
from gui_agent.grounding import FeedbackCheck
from gui_agent.models import ModelInvocationError, MultimodalRequest, MultimodalResponse
from gui_agent.perception import TextElement
from gui_agent.runtime import DesktopObserver, GuiRuntime, RunLimits, WindowInfo, WindowsWindowProbe
from scripts import week4_demo


class World:
    def __init__(self, token: str = "sample") -> None:
        self.token = token
        self.phase = 0
        self.now = 0.0
        self.captures = 0
        self.closed = False
        self.foreign = False
        self.moved_window = False
        self.reused_handle = False
        self.moved_target = False
        self.missing_target = False
        self.duplicate = False
        self.low_confidence = False
        self.resized = False
        self.offset_monitor = False
        self.stuck = False
        self.events: list[tuple[object, ...]] = []
        self.window = WindowInfo(123, 456, "Controlled test", (20, 30, 420, 230))
        self.requests: list[MultimodalRequest] = []
        self.actions: list[tuple[object, ...]] = []

    @property
    def text(self) -> str:
        return f"{('Start', 'Continue', 'Finished')[min(self.phase, 2)]} {self.token}"

    def capture(self) -> ScreenFrame:
        self.events.append(("capture", self.phase))
        self.captures += 1
        width = 900 if self.resized else 800
        image = np.full((600, width, 3), self.phase, dtype=np.uint8)
        return ScreenFrame(image, ScreenRegion(100 if self.offset_monitor else 0, 0, width, 600), float(self.captures))

    def wait(self, seconds: float) -> None:
        self.now += seconds


class FakeProbe:
    def __init__(self, world: World) -> None:
        self.world = world

    def read(self, handle: int) -> WindowInfo | None:
        w = self.world
        if w.closed:
            return None
        return WindowInfo(handle, 999 if w.reused_handle else w.window.pid, w.window.title,
                          (30, 30, 430, 230) if w.moved_window else w.window.box)

    def foreground(self) -> WindowInfo | None:
        w = self.world
        return WindowInfo(999, 888, "Other app", w.window.box) if w.foreign else self.read(w.window.handle)


class FakeRecognizer:
    def __init__(self, world: World) -> None:
        self.world = world

    def recognize(self, image: np.ndarray) -> list[TextElement]:
        w = self.world
        phase = int(image[0, 0, 0])
        w.events.append(("ocr", phase))
        if w.missing_target:
            return []
        shift = 5 if w.moved_target else 0
        box = (10 + shift, 10, 190 + shift, 40)
        text = f"{('Start', 'Continue', 'Finished')[min(phase, 2)]} {w.token}"
        elements = [TextElement(text, 0.1 if w.low_confidence else 0.99, box)]
        if w.duplicate:
            elements.append(TextElement(text, 0.99, (10, 60, 190, 90)))
        return elements


class FakeBackend:
    FAILSAFE = False

    def __init__(self, world: World) -> None:
        self.world = world
        self.error: BaseException | None = None

    def size(self) -> tuple[int, int]:
        return 800, 600

    def click(self, **kwargs: object) -> None:
        if self.error:
            raise self.error
        w = self.world
        w.events.append(("click", w.phase))
        w.actions.append(("click", kwargs))
        if not w.stuck:
            w.phase += 1

    def scroll(self, clicks: int) -> None:
        self.world.actions.append(("scroll", clicks))
        self.world.phase += 1

    def moveTo(self, *args: object, **kwargs: object) -> None:
        self.world.actions.append(("moveTo", args))

    def press(self, key: str) -> None:
        self.world.actions.append(("press", key))
        if key == "enter":
            self.world.phase += 1

    def hotkey(self, *keys: str) -> None:
        self.world.actions.append(("hotkey", *keys))
        if keys == ("alt", "f4"):
            self.world.closed = True

    def write(self, *args: object, **kwargs: object) -> None:
        pytest.fail("Strict runtime must not type or paste")


def plan_data(token: str) -> dict[str, object]:
    return {"goal": f"Reach Finished {token}", "status": "ready", "steps": [
        {"step_id": "1", "description": "Advance the interface", "depends_on": [], "success_criteria": f"Finished {token} visible"},
        {"step_id": "2", "description": "Verify visible result", "depends_on": ["1"], "success_criteria": f"Finished {token} verified"},
    ]}


class FakeClient:
    def __init__(self, world: World) -> None:
        self.world = world
        self.plan_override: object | None = None
        self.action_override: dict[str, object] | None = None
        self.error: Exception | None = None

    def generate(self, request: MultimodalRequest) -> MultimodalResponse:
        w = self.world
        w.requests.append(request)
        w.events.append(("model", w.phase))
        if self.error:
            raise self.error
        if request.system_prompt.startswith("You are a GUI task planning"):
            data = self.plan_override if self.plan_override is not None else plan_data(w.token)
        else:
            step = request.instruction.split("Current step_id: ", 1)[1].splitlines()[0]
            context = json.loads(request.instruction.split("Current observation: ", 1)[1].splitlines()[0])
            data = {"step_id": step, "observation_id": context["observation_id"]}
            if self.action_override is not None:
                data.update(self.action_override)
            elif w.phase < 2:
                target = context["elements"][0]
                data.update(status="action", action={"type": "click", "target_id": target["target_id"], "target_text": target["text"]})
            else:
                data.update(status="step_complete", checks=[{"type": "text_equals", "text": w.text, "region": (0, 0, 400, 200)}])
        text = data if isinstance(data, str) else json.dumps(data)
        return MultimodalResponse(text=text, provider="fake", model="fake")


def setup_runtime(tmp_path: Path, token: str = "sample", **limits: object):
    world = World(token)
    client = FakeClient(world)
    backend = FakeBackend(world)
    observer = DesktopObserver(world.window, tmp_path / "observations", recognizer=FakeRecognizer(world),
                               window_probe=FakeProbe(world), capture=world.capture)
    config = RunLimits(settle_seconds=0, feedback_timeout=0, **limits)
    runtime = GuiRuntime(GuiPlanningAgent(client), GuiActionAgent(client), observer, DesktopController(backend),
                         limits=config, wait=world.wait, clock=lambda: world.now)
    return world, client, backend, runtime


def goal(world: World, text: str | None = None) -> list[FeedbackCheck]:
    return [FeedbackCheck(type="text_equals", text=text or f"Finished {world.token}", region=(0, 0, 400, 200))]


def execute(runtime: GuiRuntime, world: World, **kwargs: object):
    return runtime.run(f"Reach Finished {world.token}", goal(world), execute=True,
                       confirm_plan=lambda plan, checks: True, confirm_action=lambda decision, grounded: True, **kwargs)


@pytest.mark.parametrize("token", ["query-749", "message-382"])
def test_real_runtime_fake_two_step_multi_action_loop_and_final_feedback(tmp_path: Path, token: str) -> None:
    world, client, backend, runtime = setup_runtime(tmp_path, token)
    result = execute(runtime, world)

    assert result.status == "completed"
    assert result.executed_actions == result.attempted_actions == 2
    assert result.completed_steps == ["1", "2"]
    assert result.model_decisions == 4
    assert len(world.requests) == 5  # One plan, two actions, two step checks.
    assert backend.FAILSAFE is True
    executions = [e for e in result.events if e["kind"] == "executed"]
    assert all(e["decision_observation_id"] != e["pre_execute_observation_id"] for e in executions)
    events = result.events
    first_scene = next(event for event in events if event["kind"] == "observation")
    assert first_scene["screen_size"] == (800, 600)
    assert first_scene["crop_origin"] == (20, 30)
    assert first_scene["screen_region"] == {"left": 0, "top": 0, "width": 800, "height": 600}
    assert all(event["point"] == (120, 55) for event in executions)
    for index, event in enumerate(events):
        if event["kind"] == "executed":
            assert events[index - 1]["kind"] == "observation"
            assert events[index + 1]["kind"] == "observation"
    assert len({e["observation_id"] for e in events if e["kind"] == "observation"}) == world.captures
    action_prompts = [request.instruction for request in client.world.requests[1:]]
    assert any(f"Continue {token}" in prompt for prompt in action_prompts)
    assert f'Completed steps: ["1"]' in action_prompts[-1]
    assert result.to_dict()["status"] == "completed"


def test_runtime_defaults_to_dry_run_no_confirmation_or_backend(tmp_path: Path) -> None:
    world, _, _, runtime = setup_runtime(tmp_path)
    result = runtime.run("Advance", goal(world))
    assert result.mode == "dry_run" and result.status == "blocked"
    assert result.executed_actions == 0 and world.actions == []
    assert len(world.requests) == 2


def test_execute_without_confirmation_callbacks_stops_before_observation(tmp_path: Path) -> None:
    world, _, _, runtime = setup_runtime(tmp_path)
    result = runtime.run("Advance", goal(world), execute=True)
    assert result.status == "blocked" and world.captures == 0 and world.requests == []


@pytest.mark.parametrize("plan", [
    {"goal": "Send", "status": "needs_clarification", "clarification_question": "Who?", "steps": []},
    "not json", {"goal": "Invalid", "status": "ready", "steps": []},
])
def test_clarification_and_invalid_plan_never_reach_action_or_control(tmp_path: Path, plan: object) -> None:
    world, client, _, runtime = setup_runtime(tmp_path)
    client.plan_override = plan
    result = execute(runtime, world)
    assert result.status == ("needs_clarification" if isinstance(plan, dict) and plan["status"] == "needs_clarification" else "failed")
    assert len(world.requests) == 1 and world.actions == []


@pytest.mark.parametrize("decision", [
    {"status": "action", "observation_id": "stale", "action": {"type": "key", "key": "enter", "purpose": "Proceed"}},
    {"status": "action", "action": {"type": "click", "x": 1, "y": 1}},
    {"status": "action", "action": {"type": "key", "key": "delete", "purpose": "Delete"}},
    {"status": "blocked", "reason": "Ambiguous target"},
])
def test_bad_or_blocked_model_decision_does_not_execute(tmp_path: Path, decision: dict[str, object]) -> None:
    world, client, _, runtime = setup_runtime(tmp_path)
    client.action_override = decision
    result = execute(runtime, world)
    assert result.status == ("blocked" if decision["status"] == "blocked" else "failed")
    assert world.actions == [] and len(world.requests) == 2


@pytest.mark.parametrize("when", ["plan", "action"])
@pytest.mark.parametrize("answer", [False, 1, "yes"])
def test_authorization_declined_or_not_literal_true_never_executes(tmp_path: Path, when: str, answer: object) -> None:
    world, _, _, runtime = setup_runtime(tmp_path)
    result = runtime.run("Advance", goal(world), execute=True,
                         confirm_plan=lambda plan, checks: answer if when == "plan" else True,
                         confirm_action=lambda decision, action: answer if when == "action" else True)
    assert result.status == "cancelled" and world.actions == []


@pytest.mark.parametrize("change", ["foreign", "moved_window", "reused_handle", "moved_target", "missing_target", "duplicate", "low_confidence", "resized", "offset_monitor"])
def test_changes_during_model_authorization_block_stale_execution(tmp_path: Path, change: str) -> None:
    world, _, _, runtime = setup_runtime(tmp_path)

    def confirm(decision, action):
        setattr(world, change, True)
        return True

    result = runtime.run("Advance", goal(world), execute=True,
                         confirm_plan=lambda plan, checks: True, confirm_action=confirm)
    assert result.status == "blocked" and world.actions == []
    assert result.attempted_actions == 0


def test_strict_runtime_never_infers_input_focus_from_successful_click(tmp_path: Path) -> None:
    world, client, _, runtime = setup_runtime(tmp_path)

    def confirm(decision, action):
        client.action_override = {"status": "action", "action": {
            "type": "type", "target_id": "obs-0004:text-1", "target_text": f"Continue {world.token}",
            "text": "replacement", "mode": "replace",
        }}
        return True

    result = runtime.run("Advance", goal(world), execute=True,
                         confirm_plan=lambda plan, checks: True, confirm_action=confirm)
    assert result.status == "blocked" and "Strict mode" in result.reason
    assert result.executed_actions == 1 and len(world.actions) == 1
    focus_line = world.requests[-1].instruction.split("Independent focus/selection evidence: ", 1)[1].splitlines()[0]
    assert json.loads(focus_line) == {"focus_verified": True, "focused_target_id": None, "selection_verified": False}


@pytest.mark.parametrize("action", [
    {"type": "key", "key": "enter", "purpose": "Advance"},
    {"type": "key", "key": "ctrl+a", "purpose": "Select visible content"},
    {"type": "scroll", "target_id": "obs-0002:text-1", "target_text": "Start sample", "clicks": -1},
])
def test_allowed_micro_actions_use_public_control_methods(tmp_path: Path, action: dict[str, object]) -> None:
    world, client, _, runtime = setup_runtime(tmp_path, max_actions=1)
    client.action_override = {"status": "action", "action": action}
    result = execute(runtime, world)
    assert result.executed_actions == 1
    assert world.actions[0][0] == ("moveTo" if action["type"] == "scroll" else "hotkey" if "+" in action.get("key", "") else "press")


@pytest.mark.parametrize("limit", ["max_actions", "max_step_actions"])
def test_action_limits_do_not_repeat_or_recover(tmp_path: Path, limit: str) -> None:
    world, _, _, runtime = setup_runtime(tmp_path, **{limit: 1})
    result = execute(runtime, world)
    assert result.status == "blocked" and result.executed_actions == 1
    assert len(world.actions) == 1 and "limit" in result.reason


def test_model_decision_limit_bounds_zero_action_steps(tmp_path: Path) -> None:
    world, client, _, runtime = setup_runtime(tmp_path, max_decisions=1)
    client.action_override = {"status": "step_complete", "checks": [{"type": "window_present", "window": "bound_target"}]}
    result = execute(runtime, world)
    assert result.status == "blocked" and result.model_decisions == 1 and world.actions == []


def test_model_step_complete_is_not_accepted_without_actual_feedback(tmp_path: Path) -> None:
    world, client, _, runtime = setup_runtime(tmp_path)
    client.action_override = {"status": "step_complete", "checks": [check.model_dump() for check in goal(world)]}
    result = execute(runtime, world)
    assert result.status == "inconclusive" and result.completed_steps == [] and world.actions == []


def test_completed_plan_with_weak_checks_does_not_override_independent_final_goal(tmp_path: Path) -> None:
    world, client, _, runtime = setup_runtime(tmp_path)
    client.action_override = {"status": "step_complete", "checks": [{"type": "window_present", "window": "bound_target"}]}
    result = execute(runtime, world)
    assert result.completed_steps == ["1", "2"]
    assert result.status == "inconclusive" and result.executed_actions == 0


@pytest.mark.parametrize("error", [KeyboardInterrupt("cancel"), pyautogui.FailSafeException("corner"), RuntimeError("backend failed")])
def test_backend_interruptions_preserve_attempt_and_stop_without_retry(tmp_path: Path, error: BaseException) -> None:
    world, _, backend, runtime = setup_runtime(tmp_path)
    backend.error = error
    result = execute(runtime, world)
    assert result.status == ("failed" if type(error) is RuntimeError else "cancelled")
    assert result.attempted_actions == 1 and result.executed_actions == 0
    assert len(world.requests) == 2


def test_api_error_stops_without_more_model_calls(tmp_path: Path) -> None:
    world, client, _, runtime = setup_runtime(tmp_path)
    client.error = ModelInvocationError("timeout", category="timeout")
    result = execute(runtime, world)
    assert result.status == "failed" and len(world.requests) == 1 and world.actions == []


def test_elapsed_model_call_is_checked_before_any_action(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world, client, _, runtime = setup_runtime(tmp_path, timeout_seconds=1)
    original = client.generate

    def delayed(request):
        response = original(request)
        world.now = 2
        return response

    monkeypatch.setattr(client, "generate", delayed)
    result = execute(runtime, world)
    assert result.status == "blocked" and "budget" in result.reason
    assert len(world.requests) == 1 and world.actions == []


def test_feedback_polling_observes_without_repeating_actions(tmp_path: Path) -> None:
    world, client, _, runtime = setup_runtime(tmp_path)
    runtime.limits = RunLimits(settle_seconds=0, feedback_timeout=0.5, poll_interval=0.25)
    client.action_override = {"status": "step_complete", "checks": [check.model_dump() for check in goal(world)]}
    result = execute(runtime, world)
    assert result.status == "inconclusive" and world.now == 0.5
    assert len(world.requests) == 2 and world.actions == []
    assert len([event for event in result.events if event["kind"] == "feedback"]) == 3


def test_bound_window_close_uses_absence_and_final_goal_without_calling_model_on_background(tmp_path: Path) -> None:
    world, client, _, runtime = setup_runtime(tmp_path)
    client.action_override = {"status": "action", "action": {"type": "key", "key": "alt+f4", "purpose": "Close authorized test window"}}
    result = runtime.run("Close authorized test window", [FeedbackCheck(type="window_closed", window="bound_target")], execute=True,
                         confirm_plan=lambda plan, checks: True, confirm_action=lambda decision, action: True)
    assert result.status == "completed" and world.closed and result.executed_actions == 1
    assert result.completed_steps == []  # Do not invent model step-completion decisions.
    assert len(world.requests) == 2


@pytest.mark.parametrize("checks", [[], [FeedbackCheck(type="text_present", text="Outside", region=(0, 0, 500, 200))]])
def test_missing_or_out_of_scope_final_checks_fail_before_model(tmp_path: Path, checks: list[FeedbackCheck]) -> None:
    world, _, _, runtime = setup_runtime(tmp_path)
    result = runtime.run("Advance", checks)
    assert result.status in {"failed", "blocked"} and world.requests == [] and world.actions == []


def test_observation_output_cannot_overwrite_previous_images(tmp_path: Path) -> None:
    world, _, _, runtime = setup_runtime(tmp_path)
    runtime.observer.observe()
    second = DesktopObserver(world.window, tmp_path / "observations", recognizer=FakeRecognizer(world),
                             window_probe=FakeProbe(world), capture=world.capture)
    with pytest.raises(FileExistsError):
        second.observe()


def test_key_cannot_execute_against_changed_visible_context(tmp_path: Path) -> None:
    world, client, _, runtime = setup_runtime(tmp_path)
    client.action_override = {"status": "action", "action": {"type": "key", "key": "enter", "purpose": "Advance"}}

    def confirm(decision, action):
        world.phase = 1
        return True

    result = runtime.run("Advance", goal(world), execute=True,
                         confirm_plan=lambda plan, checks: True, confirm_action=confirm)
    assert result.status == "blocked" and "context changed" in result.reason and world.actions == []


def test_cached_old_capture_is_rejected_before_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world, _, _, runtime = setup_runtime(tmp_path)
    frame = world.capture()
    monkeypatch.setattr(runtime.observer, "capture", lambda: frame)
    result = execute(runtime, world)
    assert result.status == "blocked" and "not newer" in result.reason and world.actions == []


def test_window_title_change_can_be_observed_without_rebinding_identity(tmp_path: Path) -> None:
    world, _, _, runtime = setup_runtime(tmp_path)
    world.window = WindowInfo(world.window.handle, world.window.pid, "Updated visible title", world.window.box)
    scene = runtime.observer.observe()
    assert scene.window.title == "Updated visible title"
    assert scene.window.handle == runtime.observer.bound_window.handle


def test_focus_lost_during_capture_is_detected_before_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world, _, _, runtime = setup_runtime(tmp_path)
    original = world.capture

    def lose_focus():
        frame = original()
        world.foreign = True
        return frame

    monkeypatch.setattr(runtime.observer, "capture", lose_focus)
    result = execute(runtime, world)
    assert result.status == "blocked" and world.requests == [] and world.actions == []


@pytest.mark.parametrize(("name", "value"), [
    ("max_actions", 0), ("max_actions", True), ("max_step_actions", -1), ("max_decisions", 1.5),
    ("timeout_seconds", 0), ("timeout_seconds", float("inf")), ("feedback_timeout", -1),
    ("settle_seconds", float("nan")), ("poll_interval", 0), ("poll_interval", True),
])
def test_runtime_limits_reject_invalid_values(name: str, value: object) -> None:
    with pytest.raises(ValueError):
        RunLimits(**{name: value})


class NativeFunction:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


def test_windows_probe_read_only_calls_and_pointer_sized_handles(monkeypatch: pytest.MonkeyPatch) -> None:
    handle = 0x123456789
    seen: list[tuple[object, ...]] = []

    def rect(hwnd, pointer):
        seen.append(("rect", hwnd))
        output = ctypes.cast(pointer, ctypes.POINTER(wintypes.RECT)).contents
        output.left, output.top, output.right, output.bottom = 20, 30, 420, 230
        return 1

    def pid(hwnd, pointer):
        ctypes.cast(pointer, ctypes.POINTER(wintypes.DWORD)).contents.value = 456
        return 789

    def title(hwnd, buffer, length):
        buffer.value = "Test"
        return 4

    class Api:
        GetForegroundWindow = NativeFunction(lambda: handle)
        IsWindow = NativeFunction(lambda hwnd: hwnd == handle)
        GetWindowRect = NativeFunction(rect)
        GetWindowThreadProcessId = NativeFunction(pid)
        GetWindowTextLengthW = NativeFunction(lambda hwnd: 4)
        GetWindowTextW = NativeFunction(title)

    api = Api()
    monkeypatch.setattr(runtime_module.sys, "platform", "win32")
    monkeypatch.setattr(runtime_module.ctypes, "WinDLL", lambda name, **kwargs: api, raising=False)
    probe = WindowsWindowProbe()
    assert probe.foreground() == WindowInfo(handle, 456, "Test", (20, 30, 420, 230))
    assert seen == [("rect", handle)]
    assert api.GetForegroundWindow.restype == wintypes.HWND
    assert api.IsWindow.argtypes == [wintypes.HWND]
    assert probe.read(999) is None


def test_window_probe_unsupported_platform_fails_without_loading_library(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_module.sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="Windows-only"):
        WindowsWindowProbe()


@pytest.fixture
def cli_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    world = World("cli-token")
    client = FakeClient(world)
    backend = FakeBackend(world)
    monkeypatch.setattr(week4_demo, "_client", lambda args: client)
    monkeypatch.setattr(week4_demo.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(week4_demo, "WindowsWindowProbe", lambda: FakeProbe(world))
    monkeypatch.setattr(week4_demo, "EasyOcrRecognizer", lambda **kwargs: FakeRecognizer(world))
    monkeypatch.setattr(week4_demo, "DesktopController", lambda: DesktopController(backend))
    monkeypatch.setattr(week4_demo, "DesktopObserver", lambda bound, output, **kwargs: DesktopObserver(bound, output, capture=world.capture, **kwargs))
    monkeypatch.setattr(week4_demo, "GuiRuntime", lambda *args, **kwargs: GuiRuntime(*args, **kwargs, wait=world.wait, clock=lambda: world.now))
    monkeypatch.setattr(week4_demo, "_authorize", lambda *args: True)
    checks = tmp_path / "checks.json"
    checks.write_text(json.dumps([{"type": "text_equals", "text": "Finished cli-token", "region": [0, 0, 400, 200]}]), encoding="utf-8")
    args = ["run", "--provider", "api", "--task", "Advance and verify cli-token", "--window-title", world.window.title,
            "--checks-json", str(checks), "--output-dir", str(tmp_path / "run")]
    return world, client, args


@pytest.mark.parametrize("execute", [False, True])
def test_week4_cli_real_runtime_mock_integration(cli_world, tmp_path: Path, capsys, execute: bool) -> None:
    world, _, args = cli_world
    code = week4_demo.main(args + (["--execute"] if execute else []))
    summary = json.loads((tmp_path / "run" / "run-summary.json").read_text(encoding="utf-8"))
    assert code == 0
    assert summary["status"] == ("completed" if execute else "blocked")
    assert summary["executed_actions"] == (2 if execute else 0)
    assert len(world.actions) == (2 if execute else 0)
    assert summary["model_calls"] == (5 if execute else 2)
    records = list((tmp_path / "run").glob("model-*.json"))
    assert len(records) == summary["model_calls"]
    assert all("response_before_parsing" in json.loads(p.read_text(encoding="utf-8")) for p in records)
    assert (tmp_path / "run" / "observations" / "obs-0001.png").exists()
    if not execute:
        assert "not task completion" in capsys.readouterr().out


@pytest.mark.parametrize("failure", ["wrong_title", "existing_output", "empty_checks", "invalid_checks", "blank_task"])
def test_week4_cli_preflight_stops_before_capture_or_model(cli_world, tmp_path: Path, failure: str) -> None:
    world, _, args = cli_world
    if failure == "wrong_title":
        args[args.index("--window-title") + 1] = "Other window"
    elif failure == "existing_output":
        (tmp_path / "run").mkdir()
        (tmp_path / "run" / "old.txt").write_text("Preserve", encoding="utf-8")
    elif failure in ("empty_checks", "invalid_checks"):
        (tmp_path / "checks.json").write_text("[]" if failure == "empty_checks" else '[{"type":"text_equals","text":"x"}]', encoding="utf-8")
    else:
        args[args.index("--task") + 1] = " "
    assert week4_demo.main(args + ["--execute"]) == 1
    assert world.captures == 0 and not world.requests and not world.actions
    if failure == "existing_output":
        assert (tmp_path / "run" / "old.txt").read_text(encoding="utf-8") == "Preserve"


@pytest.mark.parametrize("failure", ["clarification", "invalid_plan", "invalid_action", "denied", "ambiguous", "feedback"])
def test_week4_cli_preserves_safe_failure_summary(cli_world, tmp_path: Path, monkeypatch, failure: str) -> None:
    world, client, args = cli_world
    expected = "blocked"
    if failure == "clarification":
        client.plan_override = {"goal": "Need details", "status": "needs_clarification", "clarification_question": "Which target?", "steps": []}
        expected = "needs_clarification"
    elif failure == "invalid_plan":
        client.plan_override = "{}"
        expected = "failed"
    elif failure == "invalid_action":
        client.action_override = {"status": "action", "action": {"type": "click", "x": 1, "y": 2}}
        expected = "failed"
    elif failure == "denied":
        monkeypatch.setattr(week4_demo, "_authorize", lambda *args: False)
        expected = "cancelled"
    elif failure == "ambiguous":
        world.duplicate = True
    else:
        client.action_override = {"status": "step_complete", "checks": [{"type": "text_present", "text": "Missing", "region": [0, 0, 400, 200]}]}
        expected = "inconclusive"
    assert week4_demo.main(args + ["--execute"]) == 1
    result = json.loads((tmp_path / "run" / "run-summary.json").read_text(encoding="utf-8"))
    assert result["status"] == expected
    assert not world.actions and result["executed_actions"] == 0


@pytest.mark.parametrize("option,value", [("--start-delay", "0"), ("--max-actions", "-1"), ("--max-decisions", "0")])
def test_week4_cli_rejects_invalid_limits(cli_world, option: str, value: str) -> None:
    world, _, args = cli_world
    with pytest.raises(SystemExit):
        week4_demo.main(args + [option, value])
    assert not world.requests and world.captures == 0


def test_week4_api_requires_upload_consent_and_keeps_retry_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GUI_AGENT_API_BASE", "https://example.invalid/v1")
    monkeypatch.setenv("GUI_AGENT_API_MODEL", "glm-4v-flash")
    monkeypatch.setenv("GUI_AGENT_API_KEY", "fake-test-key")
    args = week4_demo.build_parser().parse_args(["run", "--provider", "api", "--task", "Test", "--window-title", "Test", "--checks-json", "checks.json", "--output-dir", "output"])
    with pytest.raises(ValueError, match="allow-screenshot-upload"):
        week4_demo._client(args)
    args.allow_screenshot_upload = True
    seen = []
    monkeypatch.setattr(week4_demo, "OpenAICompatibleVisionClient", lambda *a, **kw: seen.append((a, kw)))
    week4_demo._client(args)
    assert seen == [(("glm-4v-flash", "https://example.invalid/v1"), {"api_key_env": "GUI_AGENT_API_KEY", "max_retries": 0})]


def test_week4_local_model_does_not_inherit_api_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GUI_AGENT_API_MODEL", "glm-4v-flash")
    args = week4_demo.build_parser().parse_args(["run", "--provider", "local", "--cache-dir", str(tmp_path / "hub"), "--offline", "--task", "Test", "--window-title", "Test", "--checks-json", "checks.json", "--output-dir", "output"])
    seen = []
    monkeypatch.setattr(week4_demo, "TransformersVisionClient", lambda *a, **kw: seen.append((a, kw)))
    for key in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "HF_HUB_OFFLINE"):
        monkeypatch.setenv(key, "old")
    week4_demo._client(args)
    assert seen[0][0] == ("Qwen/Qwen2.5-VL-3B-Instruct", tmp_path / "hub")
    import os
    assert os.environ["HF_HUB_CACHE"] == str(tmp_path / "hub")


def test_week4_authorization_uses_pointer_sized_owner_and_defaults_to_no(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    function = NativeFunction(lambda *args: seen.append(args) or 7)
    api = type("Api", (), {"MessageBoxW": function})()
    monkeypatch.setattr(week4_demo.ctypes, "WinDLL", lambda *args, **kwargs: api)
    monkeypatch.setattr(week4_demo.time, "sleep", lambda seconds: None)
    assert week4_demo._authorize("Title", "Evidence", 0x123456789) is False
    assert seen[0][0] == 0x123456789
    assert function.argtypes[0] == wintypes.HWND
    assert seen[0][3] & 0x100


def test_week4_fixture_preparation_and_repository_guard(tmp_path: Path) -> None:
    output = tmp_path / "fixture"
    assert week4_demo.main(["prepare", "--output-dir", str(output)]) == 0
    sample = output / "week4-sample.txt"
    initial = sample.read_bytes()
    assert b"FILE TOKEN W4-OPEN-001" in initial
    assert week4_demo.main(["prepare", "--output-dir", str(output)]) == 1
    assert sample.read_bytes() == initial
    with pytest.raises(ValueError, match="outside"):
        week4_demo._external_path(week4_demo.PROJECT_ROOT / "unsafe-output")


@pytest.fixture
def local_fixture_server():
    import threading
    server = week4_demo.make_fixture_server(0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_week4_fixture_search_message_reset_and_escaping(local_fixture_server) -> None:
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen
    server = local_fixture_server
    assert server.server_address[0] == "127.0.0.1"
    base = f"http://127.0.0.1:{server.server_port}"
    token = "query <script> & sample"
    with urlopen(base + "/search?" + urlencode({"q": token})) as response:
        page = response.read().decode("utf-8")
    assert "Search results" in page and "&lt;script&gt;" in page and "<script>" not in page
    for body in ("Message alpha", "Message beta"):
        request = Request(base + "/messages", data=urlencode({"body": body}).encode(), headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urlopen(request) as response:
            assert "Message sent" in response.read().decode("utf-8")
    with urlopen(base + "/records") as response:
        records = json.load(response)
    assert records == {"searches": [token], "messages": [{"recipient": "Test Receiver", "body": body} for body in ("Message alpha", "Message beta")]}
    request = Request(base + "/reset", data=b"", headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urlopen(request) as response:
        assert "No messages sent" in response.read().decode("utf-8")
    with urlopen(base + "/records") as response:
        assert json.load(response) == {"searches": [], "messages": []}


@pytest.mark.parametrize("case,expected", [("blank", 400), ("too_long", 400), ("large_body", 413), ("cross_origin", 403), ("content_type", 415), ("unknown", 404)])
def test_week4_fixture_rejects_invalid_requests(local_fixture_server, case: str, expected: int) -> None:
    from urllib.error import HTTPError
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen
    base = f"http://127.0.0.1:{local_fixture_server.server_port}"
    body = " " if case == "blank" else ("a" * 513 if case == "too_long" else "Valid")
    data = b"a" * 8193 if case == "large_body" else urlencode({"body": body}).encode()
    headers = {"Content-Type": "text/plain" if case == "content_type" else "application/x-www-form-urlencoded"}
    if case == "cross_origin":
        headers["Origin"] = "https://example.invalid"
    request = Request(base + ("/unknown" if case == "unknown" else "/messages"), data=data, headers=headers)
    with pytest.raises(HTTPError) as error:
        urlopen(request)
    assert error.value.code == expected
    error.value.close()
    with urlopen(base + "/records") as response:
        assert json.load(response)["messages"] == []


class InputRecognizer:
    def __init__(self, world: World) -> None:
        self.world = world

    def recognize(self, image: np.ndarray) -> list[TextElement]:
        w = self.world
        if w.missing_target:
            return []
        shift = 5 if w.moved_target else 0
        elements = [TextElement("Input Target", 0.1 if w.low_confidence else 0.99, (10 + shift, 10, 190 + shift, 40)),
                    TextElement("Other Target", 0.99, (210, 10, 380, 40))]
        if w.duplicate:
            elements.append(TextElement("Input Target", 0.99, (10, 110, 190, 140)))
        if int(image[0, 0, 0]) >= 2:
            elements.append(TextElement(w.token, 0.99, (10, 60, 190, 90)))
        return elements


class InputBackend(FakeBackend):
    def write(self, text: str, **kwargs: object) -> None:
        self.world.actions.append(("write", text))
        self.world.phase = 2


class InputClient(FakeClient):
    def __init__(self, world: World) -> None:
        super().__init__(world)
        self.mode = "append"
        self.other_target = False
        self.repeat_type = False
        self.intermediate: str | None = None

    def generate(self, request: MultimodalRequest) -> MultimodalResponse:
        w = self.world
        w.requests.append(request)
        if request.system_prompt.startswith("You are a GUI task planning"):
            data = plan_data(w.token)
        else:
            context = json.loads(request.instruction.split("Current observation: ", 1)[1].splitlines()[0])
            target_text = "Other Target" if self.other_target else "Input Target"
            target = next((e for e in context["elements"] if e["text"] == target_text), {"target_id": "missing", "text": target_text})
            data = {"step_id": request.instruction.split("Current step_id: ", 1)[1].splitlines()[0],
                    "observation_id": context["observation_id"]}
            intermediate_done = any(a[0] in ("hotkey", "scroll") for a in w.actions)
            if w.phase == 0:
                data.update(status="action", action={"type": "click", "target_id": target["target_id"], "target_text": target["text"]})
            elif self.intermediate and not intermediate_done:
                action = {"type": "key", "key": "ctrl+a", "purpose": "Select input"} if self.intermediate == "key" else {
                    "type": "scroll", "target_id": target["target_id"], "target_text": target["text"], "clicks": 1}
                data.update(status="action", action=action)
            elif w.phase < 2 or self.repeat_type or (self.intermediate and not any(a[0] == "write" for a in w.actions)):
                data.update(status="action", action={"type": "type", "target_id": target["target_id"], "target_text": target["text"], "text": w.token, "mode": self.mode})
            else:
                data.update(status="step_complete", checks=[{"type": "text_present", "text": w.token, "region": [0, 0, 400, 200]}])
        return MultimodalResponse(text=json.dumps(data), provider="fake", model="fake")


def setup_input_runtime(tmp_path: Path, token: str = "input-token"):
    world = World(token)
    client = InputClient(world)
    backend = InputBackend(world)
    observer = DesktopObserver(world.window, tmp_path / "observations", recognizer=InputRecognizer(world),
                               window_probe=FakeProbe(world), capture=world.capture)
    runtime = GuiRuntime(GuiPlanningAgent(client), GuiActionAgent(client), observer, DesktopController(backend),
                         limits=RunLimits(settle_seconds=0, feedback_timeout=0), wait=world.wait, clock=lambda: world.now)
    return world, client, runtime


def run_input(runtime: GuiRuntime, world: World, **kwargs):
    return runtime.run("Enter " + world.token + " in the target input and verify it",
                       [FeedbackCheck(type="text_present", text=world.token, region=(0, 0, 400, 200))],
                       execute=True, confirm_plan=lambda *args: True,
                       confirm_action=kwargs.pop("confirm_action", lambda *args: True), **kwargs)


def test_supervised_focus_binds_agent_click_and_maps_current_observation_id(tmp_path: Path) -> None:
    world, _, runtime = setup_input_runtime(tmp_path)
    confirmations = []

    def confirm(grounded, scene):
        assert world.actions == [("click", {"x": 120, "y": 55, "button": "left", "duration": 0.0})]
        confirmations.append((grounded, scene))
        return True

    result = run_input(runtime, world, supervised_input_focus=True, confirm_input_focus=confirm)
    assert result.status == "completed"
    assert world.actions[-1] == ("write", world.token)
    assert result.executed_actions == 2 and len(confirmations) == 1
    request = world.requests[2]
    evidence = json.loads(request.instruction.split("Independent focus/selection evidence: ", 1)[1].splitlines()[0])
    context = json.loads(request.instruction.split("Current observation: ", 1)[1].splitlines()[0])
    assert evidence == {"focus_verified": True, "focused_target_id": None, "selection_verified": False,
                        "input_focus_candidate_target_id": context["elements"][0]["target_id"]}
    assert confirmations[0][0].action.target_id != evidence["input_focus_candidate_target_id"]
    confirmation = next(e for e in result.events if e["kind"] == "input_focus_confirmation")
    assert confirmation["confirmed"] is True and confirmation["selection_verified"] is False
    assert confirmation["decision_observation_id"] == context["observation_id"]
    assert len([e for e in result.events if e["kind"] == "input_focus_candidate"]) == 1


@pytest.mark.parametrize("answer", [False, None, "yes", {"target_id": "other", "x": 100}])
def test_supervised_focus_rejects_denial_or_nonboolean_evidence(tmp_path: Path, answer: object) -> None:
    world, _, runtime = setup_input_runtime(tmp_path)
    result = run_input(runtime, world, supervised_input_focus=True, confirm_input_focus=lambda *args: answer)
    assert result.status == "blocked" and result.executed_actions == 1
    assert [a[0] for a in world.actions] == ["click"]
    assert "confirmation declined" in result.reason
    assert result.model_decisions == 2 and len(world.requests) == 3


@pytest.mark.parametrize("enabled,execute_mode,callback", [(True, True, None), (True, False, lambda *a: True), ("yes", True, lambda *a: True)])
def test_supervised_focus_requires_explicit_execute_and_callback(tmp_path: Path, enabled, execute_mode, callback) -> None:
    world, _, runtime = setup_input_runtime(tmp_path)
    result = runtime.run("Test", goal(world), execute=execute_mode, confirm_plan=lambda *args: True,
                         confirm_action=lambda *args: True, supervised_input_focus=enabled, confirm_input_focus=callback)
    assert result.status in ("blocked", "failed")
    assert world.captures == 0 and not world.requests and not world.actions


def test_focus_callback_alone_does_not_enable_input(tmp_path: Path) -> None:
    world, _, runtime = setup_input_runtime(tmp_path)
    result = run_input(runtime, world, confirm_input_focus=lambda *args: pytest.fail("Strict mode must never request input focus"))
    assert result.status == "blocked" and "Strict mode" in result.reason
    assert [a[0] for a in world.actions] == ["click"]


def test_supervised_focus_cannot_authorize_type_before_agent_click(tmp_path: Path) -> None:
    world, _, runtime = setup_input_runtime(tmp_path)
    world.phase = 1
    result = run_input(runtime, world, supervised_input_focus=True,
                       confirm_input_focus=lambda *args: pytest.fail("No Agent click; no confirmation permitted"))
    assert result.status == "blocked" and not world.actions


def test_supervised_focus_keeps_replace_blocked(tmp_path: Path) -> None:
    world, client, runtime = setup_input_runtime(tmp_path)
    client.mode = "replace"
    result = run_input(runtime, world, supervised_input_focus=True, confirm_input_focus=lambda *args: True)
    assert result.status == "blocked" and "separately verified" in result.reason
    assert [a[0] for a in world.actions] == ["click"]


def test_supervised_focus_never_authorizes_another_target(tmp_path: Path) -> None:
    world, client, runtime = setup_input_runtime(tmp_path)

    def authorize(decision, grounded):
        if decision.action.type == "click":
            client.other_target = True
        return True

    result = run_input(runtime, world, supervised_input_focus=True, confirm_action=authorize,
                       confirm_input_focus=lambda *args: pytest.fail("Different target must not request focus"))
    assert result.status == "blocked" and [a[0] for a in world.actions] == ["click"]


@pytest.mark.parametrize("change", ["moved_target", "missing_target", "duplicate", "low_confidence", "moved_window", "reused_handle", "foreign", "resized", "title"])
@pytest.mark.parametrize("when", ["confirmation", "type_authorization"])
def test_supervised_focus_invalidates_on_target_window_or_geometry_change(tmp_path: Path, change: str, when: str) -> None:
    world, _, runtime = setup_input_runtime(tmp_path)

    def mutate():
        if change == "title":
            world.window = WindowInfo(world.window.handle, world.window.pid, "Changed title", world.window.box)
        else:
            setattr(world, change, True)

    def focus(*args):
        if when == "confirmation":
            mutate()
        return True

    def authorize(decision, grounded):
        if decision.action.type == "type" and when == "type_authorization":
            mutate()
        return True

    result = run_input(runtime, world, supervised_input_focus=True, confirm_input_focus=focus, confirm_action=authorize)
    assert result.status == "blocked"
    assert result.executed_actions == 1 and [a[0] for a in world.actions] == ["click"]


@pytest.mark.parametrize("action", ["key", "scroll", "type"])
def test_supervised_focus_is_consumed_by_any_subsequent_action(tmp_path: Path, action: str) -> None:
    world, client, runtime = setup_input_runtime(tmp_path)
    if action == "type":
        client.repeat_type = True
    else:
        client.intermediate = action
    result = run_input(runtime, world, supervised_input_focus=True, confirm_input_focus=lambda *args: True)
    assert result.status == "blocked" and result.executed_actions == 2
    assert sum(a[0] == "write" for a in world.actions) == (1 if action == "type" else 0)
    requests = [r for r in world.requests if "Independent focus/selection evidence: " in r.instruction]
    last = json.loads(requests[-1].instruction.split("Independent focus/selection evidence: ", 1)[1].splitlines()[0])
    assert last["focused_target_id"] is None and last["selection_verified"] is False


def test_supervised_focus_unicode_uses_controller_paste_without_real_clipboard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world, _, runtime = setup_input_runtime(tmp_path, token="\u4e2d\u6587\u8f93\u5165")

    def fake_paste(text):
        world.actions.append(("paste", text))
        world.phase = 2

    monkeypatch.setattr(runtime.controller, "paste_text", fake_paste)
    result = run_input(runtime, world, supervised_input_focus=True, confirm_input_focus=lambda *args: True)
    assert result.status == "completed" and world.actions[-1] == ("paste", world.token)


def test_focus_confirmation_is_not_task_success_evidence(tmp_path: Path) -> None:
    world, _, runtime = setup_input_runtime(tmp_path)
    result = runtime.run("Enter input-token", [FeedbackCheck(type="text_present", text="Unachieved final goal", region=(0, 0, 400, 200))],
                         execute=True, confirm_plan=lambda *args: True, confirm_action=lambda *args: True,
                         supervised_input_focus=True, confirm_input_focus=lambda *args: True)
    assert result.executed_actions == 2 and result.completed_steps == ["1", "2"]
    assert result.status == "inconclusive"


def test_week4_cli_supervision_flag_requires_execute_before_any_capture(cli_world) -> None:
    world, _, args = cli_world
    assert week4_demo.main(args + ["--supervised-input-focus"]) == 1
    assert world.captures == 0 and not world.actions and not world.requests


def test_week4_cli_supervised_mode_is_explicit_and_recorded(cli_world, tmp_path: Path, monkeypatch, capsys) -> None:
    world, _, args = cli_world
    confirmations = []
    monkeypatch.setattr(week4_demo, "_authorize", lambda *args: confirmations.append(args[0]) or True)
    assert week4_demo.main(args + ["--execute", "--supervised-input-focus"]) == 0
    summary = json.loads((tmp_path / "run" / "run-summary.json").read_text(encoding="utf-8"))
    assert summary["supervised_input_focus"] is True
    # Ordinary clicks never request focus, including this changed-text fixture.
    assert "Confirm input focus only" not in confirmations
    assert "OVERWRITES the clipboard" in capsys.readouterr().out


def test_input_authorization_precedes_focus_confirmation_and_final_revalidation(tmp_path: Path) -> None:
    world, _, runtime = setup_input_runtime(tmp_path)
    order = []
    captures_at_confirmation = []

    def authorize(decision, grounded):
        order.append("authorize_" + decision.action.type)
        return True

    def confirm(grounded, scene):
        assert grounded.action.type == "click"
        assert order == ["authorize_click", "authorize_type"]
        assert [a[0] for a in world.actions] == ["click"]
        captures_at_confirmation.append(world.captures)
        order.append("confirm_focus")
        return True

    original_write = runtime.controller.type_text

    def write(text):
        assert world.captures > captures_at_confirmation[0]
        order.append("write")
        original_write(text)

    runtime.controller.type_text = write
    result = run_input(runtime, world, supervised_input_focus=True,
                       confirm_action=authorize, confirm_input_focus=confirm)
    assert result.status == "completed"
    assert order == ["authorize_click", "authorize_type", "confirm_focus", "write"]


def test_input_action_denial_never_requests_focus_or_more_model_calls(tmp_path: Path) -> None:
    world, _, runtime = setup_input_runtime(tmp_path)
    result = run_input(
        runtime, world, supervised_input_focus=True,
        confirm_action=lambda decision, grounded: decision.action.type != "type",
        confirm_input_focus=lambda *args: pytest.fail("Denied action must not request focus"),
    )
    assert result.status == "cancelled" and result.executed_actions == 1
    assert [a[0] for a in world.actions] == ["click"]
    assert len(world.requests) == 3


def test_week4_cli_focus_denial_text_and_order_do_not_authorize_input(
    cli_world, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    world, _, args = cli_world
    client = InputClient(world)
    monkeypatch.setattr(week4_demo, "_client", lambda args: client)
    monkeypatch.setattr(week4_demo, "EasyOcrRecognizer", lambda **kwargs: InputRecognizer(world))
    monkeypatch.setattr(week4_demo, "DesktopController", lambda: DesktopController(InputBackend(world)))
    dialogs = []

    def authorize(title, text, handle):
        dialogs.append((title, text))
        return title != "Confirm input focus only"

    monkeypatch.setattr(week4_demo, "_authorize", authorize)
    assert week4_demo.main(args + ["--execute", "--supervised-input-focus"]) == 1
    assert [title for title, text in dialogs] == [
        "Authorize plan and final checks", "Authorize one GUI action",
        "Authorize one GUI action", "Confirm input focus only",
    ]
    focus_text = dialogs[-1][1]
    assert "Before the proposed text input" in focus_text
    assert "prior click is only a candidate, not focus proof" in focus_text
    assert "No stops the run" in focus_text
    assert "does NOT confirm selected text or task success" in focus_text
    summary = json.loads((tmp_path / "run" / "run-summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "blocked" and summary["executed_actions"] == 1
    assert summary["model_calls"] == 3 and [a[0] for a in world.actions] == ["click"]
    assert "No focus question after ordinary clicks" in capsys.readouterr().out


@pytest.mark.parametrize("kind", ["click", "key", "scroll", "step_complete"])
def test_non_input_decisions_do_not_request_focus(tmp_path: Path, kind: str) -> None:
    world, _, runtime = setup_input_runtime(tmp_path)
    client = FakeClient(world)
    if kind == "key":
        client.action_override = {"status": "action", "action": {"type": "key", "key": "enter", "purpose": "Advance"}}
    elif kind == "scroll":
        client.action_override = {"status": "action", "action": {"type": "scroll", "target_id": "obs-0002:text-1", "target_text": "Input Target", "clicks": 1}}
    elif kind == "step_complete":
        client.action_override = {"status": "step_complete", "checks": [{"type": "window_present", "window": "bound_target"}]}
    runtime.action_agent = GuiActionAgent(client)
    runtime.limits = RunLimits(max_actions=1, settle_seconds=0, feedback_timeout=0)
    result = run_input(runtime, world, supervised_input_focus=True,
                       confirm_input_focus=lambda *args: pytest.fail("Non-input decision must not request focus"))
    assert result.executed_actions <= 1
    assert not any(e["kind"] == "input_focus_confirmation" for e in result.events)
    assert all(a[0] != "write" for a in world.actions)


class MenuRecognizer:
    def __init__(self, world: World) -> None:
        self.world = world

    def recognize(self, image: np.ndarray) -> list[TextElement]:
        elements = [TextElement("Menu", 0.99, (10, 10, 190, 40))]
        if self.world.phase == 1:
            elements.append(TextElement("Open item", 0.99, (10, 60, 190, 90)))
        return elements


class MenuClient(FakeClient):
    def generate(self, request: MultimodalRequest) -> MultimodalResponse:
        response = super().generate(request)
        if self.world.phase == 1 and not request.system_prompt.startswith("You are a GUI task planning"):
            context = json.loads(request.instruction.split("Current observation: ", 1)[1].splitlines()[0])
            target = next(e for e in context["elements"] if e["text"] == "Open item")
            data = json.loads(response.text)
            data["action"] = {"type": "click", "target_id": target["target_id"], "target_text": target["text"]}
            return response.model_copy(update={"text": json.dumps(data)})
        return response


@pytest.mark.parametrize("dismiss_on_authorization", [False, True])
def test_menu_stays_open_without_focus_dialog_but_stale_authorized_item_is_blocked(
    tmp_path: Path, dismiss_on_authorization: bool,
) -> None:
    world, _, _, runtime = setup_runtime(tmp_path)
    runtime.observer.recognizer = MenuRecognizer(world)
    runtime.action_agent = GuiActionAgent(MenuClient(world))
    targets = []

    def authorize(decision, grounded):
        targets.append(decision.action.target_text)
        if decision.action.target_text == "Open item":
            assert world.phase == 1
            if dismiss_on_authorization:
                world.phase = 0
                return True
            return False
        return True

    result = runtime.run("Use the menu", goal(world), execute=True,
                         confirm_plan=lambda *args: True, confirm_action=authorize,
                         supervised_input_focus=True,
                         confirm_input_focus=lambda *args: pytest.fail("Menus must not request input focus"))
    assert targets == ["Menu", "Open item"]
    assert result.executed_actions == 1 and len(world.actions) == 1
    assert len(world.requests) == 3
    assert result.status == ("blocked" if dismiss_on_authorization else "cancelled")
    if dismiss_on_authorization:
        assert "Target disappeared" in result.reason
    assert not any(e["kind"] == "input_focus_confirmation" for e in result.events)


def test_input_candidate_is_invalidated_before_model_when_clicked_target_disappears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    world, _, runtime = setup_input_runtime(tmp_path)
    original_capture = runtime.observer.capture
    client = FakeClient(world)
    runtime.action_agent = GuiActionAgent(client)

    def capture():
        if world.phase == 1 and world.captures >= 4:
            world.missing_target = True
        return original_capture()

    def authorize(decision, grounded):
        client.action_override = {"status": "step_complete", "checks": [{"type": "window_present", "window": "bound_target"}]}
        return True

    monkeypatch.setattr(runtime.observer, "capture", capture)
    result = run_input(runtime, world, supervised_input_focus=True, confirm_action=authorize,
                       confirm_input_focus=lambda *args: pytest.fail("Invalidated candidate must not request focus"))
    assert result.status == "inconclusive" and result.executed_actions == 1
    assert any(e["kind"] == "input_focus_candidate_invalidated" for e in result.events)
    evidence = json.loads(world.requests[-1].instruction.split("Independent focus/selection evidence: ", 1)[1].splitlines()[0])
    assert "input_focus_candidate_target_id" not in evidence
    assert evidence["focused_target_id"] is None


class BboxJitterRecognizer(InputRecognizer):
    def __init__(self, world: World, after_box, before_box=(10, 10, 190, 40)) -> None:
        super().__init__(world)
        self.before_box = before_box
        self.after_box = after_box

    def recognize(self, image: np.ndarray) -> list[TextElement]:
        elements = super().recognize(image)
        first = elements[0]
        elements[0] = TextElement(first.text, first.confidence,
                                  self.after_box if self.world.phase else self.before_box)
        return elements


@pytest.mark.parametrize("shift", [2, 3])
def test_supervised_candidate_accepts_small_bbox_jitter_and_focus_no_stops(
    tmp_path: Path, shift: int,
) -> None:
    world, _, runtime = setup_input_runtime(tmp_path)
    runtime.observer.recognizer = BboxJitterRecognizer(world, (10, 10 - shift, 190, 40))
    questions = []

    def deny(grounded, scene):
        questions.append(scene.observation.observation_id)
        return False

    result = run_input(runtime, world, supervised_input_focus=True, confirm_input_focus=deny)
    assert result.status == "blocked" and "confirmation declined" in result.reason
    assert result.executed_actions == 1 and [a[0] for a in world.actions] == ["click"]
    assert len(questions) == 1 and len(world.requests) == 3 and result.model_decisions == 2
    evidence = json.loads(world.requests[-1].instruction.split("Independent focus/selection evidence: ", 1)[1].splitlines()[0])
    assert evidence["input_focus_candidate_target_id"] == "obs-0004:text-1"
    assert evidence["focused_target_id"] is None and evidence["selection_verified"] is False


@pytest.mark.parametrize("box,before", [
    ((14, 10, 194, 40), (10, 10, 190, 40)),
    ((30, 10, 210, 40), (10, 10, 190, 40)),
    ((11, 11, 13, 13), (10, 10, 12, 12)),
], ids=["4px-over-limit", "obvious-movement", "small-shift-insufficient-overlap"])
def test_supervised_candidate_rejects_displacement_or_low_bbox_overlap(tmp_path: Path, box, before) -> None:
    world, _, runtime = setup_input_runtime(tmp_path)
    runtime.observer.recognizer = BboxJitterRecognizer(world, box, before)
    result = run_input(runtime, world, supervised_input_focus=True,
                       confirm_input_focus=lambda *args: pytest.fail("Invalid candidate must not ask focus"))
    assert result.status == "blocked" and result.executed_actions == 1
    assert any(e["kind"] == "input_focus_unavailable" for e in result.events)
    assert [a[0] for a in world.actions] == ["click"] and len(world.requests) == 3


@pytest.mark.parametrize("change", ["text", "duplicate", "low_confidence"])
def test_supervised_jitter_does_not_accept_changed_ambiguous_or_low_confidence_target(
    tmp_path: Path, change: str,
) -> None:
    world, _, runtime = setup_input_runtime(tmp_path)

    class ChangedRecognizer(BboxJitterRecognizer):
        def recognize(self, image):
            elements = super().recognize(image)
            if world.phase:
                if change == "text":
                    elements[0] = TextElement("Changed input", 0.99, elements[0].box)
                elif change == "low_confidence":
                    elements[0] = TextElement("Input Target", 0.49, elements[0].box)
                else:
                    elements.append(TextElement("Input Target", 0.1, (10, 110, 190, 140)))
            return elements

    runtime.observer.recognizer = ChangedRecognizer(world, (10, 8, 190, 40))
    result = run_input(runtime, world, supervised_input_focus=True,
                       confirm_input_focus=lambda *args: pytest.fail("Invalid candidate must not ask focus"))
    assert result.status == "blocked" and result.executed_actions == 1
    assert any(e["kind"] == "input_focus_unavailable" for e in result.events)
    assert [a[0] for a in world.actions] == ["click"]


def test_supervised_candidate_tolerance_never_accumulates_between_observations(tmp_path: Path) -> None:
    world, _, runtime = setup_input_runtime(tmp_path)
    recognizer = BboxJitterRecognizer(world, (12, 10, 192, 40))
    runtime.observer.recognizer = recognizer

    def authorize(decision, grounded):
        if decision.action.type == "type":
            recognizer.after_box = (14, 10, 194, 40)
        return True

    result = run_input(runtime, world, supervised_input_focus=True, confirm_action=authorize,
                       confirm_input_focus=lambda *args: pytest.fail("Cumulative displacement must block"))
    assert result.status == "blocked" and result.executed_actions == 1
    assert any(e["kind"] == "input_focus_candidate_invalidated" for e in result.events)
    assert [a[0] for a in world.actions] == ["click"]


@pytest.mark.parametrize("when", ["click_authorization", "type_authorization", "focus_confirmation"])
def test_supervised_candidate_tolerance_does_not_relax_actual_action_freshness(
    tmp_path: Path, when: str,
) -> None:
    world, _, runtime = setup_input_runtime(tmp_path)
    recognizer = BboxJitterRecognizer(world, (10, 10, 190, 40))
    runtime.observer.recognizer = recognizer

    def authorize(decision, grounded):
        if when == decision.action.type + "_authorization":
            recognizer.before_box = recognizer.after_box = (10, 8, 190, 40)
        return True

    def focus(*args):
        if when == "focus_confirmation":
            recognizer.after_box = (10, 8, 190, 40)
        return True

    result = run_input(runtime, world, supervised_input_focus=True,
                       confirm_action=authorize, confirm_input_focus=focus)
    assert result.status == "blocked" and "Target moved" in result.reason
    assert [a[0] for a in world.actions] == ([] if when == "click_authorization" else ["click"])
    confirmations = [e for e in result.events if e["kind"] == "input_focus_confirmation"]
    assert len(confirmations) == (0 if when == "click_authorization" else 1)
    assert all(e["confirmed"] is True for e in confirmations)


def test_supervised_input_resolves_latest_bbox_not_original_click_coordinates(tmp_path: Path) -> None:
    world, _, runtime = setup_input_runtime(tmp_path)
    runtime.observer.recognizer = BboxJitterRecognizer(world, (10, 8, 190, 40))
    result = run_input(runtime, world, supervised_input_focus=True, confirm_input_focus=lambda *args: True)
    assert result.status == "completed" and result.executed_actions == 2
    executed = [e for e in result.events if e["kind"] == "executed"]
    assert executed[0]["point"] == (120, 55)
    assert executed[1]["point"] == (120, 54)
    assert executed[0]["decision_observation_id"] != executed[1]["pre_execute_observation_id"]
    assert world.actions[-1] == ("write", world.token)


def test_default_strict_mode_still_blocks_input_after_small_ocr_jitter(tmp_path: Path) -> None:
    world, _, runtime = setup_input_runtime(tmp_path)
    runtime.observer.recognizer = BboxJitterRecognizer(world, (10, 8, 190, 40))
    result = run_input(runtime, world,
                       confirm_input_focus=lambda *args: pytest.fail("Strict mode must never confirm focus"))
    assert result.status == "blocked" and "Strict mode" in result.reason
    assert [a[0] for a in world.actions] == ["click"]
    assert not any(e["kind"] == "input_focus_candidate" for e in result.events)


@pytest.mark.parametrize("answer", [False, None, "yes", 1, {}])
def test_supervised_append_authorization_jitter_reaches_focus_and_denial_stops(
    tmp_path: Path, answer: object,
) -> None:
    world, _, runtime = setup_input_runtime(tmp_path)
    recognizer = BboxJitterRecognizer(world, (10, 8, 190, 40))
    runtime.observer.recognizer = recognizer
    order = []

    def authorize(decision, grounded):
        order.append("authorize_" + decision.action.type)
        if decision.action.type == "type":
            recognizer.after_box = (8, 8, 190, 40)
        return True

    def confirm(grounded, scene):
        assert grounded.action.type == "click"
        assert scene.observation.targets[0].element.box == (8, 8, 190, 40)
        order.append("focus")
        return answer

    result = run_input(runtime, world, supervised_input_focus=True,
                       confirm_action=authorize, confirm_input_focus=confirm)
    assert order == ["authorize_click", "authorize_type", "focus"]
    assert result.status == "blocked" and "confirmation declined" in result.reason
    assert result.executed_actions == result.attempted_actions == 1
    assert [a[0] for a in world.actions] == ["click"]
    assert len(world.requests) == 3 and result.model_decisions == 2 and world.captures == 5
    confirmation = next(e for e in result.events if e["kind"] == "input_focus_confirmation")
    assert confirmation["confirmed"] is False and confirmation["selection_verified"] is False
    assert confirmation["observation_id"] == "obs-0005"


def test_supervised_append_authorization_jitter_yes_still_requires_strict_post_dialog_check(
    tmp_path: Path,
) -> None:
    world, _, runtime = setup_input_runtime(tmp_path)
    recognizer = BboxJitterRecognizer(world, (10, 8, 190, 40))
    runtime.observer.recognizer = recognizer
    captures_at_confirmation = []

    def authorize(decision, grounded):
        if decision.action.type == "type":
            recognizer.after_box = (8, 8, 190, 40)
        return True

    def confirm(*args):
        captures_at_confirmation.append(world.captures)
        return True

    result = run_input(runtime, world, supervised_input_focus=True,
                       confirm_action=authorize, confirm_input_focus=confirm)
    assert captures_at_confirmation == [5] and world.captures == 6
    assert result.status == "blocked" and "Target moved" in result.reason
    assert [a[0] for a in world.actions] == ["click"] and len(world.requests) == 3
    assert next(e for e in result.events if e["kind"] == "input_focus_confirmation")["confirmed"] is True


@pytest.mark.parametrize("timestamp", [3.0, 4.0])
def test_supervised_append_requires_observation_newer_than_type_decision_before_focus(
    tmp_path: Path, timestamp: float,
) -> None:
    world, _, runtime = setup_input_runtime(tmp_path)
    capture = runtime.observer.capture

    def stale_capture():
        frame = capture()
        return ScreenFrame(frame.image, frame.region, timestamp) if world.captures == 5 else frame

    runtime.observer.capture = stale_capture
    result = run_input(runtime, world, supervised_input_focus=True,
                       confirm_input_focus=lambda *args: pytest.fail("Stale type scene must not ask focus"))
    assert result.status == "blocked" and "not newer" in result.reason
    assert [a[0] for a in world.actions] == ["click"] and len(world.requests) == 3


@pytest.mark.parametrize("change", ["movement", "text", "duplicate", "low_confidence"])
def test_supervised_append_revalidates_candidate_after_authorization_before_focus(
    tmp_path: Path, change: str,
) -> None:
    world, _, runtime = setup_input_runtime(tmp_path)
    recognizer = BboxJitterRecognizer(world, (10, 10, 190, 40))
    runtime.observer.recognizer = recognizer

    def authorize(decision, grounded):
        if decision.action.type == "type":
            if change == "movement":
                recognizer.after_box = (14, 10, 194, 40)
            elif change == "text":
                class Changed(InputRecognizer):
                    def recognize(self, image):
                        return [TextElement("Changed input", 0.99, (10, 10, 190, 40))]
                runtime.observer.recognizer = Changed(world)
            else:
                setattr(world, change, True)
        return True

    result = run_input(runtime, world, supervised_input_focus=True, confirm_action=authorize,
                       confirm_input_focus=lambda *args: pytest.fail("Invalid candidate must not ask focus"))
    assert result.status == "blocked" and result.executed_actions == 1
    assert [a[0] for a in world.actions] == ["click"] and len(world.requests) == 3
    assert not any(e["kind"] == "input_focus_confirmation" for e in result.events)
