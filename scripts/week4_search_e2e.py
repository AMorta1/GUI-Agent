"""Explicit, isolated localhost search scope; not a general unattended GUI mode."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import ctypes
from ctypes import wintypes
from dataclasses import asdict, dataclass
import hashlib
import json
from math import isfinite
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import time
from typing import Callable
from urllib.parse import urlencode


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gui_agent.agent import GuiActionAgent, GuiPlanningAgent, TaskPlan
from gui_agent.control import DesktopController
from gui_agent.grounding import (
    ActionDecision, ClickAction, FeedbackCheck, GroundedAction, GroundingError,
    KeyAction, TypeAction, resolve_target, validate_decision_context,
)
from gui_agent.models import OpenAICompatibleVisionClient
from gui_agent.runtime import DesktopObserver, GuiRuntime, RunLimits, SceneObservation, WindowInfo, WindowsWindowProbe
from scripts import week4_demo as demo
from scripts import week4_focus_capability as uia


TOKEN = "W4-E2E-SEARCH-001"
TASK = (
    "Search this local test page for W4-E2E-SEARCH-001 and verify the new search results. "
    "Stay on the current test page; do not navigate away, use the address bar, or open other applications."
)
MODEL = "glm-4v-flash"
PROFILE = uia.PREP / "edge-search-test-profile"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
RESULT_URL = uia.URL + "?" + urlencode({"q": TOKEN})
LIMITS = RunLimits()
FINAL_TEXTS = ("Search results", "Local result: " + TOKEN)
ONCE_OUTPUT = ROOT.parent / ".test-tmp" / "week4-search-e2e-once-20261008"
FOCUS_RISK = (
    "UIA/window checks are not atomic with keyboard delivery; focus can change after the last "
    "check or during multi-character input. STOP cannot undo or interrupt an already started write."
)


def log_value(value):
    if isinstance(value, float) and not isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: log_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [log_value(item) for item in value]
    return value


class EvidenceLog:
    def __init__(self, output: Path):
        self.output = output

    def emit(self, kind: str, **fields) -> None:
        record = {"kind": kind, "at": time.monotonic(), **fields}
        with (self.output / "safety.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(log_value(record), ensure_ascii=False, allow_nan=False) + "\n")


def windows_arguments(command_line: str) -> list[str]:
    shell = ctypes.WinDLL("shell32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    shell.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    shell.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    count = ctypes.c_int()
    pointer = shell.CommandLineToArgvW(command_line, ctypes.byref(count))
    if not pointer:
        raise GroundingError("Browser command line unavailable")
    try:
        return [pointer[index] for index in range(count.value)]
    finally:
        kernel.LocalFree(pointer)


def read_profile(pid: int) -> dict:
    if type(pid) is not int or pid <= 0:
        raise GroundingError("Bound PID is invalid")
    command = ("[Console]::OutputEncoding=[Text.Encoding]::UTF8;$ErrorActionPreference='Stop';"
               f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}';"
               "if($null -eq $p){throw 'Bound process absent'};"
               "@{pid=[int]$p.ProcessId;exe=$p.ExecutablePath;command_line=$p.CommandLine}|ConvertTo-Json -Compress")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                            capture_output=True, encoding="utf-8", timeout=10, check=True,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    return json.loads(result.stdout.lstrip("\ufeff"))


def validate_profile(record: dict, pid: int) -> None:
    if (type(record.get("pid")) is not int or record["pid"] != pid
            or PureWindowsPath(str(record.get("exe", ""))) != PureWindowsPath(EDGE)):
        raise GroundingError("Bound process is not the dedicated Edge process")
    arguments = windows_arguments(record.get("command_line") or "")
    profiles = []
    for index, argument in enumerate(arguments):
        if argument.startswith("--user-data-dir="):
            profiles.append(argument.split("=", 1)[1])
        elif argument == "--user-data-dir" and index + 1 < len(arguments):
            profiles.append(arguments[index + 1])
    required = {"--disable-sync", "--disable-extensions", "--disable-background-networking",
                "--disable-component-update", "--disable-quic",
                "--proxy-server=http://127.0.0.1:9", "--proxy-bypass-list=127.0.0.1"}
    if (len(profiles) != 1 or PureWindowsPath(profiles[0]) != PureWindowsPath(PROFILE)
            or not required.issubset(arguments)):
        raise GroundingError("Isolated profile/background-network restrictions are missing or changed")


def validate_capture_scope(bound: WindowInfo | None, probe, log, *, profile_reader: Callable = read_profile) -> None:
    if bound is None or "week4 search" not in bound.title.casefold() or "edge" not in bound.title.casefold():
        raise GroundingError("Dedicated search window is not foreground")
    try:
        record = profile_reader(bound.pid)
        validate_profile(record, bound.pid)
    except Exception as error:
        log.emit("initial_capture_scope", passed=False, error=f"{type(error).__name__}: {error}")
        raise GroundingError("Isolated profile identity must be verified before any screenshot") from error
    if probe.read(bound.handle) != bound or probe.foreground() != bound:
        raise GroundingError("Initial window changed during profile verification")
    log.emit("initial_capture_scope", passed=True, window=asdict(bound), profile=record)


def full_window_checks(scene: SceneObservation) -> tuple[FeedbackCheck, ...]:
    GuiRuntime._require_application(scene)
    w, h = scene.observation.image_size
    left, top, right, bottom = scene.window.box
    if (scene.observation.crop_origin != (left, top) or (right - left, bottom - top) != (w, h)
            or scene.observation.allowed_box != (0, 0, w, h)):
        raise GroundingError("Initial bound crop geometry is inconsistent")
    return tuple(FeedbackCheck(type="text_present", text=text, region=(0, 0, w, h)) for text in FINAL_TEXTS)


def ocr_target(scene: SceneObservation, text: str):
    matches = [t for t in scene.observation.targets if t.element.text.strip() == text]
    if len(matches) != 1:
        raise GroundingError("Frozen OCR target is missing or ambiguous")
    target = matches[0]
    point = resolve_target(scene.observation, target.target_id, text)
    return target, point


def ocr_label(scene: SceneObservation) -> dict:
    target, point = ocr_target(scene, uia.LABEL)
    x, y = scene.observation.crop_origin
    a, b, c, d = target.element.box
    return {"text": uia.LABEL, "actual_ocr_text": target.element.text.strip(),
            "target_id": target.target_id, "observation_id": scene.observation.observation_id,
            "confidence": target.element.confidence, "screen_box": [a + x, b + y, c + x, d + y],
            "grounding_point_read_only": point}


@dataclass(frozen=True)
class Scope:
    window: WindowInfo
    checks: tuple[FeedbackCheck, ...]
    limits: RunLimits = LIMITS


@dataclass(frozen=True)
class Ticket:
    decision: ActionDecision
    grounded: GroundedAction
    scene: SceneObservation


@dataclass(frozen=True)
class InputBinding:
    input_id: tuple[int, ...]
    label_id: tuple[int, ...]


class SearchSession:
    """Test-side authorizations and evidence only; never augments model requests."""

    def __init__(self, scope: Scope, observer, probe, log, stop_file: Path, *, enabled: bool = False,
                 profile_reader: Callable = read_profile, profile_validator: Callable = validate_profile,
                 uia_reader: Callable = uia.read_uia, human_authorize: Callable = demo._authorize,
                 clock: Callable = time.monotonic):
        self.scope, self.observer, self.probe, self.log = scope, observer, probe, log
        self.stop_file, self.enabled = stop_file, enabled is True
        self.profile_reader, self.profile_validator = profile_reader, profile_validator
        self.uia_reader, self.human_authorize, self.clock = uia_reader, human_authorize, clock
        self.started_at = None
        self.plan_json = None
        self.step_ids: tuple[str, ...] = ()
        self.ticket: Ticket | None = None
        self.last_click: Ticket | None = None
        self.binding: InputBinding | None = None
        self.focus_callback_verified = False
        self.typed = False
        self.submitted = False
        self.document_id = None
        self.plan_question_asked = False

    def start(self) -> None:
        self.stop_check()
        if self.started_at is not None:
            raise GroundingError("Original runtime budget cannot be restarted")
        self.started_at = self.clock()

    def stop_check(self) -> None:
        if not self.enabled or self.stop_file.exists():
            raise GroundingError("Controlled search disabled or independent STOP requested")
        if self.started_at is not None and self.clock() - self.started_at >= self.scope.limits.timeout_seconds:
            raise GroundingError("Original runtime time budget exceeded before backend/model entry")

    def current_scene(self) -> SceneObservation:
        self.stop_check()
        scene = self.observer.latest
        if scene is None or scene.window != self.scope.window or scene.foreground != self.scope.window:
            raise GroundingError("Latest observation is missing or outside frozen window")
        self.observer.assert_current(scene)
        return scene

    def window_profile_check(self, scene: SceneObservation) -> None:
        self.stop_check()
        if (scene.window != self.scope.window or self.probe.read(self.scope.window.handle) != self.scope.window
                or self.probe.foreground() != self.scope.window):
            raise GroundingError("Frozen window identity/rectangle/foreground changed")
        try:
            record = self.profile_reader(self.scope.window.pid)
            self.profile_validator(record, self.scope.window.pid)
        except Exception as error:
            self.log.emit("profile_check", passed=False, error=f"{type(error).__name__}: {error}")
            raise GroundingError("Independent isolated profile identity is unavailable or invalid") from error
        self.log.emit("profile_check", source="process_metadata", passed=True, record=record)
        self.observer.assert_current(scene)

    def inspect(self, scene: SceneObservation, expected: str, phase: str) -> dict:
        self.window_profile_check(scene)
        try:
            data = self.uia_reader(self.scope.window.handle, expected_value=expected,
                                   allowed_urls=(uia.URL, RESULT_URL) if self.submitted else (uia.URL,))
            if (not isinstance(data, dict)
                    or any(not isinstance(data.get(key), list) for key in ("documents", "edits", "buttons"))
                    or any(not isinstance(item, dict) for key in ("documents", "edits", "buttons")
                           for item in data[key])):
                raise ValueError("UIA evidence collections are missing or invalid")
        except Exception as error:
            self.log.emit("uia_check", phase=phase, passed=False, error=f"{type(error).__name__}: {error}")
            raise GroundingError("Independent UIA evidence unavailable") from error
        self.log.emit("uia_evidence", phase=phase, source="uia_readonly_probe", data=data,
                      serialization_note="Non-finite coordinates become null in logs only; invalid target geometry is rejected")
        documents = [d for d in data.get("documents", [])
                     if d.get("name") == "Week4 Search" and d.get("offscreen") is False
                     and d.get("url_allowed") is True and uia.valid_runtime_id(d.get("runtime_id"))]
        if (len(documents) != 1 or data.get("focused_present") is not True
                or data.get("focused_in_bound") is not True or data.get("focus_stable_during_read") is not True):
            raise GroundingError("Local document/window focus evidence is missing, ambiguous or unstable")
        doc_id = tuple(documents[0]["runtime_id"])
        if self.document_id is None:
            self.document_id = doc_id
        elif not self.submitted and self.document_id != doc_id:
            raise GroundingError("Document changed before authorized submission")
        data["initial_ocr_scope_verified"] = True
        data["initial_ocr_label"] = ocr_label(scene)
        self.observer.assert_current(scene)
        self.stop_check()
        return data

    def focused_input(self, scene: SceneObservation, expected: str, phase: str) -> InputBinding:
        try:
            data = self.inspect(scene, expected, phase)
            result = uia.classify(data, require_empty=expected == "")
            focus = data.get("focused") or {}
            if result["uia_capability"] != "explicit_label_link_observed" or focus.get("value_matches_expected") is not True:
                raise GroundingError(result["reason"] or "Actual input value differs from frozen expected value")
            binding = InputBinding(tuple(focus["runtime_id"]), tuple(focus["labeled_by"]["runtime_id"]))
            if self.binding is None or binding != self.binding:
                raise GroundingError("Actual focused input is not the Agent-clicked input binding")
            self.log.emit("focus_check", source="uia_readonly_probe", phase=phase, passed=True,
                          binding=asdict(binding), observation_id=scene.observation.observation_id,
                          human_answer=None, selection_verified=False)
            return binding
        except Exception as error:
            self.log.emit("focus_check", source="uia_readonly_probe", phase=phase, passed=False,
                          reason=str(error), human_answer=None, selection_verified=False)
            if isinstance(error, GroundingError):
                raise
            raise GroundingError("Independent input focus evidence unavailable") from error

    def declaration(self, plan: TaskPlan) -> dict:
        return {"task": TASK, "model": MODEL, "plan": plan.model_dump(), "window": asdict(self.scope.window),
                "test_profile": str(PROFILE), "local_page": uia.URL, "fixed_search_word": TOKEN,
                "final_checks": [c.model_dump() for c in self.scope.checks], "budgets": asdict(self.scope.limits),
                "allowed_actions": ["click unique Search query label", "append fixed search word once into verified empty input",
                                    "submit once by unique Search button click or Enter in verified input"],
                "forbidden": "address bar/navigation, arbitrary text, replace, clipboard, scroll, other shortcuts/apps",
                "network_boundary": "No Agent external navigation; background egress is NOT proven blocked at OS level",
                "stop_file": str(self.stop_file), "pyautogui_failsafe": True,
                "residual_focus_risk": FOCUS_RISK,
                "selection_verified": False, "authorization_source": "one_human_scope_grant"}

    def confirm_plan(self, plan: TaskPlan, checks: tuple[FeedbackCheck, ...]) -> bool:
        self.stop_check()
        if self.started_at is None or self.plan_question_asked or checks != self.scope.checks or plan.status != "ready":
            raise GroundingError("Plan authorization repeated or frozen final checks changed")
        self.plan_question_asked = True
        declaration = self.declaration(plan)
        frozen_json = json.dumps(declaration, ensure_ascii=False, sort_keys=True)
        self.log.emit("scope_authorization_question", declaration=declaration)
        answer = self.human_authorize("Authorize one controlled local search scope", frozen_json, self.scope.window.handle)
        self.log.emit("scope_authorization_answer", source="human", answer=answer is True)
        if answer is not True:
            return False
        self.stop_check()
        self.plan_json = frozen_json
        self.step_ids = tuple(step.step_id for step in plan.steps)
        self.log.emit("scope_granted", sha256=hashlib.sha256(frozen_json.encode()).hexdigest(), declaration=declaration)
        return True

    def confirm_action(self, decision: ActionDecision, grounded: GroundedAction) -> bool:
        try:
            scene = self.current_scene()
            if self.plan_json is None or self.ticket is not None or self.submitted or decision.step_id not in self.step_ids:
                raise GroundingError("No unused frozen authorization for this action/step")
            validate_decision_context(decision, scene.observation, decision.step_id)
            if decision.status != "action" or decision.action != grounded.action:
                raise GroundingError("Action/grounding does not match the authorized decision")
            action = decision.action
            if isinstance(action, ClickAction):
                if action.target_text not in (uia.LABEL, "Search"):
                    raise GroundingError("Click is outside frozen search controls")
                if action.target_text == "Search" and not self.typed:
                    raise GroundingError("Submission requires the actual authorized input action first")
            elif isinstance(action, TypeAction):
                if (action.target_text != uia.LABEL or action.mode != "append" or action.text != TOKEN
                        or self.typed or self.last_click is None or self.binding is None):
                    raise GroundingError("Input is outside fixed word/append/Agent-clicked scope")
            elif isinstance(action, KeyAction):
                if action.key != "enter" or not self.typed or self.binding is None:
                    raise GroundingError("Key is outside frozen search submission scope")
            else:
                raise GroundingError("Action type is outside frozen scope")
            if not isinstance(action, KeyAction):
                if resolve_target(scene.observation, action.target_id, action.target_text) != grounded.point:
                    raise GroundingError("Authorization coordinates are not current OCR grounding")
            self.window_profile_check(scene)
            self.ticket = Ticket(decision, grounded, scene)
            self.focus_callback_verified = False
            self.log.emit("action_scope_check", source="preauthorized_scope_policy", passed=True,
                          decision=decision.model_dump(), point=grounded.point, human_answer=None)
            return True
        except Exception as error:
            self.log.emit("action_scope_check", source="preauthorized_scope_policy", passed=False, reason=str(error))
            if isinstance(error, GroundingError):
                raise
            raise GroundingError("Action scope evidence unavailable") from error

    def confirm_input_focus(self, clicked: GroundedAction, scene: SceneObservation) -> bool:
        if (self.ticket is None or not isinstance(self.ticket.decision.action, TypeAction)
                or self.last_click is None or clicked != self.last_click.grounded or scene != self.current_scene()):
            raise GroundingError("Focus check is not bound to the actual Agent click/input ticket")
        GuiRuntime._input_focus_target_id(self.last_click.scene, clicked, scene)
        self.focused_input(scene, "", "focus_callback")
        self.focus_callback_verified = True
        return True

    def consume(self, kind: str, *, point=None, text=None, key=None) -> Ticket:
        ticket = self.ticket
        if self.plan_json is None or ticket is None:
            raise GroundingError("Backend action has no frozen per-action ticket")
        action = ticket.decision.action
        if (action.type != kind or (kind == "click" and point != ticket.grounded.point)
                or (kind == "type" and (text != TOKEN or text != action.text or not self.focus_callback_verified))
                or (kind == "key" and (key != "enter" or key != action.key))):
            raise GroundingError("Backend call is outside the model-generated authorized action")
        scene = self.current_scene()
        GuiRuntime._verify_fresh_target(ticket.scene, scene, ticket.grounded)
        if kind == "type":
            if self.last_click is None:
                raise GroundingError("Agent input candidate expired at write entry")
            GuiRuntime._input_focus_target_id(self.last_click.scene, self.last_click.grounded, scene)
            self.focused_input(scene, "", "write_entry")
        elif kind == "key" or action.target_text == "Search":
            self.focused_input(scene, TOKEN, "submit_entry")
            if kind == "click":
                self.check_submit_button(scene, point)
        else:
            self.check_query_click(scene, point)
        self.ticket = None
        self.observer.assert_current(scene)
        self.stop_check()
        self.log.emit("backend_entry_checked", kind_of_action=kind, point=point, observation_id=scene.observation.observation_id)
        return ticket

    def check_query_click(self, scene: SceneObservation, point) -> None:
        data = self.inspect(scene, TOKEN if self.typed else "", "click_entry")
        binding = self.query_binding(data, scene, point)
        if self.binding is not None and binding != self.binding:
            raise GroundingError("Previously clicked input identity changed")
        self.binding = binding

    def query_binding(self, data: dict, scene: SceneObservation, point) -> InputBinding:
        documents = [list(self.document_id)]
        matches = []
        for edit in data.get("edits", []):
            if (uia.label_matches(edit.get("labeled_by")) and edit.get("in_bound") is True
                    and edit.get("control_type") == "ControlType.Edit" and edit.get("enabled") is True
                    and edit.get("offscreen") is False and edit.get("value_pattern") is True
                    and edit.get("value_read_only") is False
                    and any(a.get("runtime_id") in documents for a in edit.get("ancestors", []))):
                matches.append(edit)
        if len(matches) != 1:
            raise GroundingError("Query click has no unique explicit editable UIA label association")
        target = matches[0]
        label = ocr_label(scene)
        if (tuple(label["grounding_point_read_only"]) != point
                or not uia.label_contains_ocr_point(target["labeled_by"], label)
                or target.get("value_matches_expected") is not True
                or not uia.valid_runtime_id(target.get("runtime_id"))):
            raise GroundingError("Query click does not match current OCR/label/value evidence")
        return InputBinding(tuple(target["runtime_id"]), tuple(target["labeled_by"]["runtime_id"]))

    def check_submit_button(self, scene: SceneObservation, point) -> None:
        data = self.inspect(scene, TOKEN, "submit_button")
        document_ids = [list(self.document_id)]
        buttons = [b for b in data.get("buttons", [])
                   if b.get("name") == "Search" and b.get("control_type") == "ControlType.Button"
                   and b.get("in_bound") is True and b.get("enabled") is True and b.get("offscreen") is False
                   and any(a.get("runtime_id") in document_ids for a in b.get("ancestors", []))]
        if len(buttons) != 1 or not uia.finite_coordinates(buttons[0].get("box"), 4):
            raise GroundingError("Submit button is missing, ambiguous or invalid")
        a, b, c, d = buttons[0]["box"]
        if point is None or not (a <= point[0] < c and b <= point[1] < d):
            raise GroundingError("Submit click is outside current UIA button bounds")


class GuardedBackend:
    """Final gate immediately before the real PyAutoGUI method, without dialogs."""

    def __init__(self, raw, session: SearchSession):
        self.raw, self.session = raw, session

    @property
    def FAILSAFE(self):
        return self.raw.FAILSAFE

    @FAILSAFE.setter
    def FAILSAFE(self, value):
        if value is not True:
            raise GroundingError("Controlled search must retain PyAutoGUI FAILSAFE")
        self.raw.FAILSAFE = value

    def size(self):
        return self.raw.size()

    def click(self, *, x, y, button, duration):
        if button != "left":
            raise GroundingError("Only the authorized left click is permitted")
        ticket = self.session.consume("click", point=(x, y))
        self.session.observer.assert_current(self.session.current_scene())
        self.raw.click(x=x, y=y, button=button, duration=duration)
        if ticket.decision.action.target_text == uia.LABEL:
            self.session.last_click = ticket
        else:
            self.session.submitted = True
        self.session.log.emit("backend_executed", action=ticket.decision.action.model_dump())

    def write(self, text, *, interval):
        ticket = self.session.consume("type", text=text)
        self.session.observer.assert_current(self.session.current_scene())
        # No screenshot, prompt, model call or UI interaction after this final gate.
        self.raw.write(text, interval=interval)
        self.session.typed = True
        self.session.log.emit("backend_executed", action=ticket.decision.action.model_dump())

    def press(self, key):
        ticket = self.session.consume("key", key=key)
        self.session.observer.assert_current(self.session.current_scene())
        self.raw.press(key)
        self.session.submitted = True
        self.session.log.emit("backend_executed", action=ticket.decision.action.model_dump())

    def hotkey(self, *keys):
        raise GroundingError("Hotkeys are outside controlled search scope")

    def scroll(self, *args, **kwargs):
        raise GroundingError("Scrolling is outside controlled search scope")

    def moveTo(self, *args, **kwargs):
        raise GroundingError("Standalone movement/drag is outside controlled search scope")


class SearchController(DesktopController):
    def __init__(self, raw_backend, session: SearchSession):
        super().__init__(GuardedBackend(raw_backend, session))

    def paste_text(self, text: str) -> None:
        raise GroundingError("Clipboard input is forbidden in controlled search scope")


class SearchObserver(DesktopObserver):
    def __init__(self, *args, stop_file: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest = None
        self.stop_file = stop_file

    def observe(self):
        if self.stop_file.exists():
            raise GroundingError("Independent STOP requested before observation")
        scene = super().observe()
        self.latest = scene
        with scene.observation.image_path.with_suffix(".json").open("x", encoding="utf-8") as stream:
            json.dump(asdict(scene), stream, ensure_ascii=False, indent=2, default=str)
        return scene


class ScopedRecorder(demo.ResponseRecorder):
    def __init__(self, client, output: Path, session: SearchSession):
        super().__init__(client, output)
        self.session = session

    def generate(self, request):
        scene = self.session.current_scene()
        self.session.inspect(scene, TOKEN if self.session.typed else "", "model_request_boundary")
        response = super().generate(request)
        self.session.stop_check()
        return response


def accepted_result(result, session: SearchSession, before: dict, after: dict) -> bool:
    return (result.status == "completed" and result.executed_actions >= 3 and session.typed and session.submitted
            and before == {"searches": [], "messages": []}
            and after == {"searches": [TOKEN], "messages": []})


def save(output: Path, name: str, value) -> None:
    with (output / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def load_recognizer():
    import easyocr
    return demo.EasyOcrRecognizer(reader=easyocr.Reader(["ch_sim", "en"], gpu=False, download_enabled=False))


def run_once(args) -> int:
    required = ("controlled_local_search", "execute", "allow_screenshot_upload",
                "allow_keyboard_input", "allow_search_submit")
    if any(getattr(args, flag, False) is not True for flag in required):
        raise GroundingError("Explicit local-search, execute, screenshot-upload, keyboard-input and search-submit consent required")
    output = demo._external_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    stop_file = output / "STOP"
    log = EvidenceLog(output)
    result, session, recorder = None, None, None
    before, after = None, None
    error = None
    try:
        save(output, "attempt-claimed.json", {
            "experiment": "New scheme B GLM search E2E; historical Week4 baseline unchanged",
            "task": TASK, "model": MODEL, "fixed_search_word": TOKEN, "profile": str(PROFILE),
            "local_page": uia.URL, "budgets": asdict(LIMITS), "automatic_retry": False,
            "requires_human_plan_scope_grant": True, "residual_focus_risk": FOCUS_RISK,
            "os_level_zero_egress_guarantee": False,
        })
        sources = [Path(__file__), Path(uia.__file__), Path(demo.__file__)]
        sources.extend(ROOT / "gui_agent" / name for name in ("agent.py", "control.py", "runtime.py", "grounding.py", "models.py"))
        save(output, "source-hashes.json", {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
        before = uia.read_records()
        save(output, "service-before.json", before)
        if before != {"searches": [], "messages": []}:
            raise GroundingError("Original empty local service baseline is not present")
        print("EXPLICIT LOCAL SEARCH ONLY. No OS-level zero-egress guarantee. No automatic retry.", flush=True)
        print("One plan/scope approval; no action/focus dialogs. STOP file: " + str(stop_file), flush=True)
        print("Actual fixed-word typing/submission authorized ONLY after plan/scope approval. " + FOCUS_RISK, flush=True)
        if input("Authorize screenshot upload/preparation? Type READY; Ctrl+C cancels: ") != "READY":
            raise GroundingError("Preparation/screenshot consent declined")
        print("Loading cached CPU OCR; downloads disabled, detailed logs saved to run-console.txt.", flush=True)
        with (output / "run-console.txt").open("x", encoding="utf-8") as console:
            with redirect_stdout(console), redirect_stderr(console):
                recognizer = load_recognizer()
            print("Activate ONLY the isolated empty search browser within 8 seconds. Stay until THREE ending tones.", flush=True)
            time.sleep(8)
            probe = WindowsWindowProbe()
            bound = probe.foreground()
            validate_capture_scope(bound, probe, log)
            observer = SearchObserver(bound, output / "observations", recognizer=recognizer,
                                      window_probe=probe, stop_file=stop_file)
            with redirect_stdout(console), redirect_stderr(console):
                initial = observer.observe()
            checks = full_window_checks(initial)
            ocr_target(initial, uia.LABEL)
            if any(t.element.text.strip() in FINAL_TEXTS or TOKEN in t.element.text for t in initial.observation.targets):
                raise GroundingError("Original empty/no-results search start is missing")
            for name, value in (("final-checks.json", [c.model_dump() for c in checks]),
                                ("bound-window.json", asdict(bound))):
                with (output / name).open("x", encoding="utf-8") as stream:
                    json.dump(value, stream, ensure_ascii=False, indent=2)
            print("Frozen final checks: " + json.dumps([c.model_dump() for c in checks]), flush=True)
            session = SearchSession(Scope(bound, checks), observer, probe, log, stop_file, enabled=True)
            initial_uia = session.inspect(initial, "", "initial_scope")
            binding = session.query_binding(initial_uia, initial, ocr_target(initial, uia.LABEL)[1])
            save(output, "initial-empty-input.json", {"binding": asdict(binding), "empty": True})
            endpoint = os.environ.get("GUI_AGENT_API_BASE")
            if (not endpoint or not os.environ.get("GUI_AGENT_API_KEY")
                    or os.environ.get("GUI_AGENT_API_MODEL") != MODEL):
                raise GroundingError("Original API endpoint/key configuration is missing")
            client = OpenAICompatibleVisionClient(MODEL, endpoint, api_key_env="GUI_AGENT_API_KEY", max_retries=0)
            recorder = ScopedRecorder(client, output, session)
            controller = SearchController(demo.DesktopController()._backend, session)
            runtime = GuiRuntime(GuiPlanningAgent(recorder), GuiActionAgent(recorder), observer, controller, limits=LIMITS)
            session.start()
            with redirect_stdout(console), redirect_stderr(console):
                result = runtime.run(TASK, checks, execute=True, confirm_plan=session.confirm_plan,
                                     confirm_action=session.confirm_action, supervised_input_focus=True,
                                     confirm_input_focus=session.confirm_input_focus)
    except (Exception, KeyboardInterrupt) as exc:
        error = f"{type(exc).__name__}: {exc}"
        log.emit("entry_stopped", error=error)
    finally:
        try:
            after = uia.read_records()
            save(output, "service-after.json", after)
        except (Exception, KeyboardInterrupt) as exc:
            error = error or f"Independent records check unavailable: {type(exc).__name__}: {exc}"
        accepted = bool(error is None and result is not None and session is not None
                        and accepted_result(result, session, before, after))
        summary = {"scope": "Explicit isolated localhost search, no execution-time dialogs",
                   "runtime": result.to_dict() if result else None, "entry_error": error,
                   "model_calls": recorder.calls if recorder else 0, "accepted_success": accepted,
                   "real_input_returned": bool(session and session.typed),
                   "real_submit_returned": bool(session and session.submitted),
                   "scope_granted": bool(session and session.plan_json is not None),
                   "records_before": before, "records_after": after, "selection_verified": False,
                   "runtime_exit_code": None if result is None else 0 if result.status == "completed" else 1,
                   "entry_exit_code": 0 if accepted else 1, "residual_focus_risk": FOCUS_RISK,
                   "os_level_zero_egress_guarantee": False, "stop_file": str(stop_file)}
        with (output / "run-summary.json").open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
        print("ALL MODEL/DESKTOP WORK STOPPED; ending tones indicate completion only.", flush=True)
        try:
            uia.signal_done()
        except Exception as exc:
            log.emit("ending_tones_unavailable", error=f"{type(exc).__name__}: {exc}")
    print(f"status={result.status if result else 'preparation_blocked'} accepted_success={accepted}")
    print("reason=" + (error or (result.reason if result else "Preparation did not complete")))
    print("evidence=" + str(output))
    print("STOP: preserve evidence. No automatic retry or next task.")
    print("entry_exit_code=" + str(summary["entry_exit_code"]))
    return 0 if accepted else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-one-e2e", action="store_true", help="Claim the fixed third-step experiment directory once")
    parser.add_argument("--controlled-local-search", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-screenshot-upload", action="store_true")
    parser.add_argument("--allow-keyboard-input", action="store_true")
    parser.add_argument("--allow-search-submit", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.run_one_e2e:
        if args.output_dir is not None:
            parser.error("--run-one-e2e has a fixed output directory; --output-dir is forbidden")
        args.controlled_local_search = True
        args.output_dir = ONCE_OUTPUT
    elif not args.controlled_local_search or args.output_dir is None:
        parser.error("Use --run-one-e2e or explicit --controlled-local-search with --output-dir")
    try:
        return run_once(args)
    except Exception as exc:
        print(f"Stopped before run: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
