"""Bounded, supervised GUI execution with independent screenshot feedback."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import asdict, dataclass, field
import json
from math import isfinite
from pathlib import Path
import sys
import time
from typing import Callable, Literal, Sequence

import cv2
import pyautogui

from .agent import GuiActionAgent, GuiPlanningAgent, TaskPlan
from .capture import ScreenFrame, capture_screen
from .control import DesktopController
from .grounding import (
    ActionDecision, ClickAction, FeedbackCheck, GroundedAction, GroundingError,
    KeyAction, OcrObservation, ScrollAction, TypeAction, build_observation,
    evaluate_text_check, resolve_action, resolve_target, validate_decision_context,
)
from .perception import EasyOcrRecognizer, PixelBox


SUPERVISED_INPUT_MAX_BBOX_SHIFT = 3
SUPERVISED_INPUT_MIN_BBOX_IOU = 0.85


@dataclass(frozen=True)
class WindowInfo:
    handle: int
    pid: int
    title: str
    box: PixelBox

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in (self.handle, self.pid)):
            raise ValueError("Window handle and PID must be positive integers")
        if len(self.box) != 4 or any(type(value) is not int for value in self.box):
            raise ValueError("Window box must contain four integers")
        if self.box[0] >= self.box[2] or self.box[1] >= self.box[3]:
            raise ValueError("Window box must have positive dimensions")


class WindowsWindowProbe:
    """Read a bound/foreground window only; never activate or enumerate windows."""

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("Window safety checks are currently Windows-only")
        self._api = ctypes.WinDLL("user32", use_last_error=True)
        signatures = {
            "GetForegroundWindow": ([], wintypes.HWND),
            "IsWindow": ([wintypes.HWND], wintypes.BOOL),
            "GetWindowRect": ([wintypes.HWND, ctypes.POINTER(wintypes.RECT)], wintypes.BOOL),
            "GetWindowThreadProcessId": ([wintypes.HWND, ctypes.POINTER(wintypes.DWORD)], wintypes.DWORD),
            "GetWindowTextLengthW": ([wintypes.HWND], ctypes.c_int),
            "GetWindowTextW": ([wintypes.HWND, wintypes.LPWSTR, ctypes.c_int], ctypes.c_int),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self._api, name)
            function.argtypes = arguments
            function.restype = result

    def foreground(self) -> WindowInfo | None:
        handle = self._api.GetForegroundWindow()
        return self.read(int(handle)) if handle else None

    def read(self, handle: int) -> WindowInfo | None:
        if type(handle) is not int or handle <= 0:
            raise ValueError("Window handle must be a positive integer")
        if not self._api.IsWindow(handle):
            return None
        rect = wintypes.RECT()
        pid = wintypes.DWORD()
        if not self._api.GetWindowRect(handle, ctypes.byref(rect)):
            raise RuntimeError("Cannot read bound window rectangle")
        if not self._api.GetWindowThreadProcessId(handle, ctypes.byref(pid)) or not pid.value:
            raise RuntimeError("Cannot read bound window identity")
        length = self._api.GetWindowTextLengthW(handle)
        title = ctypes.create_unicode_buffer(max(0, length) + 1)
        self._api.GetWindowTextW(handle, title, len(title))
        if not self._api.IsWindow(handle):
            raise GroundingError("Window disappeared while reading its identity")
        return WindowInfo(handle, int(pid.value), title.value, (rect.left, rect.top, rect.right, rect.bottom))


@dataclass(frozen=True)
class SceneObservation:
    observation: OcrObservation
    window: WindowInfo | None
    foreground: WindowInfo | None
    captured_at: float

    @property
    def application_focused(self) -> bool:
        return self.window is not None and self.foreground == self.window


class DesktopObserver:
    """Save/OCR only the explicitly bound window's primary-screen rectangle."""

    def __init__(
        self, bound_window: WindowInfo, output_dir: str | Path, *,
        recognizer: EasyOcrRecognizer,
        window_probe: WindowsWindowProbe,
        capture: Callable[[], ScreenFrame] = capture_screen,
    ) -> None:
        self.bound_window = bound_window
        self.output_dir = Path(output_dir)
        self.recognizer = recognizer
        self.window_probe = window_probe
        self.capture = capture
        self._sequence = 0

    def observe(self) -> SceneObservation:
        window = self.window_probe.read(self.bound_window.handle)
        if window is not None and (window.handle, window.pid, window.box) != (
            self.bound_window.handle, self.bound_window.pid, self.bound_window.box,
        ):
            raise GroundingError("Bound window identity or rectangle changed")
        foreground = self.window_probe.foreground()
        if window is not None and foreground != window:
            raise GroundingError("Bound window is not foreground; screenshot could be occluded")
        frame = self.capture()
        if (frame.region.left, frame.region.top) != (0, 0):
            raise GroundingError("Only the origin-zero primary screen is supported")
        left, top, right, bottom = self.bound_window.box
        if not (0 <= left < right <= frame.region.width and 0 <= top < bottom <= frame.region.height):
            raise GroundingError("Bound window is outside the primary screen")
        image = frame.image[top:bottom, left:right].copy()
        elements = self.recognizer.recognize(image) if window is not None else []
        self._sequence += 1
        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_path = self.output_dir / f"obs-{self._sequence:04d}.png"
        if image_path.exists():
            raise FileExistsError("Observation output exists; do not overwrite prior evidence")
        if not cv2.imwrite(str(image_path), image):
            raise RuntimeError("Cannot save observation image")
        observation = build_observation(
            f"obs-{self._sequence:04d}", image_path, elements,
            screen_region=frame.region, screen_size=frame.region.size,
            crop_origin=(left, top), image_size=(right - left, bottom - top),
        )
        scene = SceneObservation(observation, window, foreground, frame.captured_at)
        self.assert_current(scene)
        return scene

    def assert_current(self, scene: SceneObservation) -> None:
        window = self.window_probe.read(self.bound_window.handle)
        if window != scene.window:
            raise GroundingError("Bound window changed since observation")
        if window is not None and self.window_probe.foreground() != window:
            raise GroundingError("Application focus was lost since observation")


@dataclass(frozen=True)
class RunLimits:
    max_actions: int = 20
    max_step_actions: int = 6
    max_decisions: int = 40
    timeout_seconds: float = 300.0
    settle_seconds: float = 0.25
    feedback_timeout: float = 2.0
    poll_interval: float = 0.25

    def __post_init__(self) -> None:
        for name in ("max_actions", "max_step_actions", "max_decisions"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("timeout_seconds", "settle_seconds", "feedback_timeout", "poll_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.timeout_seconds == 0 or self.poll_interval == 0:
            raise ValueError("timeout_seconds and poll_interval must be positive")


@dataclass
class RunResult:
    status: Literal["completed", "needs_clarification", "blocked", "failed", "cancelled", "inconclusive"]
    reason: str
    mode: str
    executed_actions: int = 0
    attempted_actions: int = 0
    model_decisions: int = 0
    completed_steps: list[str] = field(default_factory=list)
    plan: dict[str, object] | None = None
    events: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class GuiRuntime:
    """Strict by default; optional human focus evidence never proves selection."""

    def __init__(
        self, planning_agent: GuiPlanningAgent, action_agent: GuiActionAgent,
        observer: DesktopObserver, controller: DesktopController, *,
        limits: RunLimits | None = None,
        wait: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.planning_agent = planning_agent
        self.action_agent = action_agent
        self.observer = observer
        self.controller = controller
        self.limits = limits or RunLimits()
        self.wait = wait
        self.clock = clock

    def run(
        self, task: str, final_checks: Sequence[FeedbackCheck], *, execute: bool = False,
        confirm_plan: Callable[[TaskPlan, tuple[FeedbackCheck, ...]], bool] | None = None,
        confirm_action: Callable[[ActionDecision, GroundedAction], bool] | None = None,
        supervised_input_focus: bool = False,
        confirm_input_focus: Callable[[GroundedAction, SceneObservation], bool] | None = None,
    ) -> RunResult:
        result = RunResult("failed", "Run has not completed", "execute" if execute else "dry_run")
        deadline = self.clock() + self.limits.timeout_seconds
        pending_click: tuple[SceneObservation, GroundedAction] | None = None

        def budget() -> None:
            if self.clock() >= deadline:
                raise GroundingError("Run time budget exceeded at a call/action boundary")

        def observe() -> SceneObservation:
            nonlocal pending_click
            budget()
            scene = self.observer.observe()
            result.events.append({
                "kind": "observation", "observation_id": scene.observation.observation_id,
                "image_path": str(scene.observation.image_path), "captured_at": scene.captured_at,
                "screen_size": scene.observation.screen_size,
                "screen_region": asdict(scene.observation.screen_region),
                "crop_origin": scene.observation.crop_origin,
                "window": asdict(scene.window) if scene.window else None,
                "foreground": asdict(scene.foreground) if scene.foreground else None,
                "context": scene.observation.context(),
            })
            if pending_click is not None:
                try:
                    self._input_focus_target_id(*pending_click, scene)
                except GroundingError as exc:
                    pending_click = None
                    result.events.append({"kind": "input_focus_candidate_invalidated", "reason": str(exc),
                                          "observation_id": scene.observation.observation_id})
            budget()
            return scene

        def verify(checks: Sequence[FeedbackCheck], scene: SceneObservation) -> tuple[bool, SceneObservation]:
            until = min(deadline, self.clock() + self.limits.feedback_timeout)
            while True:
                self.observer.assert_current(scene)
                outcomes = [self._check(check, scene) for check in checks]
                result.events.append({"kind": "feedback", "observation_id": scene.observation.observation_id,
                                      "checks": [check.model_dump() for check in checks], "passed": outcomes})
                if all(outcomes):
                    return True, scene
                if self.clock() >= until:
                    return False, scene
                self.wait(min(self.limits.poll_interval, until - self.clock()))
                scene = observe()

        try:
            if type(execute) is not bool or not isinstance(task, str) or not task.strip():
                raise ValueError("execute must be boolean and task must be non-blank text")
            if execute and (confirm_plan is None or confirm_action is None):
                raise GroundingError("Execute requires plan/goal and per-action authorization callbacks")
            if type(supervised_input_focus) is not bool:
                raise ValueError("supervised_input_focus must be boolean")
            if supervised_input_focus and (not execute or not callable(confirm_input_focus)):
                raise GroundingError("Supervised input focus requires execute and a focus confirmation callback")
            if supervised_input_focus:
                result.events.append({"kind": "input_focus_mode", "supervised": True, "selection_verified": False})
            checks = tuple(FeedbackCheck.model_validate(check.model_dump()) for check in final_checks)
            if not checks:
                raise ValueError("Independent final goal checks are required")
            scene = observe()
            self._require_application(scene)
            for check in checks:
                self._validate_check_region(check, scene)
            plan = self.planning_agent.plan(
                task.strip(), image_path=scene.observation.image_path,
                context=json.dumps(scene.observation.context(), ensure_ascii=False),
            )
            budget()
            plan = TaskPlan.model_validate(plan.model_dump())
            result.plan = plan.model_dump()
            if plan.status == "needs_clarification":
                result.status, result.reason = "needs_clarification", plan.clarification_question or "Clarification required"
                return result
            if execute and confirm_plan(plan.model_copy(deep=True), checks) is not True:
                result.status, result.reason = "cancelled", "Plan/final goal authorization declined"
                return result
            budget()
            scene = observe()
            for step in plan.steps:
                step_actions = 0
                while True:
                    budget()
                    self._require_application(scene)
                    self.observer.assert_current(scene)
                    if result.model_decisions >= self.limits.max_decisions:
                        raise GroundingError("Model decision limit reached")
                    result.model_decisions += 1
                    candidate_id = self._input_focus_target_id(*pending_click, scene) if pending_click else None
                    # A prior click is only a candidate; human focus evidence is obtained on demand.
                    decision = self.action_agent.decide(
                        task.strip(), plan, step_id=step.step_id, observation=scene.observation,
                        completed_steps=tuple(result.completed_steps),
                        focus_verified=scene.application_focused,
                        focused_target_id=None, selection_verified=False,
                        input_focus_candidate_target_id=candidate_id,
                    )
                    budget()
                    decision = ActionDecision.model_validate(decision.model_dump())
                    validate_decision_context(decision, scene.observation, step.step_id)
                    result.events.append({"kind": "decision", "decision": decision.model_dump()})
                    if decision.status == "blocked":
                        result.status, result.reason = "blocked", decision.reason or "Model blocked"
                        return result
                    if decision.status == "step_complete":
                        if not execute:
                            result.status, result.reason = "blocked", "dry_run: checks proposed, no execution"
                            return result
                        passed, scene = verify(decision.checks, observe())
                        if not passed:
                            result.status, result.reason = "inconclusive", "Step evidence did not pass independent verification"
                            return result
                        result.completed_steps.append(step.step_id)
                        break
                    typing = isinstance(decision.action, TypeAction)
                    if typing:
                        if not supervised_input_focus:
                            raise GroundingError("Strict mode: input focus/selection is not independently verified")
                        if pending_click is None or candidate_id != decision.action.target_id:
                            raise GroundingError("type action requires a valid matching Agent-clicked input candidate")
                        if decision.action.mode == "replace":
                            raise GroundingError("replace requires a separately verified select-all micro-action")
                        # Preview for authorization only; resolve_action still requires confirmed focus.
                        grounded = GroundedAction(decision.action, resolve_target(
                            scene.observation, decision.action.target_id, decision.action.target_text,
                        ))
                    else:
                        grounded = resolve_action(
                            decision, scene.observation, step_id=step.step_id,
                            focus_verified=scene.application_focused,
                        )
                    if not execute:
                        result.events.append({"kind": "preview", "point": grounded.point})
                        result.status, result.reason = "blocked", "dry_run: proposed action was not executed"
                        return result
                    if result.attempted_actions >= self.limits.max_actions or step_actions >= self.limits.max_step_actions:
                        raise GroundingError("Total/per-step action limit reached")
                    if confirm_action(decision, grounded) is not True:
                        result.status, result.reason = "cancelled", "Action authorization declined"
                        return result
                    budget()
                    fresh = observe()
                    self._require_application(fresh)
                    if typing:
                        # A valid candidate permits a focus question, not execution.
                        self._verify_fresh_scene(scene, fresh)
                    else:
                        self._verify_fresh_target(scene, fresh, grounded)
                    self.observer.assert_current(fresh)
                    if typing:
                        if pending_click is None:
                            raise GroundingError("Input focus candidate expired before confirmation")
                        target_id = self._input_focus_target_id(*pending_click, fresh)
                        answer = confirm_input_focus(pending_click[1], fresh) is True
                        result.events.append({"kind": "input_focus_confirmation", "confirmed": answer,
                                              "click_observation_id": pending_click[0].observation.observation_id,
                                              "decision_observation_id": scene.observation.observation_id,
                                              "observation_id": fresh.observation.observation_id,
                                              "target_id": target_id, "target_text": decision.action.target_text,
                                              "point": grounded.point, "selection_verified": False})
                        if not answer:
                            raise GroundingError("Input focus confirmation declined or not literal True; stopped")
                        fresh = observe()
                        self._require_application(fresh)
                        self._verify_fresh_target(scene, fresh, grounded)
                        if pending_click is None:
                            raise GroundingError("Input focus candidate expired during confirmation")
                        target_id = self._input_focus_target_id(*pending_click, fresh)
                        grounded = resolve_action(
                            decision, scene.observation, step_id=step.step_id,
                            focus_verified=fresh.application_focused,
                            focused_target_id=decision.action.target_id, selection_verified=False,
                        )
                        # Preserve model IDs; execution evidence comes from the latest crop.
                        grounded = GroundedAction(grounded.action, resolve_target(
                            fresh.observation, target_id, decision.action.target_text,
                        ))
                        self.observer.assert_current(fresh)
                    budget()
                    # Every action consumes the candidate, including keys and text input.
                    pending_click = None
                    result.attempted_actions += 1
                    self._execute(grounded, input_focus_verified=typing)
                    result.executed_actions += 1
                    step_actions += 1
                    result.events.append({"kind": "executed", "step_id": step.step_id,
                                          "decision_observation_id": scene.observation.observation_id,
                                          "pre_execute_observation_id": fresh.observation.observation_id,
                                          "action": decision.action.model_dump(), "point": grounded.point})
                    self.wait(self.limits.settle_seconds)
                    clicked_scene = scene
                    scene = observe()
                    if scene.window is None:
                        passed, scene = verify(checks, scene)
                        result.status = "completed" if passed else "inconclusive"
                        result.reason = "Final goal verified after bound window closed" if passed else "Bound window closed without verified final goal"
                        return result
                    if supervised_input_focus and isinstance(decision.action, ClickAction):
                        try:
                            target_id = self._input_focus_target_id(clicked_scene, grounded, scene)
                        except GroundingError as exc:
                            result.events.append({"kind": "input_focus_unavailable", "reason": str(exc)})
                        else:
                            pending_click = (clicked_scene, grounded)
                            result.events.append({"kind": "input_focus_candidate", "focus_verified": False,
                                                  "click_observation_id": clicked_scene.observation.observation_id,
                                                  "observation_id": scene.observation.observation_id,
                                                  "target_id": target_id, "target_text": decision.action.target_text,
                                                  "point": grounded.point, "selection_verified": False})
            passed, scene = verify(checks, observe())
            result.status = "completed" if passed else "inconclusive"
            result.reason = "Independent final goal checks passed" if passed else "Final goal checks did not pass"
        except (KeyboardInterrupt, pyautogui.FailSafeException) as exc:
            result.status, result.reason = "cancelled", f"User/fail-safe interruption: {exc}"
        except GroundingError as exc:
            result.status, result.reason = "blocked", str(exc)
        except Exception as exc:
            result.status, result.reason = "failed", f"{type(exc).__name__}: {exc}"
        return result

    @staticmethod
    def _require_application(scene: SceneObservation) -> None:
        if not scene.application_focused:
            raise GroundingError("Bound application must be present and foreground")

    @staticmethod
    def _validate_check_region(check: FeedbackCheck, scene: SceneObservation) -> None:
        if check.region is not None:
            left, top, right, bottom = scene.observation.allowed_box
            a, b, c, d = check.region
            if not (left <= a < c <= right and top <= b < d <= bottom):
                raise GroundingError("Goal/check region lies outside the allowed observation")

    def _check(self, check: FeedbackCheck, scene: SceneObservation) -> bool:
        self._validate_check_region(check, scene)
        if check.type.startswith("text_"):
            return scene.window is not None and evaluate_text_check(check, scene.observation)
        if check.type == "window_closed":
            return scene.window is None
        return scene.window is not None

    @staticmethod
    def _verify_fresh_scene(before: SceneObservation, fresh: SceneObservation) -> None:
        old, new = before.observation, fresh.observation
        if not isfinite(fresh.captured_at) or fresh.captured_at <= before.captured_at:
            raise GroundingError("Pre-execution capture is not newer than the decision image")
        if before.window != fresh.window or before.foreground != fresh.foreground:
            raise GroundingError("Window identity, rectangle or focus changed before execution")
        if (old.screen_size, old.screen_region, old.crop_origin, old.image_size, old.allowed_box) != (
            new.screen_size, new.screen_region, new.crop_origin, new.image_size, new.allowed_box,
        ):
            raise GroundingError("Screen/crop geometry changed before execution")

    @staticmethod
    def _verify_fresh_target(before: SceneObservation, fresh: SceneObservation, grounded: GroundedAction) -> None:
        GuiRuntime._verify_fresh_scene(before, fresh)
        old, new = before.observation, fresh.observation
        action = grounded.action
        if isinstance(action, (ClickAction, ScrollAction, TypeAction)):
            target = next(target for target in old.targets if target.target_id == action.target_id)
            matches = [target for target in new.targets if " ".join(target.element.text.split()).casefold() == " ".join(action.target_text.split()).casefold()]
            if len(matches) != 1:
                raise GroundingError("Target disappeared or became ambiguous before execution")
            current = matches[0]
            point = resolve_target(new, current.target_id, action.target_text)
            if current.element.box != target.element.box or point != grounded.point:
                raise GroundingError("Target moved before execution; stale coordinates rejected")
        elif isinstance(action, KeyAction):
            old_text = [(target.element.text, target.element.box) for target in old.targets]
            new_text = [(target.element.text, target.element.box) for target in new.targets]
            if old_text != new_text:
                raise GroundingError("Visible OCR context changed before key execution")

    @staticmethod
    def _input_focus_target_id(clicked: SceneObservation, grounded: GroundedAction, current: SceneObservation) -> str:
        """Match a supervised click candidate, never focus proof or action permission."""
        if not isinstance(grounded.action, ClickAction):
            raise GroundingError("Input focus must be bound to an Agent click")
        GuiRuntime._verify_fresh_scene(clicked, current)
        text = " ".join(grounded.action.target_text.split()).casefold()
        matches = [target for target in current.observation.targets
                   if " ".join(target.element.text.split()).casefold() == text]
        if len(matches) != 1:
            raise GroundingError("Input candidate disappeared or became ambiguous")
        target = matches[0]
        old_point = resolve_target(clicked.observation, grounded.action.target_id, grounded.action.target_text)
        if old_point != grounded.point:
            raise GroundingError("Input candidate does not match the actual Agent click")
        resolve_target(current.observation, target.target_id, grounded.action.target_text)
        original = next(t for t in clicked.observation.targets if t.target_id == grounded.action.target_id)
        a, b = original.element.box, target.element.box
        if any(abs(old - new) > SUPERVISED_INPUT_MAX_BBOX_SHIFT for old, new in zip(a, b)):
            raise GroundingError("Input candidate bbox moved beyond the supervised tolerance")
        overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
        # Anchor every comparison to the original click, not the last jittered bbox.
        if overlap / union < SUPERVISED_INPUT_MIN_BBOX_IOU:
            raise GroundingError("Input candidate bboxes do not overlap sufficiently")
        return target.target_id

    def _execute(self, grounded: GroundedAction, *, input_focus_verified: bool = False) -> None:
        action = grounded.action
        if isinstance(action, ClickAction):
            self.controller.click(grounded.point)
        elif isinstance(action, ScrollAction):
            self.controller.scroll(action.clicks, point=grounded.point)
        elif isinstance(action, KeyAction):
            keys = action.key.split("+")
            if len(keys) == 1:
                self.controller.press(keys[0])
            else:
                self.controller.hotkey(*keys)
        elif isinstance(action, TypeAction) and input_focus_verified and action.mode == "append":
            if action.text.isascii():
                self.controller.type_text(action.text)
            else:
                self.controller.paste_text(action.text)
        else:
            raise GroundingError("Strict mode forbids unverified text input")
