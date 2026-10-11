from __future__ import annotations

from typing import get_args

import pytest

import gui_agent.control as control_module
from gui_agent.capture import ScreenRegion
from gui_agent.control import DesktopController, VALID_HOTKEYS, VALID_KEYS
from gui_agent.grounding import ActionDecision, AllowedKey, GroundingError, build_observation, resolve_action
from gui_agent.perception import TextElement


class FakePyAutoGUI:
    def __init__(self) -> None:
        self.FAILSAFE = False
        self.calls: list[tuple[object, ...]] = []

    def size(self) -> tuple[int, int]:
        return 800, 600

    def click(self, **kwargs: object) -> None:
        self.calls.append(("click", kwargs))

    def write(self, text: str, *, interval: float) -> None:
        self.calls.append(("write", text, interval))

    def hotkey(self, *keys: str) -> None:
        self.calls.append(("hotkey", *keys))

    def press(self, key: str) -> None:
        self.calls.append(("press", key))

    def moveTo(self, x: int, y: int, *, duration: float) -> None:
        self.calls.append(("moveTo", x, y, duration))

    def scroll(self, clicks: int) -> None:
        self.calls.append(("scroll", clicks))

    def dragTo(self, x: int, y: int, *, duration: float, button: str) -> None:
        self.calls.append(("dragTo", x, y, duration, button))


@pytest.fixture
def backend() -> FakePyAutoGUI:
    return FakePyAutoGUI()


@pytest.fixture
def controller(backend: FakePyAutoGUI) -> DesktopController:
    return DesktopController(backend=backend)


def test_controller_enables_failsafe(backend: FakePyAutoGUI) -> None:
    DesktopController(backend=backend)

    assert backend.FAILSAFE is True


def test_click_calls_backend(controller: DesktopController, backend: FakePyAutoGUI) -> None:
    controller.click((100, 200), duration=0.1)

    assert backend.calls == [
        ("click", {"x": 100, "y": 200, "button": "left", "duration": 0.1})
    ]


def test_type_text_calls_backend(
    controller: DesktopController,
    backend: FakePyAutoGUI,
) -> None:
    controller.type_text("GUI Agent 123", interval=0.05)

    assert backend.calls == [("write", "GUI Agent 123", 0.05)]


def test_paste_text_uses_clipboard_and_ctrl_v(
    controller: DesktopController,
    backend: FakePyAutoGUI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clipboard_values: list[str] = []
    monkeypatch.setattr(control_module.pyperclip, "copy", clipboard_values.append)
    monkeypatch.setattr(control_module.sys, "platform", "win32")

    controller.paste_text("中文输入验证")

    assert clipboard_values == ["中文输入验证"]
    assert backend.calls == [("hotkey", "ctrl", "v")]


def test_paste_text_uses_command_v_on_macos(
    controller: DesktopController,
    backend: FakePyAutoGUI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(control_module.pyperclip, "copy", lambda text: None)
    monkeypatch.setattr(control_module.sys, "platform", "darwin")

    controller.paste_text("中文")

    assert backend.calls == [("hotkey", "command", "v")]


def test_scroll_moves_to_target_first(
    controller: DesktopController,
    backend: FakePyAutoGUI,
) -> None:
    controller.scroll(-3, point=(300, 400))

    assert backend.calls == [
        ("moveTo", 300, 400, 0.0),
        ("scroll", -3),
    ]


def test_drag_moves_to_start_then_drags(
    controller: DesktopController,
    backend: FakePyAutoGUI,
) -> None:
    controller.drag((100, 100), (500, 300), duration=0.4)

    assert backend.calls == [
        ("moveTo", 100, 100, 0.0),
        ("dragTo", 500, 300, 0.4, "left"),
    ]


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (lambda item: item.click((-1, 10)), ValueError, "outside"),
        (lambda item: item.click((800, 10)), ValueError, "outside"),
        (lambda item: item.click((10.5, 10)), TypeError, "integer"),
        (lambda item: item.click((10, 10), button="other"), ValueError, "button"),
        (lambda item: item.click((10, 10), duration=-1), ValueError, "non-negative"),
        (lambda item: item.type_text(""), ValueError, "empty"),
        (lambda item: item.type_text("中文"), ValueError, "ASCII"),
        (lambda item: item.paste_text(""), ValueError, "empty"),
        (lambda item: item.paste_text(123), TypeError, "string"),
        (lambda item: item.scroll(0), ValueError, "zero"),
        (lambda item: item.scroll(1.5), TypeError, "integer"),
    ],
)
def test_invalid_actions_are_rejected(
    controller: DesktopController,
    call: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        call(controller)


@pytest.mark.parametrize("key", sorted(VALID_KEYS))
def test_press_calls_backend_once(
    key: str, controller: DesktopController, backend: FakePyAutoGUI,
) -> None:
    controller.press(key)
    assert backend.calls == [("press", key)]
    assert backend.FAILSAFE is True


@pytest.mark.parametrize("keys", sorted(VALID_HOTKEYS))
def test_hotkey_calls_backend_once_in_order(
    keys: tuple[str, ...], controller: DesktopController, backend: FakePyAutoGUI,
) -> None:
    controller.hotkey(*keys)
    assert backend.calls == [("hotkey", *keys)]
    assert backend.FAILSAFE is True


def test_controller_keys_match_action_contract() -> None:
    assert VALID_KEYS | {"+".join(keys) for keys in VALID_HOTKEYS} == set(get_args(AllowedKey))


@pytest.mark.parametrize(("key", "error"), [
    ("", ValueError), ("return", ValueError), ("delete", ValueError), ("ctrl+a", ValueError),
    (["enter"], TypeError), (None, TypeError), (True, TypeError),
])
def test_invalid_press_never_calls_backend(
    key: object, error: type[Exception], controller: DesktopController, backend: FakePyAutoGUI,
) -> None:
    with pytest.raises(error):
        controller.press(key)
    assert backend.calls == []


@pytest.mark.parametrize(("keys", "error"), [
    ((), ValueError), (("ctrl",), ValueError), (("win", "r"), ValueError),
    (("ctrl", "v"), ValueError), (("alt", "f4", "enter"), ValueError),
    (("a", "ctrl"), ValueError), ((["ctrl", "a"],), TypeError),
    (("ctrl", None), TypeError), (("CTRL", "a"), ValueError),
])
def test_invalid_hotkey_never_calls_backend(
    keys: tuple[object, ...], error: type[Exception],
    controller: DesktopController, backend: FakePyAutoGUI,
) -> None:
    with pytest.raises(error):
        controller.hotkey(*keys)
    assert backend.calls == []


@pytest.mark.parametrize("method", ["press", "hotkey"])
@pytest.mark.parametrize("error_type", [control_module.pyautogui.FailSafeException, KeyboardInterrupt])
def test_key_backend_interrupts_propagate_without_retry(
    method: str, error_type: type[BaseException], controller: DesktopController,
    backend: FakePyAutoGUI, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    error = error_type("Stopped by user")

    def stop(*keys: str) -> None:
        calls.append(keys)
        raise error

    monkeypatch.setattr(backend, method, stop)
    keys = ("enter",) if method == "press" else ("ctrl", "a")
    with pytest.raises(error_type) as caught:
        getattr(controller, method)(*keys)
    assert caught.value is error
    assert calls == [keys]


@pytest.mark.parametrize("focus_verified", [False, None, 1, "true"])
def test_unverified_key_focus_stops_before_controller(
    focus_verified: object, controller: DesktopController, backend: FakePyAutoGUI,
) -> None:
    observation = build_observation(
        "obs-1", "screen.png", [TextElement("INPUT", 0.95, (10, 10, 100, 50))],
        screen_region=ScreenRegion(0, 0, 800, 600), screen_size=(800, 600),
    )
    decision = ActionDecision.model_validate({
        "status": "action", "step_id": "step-1", "observation_id": "obs-1",
        "action": {"type": "key", "key": "ctrl+a", "purpose": "Select old text"},
    })
    with pytest.raises(GroundingError, match="focus"):
        resolved = resolve_action(
            decision, observation, step_id="step-1", focus_verified=focus_verified,
        )
        controller.hotkey(*resolved.action.key.split("+"))
    assert backend.calls == []


def test_verified_select_all_and_replacement_use_public_controller(
    controller: DesktopController, backend: FakePyAutoGUI,
) -> None:
    observation = build_observation(
        "obs-1", "screen.png", [TextElement("INPUT", 0.95, (10, 10, 100, 50))],
        screen_region=ScreenRegion(0, 0, 800, 600), screen_size=(800, 600),
    )
    decision = ActionDecision.model_validate({
        "status": "action", "step_id": "step-1", "observation_id": "obs-1",
        "action": {"type": "key", "key": "ctrl+a", "purpose": "Select old text"},
    })
    resolved = resolve_action(decision, observation, step_id="step-1", focus_verified=True)
    controller.hotkey(*resolved.action.key.split("+"))
    replacement = ActionDecision.model_validate({
        "status": "action", "step_id": "step-1", "observation_id": "obs-1",
        "action": {"type": "type", "target_id": "obs-1:text-1", "target_text": "INPUT",
                   "text": "Week4 Key Smoke", "mode": "replace"},
    })
    with pytest.raises(GroundingError, match="select-all"):
        resolve_action(
            replacement, observation, step_id="step-1", focus_verified=True,
            focused_target_id="obs-1:text-1",
        )
    assert backend.calls == [("hotkey", "ctrl", "a")]
    typed = resolve_action(
        replacement, observation, step_id="step-1", focus_verified=True,
        focused_target_id="obs-1:text-1", selection_verified=True,
    )
    controller.type_text(typed.action.text)
    assert backend.calls == [("hotkey", "ctrl", "a"), ("write", "Week4 Key Smoke", 0.02)]
