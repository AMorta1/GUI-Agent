from __future__ import annotations

import pytest

import gui_agent.control as control_module
from gui_agent.control import DesktopController


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
