"""Validated mouse and keyboard actions for the local desktop."""

from __future__ import annotations

from math import isfinite
from numbers import Real
import sys
from typing import Any, TypeAlias

import pyautogui
import pyperclip


ScreenPoint: TypeAlias = tuple[int, int]
VALID_BUTTONS = {"left", "middle", "right"}


class DesktopController:
    """Run explicit PyAutoGUI actions while retaining its corner fail-safe."""

    def __init__(self, backend: Any = None) -> None:
        self._backend = backend if backend is not None else pyautogui
        self._backend.FAILSAFE = True

    def click(
        self,
        point: ScreenPoint,
        *,
        button: str = "left",
        duration: float = 0.0,
    ) -> None:
        x, y = self._validate_point(point)
        self._validate_button(button)
        self._validate_duration(duration)
        self._backend.click(x=x, y=y, button=button, duration=float(duration))

    def type_text(self, text: str, *, interval: float = 0.02) -> None:
        if not isinstance(text, str):
            raise TypeError("Text input must be a string")
        if not text:
            raise ValueError("Text input must not be empty")
        if not text.isascii():
            raise ValueError("Week 2 text input supports ASCII only")
        self._validate_duration(interval, name="Typing interval")
        self._backend.write(text, interval=float(interval))

    def paste_text(self, text: str) -> None:
        """Paste Unicode text through the system clipboard.

        The clipboard is intentionally not restored after the paste operation.
        """

        if not isinstance(text, str):
            raise TypeError("Text input must be a string")
        if not text:
            raise ValueError("Text input must not be empty")

        pyperclip.copy(text)
        modifier = "command" if sys.platform == "darwin" else "ctrl"
        self._backend.hotkey(modifier, "v")

    def scroll(self, clicks: int, *, point: ScreenPoint | None = None) -> None:
        if not isinstance(clicks, int) or isinstance(clicks, bool):
            raise TypeError("Scroll clicks must be an integer")
        if clicks == 0:
            raise ValueError("Scroll clicks must not be zero")
        if point is not None:
            x, y = self._validate_point(point)
            self._backend.moveTo(x, y, duration=0.0)
        self._backend.scroll(clicks)

    def drag(
        self,
        start: ScreenPoint,
        end: ScreenPoint,
        *,
        duration: float = 0.5,
        button: str = "left",
    ) -> None:
        start_x, start_y = self._validate_point(start)
        end_x, end_y = self._validate_point(end)
        self._validate_button(button)
        self._validate_duration(duration)
        self._backend.moveTo(start_x, start_y, duration=0.0)
        self._backend.dragTo(
            end_x,
            end_y,
            duration=float(duration),
            button=button,
        )

    def _validate_point(self, point: ScreenPoint) -> ScreenPoint:
        if len(point) != 2 or any(
            not isinstance(value, int) or isinstance(value, bool) for value in point
        ):
            raise TypeError("Screen point must contain integer x and y coordinates")
        x, y = point
        screen_width, screen_height = self._backend.size()
        if not 0 <= x < screen_width or not 0 <= y < screen_height:
            raise ValueError("Screen point lies outside the primary screen")
        return x, y

    @staticmethod
    def _validate_button(button: str) -> None:
        if button not in VALID_BUTTONS:
            raise ValueError(f"Mouse button must be one of: {sorted(VALID_BUTTONS)}")

    @staticmethod
    def _validate_duration(duration: float, *, name: str = "Duration") -> None:
        if (
            not isinstance(duration, Real)
            or isinstance(duration, bool)
            or not isfinite(float(duration))
            or duration < 0
        ):
            raise ValueError(f"{name} must be a finite non-negative number")
