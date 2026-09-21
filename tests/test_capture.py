from __future__ import annotations

import numpy as np
import pytest

from gui_agent import capture as capture_module
from gui_agent.capture import ScreenCapture, map_box, map_point, resize_image


class FakeMss:
    def __init__(self) -> None:
        self.monitors = [
            {"left": 0, "top": 0, "width": 6, "height": 4},
            {"left": 10, "top": 20, "width": 3, "height": 2},
        ]
        self.closed = False
        self.grab_count = 0

    def grab(self, monitor: dict[str, int]) -> np.ndarray:
        self.grab_count += 1
        image = np.zeros((monitor["height"], monitor["width"], 4), dtype=np.uint8)
        image[..., :3] = (10, 20, 30)
        image[..., 3] = 255
        return image

    def close(self) -> None:
        self.closed = True


def test_capture_returns_bgr_frame_and_reuses_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = FakeMss()
    monkeypatch.setattr(capture_module.mss, "mss", lambda: backend)

    with ScreenCapture() as capture:
        first = capture.capture()
        second = capture.capture()
        assert capture.monitor_count == 1

    assert backend.grab_count == 2
    assert backend.closed is True
    assert first.region.left == 10
    assert first.region.top == 20
    assert first.region.size == (3, 2)
    assert first.image.shape == (2, 3, 3)
    assert first.image[0, 0].tolist() == [10, 20, 30]
    assert second.captured_at >= first.captured_at


def test_capture_rejects_invalid_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(capture_module.mss, "mss", FakeMss)

    with ScreenCapture() as capture:
        with pytest.raises(ValueError, match="available range"):
            capture.capture(2)
        with pytest.raises(TypeError, match="integer"):
            capture.capture(True)


def test_capture_requires_open_context() -> None:
    with pytest.raises(RuntimeError, match="context manager"):
        ScreenCapture().capture()


def test_resize_image_preserves_channels_and_dtype() -> None:
    image = np.zeros((600, 800, 3), dtype=np.uint8)

    resized = resize_image(image, (400, 300))

    assert resized.shape == (300, 400, 3)
    assert resized.dtype == np.uint8


@pytest.mark.parametrize(
    ("source_size", "target_size", "point", "expected"),
    [
        ((1920, 1080), (1280, 720), (960, 540), (640, 360)),
        ((2560, 1600), (1280, 800), (2560, 1600), (1280, 800)),
        ((800, 600), (1600, 1200), (0, 0), (0, 0)),
    ],
)
def test_map_point_between_resolutions(
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    point: tuple[int, int],
    expected: tuple[int, int],
) -> None:
    assert map_point(point, source_size, target_size) == expected


def test_point_round_trip() -> None:
    original = (1200, 700)
    processed = map_point(original, (2560, 1600), (1280, 800))

    assert map_point(processed, (1280, 800), (2560, 1600)) == original


def test_map_box_between_resolutions() -> None:
    assert map_box((100, 50, 500, 250), (1000, 500), (2000, 1000)) == (
        200,
        100,
        1000,
        500,
    )


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: map_point((1, 1), (0, 10), (20, 20)), "positive"),
        (lambda: map_point((101, 1), (100, 100), (20, 20)), "within"),
        (lambda: map_box((5, 5, 4, 10), (100, 100), (20, 20)), "positive"),
        (lambda: resize_image(np.array([]), (20, 20)), "non-empty"),
    ],
)
def test_invalid_geometry_is_rejected(call: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
