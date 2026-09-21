"""Cross-platform in-memory screen capture and coordinate conversion."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
import time
from typing import TypeAlias

import cv2
import mss
import numpy as np
from numpy.typing import NDArray


Size: TypeAlias = tuple[int, int]
Point: TypeAlias = tuple[float, float]
BoundingBox: TypeAlias = tuple[float, float, float, float]


@dataclass(frozen=True)
class ScreenRegion:
    """A rectangle in desktop coordinates."""

    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Screen region width and height must be positive")

    @property
    def size(self) -> Size:
        return self.width, self.height


@dataclass(frozen=True)
class ScreenFrame:
    """A captured BGR image and its location on the desktop."""

    image: NDArray[np.uint8]
    region: ScreenRegion
    captured_at: float

    def __post_init__(self) -> None:
        if not isinstance(self.image, np.ndarray):
            raise TypeError("Screen frame image must be a NumPy array")
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError("Screen frame image must have BGR shape (height, width, 3)")
        if self.image.shape[:2] != (self.region.height, self.region.width):
            raise ValueError("Screen frame image size does not match its region")


class ScreenCapture:
    """Reuse one mss session for one or more screen captures."""

    def __init__(self) -> None:
        self._backend: object | None = None

    def __enter__(self) -> ScreenCapture:
        if self._backend is not None:
            raise RuntimeError("ScreenCapture is already open")
        self._backend = mss.mss()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    @property
    def monitor_count(self) -> int:
        backend = self._require_backend()
        return max(0, len(backend.monitors) - 1)

    def capture(self, monitor_index: int = 1) -> ScreenFrame:
        """Capture a monitor as BGR; index 0 represents the virtual desktop."""

        backend = self._require_backend()
        monitors = backend.monitors
        if not isinstance(monitor_index, int) or isinstance(monitor_index, bool):
            raise TypeError("Monitor index must be an integer")
        if monitor_index < 0 or monitor_index >= len(monitors):
            raise ValueError(
                f"Monitor index {monitor_index} is outside the available range "
                f"0..{len(monitors) - 1}"
            )

        monitor = monitors[monitor_index]
        bgra = np.asarray(backend.grab(monitor))
        if bgra.ndim != 3 or bgra.shape[2] != 4:
            raise RuntimeError("mss returned an image without BGRA channels")

        image = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        region = ScreenRegion(
            left=int(monitor["left"]),
            top=int(monitor["top"]),
            width=int(monitor["width"]),
            height=int(monitor["height"]),
        )
        return ScreenFrame(image=image, region=region, captured_at=time.time())

    def _require_backend(self) -> object:
        if self._backend is None:
            raise RuntimeError("ScreenCapture must be used as a context manager")
        return self._backend


def capture_screen(monitor_index: int = 1) -> ScreenFrame:
    """Capture one screen frame without manually managing a session."""

    with ScreenCapture() as capture:
        return capture.capture(monitor_index)


def resize_image(image: NDArray[np.uint8], target_size: Size) -> NDArray[np.uint8]:
    """Resize an in-memory image to ``(width, height)``."""

    width, height = _validate_size(target_size, "Target size")
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array")
    if image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("Image must be a non-empty 2D or 3D array")

    source_height, source_width = image.shape[:2]
    interpolation = (
        cv2.INTER_AREA
        if width < source_width or height < source_height
        else cv2.INTER_LINEAR
    )
    return cv2.resize(image, (width, height), interpolation=interpolation)


def map_point(point: Point, source_size: Size, target_size: Size) -> tuple[int, int]:
    """Map an image-edge coordinate between two resolutions."""

    source_width, source_height = _validate_size(source_size, "Source size")
    target_width, target_height = _validate_size(target_size, "Target size")
    x, y = _validate_point(point)
    if not 0 <= x <= source_width or not 0 <= y <= source_height:
        raise ValueError("Point must lie within the source size")

    return (
        round(x * target_width / source_width),
        round(y * target_height / source_height),
    )


def map_box(
    box: BoundingBox,
    source_size: Size,
    target_size: Size,
) -> tuple[int, int, int, int]:
    """Map a ``(left, top, right, bottom)`` box between two resolutions."""

    if len(box) != 4 or any(not isinstance(value, Real) for value in box):
        raise TypeError("Bounding box must contain four numeric coordinates")
    left, top, right, bottom = box
    if left >= right or top >= bottom:
        raise ValueError("Bounding box must have positive width and height")

    mapped_left, mapped_top = map_point((left, top), source_size, target_size)
    mapped_right, mapped_bottom = map_point((right, bottom), source_size, target_size)
    return mapped_left, mapped_top, mapped_right, mapped_bottom


def _validate_size(size: Size, name: str) -> Size:
    if len(size) != 2:
        raise TypeError(f"{name} must contain width and height")
    width, height = size
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or isinstance(height, bool)
    ):
        raise TypeError(f"{name} width and height must be integers")
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} width and height must be positive")
    return width, height


def _validate_point(point: Point) -> Point:
    if len(point) != 2 or any(not isinstance(value, Real) for value in point):
        raise TypeError("Point must contain two numeric coordinates")
    return float(point[0]), float(point[1])
