"""Desktop perception and control building blocks for GUI Agent."""

from .capture import (
    ScreenCapture,
    ScreenFrame,
    ScreenRegion,
    capture_screen,
    map_box,
    map_point,
    resize_image,
)
from .perception import (
    EasyOcrRecognizer,
    TextElement,
    draw_text_boxes,
    map_text_elements,
    normalize_ocr_results,
)
from .control import DesktopController

__all__ = [
    "ScreenCapture",
    "ScreenFrame",
    "ScreenRegion",
    "capture_screen",
    "map_box",
    "map_point",
    "resize_image",
    "EasyOcrRecognizer",
    "TextElement",
    "draw_text_boxes",
    "map_text_elements",
    "normalize_ocr_results",
    "DesktopController",
]
